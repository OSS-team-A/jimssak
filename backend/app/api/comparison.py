import json
from collections import Counter
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status

from ..schemas.analysis import AnalysisError
from ..schemas.comparison import (
    ComparisonCheck,
    ComparisonItem,
    ComparisonOut,
    ComparisonPhoto,
    ComparisonSelection,
    ComparisonSummary,
)
from ..schemas.comparison_chat import ComparisonChatInput, ComparisonChatReply
from ..services.analysis import StaleAnalysis
from ..services.comparison import NEEDS_CHECK, ComparisonReviewed, ComparisonTask
from ..services.inventory import load_inventory
from ..services.move_chat import ChatProviderError, ChatUnavailable
from .moves import (
    AccessibleMove,
    AnalysisDep,
    ComparisonDep,
    GuardedRoute,
    MediaStorageDep,
    MoveStoreDep,
)

router = APIRouter(prefix="/api/v1/moves", tags=["after-move comparison"],
                   route_class=GuardedRoute)
ACCESS = {
    401: {"description": "인증 정보가 없거나 유효하지 않음"},
    404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
}
STALE_RESULT = "비교 결과가 바뀌었어요. 결과를 다시 불러와 주세요."
REVIEW_FINISHED = "이미 검토를 마친 이사예요."


def _photos(snapshots: list[dict]) -> list[ComparisonPhoto]:
    return [ComparisonPhoto(time_ms=s["time_ms"], image_url=s.get("image_url"))
            for s in snapshots]


def comparison_out(task: ComparisonTask | None) -> ComparisonOut:
    empty = ComparisonSummary(ok=0, changed=0, missing=0, unchecked=0)
    if task is None:
        return ComparisonOut(status="not_started", comparison_id=None, progress=0.0,
                             demo=False, error=None, items=[], summary=empty, reviewed_at=None)
    results = task.results() if task.status == "completed" else {}
    items = []
    for listed in task.inventory if results else []:
        found = results.get(listed["item_id"])
        if found is None:
            continue
        needs_check = found["result"] in NEEDS_CHECK
        items.append(ComparisonItem(
            item_id=listed["item_id"], label_ko=listed["label_ko"], quantity=listed["quantity"],
            dimensions_cm=listed.get("dimensions_cm") or {},
            disassembly_possible=listed.get("disassembly_possible"),
            result=found["result"], reasons=found.get("reasons", []),
            before_photos=_photos(found.get("before_snapshots", [])),
            after_photos=_photos(found.get("after_snapshots", [])),
            needs_check=needs_check,
            user_check=task.checks.get(listed["item_id"]) if needs_check else None,
        ))
    summary = ComparisonSummary(
        ok=sum(i.result == "ok" for i in items),
        changed=sum(i.result == "changed" for i in items),
        missing=sum(i.result == "missing" for i in items),
        unchecked=sum(i.needs_check and i.user_check is None for i in items),
        confirmed=sum(i.user_check == "confirmed" for i in items),
        needs_attention=sum(i.needs_check and i.user_check != "confirmed" for i in items),
    )
    return ComparisonOut(
        status=task.status, comparison_id=task.comparison_id, progress=task.progress,
        demo=task.demo,
        error=AnalysisError(**json.loads(task.error_json)) if task.error_json else None,
        items=items, summary=summary,
        reviewed_at=(datetime.fromtimestamp(task.reviewed_at, UTC)
                     if task.reviewed_at is not None else None),
    )


@router.get(
    "/{move_id}/comparison",
    summary="이사 후 비교 결과 (SC-16)",
    description=(
        "이사 후 영상(`POST /moves/{move_id}/after-media`)과 확정한 이삿짐 목록의 비교 결과. "
        "`status`가 queued·running이면 2~3초마다 다시 조회한다. 완료되면 물품별로 "
        "`ok`(이상 없음) / `changed`(상태 변화 의심) / `missing`(찾지 못함)과 이사 전후 사진이 "
        "나온다. `needs_check`가 true인 물품은 사용자에게 확인 답을 받는다."
    ),
    responses=ACCESS,
)
def get_comparison(
    move: AccessibleMove, analyses: AnalysisDep, comparisons: ComparisonDep,
) -> ComparisonOut:
    job = analyses.current(move.id)
    return comparison_out(comparisons.current(move.id, job.analysis_id if job else None))


@router.post(
    "/{move_id}/comparison/retry",
    status_code=status.HTTP_202_ACCEPTED,
    summary="실패한 이사 후 비교 재시도 (SC-15F)",
    description=(
        "`comparison_id`로 현재 실패한 비교를 지정한다. 재시도 가능한 실패라면 저장해 둔 "
        "이사 후 영상으로 새 비교를 시작한다. 같은 영상에서 다시 실패하면 영상을 다시 촬영한다."
    ),
    responses={**ACCESS, 409: {"description": "다른 비교가 시작되었거나 재시도할 수 없음"}},
)
def retry_comparison(
    move: AccessibleMove, body: ComparisonSelection, analyses: AnalysisDep,
    comparisons: ComparisonDep, moves: MoveStoreDep, storage: MediaStorageDep,
) -> ComparisonOut:
    job = analyses.current(move.id)
    # Settle a timed-out job before opening the transaction that starts its replacement.
    shown = comparisons.current(move.id, job.analysis_id if job else None)
    if shown is None or shown.comparison_id != body.comparison_id:
        raise HTTPException(status.HTTP_409_CONFLICT, STALE_RESULT)
    retry_unavailable = HTTPException(
        status.HTTP_409_CONFLICT, "이 비교를 다시 시작할 수 없어요. 이사 후 영상을 확인해 주세요."
    )
    if shown.status != "failed" or not json.loads(shown.error_json or "{}").get("retryable"):
        raise retry_unavailable
    with moves.db.connect() as conn:
        conn.execute("SELECT 1 FROM moves WHERE id = %s FOR UPDATE", (move.id,))
        job = analyses.latest(move.id, conn=conn)
        if job is None or job.analysis_id != shown.analysis_id:
            raise HTTPException(status.HTTP_409_CONFLICT, STALE_RESULT)
        current = comparisons.latest(move.id, job.analysis_id, conn=conn)
        if current is None or current.comparison_id != shown.comparison_id:
            raise HTTPException(status.HTTP_409_CONFLICT, STALE_RESULT)
        after = moves.get_after_video(move.id, conn=conn)
        inventory = load_inventory(analyses.db, job, conn=conn, for_share=True)
        if (
            after is None or after.id != current.after_media_id
            or inventory.status != "confirmed" or not storage.path(after).is_file()
        ):
            raise retry_unavailable
        cancelled = comparisons.cancel_for_move(move.id, conn=conn)
        task = comparisons.start(
            move.id, job.analysis_id, job.media_id, after.id,
            [item.model_dump(mode="json") for item in inventory.items],
            demo=current.demo, demo_fail=current.demo_fail, conn=conn,
        )
    for comparison_id in cancelled:
        storage.discard_snapshots(move.id, comparison_id)
    return comparison_out(task)


def _completed(move, analyses, comparisons, shown: str) -> ComparisonTask:
    """The current comparison, which must be finished and the one the screen is showing."""
    job = analyses.current(move.id)
    task = comparisons.current(move.id, job.analysis_id if job else None)
    if task is not None and task.comparison_id != shown:
        raise HTTPException(status.HTTP_409_CONFLICT, STALE_RESULT)
    if task is None or task.status != "completed":
        raise HTTPException(status.HTTP_409_CONFLICT, "이사 후 비교가 끝난 뒤에 확인할 수 있어요.")
    return task


def _chat_candidates(task: ComparisonTask) -> list[dict]:
    results = task.results()
    label_counts = Counter(item["label_ko"] for item in task.inventory)
    candidates = []
    for number, item in enumerate(task.inventory, start=1):
        item_id = item["item_id"]
        found = results.get(item_id)
        if found is None or found["result"] not in NEEDS_CHECK:
            continue
        label = item["label_ko"]
        candidates.append({
            "item_id": item_id, "item_number": number, "label_ko": label,
            "same_name_count": label_counts[label],
            "quantity": item["quantity"], "dimensions_cm": item.get("dimensions_cm"),
            "result": found["result"], "user_check": task.checks.get(item_id),
        })
    return candidates


@router.post(
    "/{move_id}/comparison/chat",
    summary="채팅으로 이사 후 물품 확인 답 제안 (SC-16)",
    description=(
        "현재 `comparison_id`와 사용자 메시지, 필요하면 최근 대화 8개를 보낸다. "
        "확인 대상의 `confirmed` 또는 `not_found` 답만 제안한다. "
        "`checks`가 비어 있으면 질문이나 안내를 표시한다. 사용자가 제안을 확인하면 "
        "각 물품의 기존 PATCH에 `comparison_id`와 `user_check`를 보내 저장한다."
    ),
    responses={
        **ACCESS,
        409: {"description": "비교 미완료·변경 또는 이미 검토 완료"},
        502: {"description": "채팅 모델 호출 실패 또는 올바르지 않은 제안"},
        503: {"description": "서버 채팅 모델 설정이 누락됨"},
    },
)
def suggest_comparison_checks(
    move: AccessibleMove, body: ComparisonChatInput, analyses: AnalysisDep,
    comparisons: ComparisonDep, request: Request,
) -> ComparisonChatReply:
    task = _completed(move, analyses, comparisons, body.comparison_id)
    if task.reviewed_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, REVIEW_FINISHED)
    candidates = _chat_candidates(task)
    if not candidates:
        return ComparisonChatReply(
            comparison_id=task.comparison_id, message="따로 확인할 물품이 없어요."
        )

    try:
        proposal = request.app.state.move_chat.interpret_comparison(body, candidates)
    except ChatUnavailable as exc:
        raise HTTPException(503, "이사 후 확인 채팅이 아직 설정되지 않았어요.") from exc
    except ChatProviderError as exc:
        raise HTTPException(502, "답변을 가져오지 못했어요. 다시 시도해 주세요.") from exc
    # A retake or review completion may occur while the model is responding.
    current = _completed(move, analyses, comparisons, body.comparison_id)
    if current.reviewed_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, REVIEW_FINISHED)
    eligible_ids = {candidate["item_id"] for candidate in candidates}
    if any(check.item_id not in eligible_ids for check in proposal.checks):
        raise HTTPException(502, "현재 확인 대상에 맞는 답변을 제안하지 못했어요.")
    return ComparisonChatReply(
        comparison_id=current.comparison_id, message=proposal.message, checks=proposal.checks,
    )


@router.patch(
    "/{move_id}/comparison/items/{item_id}",
    summary="물품 확인 답 저장 (SC-16)",
    description=(
        "확인이 필요한 물품(`needs_check: true`)에 사용자 답을 저장한다. "
        "`confirmed`: 직접 확인했어요, `not_found`: 아직 못 찾았어요. 검토를 마치기 전까지는 "
        "답을 바꿀 수 있다. 저장 후의 전체 비교 결과를 돌려준다."
    ),
    responses={
        **ACCESS,
        404: {"description": "이사가 없거나 비교 결과에 없는 물품"},
        409: {"description": "비교가 끝나지 않음, 화면의 비교가 바뀜(`comparison_id` 불일치), "
                           "확인이 필요 없는 물품, 또는 검토를 이미 마침"},
    },
)
def check_item(
    move: AccessibleMove, item_id: str, body: ComparisonCheck,
    analyses: AnalysisDep, comparisons: ComparisonDep,
) -> ComparisonOut:
    task = _completed(move, analyses, comparisons, body.comparison_id)
    found = task.results().get(item_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "비교 결과에 없는 물품입니다.")
    if found["result"] not in NEEDS_CHECK:
        raise HTTPException(status.HTTP_409_CONFLICT, "이상 없는 물품은 확인할 필요가 없어요.")
    try:
        saved = comparisons.set_check(task.comparison_id, item_id, body.user_check)
    except StaleAnalysis as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, STALE_RESULT) from exc
    except ComparisonReviewed as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, REVIEW_FINISHED) from exc
    return comparison_out(saved)


@router.post(
    "/{move_id}/comparison/complete",
    summary="이사 후 검토 마치기 (SC-16)",
    description=(
        "검토를 마친다(`reviewed_at` 기록, 나의 이사 기록 `stage: done`). 미응답·"
        "아직 못 찾은 물품은 확인 필요로 남긴다. 이미 마쳤으면 그대로 성공한다."
    ),
    responses={
        **ACCESS,
        409: {"description": "비교가 끝나지 않았거나 화면의 비교가 바뀌었음"},
    },
)
def complete_review(
    move: AccessibleMove, body: ComparisonSelection, analyses: AnalysisDep,
    comparisons: ComparisonDep,
) -> ComparisonOut:
    task = _completed(move, analyses, comparisons, body.comparison_id)
    try:
        saved = comparisons.finish_review(task.comparison_id)
    except StaleAnalysis as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, STALE_RESULT) from exc
    return comparison_out(saved)

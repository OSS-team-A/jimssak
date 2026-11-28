import json
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status

from ..schemas.analysis import AnalysisError
from ..schemas.comparison import (
    ComparisonCheck,
    ComparisonComplete,
    ComparisonItem,
    ComparisonOut,
    ComparisonPhoto,
    ComparisonSummary,
)
from ..services.comparison import NEEDS_CHECK, ComparisonReviewed, ComparisonTask
from .moves import AccessibleMove, AnalysisDep, ComparisonDep, GuardedRoute

router = APIRouter(prefix="/api/v1/moves", tags=["after-move comparison"],
                   route_class=GuardedRoute)
ACCESS = {
    401: {"description": "인증 정보가 없거나 유효하지 않음"},
    404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
}


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


def _completed(move, analyses, comparisons, shown: str) -> ComparisonTask:
    """The current comparison, which must be finished and the one the screen is showing."""
    job = analyses.current(move.id)
    task = comparisons.current(move.id, job.analysis_id if job else None)
    if task is not None and task.comparison_id != shown:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "비교 결과가 바뀌었어요. 결과를 다시 불러와 주세요."
        )
    if task is None or task.status != "completed":
        raise HTTPException(status.HTTP_409_CONFLICT, "이사 후 비교가 끝난 뒤에 확인할 수 있어요.")
    return task


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
        comparisons.set_check(task.comparison_id, item_id, body.user_check)
    except ComparisonReviewed as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 검토를 마친 이사예요.") from exc
    return comparison_out(comparisons.get(task.comparison_id))


@router.post(
    "/{move_id}/comparison/complete",
    summary="이사 후 검토 마치기 (SC-16)",
    description=(
        "확인이 필요한 물품에 모두 답했으면 검토를 마친다(`reviewed_at` 기록, 나의 이사 기록 "
        "`stage: done`). 이미 마쳤으면 그대로 성공한다."
    ),
    responses={
        **ACCESS,
        409: {"description": "비교가 끝나지 않음, 화면의 비교가 바뀜, 또는 아직 답하지 "
                           "않은 물품이 있음"},
    },
)
def complete_review(
    move: AccessibleMove, body: ComparisonComplete, analyses: AnalysisDep,
    comparisons: ComparisonDep,
) -> ComparisonOut:
    task = _completed(move, analyses, comparisons, body.comparison_id)
    needs_check = [item_id for item_id, found in task.results().items()
                   if found["result"] in NEEDS_CHECK]
    if not comparisons.finish_review(task.comparison_id, needs_check):
        raise HTTPException(status.HTTP_409_CONFLICT, "확인이 필요한 물품에 모두 답해주세요.")
    return comparison_out(comparisons.get(task.comparison_id))

import json
from datetime import UTC, datetime

from fastapi import APIRouter

from ..schemas.analysis import AnalysisError
from ..schemas.comparison import (
    ComparisonItem,
    ComparisonOut,
    ComparisonPhoto,
    ComparisonSummary,
)
from ..services.comparison import NEEDS_CHECK, ComparisonTask
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

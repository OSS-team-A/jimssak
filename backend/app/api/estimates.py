from datetime import date
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from ..schemas.estimates import EstimateOptions, MoveEstimate
from ..services.estimates import EstimateCapacityExceeded, calculate_estimate
from ..services.inventory import load_inventory
from .moves import AccessibleMove, AnalysisDep, GuardedRoute, MoveStoreDep, _day_info

router = APIRouter(prefix="/api/v1/moves", tags=["estimates"], route_class=GuardedRoute)


@router.post(
    "/{move_id}/estimate",
    summary="SC-09 원룸 당일 이사 카드 예상가",
    description=(
        "이 이사의 확정 목록과 저장된 주소·도로거리·일정으로 아정당과 예스2404의 "
        "일반·반포장·포장 카드 금액을 원 단위 숫자로 반환한다. 본문은 선택이다. "
        "아정당 반포장·포장은 예상 범위의 중간값이며, 일반은 공개 시작가를 기준으로 산정한다. "
        "목록 부피 10㎥당 1톤 차량 1대를 기준으로 산정한다. 누락값과 계산에 쓴 가정은 "
        "`assumptions`에 담는다. 사용자 화면에는 ‘데모 예상가’를 표시한다."
    ),
    responses={
        401: {"description": "인증 정보가 없거나 유효하지 않음"},
        404: {"description": "이사가 없거나 접근할 수 없음"},
        409: {"description": "목록 미확정, 당일 수령 아님 또는 지원하는 적재량 초과"},
        422: {"description": "잘못된 이사 ID·작업 옵션이나 현재 목록에 없는 돌침대"},
    },
)
def create_estimate(
    move: AccessibleMove,
    moves: MoveStoreDep,
    analyses: AnalysisDep,
    body: Annotated[EstimateOptions | None, Body()] = None,
) -> MoveEstimate:
    options = body or EstimateOptions()
    with moves.db.connect() as db:
        db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        job = analyses.latest(move.id, conn=db)
        inventory = load_inventory(moves.db, job, conn=db)
        if inventory.status != "confirmed":
            raise HTTPException(409, "견적을 계산하기 전에 이삿짐 목록을 확정해 주세요.")
        places, schedule = moves.get_details(move.id, conn=db)
        review = moves.get_review(move.id, conn=db)
        if (
            schedule and review and review.delivery_date
            and review.delivery_date != date.fromisoformat(schedule.move_date)
        ):
            raise HTTPException(409, "당일 수령하는 이사만 견적을 계산할 수 있습니다.")
        route = moves.get_route(move.id, conn=db)
        lookups = moves.get_lookups(move.id, conn=db)
        date_info = moves.get_date_info(move.id, conn=db)

    items = {item.item_id: item for item in inventory.items}
    day = _day_info(date_info) if (
        schedule and date_info and schedule.move_date == date_info.move_date
    ) else None
    try:
        return calculate_estimate(places, lookups, route, schedule, day, options, items)
    except EstimateCapacityExceeded as error:
        raise HTTPException(409, str(error)) from error

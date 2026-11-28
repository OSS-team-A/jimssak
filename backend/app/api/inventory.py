from contextlib import contextmanager
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from ..schemas.analysis import MAX_INVENTORY_ITEMS
from ..schemas.inventory import Inventory, InventoryItem, ItemCreate, ItemPatch
from ..services.inventory import edit_inventory, load_inventory
from .moves import AccessibleMove, AnalysisDep

router = APIRouter(
    prefix="/api/v1/moves/{move_id}/inventory",
    tags=["inventory"],
    responses={
        401: {"description": "로그인 또는 비회원 토큰이 없거나 로그인 토큰이 유효하지 않음"},
        404: {"description": "이사나 물품을 찾을 수 없거나 이사에 접근할 수 없음"},
    },
)
EDIT_CONFLICT = {
    409: {"description": "영상 분석 미완료, 목록 변경 또는 이미 확정된 목록"},
}
CONFIRM_CONFLICT = {409: {"description": "영상 분석 미완료 또는 목록 변경"}}


@router.get(
    "", summary="이삿짐 목록 조회",
    description="최신 영상의 목록과 상태를 조회한다. 목록이 준비되기 전에는 `empty`를 반환한다.",
)
def get_inventory(move: AccessibleMove, analyses: AnalysisDep) -> Inventory:
    return load_inventory(analyses.db, analyses.current(move.id))


@contextmanager
def editing(move, analyses, *, allow_confirmed=False):
    job = analyses.current(move.id)
    if job is None or job.status != "completed":
        raise HTTPException(409, "영상 분석이 완료된 후 목록을 편집할 수 있습니다.")
    with edit_inventory(analyses.db, job.analysis_id) as row:
        if row is None:
            raise HTTPException(409, "현재 영상의 이삿짐 목록을 다시 확인해주세요.")
        if row["confirmed"] and not allow_confirmed:
            raise HTTPException(409, "이미 확정한 이삿짐 목록입니다.")
        yield row


def find_item(items: list[dict], item_id: str) -> dict:
    for item in items:
        if item["item_id"] == item_id:
            return item
    raise HTTPException(404, "물품을 찾을 수 없습니다.")


def append_item(items: list[dict], body: ItemCreate) -> InventoryItem:
    if len(items) >= MAX_INVENTORY_ITEMS:
        raise HTTPException(409, f"물품은 최대 {MAX_INVENTORY_ITEMS}개까지 등록할 수 있습니다.")
    item = InventoryItem(item_id=uuid4().hex, source="user", **body.model_dump())
    items.append(item.model_dump())
    return item


def patch_item(item: dict, body: ItemPatch) -> InventoryItem:
    changes = body.model_dump(exclude_unset=True)
    if "dimensions_cm" in changes:
        item["dimensions_cm"].update(changes.pop("dimensions_cm"))
    item.update(changes)
    return InventoryItem.model_validate(item)


@router.post(
    "/items", status_code=201, summary="물품 추가",
    description=f"확정 전 목록에 물품을 추가한다. 최대 {MAX_INVENTORY_ITEMS}개까지 등록할 수 있다.",
    responses={409: {"description": "목록을 편집할 수 없거나 물품 등록 한도 초과"}},
)
def add_item(move: AccessibleMove, analyses: AnalysisDep, body: ItemCreate) -> InventoryItem:
    with editing(move, analyses) as row:
        return append_item(row["items"], body)


@router.patch(
    "/items/{item_id}", summary="수량·치수·특이사항·분해 여부 수정",
    description="확정 전 목록에서 요청에 포함한 필드만 수정하고 수정된 물품을 반환한다.",
    responses=EDIT_CONFLICT,
)
def update_item(
    move: AccessibleMove, analyses: AnalysisDep, item_id: str, body: ItemPatch,
) -> InventoryItem:
    with editing(move, analyses) as row:
        return patch_item(find_item(row["items"], item_id), body)


@router.delete(
    "/items/{item_id}", status_code=204, summary="물품 삭제",
    description="확정 전 목록에서 물품을 삭제한다. 성공 시 응답 본문은 없다.",
    responses=EDIT_CONFLICT,
)
def delete_item(move: AccessibleMove, analyses: AnalysisDep, item_id: str):
    with editing(move, analyses) as row:
        row["items"].remove(find_item(row["items"], item_id))


@router.post(
    "/confirm", status_code=204, summary="현재 이삿짐 목록 확정",
    description="목록을 `confirmed` 상태로 전환한다. 성공 시 응답 본문은 없다.",
    responses=CONFIRM_CONFLICT,
)
def confirm_inventory(move: AccessibleMove, analyses: AnalysisDep):
    with editing(move, analyses, allow_confirmed=True) as row:
        row["confirmed"] = True


@router.delete(
    "/confirm", status_code=204, summary="이삿짐 목록 확정 취소",
    description="목록을 `draft` 상태로 전환해 다시 편집할 수 있게 한다. 성공 시 응답 본문은 없다.",
    responses=CONFIRM_CONFLICT,
)
def unconfirm_inventory(move: AccessibleMove, analyses: AnalysisDep):
    with editing(move, analyses, allow_confirmed=True) as row:
        row["confirmed"] = False

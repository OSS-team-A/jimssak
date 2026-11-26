from contextlib import contextmanager
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from ..schemas.analysis import MAX_INVENTORY_ITEMS
from ..schemas.inventory import Inventory, InventoryItem, ItemCreate, ItemPatch
from ..services.inventory import edit_inventory, load_inventory
from .moves import AccessibleMove, AnalysisDep

router = APIRouter(prefix="/api/v1/moves/{move_id}/inventory", tags=["inventory"])


@router.get("", summary="이삿짐 목록 조회")
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


@router.post("/items", status_code=201, summary="물품 추가")
def add_item(move: AccessibleMove, analyses: AnalysisDep, body: ItemCreate) -> InventoryItem:
    item = InventoryItem(item_id=uuid4().hex, source="user", **body.model_dump())
    with editing(move, analyses) as row:
        if len(row["items"]) >= MAX_INVENTORY_ITEMS:
            raise HTTPException(409, f"물품은 최대 {MAX_INVENTORY_ITEMS}개까지 등록할 수 있습니다.")
        row["items"].append(item.model_dump())
    return item


@router.patch("/items/{item_id}", summary="수량·치수·특이사항·분해 여부 수정")
def update_item(
    move: AccessibleMove, analyses: AnalysisDep, item_id: str, body: ItemPatch,
) -> InventoryItem:
    with editing(move, analyses) as row:
        item = find_item(row["items"], item_id)
        changes = body.model_dump(exclude_unset=True)
        if "dimensions_cm" in changes:
            item["dimensions_cm"].update(changes.pop("dimensions_cm"))
        item.update(changes)
        return InventoryItem.model_validate(item)


@router.delete("/items/{item_id}", status_code=204, summary="물품 삭제")
def delete_item(move: AccessibleMove, analyses: AnalysisDep, item_id: str):
    with editing(move, analyses) as row:
        row["items"].remove(find_item(row["items"], item_id))


@router.post("/confirm", status_code=204, summary="현재 이삿짐 목록 확정")
def confirm_inventory(move: AccessibleMove, analyses: AnalysisDep):
    with editing(move, analyses, allow_confirmed=True) as row:
        row["confirmed"] = True


@router.delete("/confirm", status_code=204, summary="이삿짐 목록 확정 취소")
def unconfirm_inventory(move: AccessibleMove, analyses: AnalysisDep):
    with editing(move, analyses, allow_confirmed=True) as row:
        row["confirmed"] = False

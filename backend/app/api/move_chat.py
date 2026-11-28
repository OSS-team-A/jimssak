import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Request
from psycopg import Connection
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from ..schemas.inventory import Inventory
from ..schemas.move_chat import (
    ChatAddItem,
    ChatApplied,
    ChatApply,
    ChatChanges,
    ChatEditItem,
    ChatInput,
    ChatInterpretation,
    ChatReply,
)
from ..schemas.moves import MoveUpdate, PlaceInput, ReviewDetails, ScheduleInput
from ..services.analysis import AnalysisJob, AnalysisStore
from ..services.inventory import edit_inventory, inventory_from_row, load_inventory
from ..services.move_chat import ChatProviderError, ChatUnavailable
from ..services.moves import MoveStore, PlaceRecord, ScheduleRecord, place_ladder
from .inventory import append_item, find_item, patch_item
from .moves import (
    AccessibleMove,
    AnalysisDep,
    CalendarDep,
    MoveStoreDep,
    RegisterDep,
    RouterDep,
    prepare_move_update,
    validate_move_review,
)

router = APIRouter(prefix="/api/v1/moves/{move_id}/chat", tags=["move chat"])
STALE = "목록이나 이사 정보가 바뀌었어요. 메시지를 다시 보내 변경 내용을 확인해주세요."
ACCESS_RESPONSES = {
    401: {"description": "인증 정보가 없거나 유효하지 않음"},
    404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
}
EXAMPLE_PROPOSAL_ID = "f130b544-5aed-4d25-b342-5cbf0becde89"


@dataclass
class ReviewState:
    inventory: Inventory
    places: dict[str, PlaceRecord]
    schedule: ScheduleRecord | None
    review: ReviewDetails | None

    def fingerprint(self, analysis_id: str) -> str:
        data = {
            "analysis_id": analysis_id,
            "inventory": self.inventory.model_dump(mode="json"),
            "places": {role: asdict(place) for role, place in self.places.items()},
            "schedule": asdict(self.schedule) if self.schedule else None,
            "review": self.review.model_dump(mode="json") if self.review else None,
        }
        return sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def model_context(self) -> dict:
        return {
            "items": [item.model_dump(mode="json", exclude={"snapshots", "confidence"})
                      for item in self.inventory.items],
            "schedule": asdict(self.schedule) if self.schedule else None,
            "review": self.review.model_dump(mode="json") if self.review else None,
            "places": {role: {"floor": place.floor, "elevator": place.elevator,
                              "ladder": place_ladder(place, None)[0]}
                       for role, place in self.places.items()},
        }


def _current_job(analyses: AnalysisStore, move_id: str) -> AnalysisJob:
    job = analyses.current(move_id)
    if job is None or job.status != "completed":
        raise HTTPException(409, "영상 분석이 완료된 후 이삿짐을 검토할 수 있어요.")
    return job


def _state(
    moves: MoveStore, move_id: str, inventory: Inventory, conn: Connection | None = None,
) -> ReviewState:
    places, schedule = moves.get_details(move_id, conn=conn)
    return ReviewState(inventory, places, schedule, moves.get_review(move_id, conn=conn))


def _apply_items(items: list[dict], changes: ChatChanges) -> list[str]:
    summary = []
    seen = set()
    for operation in changes.items:
        if isinstance(operation, ChatAddItem):
            item = append_item(items, operation.item)
            details = _item_summary(
                {"dimensions_cm": {}}, item.model_dump(),
                operation.item.model_fields_set - {"label_ko", "quantity"},
            )
            suffix = f" · {', '.join(details)}" if details else ""
            summary.append(f"{item.label_ko} {item.quantity}개 추가{suffix}")
            continue
        target = find_item(items, operation.item_id)
        if operation.item_id in seen:
            raise HTTPException(422, "같은 품목의 변경을 한 항목에 모아주세요.")
        seen.add(operation.item_id)
        label = target["label_ko"]
        if isinstance(operation, ChatEditItem):
            before = deepcopy(target)
            item = patch_item(target, operation.changes)
            details = _item_summary(before, item.model_dump(), operation.changes.model_fields_set)
            if details:
                summary.append(f"{label} · " + ", ".join(details))
        else:
            items.remove(target)
            summary.append(f"{label} {target['quantity']}개 제외")
    return summary


def _item_summary(before: dict, after: dict, fields: set[str]) -> list[str]:
    details = []
    if "label_ko" in fields and before["label_ko"] != after["label_ko"]:
        details.append(f"이름 → {after['label_ko']}")
    if "quantity" in fields and before["quantity"] != after["quantity"]:
        details.append(f"수량 {before['quantity']}개 → {after['quantity']}개")
    if "dimensions_cm" in fields:
        before_dimensions, after_dimensions = before["dimensions_cm"], after["dimensions_cm"]
        for axis, label in (("width", "가로"), ("depth", "깊이"), ("height", "높이")):
            if before_dimensions.get(axis) != after_dimensions.get(axis):
                value = after_dimensions.get(axis)
                details.append(f"{label} → {f'{value:g}cm' if value is not None else '미입력'}")
    if "notes" in fields and before.get("notes") != after["notes"]:
        details.append(f"특이사항 → {after['notes'] or '삭제'}")
    for field, label in (
        ("disassembly_possible", "분리 가능"), ("disassembly_required", "분리 필요"),
    ):
        if field in fields and before.get(field) != after[field]:
            value = {True: "예", False: "아니요", None: "확인 필요"}[after[field]]
            details.append(f"{label} → {value}")
    return details


def _additions_only(changes: ChatChanges) -> bool:
    return (
        bool(changes.items)
        and all(isinstance(item, ChatAddItem) for item in changes.items)
        and not any((changes.origin, changes.destination, changes.move_date,
                     changes.start_time, changes.review))
    )


def _has_size(value: str) -> bool:
    return bool(re.search(
        r"\d+(?:\.\d+)?\s*(?:mm|cm|인치|inch)|"
        r"\d+(?:\.\d+)?\s*[x×*]\s*\d+|"
        r"(?:가로|세로|깊이|높이)\s*\d+",
        value, re.IGNORECASE,
    ))


def _missing_addition_sizes(changes: ChatChanges, message: str) -> list[str]:
    missing = []
    for change in changes.items:
        if not isinstance(change, ChatAddItem):
            continue
        item = change.item
        known = any(value is not None for value in item.dimensions_cm.model_dump().values())
        known |= _has_size(item.notes or "")
        if len(changes.items) == 1:
            known |= _has_size(message)
        if not known:
            missing.append(item.label_ko)
    return missing


def _move_update(
    changes: ChatChanges, state: ReviewState, moves: MoveStore, move_id: str,
) -> tuple[MoveUpdate | None, list[str]]:
    updates = {}
    summary = []
    lookups = moves.get_lookups(move_id) if changes.origin or changes.destination else {}
    for role, title in (("origin", "출발지"), ("destination", "도착지")):
        proposed = getattr(changes, role)
        if proposed is None:
            continue
        new_fields = proposed.model_dump(exclude_unset=True)
        place = state.places.get(role)
        if not new_fields:
            continue
        same_place = place and (new_fields.get("road_address", place.road_address)
                                == place.road_address)
        values = {}
        if same_place:
            values = {"road_address": place.road_address, "detail_address": place.detail_address,
                      "floor": place.floor}
            lookup = lookups.get(role)
            if lookup and lookup.adm_cd and lookup.lot_main is not None:
                values.update(adm_cd=lookup.adm_cd, lot_main=lookup.lot_main,
                              lot_sub=lookup.lot_sub or 0, mountain=lookup.mountain)
        values.update(new_fields)
        updates[role] = PlaceInput.model_validate(values)
        labels = {"road_address": "주소", "detail_address": "상세 주소", "floor": "층",
                  "elevator": "승강기", "ladder": "사다리차"}
        for name, value in new_fields.items():
            if name == "elevator":
                value = {"available": "있음", "unavailable": "없음",
                         "unknown": "확인 필요", None: "확인 필요"}[value]
            elif name == "ladder":
                value = {"use": "사용", "not_use": "사용 안 함", None: "기본값"}[value]
            summary.append(f"{title} {labels[name]} → {value if value is not None else '미입력'}")
        if place and not same_place:
            summary.append(f"{title} 변경 후 상세 주소·층·승강기 정보 확인 필요")
    if changes.move_date or changes.start_time:
        current = asdict(state.schedule) if state.schedule else {}
        if changes.move_date:
            current["move_date"] = changes.move_date
            summary.append(f"짐 빼는 날 → {changes.move_date.isoformat()}")
        if changes.start_time:
            current["start_time"] = changes.start_time
            summary.append(f"이사 시작 시간 → {changes.start_time:%H:%M}")
        updates["schedule"] = ScheduleInput.model_validate(current)
    if changes.review and changes.review.model_fields_set:
        patch = changes.review.model_dump(exclude_unset=True)
        current = state.review.model_dump() if state.review else {}
        updates["review"] = ReviewDetails.model_validate(current | patch)
        if "delivery_date" in patch:
            summary.append(f"짐 받는 날 → {patch['delivery_date'] or '미입력'}")
        if "notes" in patch:
            summary.append(f"이사 특이사항 → {patch['notes'] or '삭제'}")
    if not updates:
        return None, summary
    update = MoveUpdate(**updates)
    validate_move_review(update, state.schedule, state.review)
    return update, summary


@router.post(
    "",
    summary="SC-07 채팅으로 목록·이사 정보 변경 제안",
    description=(
        "분석을 완료한 이사의 목록이 `draft`일 때 사용한다. 품목 추가·제외·수정과 "
        "출발지·도착지·일정·특이사항을 한 메시지에서 요청할 수 있다. "
        "회원은 `Authorization: Bearer`, 비회원은 `X-Guest-Token`을 보낸다.\n\n"
        "크기가 확인된 물품 추가만 요청한 경우 즉시 저장하고 `inventory`에 최신 목록을 반환한다. "
        "크기가 빠진 추가 요청은 필요한 크기를 질문한다. 다른 변경은 "
        "`proposal_id`가 있으면 `summary`로 변경 내용을 표시하고, 사용자 확인 후 "
        "`POST /api/v1/moves/{move_id}/chat/apply`를 호출한다. "
        "`proposal_id`가 null이면 `message`와 `choices`를 채팅에 표시한다. "
        "`advance`가 true면 다음 검토 질문으로 이동한다."
    ),
    responses={
        **ACCESS_RESPONSES,
        200: {
            "description": "변경 제안 또는 채팅 답변",
            "content": {"application/json": {"examples": {
                "proposal": {
                    "summary": "수량 변경안",
                    "value": {
                        "message": "이 내용을 반영할까요?",
                        "proposal_id": EXAMPLE_PROPOSAL_ID,
                        "summary": ["의자 · 수량 1개 → 3개"],
                        "choices": [], "advance": False,
                    },
                },
                "question": {
                    "summary": "추가 정보 요청",
                    "value": {
                        "message": "어떤 의자의 수량을 바꿀까요?",
                        "proposal_id": None, "summary": [],
                        "choices": ["식탁 의자", "책상 의자"], "advance": False,
                    },
                },
            }}},
        },
        409: {"description": "영상 분석 미완료 또는 목록을 편집할 수 없는 상태"},
        502: {"description": "모델 호출에 실패했거나 응답 형식이 올바르지 않음"},
        503: {"description": "서버의 이삿짐 채팅 설정이 누락됨"},
    },
)
def propose(
    move: AccessibleMove, moves: MoveStoreDep, analyses: AnalysisDep,
    body: Annotated[ChatInput, Body(openapi_examples={
        "quantity": {
            "summary": "품목 수량 수정",
            "value": {"message": "의자 3개로 바꿔줘"},
        },
        "follow_up": {
            "summary": "분리 가능 여부 답변",
            "value": {
                "message": "분리할 수 있어요",
                "history": [
                    {"role": "user", "content": "침대를 돌침대로 바꿔줘"},
                    {"role": "assistant", "content": "돌침대를 분리해서 옮길 수 있나요?"},
                ],
            },
        },
    })],
    request: Request,
) -> ChatReply:
    job = _current_job(analyses, move.id)
    state = _state(moves, move.id, load_inventory(analyses.db, job))
    if state.inventory.status != "draft":
        raise HTTPException(409, "이삿짐 목록을 다시 편집 상태로 열어주세요.")
    try:
        interpretation = request.app.state.move_chat.interpret(body, state.model_context())
    except ChatUnavailable as exc:
        raise HTTPException(503, "이삿짐 채팅이 아직 설정되지 않았어요.") from exc
    except ChatProviderError as exc:
        raise HTTPException(502, "답변을 가져오지 못했어요. 다시 시도해주세요.") from exc
    changes = interpretation.changes
    if changes is None:
        return ChatReply(message=interpretation.message, choices=interpretation.choices,
                         advance=interpretation.advance)
    if _additions_only(changes):
        missing = _missing_addition_sizes(changes, body.message)
        if missing:
            names = ", ".join(missing)
            return ChatReply(
                message=f"{names}의 크기를 알려주세요. 알고 있는 치수를 cm나 mm로 적어주세요."
            )
        with edit_inventory(analyses.db, job.analysis_id) as row:
            if row is None or row["confirmed"] or inventory_from_row(row) != state.inventory:
                raise HTTPException(409, STALE)
            for change in changes.items:
                append_item(row["items"], change.item)
            saved_inventory = inventory_from_row(row)
        return ChatReply(message="이삿짐 목록에 추가했어요.", inventory=saved_inventory)
    try:
        items = [item.model_dump() for item in state.inventory.items]
        summary = _apply_items(items, changes)
        _, move_summary = _move_update(changes, state, moves, move.id)
        summary.extend(move_summary)
    except HTTPException as exc:
        return ChatReply(message=exc.detail)
    except ValidationError:
        return ChatReply(message="변경할 항목과 값을 한 번 더 정확하게 알려주세요.")
    if not summary:
        return ChatReply(message="이미 목록에 반영된 내용이에요.", advance=True)
    proposal_id = uuid4()
    with moves.db.connect() as conn:
        conn.execute(
            "INSERT INTO move_chat_proposals(move_id, proposal_id, snapshot, changes) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT(move_id) DO UPDATE SET "
            "proposal_id = excluded.proposal_id, snapshot = excluded.snapshot, "
            "changes = excluded.changes, applied = FALSE",
            (move.id, str(proposal_id), state.fingerprint(job.analysis_id),
             Jsonb(interpretation.model_dump(mode="json", exclude_unset=True))),
        )
    return ChatReply(
        message="이 내용을 반영할까요?", proposal_id=proposal_id, summary=summary,
    )


def _proposal(conn: Connection, move_id: str, proposal_id: str, *, lock=False) -> dict:
    query = "SELECT snapshot, changes, applied FROM move_chat_proposals WHERE move_id = %s "
    query += "AND proposal_id = %s"
    row = conn.execute(query + (" FOR UPDATE" if lock else ""), (move_id, proposal_id)).fetchone()
    if row is None:
        raise HTTPException(409, STALE)
    return row


@router.post(
    "/apply",
    summary="SC-07에서 확인한 변경 적용",
    description=(
        "사용자가 확인한 변경안의 `proposal_id`를 보낸다. 품목과 이사 정보 변경을 한 번에 "
        "저장하고, 저장 후의 전체 목록을 `inventory`로 반환한다. 새 변경안이 생기거나 목록·"
        "이사 정보가 바뀌어 기존 변경안을 적용할 수 없으면 409를 반환한다. "
        "이사 조건 화면은 `GET /api/v1/moves/{move_id}`로 최신 정보를 다시 불러온다."
    ),
    responses={
        **ACCESS_RESPONSES,
        200: {
            "description": "변경 적용 결과",
            "content": {"application/json": {"example": {
                "inventory": {
                    "status": "draft", "item_count": 1, "total_quantity": 3,
                    "items": [{
                        "item_id": "chair-1", "source": "demo", "label_ko": "의자",
                        "quantity": 3, "dimensions_cm": {"width": 45, "depth": 50, "height": 90},
                        "dimensions_status": "estimated", "confidence": 0.9,
                        "needs_review": False, "review_reasons": [], "notes": None,
                        "disassembly_required": None, "disassembly_possible": None,
                        "snapshots": [],
                    }],
                },
                "message": "변경 내용을 반영했어요.", "choices": [], "advance": True,
            }}},
        },
        409: {"description": "분석 미완료, 목록 확정 또는 변경안이 현재 데이터와 일치하지 않음"},
    },
)
def apply_chat(
    move: AccessibleMove, moves: MoveStoreDep, analyses: AnalysisDep,
    body: Annotated[ChatApply, Body(openapi_examples={
        "confirm": {
            "summary": "사용자가 확인한 변경안",
            "value": {"proposal_id": EXAMPLE_PROPOSAL_ID},
        },
    })],
    register: RegisterDep, route: RouterDep, calendar: CalendarDep,
) -> ChatApplied:
    job = _current_job(analyses, move.id)
    proposal_id = str(body.proposal_id)
    with moves.db.connect() as conn:
        record = _proposal(conn, move.id, proposal_id)
    interpretation = ChatInterpretation.model_validate(record["changes"])
    changes = interpretation.changes
    assert changes is not None
    applied_message = "변경 내용을 반영했어요."
    if interpretation.follow_up:
        applied_message += f" {interpretation.follow_up}"

    state = _state(moves, move.id, load_inventory(analyses.db, job))
    save = None
    if not record["applied"]:
        if state.fingerprint(job.analysis_id) != record["snapshot"]:
            raise HTTPException(409, STALE)
        try:
            update, _ = _move_update(changes, state, moves, move.id)
        except ValidationError as exc:
            raise HTTPException(409, STALE) from exc
        if update:
            save = prepare_move_update(update, move, moves, register, route, calendar)
    with moves.db.connect() as conn:
        conn.execute("SELECT 1 FROM moves WHERE id = %s FOR NO KEY UPDATE", (move.id,))
        record = _proposal(conn, move.id, proposal_id, lock=True)
        video = conn.execute("SELECT id FROM move_videos WHERE move_id = %s", (move.id,)).fetchone()
        if video is None or video["id"] != job.media_id:
            raise HTTPException(409, STALE)
        with edit_inventory(analyses.db, job.analysis_id, conn=conn) as row:
            inventory = inventory_from_row(row)
            if not record["applied"]:
                locked_state = _state(moves, move.id, inventory, conn)
                if row is None or row["confirmed"] or (
                    locked_state.fingerprint(job.analysis_id) != record["snapshot"]
                ):
                    raise HTTPException(409, STALE)
                _apply_items(row["items"], changes)
                if save:
                    save(conn)
                conn.execute("UPDATE move_chat_proposals SET applied = TRUE WHERE move_id = %s",
                             (move.id,))
                inventory = inventory_from_row(row)
    return ChatApplied(
        inventory=inventory, message=applied_message,
        choices=interpretation.choices if interpretation.follow_up else [],
        advance=not interpretation.follow_up,
    )

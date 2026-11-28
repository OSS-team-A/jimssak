from datetime import date, time
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .inventory import Inventory, ItemCreate, ItemPatch
from .moves import (
    MAX_FLOOR,
    MIN_FLOOR,
    Address,
    DetailAddress,
    Elevator,
    Ladder,
    ReviewDetails,
)

ChatText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(description="사용자 또는 짐싹")
    content: ChatText = Field(description="이전 대화 메시지")


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: ChatText = Field(description="사용자가 입력하거나 선택한 답변. 1~1000자")
    history: list[ChatTurn] = Field(
        default_factory=list, max_length=8,
        description="최근 대화 최대 8개. 오래된 메시지부터 순서대로 보내며, 첫 요청에는 생략 가능",
    )


class ChatAddItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add"]
    item: ItemCreate


class ChatEditItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["update"]
    item_id: str
    changes: ItemPatch


class ChatRemoveItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["exclude"]
    item_id: str


ItemChange = Annotated[ChatAddItem | ChatEditItem | ChatRemoveItem, Field(discriminator="action")]


class PlaceChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    road_address: Address | None = None
    detail_address: DetailAddress | None = None
    floor: int | None = Field(default=None, ge=MIN_FLOOR, le=MAX_FLOOR)
    elevator: Elevator | None = None
    ladder: Ladder | None = None


class ChatChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ItemChange] = Field(default_factory=list, max_length=20)
    origin: PlaceChanges | None = None
    destination: PlaceChanges | None = None
    move_date: date | None = None
    start_time: time | None = None
    review: ReviewDetails | None = None


class ChatInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: ChatText
    changes: ChatChanges | None = None
    follow_up: ChatText | None = None
    choices: list[Annotated[str, StringConstraints(min_length=1, max_length=40)]] = Field(
        default_factory=list, max_length=3
    )
    advance: bool = False


class ChatReply(BaseModel):
    message: str = Field(description="채팅에 표시할 문구")
    inventory: Inventory | None = Field(
        default=None, description="확인 없이 추가한 물품이 반영된 최신 목록",
    )
    proposal_id: UUID | None = Field(
        default=None, description="확인할 변경안의 ID. null이면 적용 요청 없이 답변을 표시",
    )
    summary: list[str] = Field(default_factory=list, description="사용자에게 보여줄 변경 내역")
    choices: list[str] = Field(
        default_factory=list,
        description="추가 질문의 답변 후보. 선택한 문자열을 다음 message로 전송",
    )
    advance: bool = Field(default=False, description="true면 다음 검토 질문으로 이동")


class ChatApply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: UUID = Field(description="사용자가 반영을 확인한 변경안의 ID")


class ChatApplied(BaseModel):
    inventory: Inventory = Field(description="저장 후 전체 이삿짐 목록. 화면의 목록과 요약을 갱신")
    message: str = Field(description="저장 결과와, 필요한 경우 다음에 물어볼 추가 질문")
    choices: list[str] = Field(
        default_factory=list,
        description="추가 질문의 답변 후보. 선택한 문자열을 다음 message로 전송",
    )
    advance: bool = Field(default=True, description="true면 다음 검토 질문으로 이동")

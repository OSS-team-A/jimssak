from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# Moves happen in Korea, which has no daylight saving time.
KST = timezone(timedelta(hours=9))
MIN_FLOOR, MAX_FLOOR = -5, 80

Owner = Literal["member", "guest"]
Elevator = Literal["available", "unavailable", "unknown"]
# Where the elevator value came from: the user, or the building register lookup (later).
ElevatorSource = Literal["user", "building_register"]
Address = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
DetailAddress = Annotated[str, StringConstraints(strip_whitespace=True, max_length=50)]
# Item keys in the order the chat asks for them.
Step = Literal["room_video", "origin", "destination", "schedule"]


class MoveCreated(BaseModel):
    move_id: UUID
    owner: Owner
    guest_token: str | None = Field(
        default=None,
        description="Guests only. Send it as X-Guest-Token on later requests for this move.",
    )
    expires_at: datetime | None = Field(
        default=None, description="Guests only. Extended on every request for this move."
    )


class Section(BaseModel):
    status: Literal["empty", "not_started"]


class MediaInfo(BaseModel):
    media_id: UUID
    filename: str
    content_type: Literal["video/mp4", "video/quicktime"]
    size_bytes: int
    created_at: datetime


class UploadedVideo(MediaInfo):
    replaced: bool = Field(description="True when this upload replaced an earlier video.")


class RoomVideo(BaseModel):
    """One video per move: small one-room moves are filmed in a single take."""

    status: Literal["empty", "uploaded"]
    video: MediaInfo | None = None


class PlaceInput(BaseModel):
    """One end of the move as the SC-06 form sends it."""

    model_config = ConfigDict(extra="forbid")

    road_address: Address = Field(description="선택한 도로명주소, 예: 서울 중구 퇴계로36길 2")
    detail_address: DetailAddress | None = Field(default=None, description="동·호, 예: 302호")
    floor: int | None = Field(
        default=None, ge=MIN_FLOOR, le=MAX_FLOOR,
        description="이사하는 층. 지하는 음수(지하 1층 = -1). 0은 없다.",
    )
    elevator: Elevator | None = Field(
        default=None,
        description=(
            "최종 확인에서 사용자가 승강기 값을 고칠 때만 보낸다. 보내지 않으면 주소가 같을 때 "
            "기존 값을 유지하고, 주소가 바뀌면 unknown(확인 필요)으로 돌아간다."
        ),
    )

    @field_validator("floor")
    @classmethod
    def no_zero_floor(cls, floor: int | None) -> int | None:
        if floor == 0:
            raise ValueError("0층은 없습니다. 지하 1층은 -1로 보내주세요.")
        return floor

    @field_validator("detail_address")
    @classmethod
    def blank_detail_is_none(cls, detail: str | None) -> str | None:
        return detail or None


class ScheduleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move_date: date = Field(description="이사 날짜, 예: 2026-12-20")
    start_time: time = Field(description="이사 시작 시간. 1시간 단위(정시), 예: 09:00, 13:00")

    @field_validator("move_date")
    @classmethod
    def not_in_past(cls, move_date: date) -> date:
        if move_date < datetime.now(KST).date():
            raise ValueError("지난 날짜는 선택할 수 없습니다.")
        return move_date

    @field_validator("start_time")
    @classmethod
    def on_the_hour(cls, start_time: time) -> time:
        # Movers book start times on the hour, as the SC-06 picker shows (오전 9:00).
        if start_time.minute or start_time.second or start_time.microsecond:
            raise ValueError("시작 시간은 1시간 단위(정시)로 보내주세요.")
        return start_time.replace(tzinfo=None)


class MoveUpdate(BaseModel):
    """Send only the sections that changed. A section sent as null is cleared."""

    model_config = ConfigDict(extra="forbid")

    origin: PlaceInput | None = None
    destination: PlaceInput | None = None
    schedule: ScheduleInput | None = None

    @model_validator(mode="after")
    def something_to_change(self):
        if not self.model_fields_set:
            raise ValueError("origin, destination, schedule 중 하나 이상을 보내주세요.")
        return self


class Place(BaseModel):
    road_address: str
    detail_address: str | None
    floor: int | None
    elevator: Elevator
    elevator_source: ElevatorSource | None = Field(
        description="user: 사용자 입력, building_register: 건축물대장 조회(후속). 모르면 null."
    )


class PlaceSection(BaseModel):
    status: Literal["empty", "partial", "filled"] = Field(
        description="partial: 주소는 있지만 층이 아직 없음"
    )
    value: Place | None = None


class Schedule(BaseModel):
    move_date: date
    start_time: time


class ScheduleSection(BaseModel):
    status: Literal["empty", "filled"]
    value: Schedule | None = None


class MoveData(BaseModel):
    room_video: RoomVideo
    analysis: Section
    inventory: Section
    origin: PlaceSection
    destination: PlaceSection
    schedule: ScheduleSection


class MoveSnapshot(BaseModel):
    move_id: UUID
    owner: Owner
    created_at: datetime
    expires_at: datetime | None
    stage: Literal["collecting"] = "collecting"
    data: MoveData
    missing: list[Step] = Field(description="User inputs still empty, in asking order.")
    next_step: Step | None = Field(description="First item in missing, or null when complete.")

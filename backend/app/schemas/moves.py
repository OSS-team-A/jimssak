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
ELEVATOR_DOC = (
    "이삿날 승강기를 쓸 수 있는지. 화면에는 아래 문구로 바꿔 보여 준다.\n\n"
    "| 값 | 화면 표시 | 언제 |\n"
    "|---|---|---|\n"
    "| `available` | 승강기 있음 | 대장에 승용 또는 비상용 승강기가 1대 이상, "
    "또는 사용자가 ‘있음’을 고름 |\n"
    "| `unavailable` | 승강기 없음 | 5층 이하이고 대장상 0대, 또는 사용자가 ‘없음’을 고름 |\n"
    "| `unknown` | 승강기 확인 필요 | 아직 모름: 직접 입력한 주소(조회 안 함), 조회 실패, "
    "건물 특정 실패(동 미입력), 대장 항목 누락, 6층 이상인데 대장상 0대, 주소 변경 직후 |\n\n"
    "`unknown`이면 사용자에게 있음/없음을 골라 달라고 안내한다."
)
# Where the elevator value came from: the user, or the building register lookup (later).
ElevatorSource = Literal["user", "building_register"]
Address = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
DetailAddress = Annotated[str, StringConstraints(strip_whitespace=True, max_length=50)]
BuildingName = Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]
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
            "최종 확인에서 사용자가 승강기 값을 고칠 때만 보낸다(`available` 있음 / "
            "`unavailable` 없음 / `unknown` 확인 필요). 보내지 않으면 주소가 같을 때 "
            "기존 값을 유지하고, 주소가 바뀌면 `unknown`(확인 필요)으로 돌아간다. "
            "사용자가 고친 값은 같은 주소인 동안 건축물대장 조회 결과로 덮어쓰지 않는다."
        ),
    )

    # Copied as-is from the address search result the user picked. Without them the
    # building register is not looked up (e.g. an address typed by hand).
    building_name: BuildingName | None = Field(
        default=None, description="주소 검색 후보의 building_name"
    )
    adm_cd: Annotated[str, StringConstraints(pattern=r"^\d{10}$")] | None = Field(
        default=None, description="주소 검색 후보의 adm_cd (행정구역코드 10자리)"
    )
    mountain: bool = Field(default=False, description="주소 검색 후보의 mountain (산 지번)")
    lot_main: int | None = Field(
        default=None, ge=0, le=9999, description="주소 검색 후보의 lot_main"
    )
    lot_sub: int = Field(default=0, ge=0, le=9999, description="주소 검색 후보의 lot_sub")

    @model_validator(mode="after")
    def code_and_lot_together(self):
        if (self.adm_cd is None) != (self.lot_main is None):
            raise ValueError("adm_cd와 lot_main은 주소 검색 결과에서 함께 보내주세요.")
        return self

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


class BuildingInfo(BaseModel):
    """What the building register says about this address (SC-06 최종 확인)."""

    status: Literal["found", "ambiguous", "not_found", "failed", "not_requested"] = Field(
        description=(
            "found: 건물을 찾음, ambiguous: 같은 지번에 건물이 여러 개라 특정 못 함(동 입력 필요), "
            "not_found: 대장에 없음, failed: 조회 실패(다시 저장하면 재시도), "
            "not_requested: 주소 검색 코드 없이 입력해 조회하지 않음"
        )
    )
    building_name: str | None = None
    dong_name: str | None = None
    above_ground_floors: int | None = None
    ride_elevators: int | None = Field(default=None, description="승용승강기 수")
    emergency_elevators: int | None = Field(default=None, description="비상용승강기 수")
    checked_at: datetime | None = None


class Place(BaseModel):
    road_address: str
    detail_address: str | None
    floor: int | None
    elevator: Elevator = Field(description=ELEVATOR_DOC)
    elevator_source: ElevatorSource | None = Field(
        description=(
            "승강기 값의 출처.\n\n"
            "| 값 | 의미 | 최종 확인 화면 문구 예 |\n"
            "|---|---|---|\n"
            "| `building_register` | 건축물대장 조회로 채움 | 건물 정보에서 찾았어요 · "
            "실제 조건과 다르면 수정해주세요 |\n"
            "| `user` | 사용자가 직접 고름 | (사용자가 고른 값 그대로) |\n"
            "| `null` | 아직 모름 (`elevator`가 `unknown`) | 승강기 확인 필요 |"
        )
    )
    building: BuildingInfo


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


class RouteSection(BaseModel):
    """Road distance between origin and destination, for the quote's distance rules."""

    status: Literal["not_ready", "found", "not_found", "failed"] = Field(
        description=(
            "not_ready: 출발지·도착지 중 하나가 아직 없음, found: 계산됨, "
            "not_found: 주소의 좌표를 찾지 못함(주소 확인 필요), "
            "failed: 조회 실패(같은 본문으로 다시 저장하면 재계산)"
        )
    )
    distance_m: int | None = Field(default=None, description="자동차 도로거리(m). 화면은 km로 변환")
    duration_s: int | None = Field(default=None, description="예상 이동 시간(초)")
    basis: Literal["kakao_car_recommend"] | None = Field(
        default=None, description="거리 기준: 카카오 자동차 길찾기 추천 경로"
    )
    checked_at: datetime | None = None


class MoveData(BaseModel):
    room_video: RoomVideo
    analysis: Section
    inventory: Section
    origin: PlaceSection
    destination: PlaceSection
    route: RouteSection
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

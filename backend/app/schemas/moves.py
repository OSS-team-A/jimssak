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

from .analysis import AnalysisSection
from .inventory import InventorySection

# Moves happen in Korea, which has no daylight saving time.
KST = timezone(timedelta(hours=9))
MIN_FLOOR, MAX_FLOOR = -5, 80
# A move date may be at most this many days ahead.
MAX_DAYS_AHEAD = 365

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
STEP_DOC = (
    "사용자가 아직 입력하지 않은 항목. 채팅·입력 화면이 다음에 물어볼 순서대로 나온다.\n\n"
    "| 값 | 뜻 | 채우는 API |\n"
    "|---|---|---|\n"
    "| `room_video` | 방 영상 | `POST /moves/{move_id}/media` |\n"
    "| `origin` | 출발지 주소와 층 | `PATCH /moves/{move_id}`의 `origin` |\n"
    "| `destination` | 도착지 주소와 층 | `PATCH /moves/{move_id}`의 `destination` |\n"
    "| `schedule` | 이사 날짜와 시작 시간 | `PATCH /moves/{move_id}`의 `schedule` |"
)


class MoveCreated(BaseModel):
    """이사 생성 결과."""

    move_id: UUID = Field(description="이사 ID(UUID). 이후 `/moves/{move_id}` 경로에 쓴다.")
    owner: Owner = Field(
        description="member: 로그인한 회원의 이사(계정에 영구 저장), guest: 비회원 이사(임시)"
    )
    guest_token: str | None = Field(
        default=None,
        description=(
            "비회원만. 이 이사를 다시 열 때 `X-Guest-Token` 헤더로 보낸다. 프론트는 "
            "`sessionStorage`에 보관하고, 로그인 후 `POST /moves/{move_id}/claim`으로 "
            "계정에 연결하면 지운다. 회원 이사는 null."
        ),
    )
    expires_at: datetime | None = Field(
        default=None,
        description=(
            "비회원만. 이 시각이 지나면 이사를 열 수 없다(UTC). 이 이사에 요청할 때마다 "
            "마지막 사용 시각 기준으로 다시 늘어난다(기본 24시간). 회원 이사는 null."
        ),
    )


class MediaInfo(BaseModel):
    """업로드한 방 영상."""

    media_id: UUID = Field(description="영상 ID(UUID)")
    filename: str = Field(description="사용자가 올린 파일 이름(표시용). 저장 경로와는 무관하다.")
    content_type: Literal["video/mp4", "video/quicktime"] = Field(
        description="video/mp4(.mp4) 또는 video/quicktime(.mov)"
    )
    size_bytes: int = Field(description="파일 크기(바이트)")
    created_at: datetime = Field(description="업로드 시각(UTC)")


class UploadedVideo(MediaInfo):
    """영상 업로드 결과."""

    replaced: bool = Field(
        description=(
            "이사에 이미 있던 영상을 이번 업로드로 교체했으면 true(재촬영). 첫 업로드는 false."
        )
    )


class RoomVideo(BaseModel):
    """방 영상. 원룸 이사라 이사 한 건에 영상은 하나이며, 다시 올리면 교체된다."""

    status: Literal["empty", "uploaded"] = Field(
        description="empty: 아직 영상 없음, uploaded: 영상 있음"
    )
    video: MediaInfo | None = Field(default=None, description="영상 정보. 없으면 null")


class PlaceInput(BaseModel):
    """출발지 또는 도착지 입력(SC-06).

    주소 검색(`GET /addresses`)에서 고른 후보의 `road_address`, `building_name`,
    `adm_cd`, `mountain`, `lot_main`, `lot_sub`에 사용자가 입력한 `detail_address`,
    `floor`를 더해 보낸다. 주소 검색 코드가 있으면 건축물대장을 조회해 건물 정보와
    승강기를 채운다. 보낸 항목은 통째로 저장되며 알 수 없는 칸은 422다.
    """

    model_config = ConfigDict(extra="forbid")

    road_address: Address = Field(
        description=(
            "주소 검색 후보의 road_address(1~200자). "
            "예: 서울특별시 중구 퇴계로36길 2 (필동2가)"
        )
    )
    detail_address: DetailAddress | None = Field(
        default=None,
        description=(
            "사용자가 입력한 상세주소(50자 이하). 예: 302호, 101동 502호. 아파트는 동까지 "
            "넣어야 같은 지번의 여러 동 중 해당 건물을 찾는다. 빈 문자열은 null로 저장된다."
        ),
    )
    floor: int | None = Field(
        default=None, ge=MIN_FLOOR, le=MAX_FLOOR,
        description=(
            "이사하는 층(-5~80). 지하는 음수(지하 1층 = -1)이고 0은 없다. "
            "보내지 않으면 주소만 저장되고 상태가 partial로 남는다."
        ),
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
    """이사 일정 입력(SC-06 일정 선택). 저장하면 그 날짜의 할증 조건을 판정한다."""

    model_config = ConfigDict(extra="forbid")

    move_date: date = Field(
        description="이사 날짜(YYYY-MM-DD). 한국 시간 기준 오늘부터 365일 이내. 예: 2026-12-20"
    )
    start_time: time = Field(description="이사 시작 시간. 1시간 단위(정시), 예: 09:00, 13:00")

    @field_validator("move_date")
    @classmethod
    def within_a_year(cls, move_date: date) -> date:
        today = datetime.now(KST).date()
        if move_date < today:
            raise ValueError("지난 날짜는 선택할 수 없습니다.")
        if move_date > today + timedelta(days=MAX_DAYS_AHEAD):
            # Holidays further out may not be published, and price tables change.
            raise ValueError(f"이사 날짜는 오늘부터 {MAX_DAYS_AHEAD}일 이내로 선택해 주세요.")
        return move_date

    @field_validator("start_time")
    @classmethod
    def on_the_hour(cls, start_time: time) -> time:
        # Movers book start times on the hour, as the SC-06 picker shows (오전 9:00).
        if start_time.minute or start_time.second or start_time.microsecond:
            raise ValueError("시작 시간은 1시간 단위(정시)로 보내주세요.")
        return start_time.replace(tzinfo=None)


class MoveUpdate(BaseModel):
    """이사 기본정보 저장 본문. 바뀐 항목만 보내고, null로 보낸 항목은 지운다. 하나 이상 필요."""

    model_config = ConfigDict(extra="forbid")

    origin: PlaceInput | None = Field(default=None, description="출발지. null이면 지움")
    destination: PlaceInput | None = Field(default=None, description="도착지. null이면 지움")
    schedule: ScheduleInput | None = Field(default=None, description="이사 일정. null이면 지움")

    @model_validator(mode="after")
    def something_to_change(self):
        if not self.model_fields_set:
            raise ValueError("origin, destination, schedule 중 하나 이상을 보내주세요.")
        return self


class BuildingInfo(BaseModel):
    """건축물대장(표제부) 조회 결과. SC-06 최종 확인 화면의 건물 정보."""

    status: Literal["found", "ambiguous", "not_found", "failed", "not_requested"] = Field(
        description=(
            "found: 건물을 찾음, ambiguous: 같은 지번에 건물이 여러 개라 특정 못 함(동 입력 필요), "
            "not_found: 대장에 없음, failed: 조회 실패(다시 저장하면 재시도), "
            "not_requested: 주소 검색 코드 없이 입력해 조회하지 않음"
        )
    )
    building_name: str | None = Field(
        default=None, description="대장의 건물명. 예: 대치아파트209동. 없으면 null"
    )
    dong_name: str | None = Field(
        default=None, description="대장의 동명칭. 예: 209, 717동. 단독 건물은 보통 null"
    )
    above_ground_floors: int | None = Field(
        default=None, description="지상층수(옥탑 제외). 사용자가 입력한 층과 비교할 수 있다."
    )
    ride_elevators: int | None = Field(default=None, description="대장상 승용승강기 수")
    emergency_elevators: int | None = Field(default=None, description="대장상 비상용승강기 수")
    checked_at: datetime | None = Field(
        default=None,
        description=(
            "건축물대장을 조회한 시각(UTC). 같은 지번은 다시 조회하지 않으므로, 층·상세주소만 "
            "바꿔 저장하면 이 값이 그대로다."
        ),
    )


class Place(BaseModel):
    """저장된 출발지 또는 도착지."""

    road_address: str = Field(description="도로명주소")
    detail_address: str | None = Field(description="상세주소(동·호). 없으면 null")
    floor: int | None = Field(description="이사하는 층. 지하는 음수. 아직 없으면 null")
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
    building: BuildingInfo = Field(description="건축물대장 조회 결과")


class PlaceSection(BaseModel):
    """출발지 또는 도착지 입력 상태."""

    status: Literal["empty", "partial", "filled"] = Field(
        description=(
            "empty: 입력 없음, partial: 주소는 있지만 층이 아직 없음, filled: 주소와 층 있음"
        )
    )
    value: Place | None = Field(default=None, description="저장된 값. empty면 null")


class Schedule(BaseModel):
    """저장된 이사 일정."""

    move_date: date = Field(description="이사 날짜(YYYY-MM-DD)")
    start_time: time = Field(description="시작 시간(HH:MM:SS, 정시)")


class LunarDateOut(BaseModel):
    """이사 날짜의 음력."""

    month: int = Field(description="음력 월")
    day: int = Field(description="음력 일. 9·10·19·20·29·30이면 손 없는 날")
    leap: bool = Field(description="윤달이면 true")


class DayInfoOut(BaseModel):
    """이사 날짜가 해당하는 할증 조건. 견적에서 날짜 요금을 고를 때 쓴다.

    공휴일·음력은 한국천문연구원 자료로 판정한다. 모르는 값은 false가 아니라 null이다.
    """

    weekend: bool = Field(description="토·일")
    friday: bool = Field(description="금요일 (업체에 따라 주말 요금)")
    month_end: bool = Field(description="월말: 그달 마지막 3일")
    holiday: bool | None = Field(description="공휴일·대체공휴일. 그해 공휴일을 모르면 null")
    holiday_name: str | None = Field(default=None, description="예: 기독탄신일")
    son_eomneun: bool | None = Field(
        description="손 없는 날: 음력 9·10·19·20·29·30일. 음력을 못 구하면 null"
    )
    lunar: LunarDateOut | None = Field(default=None, description="음력 날짜. 못 구하면 null")
    pre_holiday: bool | None = Field(
        description=(
            "연휴 전날: 공휴일이 하나 이상 낀 이틀 이상 쉬는 날의 전날(평범한 주말은 제외). "
            "공휴일을 모르면 null"
        )
    )
    holiday_status: Literal["confirmed", "unpublished", "failed"] = Field(
        description=(
            "confirmed: 공휴일 확인됨, unpublished: 그해 공휴일 자료가 아직 없음(공휴일 미확정), "
            "failed: 조회 실패(같은 본문으로 다시 저장하면 재조회)"
        )
    )
    lunar_status: Literal["found", "failed"] = Field(
        description="found: 음력 확인됨, failed: 조회 실패(같은 본문으로 다시 저장하면 재조회)"
    )
    checked_at: datetime = Field(description="판정한 시각(UTC). 날짜가 바뀔 때만 다시 판정한다.")


class ScheduleSection(BaseModel):
    """이사 일정 입력 상태."""

    status: Literal["empty", "filled"] = Field(description="empty: 일정 없음, filled: 일정 있음")
    value: Schedule | None = Field(default=None, description="저장된 일정. empty면 null")
    day: DayInfoOut | None = Field(
        default=None, description="이사 날짜가 해당하는 할증 조건. 일정을 저장하면 채워진다."
    )


class RouteSection(BaseModel):
    """출발지와 도착지 사이 자동차 도로거리. 견적의 기본거리 조건(예: 10km 이내) 판정에 쓴다.

    두 도로명주소 중 하나가 바뀔 때만 다시 계산한다. 직선거리로 대신하지 않는다.
    """

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
    checked_at: datetime | None = Field(default=None, description="계산한 시각(UTC)")


class MoveData(BaseModel):
    """이사 한 건의 항목별 상태. 각 항목은 status와 값으로 이루어진다."""

    room_video: RoomVideo = Field(description="방 영상(SC-05)")
    analysis: AnalysisSection = Field(description="영상 분석(SC-06 분석 대기·완료·실패)")
    inventory: InventorySection = Field(description="이삿짐 목록 상태, 항목 수와 총수량(SC-07)")
    origin: PlaceSection = Field(description="출발지(SC-06)")
    destination: PlaceSection = Field(description="도착지(SC-06)")
    route: RouteSection = Field(description="출발지→도착지 자동차 거리")
    schedule: ScheduleSection = Field(description="이사 일정과 날짜 할증 조건(SC-06)")


class MoveSnapshot(BaseModel):
    """이사 한 건의 현재 상태. 조회·저장·연결 응답이 모두 이 형태다."""

    move_id: UUID = Field(description="이사 ID(UUID)")
    owner: Owner = Field(description="member: 회원 이사, guest: 비회원 이사")
    created_at: datetime = Field(description="이사를 만든 시각(UTC)")
    expires_at: datetime | None = Field(
        description="비회원 이사만. 이 시각이 지나면 열 수 없다(UTC). 회원 이사는 null."
    )
    stage: Literal["collecting"] = Field(
        default="collecting",
        description="진행 단계. 지금은 collecting(이사 정보 모으는 중)만 있다.",
    )
    data: MoveData = Field(description="항목별 상태와 값")
    missing: list[Step] = Field(description=STEP_DOC)
    next_step: Step | None = Field(
        description="missing의 첫 항목. 채팅 화면이 다음에 물어볼 것. 모두 입력했으면 null."
    )


class MoveSummary(BaseModel):
    """나의 이사 기록(SC-13)의 카드 하나."""

    move_id: UUID = Field(description="이사 ID. 상세는 `GET /moves/{move_id}`")
    created_at: datetime = Field(description="이사를 만든 시각(UTC)")
    move_date: date | None = Field(description="이사 날짜. 아직 안 정했으면 null")
    start_time: time | None = Field(default=None, description="시작 시간. 없으면 null")
    origin_address: str | None = Field(description="출발지 도로명주소. 없으면 null")
    destination_address: str | None = Field(description="도착지 도로명주소. 없으면 null")
    stage: Literal["preparing"] = Field(
        default="preparing",
        description=(
            "진행 상태. 지금은 preparing(이사 준비 중)만 있다. 이사 후 확인 기능이 생기면 "
            "after_move(이사 후 확인 전), comparing(전후 비교 확인 중), done(이사 기록 완료)이 "
            "추가된다."
        ),
    )


class MoveList(BaseModel):
    """나의 이사 기록 목록."""

    moves: list[MoveSummary] = Field(description="최신순")
    total: int = Field(description="이 회원의 전체 이사 수(페이지와 무관)")
    limit: int = Field(description="요청한 최대 개수")
    offset: int = Field(description="건너뛴 개수")

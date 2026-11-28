from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .moves import MAX_FLOOR, MIN_FLOOR

Fee = Annotated[int, Field(strict=True, ge=0, le=10_000_000)]
Transport = Literal["ground_floor", "stairs", "elevator", "ladder", "mixed"]
VEHICLE_CAPACITY_M3 = 10
MAX_VEHICLES = 6


class EstimatePlace(BaseModel):
    """한 장소의 작업 조건. 빠진 항목은 저장 정보와 데모 단가로 계산한다."""

    model_config = ConfigDict(extra="forbid")

    method: Transport | None = Field(
        default=None,
        description=(
            "운반 방식을 직접 정할 때만 보낸다. 없으면 저장된 승강기·사다리차 값으로 정한다."
        ),
    )
    entry_floor: int = Field(default=1, ge=MIN_FLOOR, le=MAX_FLOOR)
    handling_fee_krw: Fee | None = Field(default=None, description="확인된 장비·층 작업비 합계")
    carry_distance_m: int | None = Field(default=None, ge=0, le=2_000)
    long_carry: bool = Field(
        default=False, description="현장 이송 10m 초과, 거리 미입력 시 20m 가정"
    )
    carry_fee_krw: Fee | None = Field(default=None, description="확인된 별도 현장 이송비")

    @model_validator(mode="after")
    def validate_place(self):
        if self.entry_floor == 0:
            raise ValueError("0층은 없습니다.")
        return self


class EstimateOptions(BaseModel):
    """추가 작업 선택값. 본문 없이 호출하면 저장 정보만으로 계산한다."""

    model_config = ConfigDict(extra="forbid")

    origin: EstimatePlace = Field(default_factory=EstimatePlace)
    destination: EstimatePlace = Field(default_factory=EstimatePlace)
    workers: int = Field(default=1, ge=1, le=10, description="전체 작업자 수, 기본 1명")
    extra_vehicles: int = Field(
        default=0, ge=0, le=MAX_VEHICLES - 1, description="추가 요청한 1톤 차량 최소 대수"
    )
    load_volume_m3: float | None = Field(
        default=None, gt=0, le=VEHICLE_CAPACITY_M3 * MAX_VEHICLES,
        description="확인한 적재 부피(㎥). 입력하면 목록의 외곽 부피 대신 적용한다.",
    )
    night_work: bool | None = Field(
        default=None, description="미입력 시 시작 시각으로 야간 여부 판정"
    )
    bad_weather: bool = False
    night_fee_krw: Fee | None = Field(default=None, description="확인된 야간·악천후 공통 비용")
    waiting_requested: bool = False
    waiting_minutes: int | None = Field(default=None, ge=0, le=1_440)
    confirmed_wait_minutes: int | None = Field(
        default=None, ge=0, le=1_440, description="예스2404가 확인한 고객 사유 지연 시간(분)"
    )
    deposit_krw: Fee | None = Field(default=None, description="예스2404 계약금(원)")
    waiting_fee_krw: Fee | None = None
    toll_road: bool = False
    toll_fee_krw: Fee | None = None
    other_service_fees_krw: list[Fee | None] = Field(
        default_factory=list, max_length=20,
        description="사용자가 요청한 별도 작업별 확인 금액. 금액 미확인 작업은 항목당 5만 원.",
    )


class EstimateCards(BaseModel):
    """가격 비교 카드에 표시할 금액. 단위는 원이다."""

    general: int = Field(ge=0, description="일반이사 카드 금액")
    semi_pack: int = Field(ge=0, description="반포장이사 카드 금액")
    full_pack: int = Field(ge=0, description="포장이사 카드 금액")


class MoveEstimate(BaseModel):
    """아정당과 예스2404의 시연용 카드 금액."""

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "ajd": {"general": 304000, "semi_pack": 534000, "full_pack": 734000},
        "yes2404": {"general": 330000, "semi_pack": 560000, "full_pack": 580000},
        "assumptions": [
            "출발지 현장 이송거리 10m 이내 가정",
            "도착지 현장 이송거리 10m 이내 가정",
        ],
    }]})

    ajd: EstimateCards
    yes2404: EstimateCards
    assumptions: list[str] = Field(description="가격 계산에 사용한 미확인 조건과 적재량 추정")

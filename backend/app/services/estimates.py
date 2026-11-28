from calendar import monthrange
from datetime import date, time
from fractions import Fraction
from math import ceil, isfinite

from ..schemas.estimates import (
    MAX_VEHICLES,
    VEHICLE_CAPACITY_M3,
    EstimateCards,
    EstimateOptions,
    EstimatePlace,
    MoveEstimate,
)
from ..schemas.inventory import InventoryItem
from ..schemas.moves import DayInfoOut
from .moves import LookupRecord, PlaceRecord, RouteRecord, ScheduleRecord, place_ladder

DEMO_DISTANCE_M = 18_000
ELEVATOR_FEE = 50_000
SPECIAL_WORK_ALLOWANCE = 50_000
EXTRA_VEHICLE_FEE = 200_000
SEOUL = "11"
METROS = {
    "서울": SEOUL, "서울특별시": SEOUL, "부산": "26", "부산광역시": "26",
    "대구": "27", "대구광역시": "27", "인천": "28", "인천광역시": "28",
    "광주": "29", "광주광역시": "29", "대전": "30", "대전광역시": "30",
    "울산": "31", "울산광역시": "31", "세종": "36", "세종특별자치시": "36",
}
PROVINCES = {
    "경기": "41", "경기도": "41", "강원": "51", "강원도": "51", "강원특별자치도": "51",
    "충북": "43", "충청북도": "43", "충남": "44", "충청남도": "44",
    "전북": "52", "전라북도": "52", "전북특별자치도": "52", "전남": "46", "전라남도": "46",
    "경북": "47", "경상북도": "47", "경남": "48", "경상남도": "48",
    "제주": "50", "제주도": "50", "제주특별자치도": "50",
}

# The public 1-ton transport prices are reference points. Only their difference is added.
FREIGHT_RATES = (
    (10_000, 40_000), (20_000, 45_000), (30_000, 50_000), (40_000, 60_000),
    (50_000, 70_000), (60_000, 75_000), (100_000, 100_000), (150_000, 130_000),
    (200_000, 150_000), (300_000, 220_000), (400_000, 260_000), (500_000, 310_000),
)
SEOUL_TEAM_RATES = ((10_000, 0), (20_000, 30_000), (40_000, 50_000), (60_000, 80_000))
LADDER_RATES = (
    (5, 150_000), (7, 160_000), (9, 170_000), (11, 180_000), (13, 190_000),
    (14, 200_000), (15, 210_000), (16, 220_000), (17, 230_000), (18, 250_000),
    (19, 260_000), (20, 280_000), (21, 310_000), (22, 340_000), (23, 370_000), (24, 400_000),
)


class EstimateCapacityExceeded(ValueError):
    """The load exceeds the supported 1-ton vehicle count."""


def _vehicle_count(
    items: dict[str, InventoryItem], options: EstimateOptions, assumptions: list[str],
) -> int:
    if options.load_volume_m3 is not None:
        volume = Fraction(str(options.load_volume_m3))
    else:
        volume = Fraction(0)
        missing_quantity = 0
        for item in items.values():
            dimensions = item.dimensions_cm
            width, depth, height = dimensions.width, dimensions.depth, dimensions.height
            if (
                width is None or depth is None or height is None
                or not all(isfinite(dimension) and dimension > 0
                           for dimension in (width, depth, height))
            ):
                missing_quantity += item.quantity
                continue
            volume += (
                Fraction(str(width)) * Fraction(str(depth)) * Fraction(str(height))
                * item.quantity / 1_000_000
            )
        if missing_quantity:
            assumptions.append(
                f"치수가 없거나 유효하지 않은 물품 {missing_quantity}개는 부피 합계에서 제외"
            )

    vehicles = max(1 + options.extra_vehicles, ceil(volume / VEHICLE_CAPACITY_M3))
    if vehicles > MAX_VEHICLES:
        raise EstimateCapacityExceeded(
            f"1톤 차량 {MAX_VEHICLES}대({MAX_VEHICLES * VEHICLE_CAPACITY_M3}㎥)를 "
            "넘는 짐은 별도 견적이 필요해요."
        )
    if options.load_volume_m3 is None:
        if volume == 0:
            assumptions.append(f"계산 가능한 물품 부피가 없어 1톤 차량 {vehicles}대 가정")
        else:
            assumptions.append(
                f"목록의 외곽 부피 약 {float(volume):.6g}㎥, 1톤 차량 {vehicles}대 추정. "
                "포장재·적재 여유 공간은 미포함"
            )
    return vehicles


def _municipality(place: PlaceRecord | None, lookup: LookupRecord | None) -> str | None:
    if place is None:
        return None
    parts = place.road_address.split(maxsplit=2)
    region, area = (parts + [""])[:2]
    if region in METROS:
        return f"{METROS[region]}:{area}" if area.endswith("군") else METROS[region]
    if region in PROVINCES and area.endswith(("시", "군")):
        return f"{PROVINCES[region]}:{area}"
    code = lookup.adm_cd if lookup else None
    if code and code[:2] in METROS.values():
        return code[:2]
    return None


def _freight_rate(distance: int) -> Fraction:
    if distance <= 10_000:
        return Fraction(40_000)
    for (start, low), (end, high) in zip(FREIGHT_RATES, FREIGHT_RATES[1:]):
        if distance <= end:
            return low + Fraction((distance - start) * (high - low), end - start)
    return 310_000 + Fraction((distance - 500_000) * 620, 1_000)


def _distance_fees(distance: int, origin: str | None, destination: str | None) -> tuple[int, int]:
    rate = _freight_rate(distance)
    ajd = ceil((rate - 40_000) / 1_000) * 1_000
    if origin in (None, SEOUL) and destination in (None, SEOUL):
        for limit, fee in SEOUL_TEAM_RATES:
            if distance <= limit:
                return ajd, fee
        return ajd, 80_000 + ceil((rate - 75_000) / 1_000) * 1_000
    different_city = origin is not None and destination is not None and origin != destination
    return ajd, max(ajd, 10_000) if different_city else ajd


def _place_fee(
    place: PlaceRecord | None, lookup: LookupRecord | None, work: EstimatePlace, side: str,
    assumptions: list[str],
) -> int:
    if work.carry_fee_krw is not None:
        carry = work.carry_fee_krw
    else:
        distance = work.carry_distance_m
        if distance is None:
            distance = 20 if work.long_carry else 10
            condition = "20m" if work.long_carry else "10m 이내"
            assumptions.append(f"{side} 현장 이송거리 {condition} 가정")
        carry = max(0, (distance - 1) // 10) * 10_000
    if work.handling_fee_krw is not None:
        return work.handling_fee_krw + carry

    floor = place.floor if place else None
    method = work.method
    if method is None:
        if floor is None or floor == work.entry_floor:
            method = "ground_floor"
            if floor is None:
                assumptions.append(f"{side} 층수 미확인으로 진입층에서 바로 운반 가정")
        elif place_ladder(place, lookup)[0] == "use":
            method = "ladder"
            if place_ladder(place, lookup)[1] == "default":
                assumptions.append(f"{side} 5층 이상·승강기 미확인으로 사다리차 사용 가정")
        elif place.elevator == "available":
            method = "elevator"
            assumptions.append(f"{side} 승강기 사용 가정")
        else:
            method = "stairs"
            assumptions.append(f"{side} 계단 사용 가정")

    if method == "stairs":
        if floor is None:
            floors = 1
            assumptions.append(f"{side} 계단 한 층 이동 가정")
        else:
            floors = abs(floor - work.entry_floor) - (floor * work.entry_floor < 0)
        return floors * 15_000 + carry
    if method == "ladder":
        if floor is None:
            floor = 5
            assumptions.append(f"{side} 사다리차 5층 작업 가정")
        if floor < 2:
            return SPECIAL_WORK_ALLOWANCE + carry
        return next((fee for limit, fee in LADDER_RATES if floor <= limit), 450_000) + carry
    return {"ground_floor": 0, "elevator": ELEVATOR_FEE, "mixed": SPECIAL_WORK_ALLOWANCE}[
        method
    ] + carry


def _date_rates(
    schedule: ScheduleRecord | None, day: DayInfoOut | None,
) -> tuple[int, int, int, bool]:
    if schedule is None:
        return 370_000, 0, 0, False
    moving_day = date.fromisoformat(schedule.move_date)
    friday_or_weekend = moving_day.weekday() >= 4
    month_end = moving_day.day > monthrange(moving_day.year, moving_day.month)[1] - 3
    holiday = day is not None and day.holiday is True
    popular_day = day is not None and day.son_eomneun is True
    before_holiday = day is not None and day.pre_holiday is True
    yes_base = 370_000
    if popular_day:
        yes_base = 440_000
    elif friday_or_weekend or month_end or holiday:
        yes_base = 400_000
    ajd_flat = 25_000 if friday_or_weekend or holiday or before_holiday else 0
    ajd_percent = 25 if popular_day or month_end else 0
    return yes_base, ajd_flat, ajd_percent, time.fromisoformat(schedule.start_time).hour >= 19


def _waiting_fee(options: EstimateOptions, *, yes2404: bool = False) -> int:
    if options.waiting_fee_krw is not None:
        return options.waiting_fee_krw
    if yes2404 and options.deposit_krw is not None and options.confirmed_wait_minutes is not None:
        # Yes2404 moving terms, art. 30(1): half the deposit per full hour of
        # customer-caused delay, capped at twice the deposit.
        hours = options.confirmed_wait_minutes // 60
        return min((options.deposit_krw * hours + 1) // 2, 2 * options.deposit_krw)
    minutes = options.waiting_minutes
    if minutes is None:
        minutes = options.confirmed_wait_minutes
    if minutes is None:
        minutes = 60 if options.waiting_requested else 0
    return 20_000 * ((minutes + 59) // 60)


def _common_fees(options: EstimateOptions, vehicles: int) -> int:
    total = (options.workers - 1) * 80_000 + (vehicles - 1) * EXTRA_VEHICLE_FEE
    total += (
        options.toll_fee_krw if options.toll_fee_krw is not None
        else 10_000 if options.toll_road else 0
    )
    total += sum(
        SPECIAL_WORK_ALLOWANCE if fee is None else fee for fee in options.other_service_fees_krw
    )
    return total


def calculate_estimate(
    places: dict[str, PlaceRecord],
    lookups: dict[str, LookupRecord],
    route: RouteRecord | None,
    schedule: ScheduleRecord | None,
    day: DayInfoOut | None,
    options: EstimateOptions,
    items: dict[str, InventoryItem],
) -> MoveEstimate:
    assumptions: list[str] = []
    vehicles = _vehicle_count(items, options, assumptions)
    origin, destination = places.get("origin"), places.get("destination")
    if (
        route is not None and route.status == "found" and route.distance_m is not None
        and origin is not None and destination is not None
        and (route.origin_address, route.destination_address)
        == (origin.road_address, destination.road_address)
    ):
        distance = route.distance_m
    else:
        distance = DEMO_DISTANCE_M
        assumptions.append("자동차 경로 미확인으로 거리 18km 가정")
    origin_city = _municipality(origin, lookups.get("origin"))
    destination_city = _municipality(destination, lookups.get("destination"))
    if None in (origin_city, destination_city):
        if origin_city in (None, SEOUL) and destination_city in (None, SEOUL):
            assumptions.append("지역 미확인으로 예스2404 서울 시내 거리 요율 적용")
        else:
            assumptions.append("지역 미확인으로 예스2404 시외 최소 추가비 미적용")
    ajd_distance, yes_distance = _distance_fees(
        distance, origin_city, destination_city,
    )
    if schedule is None:
        assumptions.append("일정 미입력으로 일반 평일 10시 시작 가정")
    elif day is None or day.holiday is None or day.son_eomneun is None or day.pre_holiday is None:
        assumptions.append("공휴일·연휴 전날·손 없는 날 중 미확인 조건은 날짜 할증에서 제외")
    yes_base, ajd_flat, ajd_percent, late = _date_rates(schedule, day)
    night = options.night_work if options.night_work is not None else late
    common = _common_fees(options, vehicles)
    common += _place_fee(origin, lookups.get("origin"), options.origin, "출발지", assumptions)
    common += _place_fee(
        destination, lookups.get("destination"), options.destination, "도착지", assumptions,
    )
    ajd_extras = ajd_distance + _waiting_fee(options)
    yes_extras = yes_distance + _waiting_fee(options, yes2404=True)

    def total(base: int, extras: int) -> int:
        night_fee = options.night_fee_krw
        if night_fee is None:
            night_fee = (base + 1) // 2 if night or options.bad_weather else 0
        return base + extras + common + night_fee

    def ajd_total(base: int) -> int:
        surcharge = max(ajd_flat, (base * ajd_percent + 99) // 100)
        return total(base + surcharge, ajd_extras)

    yes_date_extra = yes_base - 370_000
    return MoveEstimate(
        ajd=EstimateCards(
            general=ajd_total(120_000),
            semi_pack=(ajd_total(300_000) + ajd_total(400_000) + 1) // 2,
            full_pack=(ajd_total(500_000) + ajd_total(600_000) + 1) // 2,
        ),
        yes2404=EstimateCards(
            general=total(120_000 + yes_date_extra, yes_extras),
            semi_pack=total(350_000 + yes_date_extra, yes_extras),
            full_pack=total(yes_base, yes_extras),
        ),
        assumptions=assumptions,
    )

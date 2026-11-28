"""Building facts from 국토교통부 건축HUB 건축물대장 표제부.

One lot (지번) can hold many buildings, e.g. a whole apartment complex. We fetch every
표제부 row for the lot once, keep the rows, and pick the user's building from them, so a
changed 동 in the detail address can be re-matched without calling the API again.
"""

import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

# Only these fields are kept from each 표제부 row.
KEPT_FIELDS = (
    "mgmBldrgstPk", "mainAtchGbCd", "bldNm", "dongNm", "newPlatPlc", "mainPurpsCdNm",
    "grndFlrCnt", "rideUseElvtCnt", "emgenUseElvtCnt",
)
ROWS_PER_PAGE = 100
MAX_PAGES = 10
# The service sometimes answers 200 with an empty body or 503 SERVICETIMEOUT_ERROR.
ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 0.5
# A building this tall recorded with no elevator is more likely a gap in the record, so the
# user is asked to check. Without a usable elevator, moving at this height defaults to a
# ladder truck.
TALL_BUILDING_FLOORS = 5


@dataclass(frozen=True)
class LotCode:
    adm_cd: str  # 10 digits: 시군구코드(5) + 법정동코드(5)
    mountain: bool
    lot_main: int
    lot_sub: int

    def params(self) -> dict[str, str]:
        return {
            "sigunguCd": self.adm_cd[:5],
            "bjdongCd": self.adm_cd[5:],
            "platGbCd": "1" if self.mountain else "0",
            "bun": f"{self.lot_main:04d}",
            "ji": f"{self.lot_sub:04d}",
        }


class BuildingLookupFailed(Exception):
    """The register could not be read right now."""


class BuildingRegister(Protocol):
    def fetch(self, lot: LotCode) -> list[dict]: ...


class HubBuildingRegister:
    def __init__(self, url: str, service_key: str | None, timeout: float, budget: float):
        self.url = url
        self.service_key = service_key
        # Per request, and for the whole lookup including retries: the user is waiting
        # on the save button, and a failed lookup is retried on the next save anyway.
        self.timeout = timeout
        self.budget = budget

    def fetch(self, lot: LotCode) -> list[dict]:
        if not self.service_key:
            raise BuildingLookupFailed("DATA_GO_KR_SERVICE_KEY is not set")
        deadline = time.monotonic() + self.budget
        rows: list[dict] = []
        for page in range(1, MAX_PAGES + 1):
            items, total = self._page(lot, page, deadline)
            rows.extend(items)
            if len(rows) >= total or not items:
                return rows
        return rows

    def _page(self, lot: LotCode, page: int, deadline: float) -> tuple[list[dict], int]:
        params = {
            "serviceKey": self.service_key, **lot.params(),
            "numOfRows": ROWS_PER_PAGE, "pageNo": page, "_type": "json",
        }
        last_error = "out of time"
        for attempt in range(ATTEMPTS):
            if attempt:
                if deadline - time.monotonic() <= RETRY_PAUSE_SECONDS:
                    break
                time.sleep(RETRY_PAUSE_SECONDS)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = httpx.get(
                    self.url, params=params, timeout=min(self.timeout, remaining)
                )
                if response.status_code != 200 or not response.content:
                    last_error = f"HTTP {response.status_code}, {len(response.content)} bytes"
                    continue
                return parse_page(response.json())
            except (httpx.HTTPError, ValueError) as exc:
                last_error = type(exc).__name__
            except BuildingLookupFailed as exc:
                last_error = str(exc)
        raise BuildingLookupFailed(f"building register failed: {last_error}")


def parse_page(body: object) -> tuple[list[dict], int]:
    try:
        header = body["response"]["header"]
        if str(header.get("resultCode")) != "00":
            raise BuildingLookupFailed(f"result code {header.get('resultCode')}")
        content = body["response"]["body"]
        total = int(content.get("totalCount") or 0)
        items = (content.get("items") or {}) if isinstance(content.get("items"), dict) else {}
        found = items.get("item") or []
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise BuildingLookupFailed("unexpected building register response") from exc
    if isinstance(found, dict):
        found = [found]
    return [{key: row.get(key) for key in KEPT_FIELDS} for row in found], total


# --- picking the user's building out of the lot ---

@dataclass(frozen=True)
class Match:
    status: str  # found / ambiguous / not_found
    row: dict | None = None


def _plain(text: object) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def road_core(address: str | None) -> str:
    """'서울특별시 강남구 개포로109길 5 (개포동, 대치아파트)' -> '서울특별시강남구개포로109길5'."""
    return _plain(re.sub(r"\(.*?\)", "", address or ""))


def dong_from(detail_address: str | None) -> str | None:
    """'209동 1203호' -> '209', 'A동 3층' -> 'A', '가동' -> '가'."""
    found = re.search(r"(\d+|[A-Za-z]|[가-힣])\s*동", detail_address or "")
    return found.group(1) if found else None


def select_building(
    rows: list[dict], road_address: str, building_name: str | None, detail_address: str | None
) -> Match:
    if not rows:
        return Match("not_found")
    # Annex buildings (경비실, 경로당 …) are not where anyone moves in or out.
    mains = [row for row in rows if str(row.get("mainAtchGbCd")) == "0"] or rows
    candidates = mains
    core = road_core(road_address)
    on_road = [row for row in candidates if road_core(row.get("newPlatPlc")) == core]
    if on_road:
        candidates = on_road
    if len(candidates) == 1:
        return Match("found", candidates[0])
    dong = dong_from(detail_address)
    if dong:
        def in_dong(rows: list[dict]) -> list[dict]:
            return [
                row for row in rows
                if _plain(row.get("dongNm")) in (dong, f"{dong}동")
                or _plain(row.get("bldNm")).endswith(f"{dong}동")
            ]

        # A complex has several road addresses, and people pick any of them in search,
        # so a 동 not on the chosen road is looked for across the lot's main buildings.
        found = in_dong(candidates) or in_dong(mains)
        if len(found) == 1:
            return Match("found", found[0])
        if found:
            candidates = found
    if building_name:
        named = [row for row in candidates if _plain(row.get("bldNm")) == _plain(building_name)]
        if len(named) == 1:
            return Match("found", named[0])
    return Match("ambiguous")


def count(value: object) -> int | None:
    text = str(value).strip() if value is not None else ""
    return int(text) if text.isdigit() else None


def decide_elevator(row: dict) -> str:
    """available / unavailable / unknown, from one 표제부 row."""
    ride, emergency = count(row.get("rideUseElvtCnt")), count(row.get("emgenUseElvtCnt"))
    if ride is None or emergency is None:
        return "unknown"
    # Apartments often record their everyday lifts as emergency elevators (비상용 겸용),
    # so either kind counts as having an elevator.
    if ride >= 1 or emergency >= 1:
        return "available"
    floors = count(row.get("grndFlrCnt"))
    if floors is not None and floors >= TALL_BUILDING_FLOORS:
        # A tall building recorded with no elevator is more likely a gap in the record.
        return "unknown"
    return "unavailable"


def default_ladder(elevator: str, building_floors: int | None, floor: int | None) -> str:
    """use / not_use when the user has not chosen: a tall building with no usable elevator."""
    floors = building_floors if building_floors is not None else floor
    tall = floors is not None and floors >= TALL_BUILDING_FLOORS
    return "use" if tall and elevator in ("unavailable", "unknown") else "not_use"

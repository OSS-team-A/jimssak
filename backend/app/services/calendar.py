"""Which price-raising conditions a move date falls on.

Movers charge more on weekends (and often Fridays), on the last days of a month, on
public holidays, on 손 없는 날, and on the day before a long holiday. Holidays and lunar
dates come from 한국천문연구원 (KASI) via data.go.kr; the rest is plain calendar math.
"""

import calendar as month_calendar
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import httpx

ATTEMPTS = 2
RETRY_PAUSE_SECONDS = 0.3
# Movers' month-end rate starts two days before the last day (예스2404 기준).
MONTH_END_DAYS = 3
# Lunar days on which no evil spirit is said to roam: 9·10·19·20·29·30.
SON_EOMNEUN_DAYS = {9, 10, 19, 20, 29, 30}
# A long holiday is followed for at most this many days after the move date.
HOLIDAY_RUN_LOOKAHEAD_DAYS = 15


class CalendarLookupFailed(Exception):
    """KASI could not be read right now; worth retrying later."""


@dataclass(frozen=True)
class LunarDate:
    month: int
    day: int
    leap: bool  # 윤달


class Calendar(Protocol):
    def year_holidays(self, year: int) -> dict[str, str]:
        """ISO date -> holiday name for the whole year; empty if not published yet."""
        ...

    def month_lunar(self, year: int, month: int) -> dict[str, LunarDate]:
        """ISO date -> lunar date for every day of the month."""
        ...


class KasiCalendar:
    def __init__(
        self, holiday_url: str, lunar_url: str, service_key: str | None,
        timeout: float, budget: float,
    ):
        self.holiday_url = holiday_url
        self.lunar_url = lunar_url
        self.service_key = service_key
        self.timeout = timeout
        self.budget = budget

    def year_holidays(self, year: int) -> dict[str, str]:
        items = self._items(self.holiday_url, {"solYear": year, "numOfRows": 100})
        return {
            _iso(item["locdate"]): str(item.get("dateName", "")).strip()
            for item in items if str(item.get("isHoliday", "Y")) == "Y"
        }

    def month_lunar(self, year: int, month: int) -> dict[str, LunarDate]:
        items = self._items(
            self.lunar_url, {"solYear": year, "solMonth": f"{month:02d}", "numOfRows": 31}
        )
        try:
            return {
                date(int(item["solYear"]), int(item["solMonth"]), int(item["solDay"]))
                .isoformat(): LunarDate(
                    int(item["lunMonth"]), int(item["lunDay"]), item.get("lunLeapmonth") == "윤"
                )
                for item in items
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise CalendarLookupFailed("unexpected lunar response") from exc

    def _items(self, url: str, params: dict) -> list[dict]:
        if not self.service_key:
            raise CalendarLookupFailed("DATA_GO_KR_SERVICE_KEY is not set")
        deadline = time.monotonic() + self.budget
        params = {"serviceKey": self.service_key, "_type": "json", **params}
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
                response = httpx.get(url, params=params, timeout=min(self.timeout, remaining))
                if response.status_code != 200 or not response.content:
                    last_error = f"HTTP {response.status_code}, {len(response.content)} bytes"
                    continue
                return parse_items(response.json())
            except (httpx.HTTPError, ValueError) as exc:
                last_error = type(exc).__name__
            except CalendarLookupFailed as exc:
                last_error = str(exc)
        raise CalendarLookupFailed(f"KASI request failed: {last_error}")


def parse_items(body: object) -> list[dict]:
    try:
        header = body["response"]["header"]
        if str(header.get("resultCode")) != "00":
            raise CalendarLookupFailed(f"result code {header.get('resultCode')}")
        content = body["response"]["body"]
        items = content.get("items")
        found = items.get("item") if isinstance(items, dict) else None
    except (KeyError, TypeError, AttributeError) as exc:
        raise CalendarLookupFailed("unexpected KASI response") from exc
    if not found:
        return []
    return found if isinstance(found, list) else [found]


def _iso(locdate: object) -> str:
    text = str(locdate)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


# --- judging one date ---

@dataclass(frozen=True)
class DayInfo:
    weekend: bool
    friday: bool
    month_end: bool
    # None when that year's holidays are not known (not published or not fetched).
    holiday: bool | None
    holiday_name: str | None
    son_eomneun: bool | None
    lunar: LunarDate | None
    pre_holiday: bool | None


def judge(
    day: date, holidays_by_year: dict[int, dict[str, str] | None], lunar: LunarDate | None
) -> DayInfo:
    """Holiday-dependent answers are None when a needed year is unknown."""
    last_day = month_calendar.monthrange(day.year, day.month)[1]
    this_year = holidays_by_year.get(day.year)
    holiday_name = this_year.get(day.isoformat()) if this_year is not None else None
    return DayInfo(
        weekend=day.weekday() >= 5,
        friday=day.weekday() == 4,
        month_end=day.day > last_day - MONTH_END_DAYS,
        holiday=None if this_year is None else holiday_name is not None,
        holiday_name=holiday_name,
        son_eomneun=None if lunar is None else lunar.day in SON_EOMNEUN_DAYS,
        lunar=lunar,
        pre_holiday=_pre_holiday(day, holidays_by_year),
    )


def years_needed(day: date) -> set[int]:
    return {day.year, (day + timedelta(days=HOLIDAY_RUN_LOOKAHEAD_DAYS)).year}


def _pre_holiday(day: date, holidays_by_year: dict[int, dict[str, str] | None]) -> bool | None:
    """True on the working day right before 2+ days off that include a public holiday.

    A plain weekend alone is not a long holiday.
    """
    if any(holidays_by_year.get(year) is None for year in years_needed(day)):
        return None

    def is_holiday(when: date) -> bool:
        return when.isoformat() in holidays_by_year[when.year]

    def is_off(when: date) -> bool:
        return when.weekday() >= 5 or is_holiday(when)

    if is_off(day):
        return False
    run, cursor = [], day + timedelta(days=1)
    while is_off(cursor) and len(run) < HOLIDAY_RUN_LOOKAHEAD_DAYS:
        run.append(cursor)
        cursor += timedelta(days=1)
    return len(run) >= 2 and any(is_holiday(when) for when in run)

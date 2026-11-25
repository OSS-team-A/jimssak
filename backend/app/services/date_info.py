"""Judge a move date using KASI data shared by every move.

Holidays are cached per year and lunar dates per month, so a month is fetched once no
matter how many moves fall in it.
"""

import json
import time
from dataclasses import asdict
from datetime import date

from .calendar import Calendar, CalendarLookupFailed, LunarDate, judge, years_needed
from .moves import CalendarEntry, DateInfoRecord, MoveStore

# Published holidays are re-read weekly to pick up 임시공휴일 declared later.
HOLIDAYS_FRESH_SECONDS = 7 * 24 * 60 * 60
# A year with nothing published yet is asked again the next day.
UNPUBLISHED_FRESH_SECONDS = 24 * 60 * 60


def _holidays(store: MoveStore, calendar: Calendar, year: int) -> tuple[str, dict | None]:
    """(confirmed | unpublished | failed, ISO date -> name or None if unknown)."""
    key = f"holidays:{year}"
    cached = store.get_calendar(key)
    now = int(time.time())
    if cached is not None:
        fresh_for = (
            HOLIDAYS_FRESH_SECONDS if cached.status == "confirmed" else UNPUBLISHED_FRESH_SECONDS
        )
        if now - cached.fetched_at < fresh_for:
            return _known(cached)
    try:
        found = calendar.year_holidays(year)
    except CalendarLookupFailed:
        # A stale copy beats nothing; with none, the year stays unknown for now.
        return _known(cached) if cached is not None else ("failed", None)
    entry = CalendarEntry(
        "confirmed" if found else "unpublished", json.dumps(found, ensure_ascii=False), now
    )
    store.put_calendar(key, entry)
    return _known(entry)


def _known(entry: CalendarEntry) -> tuple[str, dict | None]:
    data = json.loads(entry.data_json)
    return entry.status, data if entry.status == "confirmed" else None


def _lunar(store: MoveStore, calendar: Calendar, day: date) -> LunarDate | None:
    key = f"lunar:{day.year}-{day.month:02d}"
    cached = store.get_calendar(key)
    if cached is None:
        try:
            month = calendar.month_lunar(day.year, day.month)
        except CalendarLookupFailed:
            return None
        # Lunar dates never change, so a fetched month is kept for good.
        cached = CalendarEntry(
            "found", json.dumps({k: asdict(v) for k, v in month.items()}), int(time.time())
        )
        store.put_calendar(key, cached)
    found = json.loads(cached.data_json).get(day.isoformat())
    return LunarDate(**found) if found else None


def judge_move_date(store: MoveStore, calendar: Calendar, day: date) -> DateInfoRecord:
    statuses, holidays_by_year = [], {}
    for year in sorted(years_needed(day)):
        status, holidays = _holidays(store, calendar, year)
        statuses.append(status)
        holidays_by_year[year] = holidays
    lunar = _lunar(store, calendar, day)
    info = judge(day, holidays_by_year, lunar)
    if "failed" in statuses:
        holiday_status = "failed"
    elif "unpublished" in statuses:
        holiday_status = "unpublished"
    else:
        holiday_status = "confirmed"
    return DateInfoRecord(
        move_date=day.isoformat(),
        holiday_status=holiday_status,
        lunar_status="found" if lunar is not None else "failed",
        info_json=json.dumps(asdict(info), ensure_ascii=False),
        checked_at=int(time.time()),
    )


def is_complete(record: DateInfoRecord) -> bool:
    return record.holiday_status == "confirmed" and record.lunar_status == "found"

import json
from datetime import date, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db import Database
from backend.app.main import create_app
from backend.app.schemas.moves import KST
from backend.app.services import calendar as calendar_module
from backend.app.services.calendar import (
    CalendarLookupFailed,
    KasiCalendar,
    LunarDate,
    judge,
)
from backend.app.services.date_info import judge_move_date
from backend.app.services.moves import MoveStore

# Real data from KASI, checked 2026-11-25.
H2026 = {"2026-09-24": "추석", "2026-09-25": "추석", "2026-09-26": "추석",
         "2026-10-03": "개천절", "2026-10-05": "대체공휴일(개천절)", "2026-10-09": "한글날",
         "2026-12-25": "기독탄신일"}
H2027 = {"2027-01-01": "1월1일"}
BOTH = {2026: H2026, 2027: H2027}
LUNAR_NOV = {25: (10, 17), 27: (10, 19), 28: (10, 20), 29: (10, 21)}


def lunar(day):
    month, lday = LUNAR_NOV[day]
    return LunarDate(month, lday, False)


# --- judging a date ---

def test_plain_wednesday():
    info = judge(date(2026, 11, 25), BOTH, lunar(25))
    assert (info.weekend, info.friday, info.month_end, info.holiday, info.son_eomneun,
            info.pre_holiday) == (False, False, False, False, False, False)


def test_saturday_month_end_son_eomneun():
    info = judge(date(2026, 11, 28), BOTH, lunar(28))
    assert info.weekend and info.month_end and info.son_eomneun
    assert info.pre_holiday is False  # already a day off


def test_friday_before_plain_weekend_is_not_pre_holiday():
    info = judge(date(2026, 11, 27), BOTH, lunar(27))
    assert info.friday and info.son_eomneun
    assert info.month_end is False  # month end is the last three days: 28-30
    assert info.pre_holiday is False


def test_holiday_and_the_day_before():
    christmas = judge(date(2026, 12, 25), BOTH, None)
    assert christmas.holiday and christmas.holiday_name == "기독탄신일"
    assert judge(date(2026, 12, 24), BOTH, None).pre_holiday is True


@pytest.mark.parametrize("day", [
    date(2026, 9, 23),   # before 추석 24-26 + Sunday
    date(2026, 10, 2),   # before 개천절 (Sat), Sunday, 대체공휴일 (Mon)
    date(2026, 12, 31),  # before 1월1일 2027 (Fri) + weekend: crosses the year
])
def test_pre_holiday(day):
    assert judge(day, BOTH, None).pre_holiday is True


def test_unknown_years_stay_unknown():
    next_year_unknown = judge(date(2026, 12, 31), {2026: H2026, 2027: None}, None)
    assert next_year_unknown.holiday is False      # 2026 itself is known
    assert next_year_unknown.pre_holiday is None   # the run reaches into 2027
    this_year_unknown = judge(date(2026, 11, 25), {2026: None}, None)
    assert this_year_unknown.holiday is None and this_year_unknown.pre_holiday is None
    assert this_year_unknown.son_eomneun is None   # no lunar date either


# --- caching and judging with the shared store ---

class FakeCalendar:
    def __init__(self, holidays=None, fail=False):
        self.holidays = BOTH if holidays is None else holidays
        self.fail = fail
        self.calls = []

    def year_holidays(self, year):
        self.calls.append(("holidays", year))
        if self.fail:
            raise CalendarLookupFailed("down")
        return self.holidays.get(year, {})

    def month_lunar(self, year, month):
        self.calls.append(("lunar", year, month))
        if self.fail:
            raise CalendarLookupFailed("down")
        start = date(year, month, 1)
        days = {}
        while start.month == month:
            days[start.isoformat()] = LunarDate(10, start.day % 30 or 30, False)
            start += timedelta(days=1)
        return days


@pytest.fixture
def store(database_url):
    db = Database(database_url)
    db.open()
    db.initialize()
    yield MoveStore(db, 600)
    db.close()


def test_year_and_month_are_fetched_once(store):
    calendar = FakeCalendar()
    judge_move_date(store, calendar, date(2026, 11, 25))
    judge_move_date(store, calendar, date(2026, 11, 28))
    assert calendar.calls == [("holidays", 2026), ("lunar", 2026, 11)]
    record = judge_move_date(store, calendar, date(2026, 12, 24))  # also needs 2027
    assert ("holidays", 2027) in calendar.calls and ("lunar", 2026, 12) in calendar.calls
    assert record.holiday_status == "confirmed" and record.lunar_status == "found"


def test_unpublished_year(store):
    record = judge_move_date(store, FakeCalendar(holidays={2026: H2026}), date(2026, 12, 24))
    assert record.holiday_status == "unpublished"
    assert json.loads(record.info_json)["pre_holiday"] is None


def test_failure_then_retry(store):
    failing = judge_move_date(store, FakeCalendar(fail=True), date(2026, 11, 25))
    assert (failing.holiday_status, failing.lunar_status) == ("failed", "failed")
    working = FakeCalendar()
    retried = judge_move_date(store, working, date(2026, 11, 25))
    assert (retried.holiday_status, retried.lunar_status) == ("confirmed", "found")
    assert working.calls  # failures were not cached


def test_stale_holidays_are_used_when_refresh_fails(store):
    judge_move_date(store, FakeCalendar(), date(2026, 11, 25))
    with store.db.connect() as db:
        db.execute("UPDATE calendar_cache SET fetched_at = 0")  # a week old and more
    record = judge_move_date(store, FakeCalendar(fail=True), date(2026, 11, 25))
    assert record.holiday_status == "confirmed"


# --- through the API ---

MOVES = "/api/v1/moves"


def in_days(n):
    return (datetime.now(KST).date() + timedelta(days=n)).isoformat()


@pytest.fixture
def client(tmp_path):
    settings = Settings(media_dir=tmp_path / "media")
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def move(client):
    body = client.post(MOVES).json()
    return f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}


def schedule(client, move, day, **extra):
    url, headers = move
    response = client.patch(url, headers=headers, json={
        "schedule": {"move_date": day, "start_time": "09:00"}, **extra})
    return response


def test_saving_a_date_fills_day_info(client, move):
    day = date.fromisoformat(in_days(30))
    client.app.state.calendar = FakeCalendar(holidays={
        day.year: {day.isoformat(): "테스트공휴일"},
        day.year + 1: {f"{day.year + 1}-01-01": "1월1일"},
    })
    info = schedule(client, move, day.isoformat()).json()["data"]["schedule"]["day"]
    assert info["holiday"] is True and info["holiday_name"] == "테스트공휴일"
    assert info["weekend"] == (day.weekday() >= 5)
    assert info["holiday_status"] == "confirmed" and info["lunar_status"] == "found"
    assert info["lunar"] is not None and info["checked_at"]


def test_only_a_new_date_calls_out(client, move):
    calendar = client.app.state.calendar = FakeCalendar(
        holidays={year: {"x": "y"} for year in range(2026, 2030)})
    schedule(client, move, in_days(30))
    calls = len(calendar.calls)
    schedule(client, move, in_days(30))  # same date
    url, headers = move
    client.patch(url, headers=headers, json={"origin": {"road_address": "서울", "floor": 3}})
    assert len(calendar.calls) == calls

    # A new date re-judges, but only fetches the lunar month if it is a new month;
    # its holiday years are already cached.
    first, second = (date.fromisoformat(in_days(n)) for n in (30, 31))
    schedule(client, move, second.isoformat())
    new_month = (second.year, second.month) != (first.year, first.month)
    assert calendar.calls[calls:] == ([("lunar", second.year, second.month)] if new_month else [])


def test_failed_date_lookup_is_retried(client, move):
    client.app.state.calendar = FakeCalendar(fail=True)
    first = schedule(client, move, in_days(30))
    assert first.status_code == 200
    assert first.json()["data"]["schedule"]["day"]["holiday_status"] == "failed"
    client.app.state.calendar = FakeCalendar(
        holidays={year: {"x": "y"} for year in range(2026, 2030)})
    again = schedule(client, move, in_days(30)).json()["data"]["schedule"]["day"]
    assert again["holiday_status"] == "confirmed"


def test_clearing_the_schedule_drops_day_info(client, move):
    client.app.state.calendar = FakeCalendar()
    schedule(client, move, in_days(30))
    url, headers = move
    data = client.patch(url, headers=headers, json={"schedule": None}).json()["data"]
    assert data["schedule"] == {"status": "empty", "value": None, "day": None}


def test_more_than_a_year_ahead_is_rejected(client, move):
    assert schedule(client, move, in_days(365)).status_code == 200
    response = schedule(client, move, in_days(366))
    assert response.status_code == 422
    assert "365일" in response.text


# --- the KASI client, with the network replaced ---

HOLIDAY_URL = "https://kasi.example/holidays"
LUNAR_URL = "https://kasi.example/lunar"


def kasi(key="k"):
    return KasiCalendar(HOLIDAY_URL, LUNAR_URL, key, 2.0, 10.0)


def ok(items):
    return {"response": {"header": {"resultCode": "00"},
                         "body": {"items": {"item": items} if items else "", "totalCount": 0}}}


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(calendar_module, "RETRY_PAUSE_SECONDS", 0)


def respond(monkeypatch, *bodies):
    queue = list(bodies)
    seen = []

    def fake_get(url, params, timeout):
        seen.append((url, params))
        body = queue.pop(0)
        request = httpx.Request("GET", url)
        if isinstance(body, int):
            return httpx.Response(body, request=request)
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx, "get", fake_get)
    return seen


def test_kasi_holidays(monkeypatch):
    seen = respond(monkeypatch, ok([
        {"locdate": 20261225, "dateName": "기독탄신일", "isHoliday": "Y"},
        {"locdate": 20261226, "dateName": "기념일", "isHoliday": "N"},
    ]))
    assert kasi().year_holidays(2026) == {"2026-12-25": "기독탄신일"}
    assert seen[0][1]["solYear"] == 2026 and seen[0][1]["serviceKey"] == "k"


def test_kasi_single_item_and_empty_year(monkeypatch):
    respond(monkeypatch, ok({"locdate": 20290101, "dateName": "1월1일", "isHoliday": "Y"}), ok([]))
    assert kasi().year_holidays(2029) == {"2029-01-01": "1월1일"}
    assert kasi().year_holidays(2030) == {}


def test_kasi_lunar_month(monkeypatch):
    respond(monkeypatch, ok([
        {"solYear": 2026, "solMonth": 11, "solDay": 28, "lunMonth": 10, "lunDay": 20,
         "lunLeapmonth": "평"},
        {"solYear": 2026, "solMonth": 11, "solDay": 29, "lunMonth": 10, "lunDay": 21,
         "lunLeapmonth": "윤"},
    ]))
    month = kasi().month_lunar(2026, 11)
    assert month["2026-11-28"] == LunarDate(10, 20, False)
    assert month["2026-11-29"].leap is True


def test_kasi_retries_then_succeeds(monkeypatch):
    respond(monkeypatch, 503, ok([{"locdate": 20261225, "dateName": "기독탄신일"}]))
    assert kasi().year_holidays(2026) == {"2026-12-25": "기독탄신일"}


def test_kasi_gives_up(monkeypatch):
    respond(monkeypatch, 503, 503)
    with pytest.raises(CalendarLookupFailed):
        kasi().year_holidays(2026)


def test_kasi_without_key_never_calls_out(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("network called"))
    with pytest.raises(CalendarLookupFailed):
        kasi(key=None).year_holidays(2026)

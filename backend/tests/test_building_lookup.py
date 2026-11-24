import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services import building_register as br
from backend.app.services.building_register import (
    BuildingLookupFailed,
    HubBuildingRegister,
    LotCode,
    decide_elevator,
    dong_from,
    parse_page,
    road_core,
    select_building,
)

MOVES = "/api/v1/moves"
TOEGYE_LOT = LotCode("1114013800", False, 82, 1)


def row(name, dong, road, floors, ride, emergency, main="0"):
    return {"mgmBldrgstPk": f"pk-{name}-{dong}", "mainAtchGbCd": main, "bldNm": name,
            "dongNm": dong, "newPlatPlc": road, "mainPurpsCdNm": "", "grndFlrCnt": floors,
            "rideUseElvtCnt": ride, "emgenUseElvtCnt": emergency}


# Rows as the register returned them on 2026-11-24.
TOEGYE_ROAD = "서울특별시 중구 퇴계로36길 2 (필동2가)"
TOEGYE = [row("동국대학교 충무로 영상센터", " ", TOEGYE_ROAD, 11, 3, 3)]
GAEPO = [
    row("대치아파트209동", "209", "서울특별시 강남구 개포로109길 9 (개포동)", 15, 0, 0),
    row("대치아파트210동", "210", "서울특별시 강남구 개포로109길 9 (개포동)", 15, 0, 0),
    row("대치종합상가", "대치종합상가", "서울특별시 강남구 개포로109길 9 (개포동)", 3, 0, 0),
    row("대치아파트101동", "101", "서울특별시 강남구 개포로109길 5 (개포동)", 15, 0, 0),
    row("대치단지", "통합경비실-1", " ", 1, 0, 0, main="1"),
]

TOEGYE_PLACE = {
    "road_address": TOEGYE_ROAD, "building_name": "동국대학교 충무로 영상센터",
    "adm_cd": "1114013800", "mountain": False, "lot_main": 82, "lot_sub": 1, "floor": 3,
}
GAEPO_PLACE = {
    "road_address": "서울특별시 강남구 개포로109길 9 (개포동, 대치아파트)",
    "building_name": "대치아파트",
    "adm_cd": "1168010300", "mountain": False, "lot_main": 12, "lot_sub": 0, "floor": 12,
}


class FakeRegister:
    def __init__(self, rows_by_lot, fail_times=0):
        self.rows_by_lot, self.fail_times, self.calls = rows_by_lot, fail_times, []

    def fetch(self, lot):
        self.calls.append(lot)
        if self.fail_times:
            self.fail_times -= 1
            raise BuildingLookupFailed("register down")
        return self.rows_by_lot.get((lot.adm_cd, lot.lot_main, lot.lot_sub), [])


@pytest.fixture
def client(tmp_path):
    settings = Settings(auth_db_path=tmp_path / "app.sqlite3", media_dir=tmp_path / "media",
                        data_go_kr_service_key=None, address_search_api_key=None)
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def register(client):
    fake = FakeRegister({("1114013800", 82, 1): TOEGYE, ("1168010300", 12, 0): GAEPO})
    client.app.state.building_register = fake
    return fake


@pytest.fixture
def move(client):
    body = client.post(MOVES).json()
    return f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}


def save(client, move, **sections):
    url, headers = move
    response = client.patch(url, headers=headers, json=sections)
    assert response.status_code == 200, response.text
    return response.json()["data"]


# --- through the API ---

def test_found_building_fills_elevator(client, register, move):
    origin = save(client, move, origin=TOEGYE_PLACE)["origin"]["value"]
    assert origin["elevator"] == "available"
    assert origin["elevator_source"] == "building_register"
    building = origin["building"]
    assert building["status"] == "found"
    assert building["building_name"] == "동국대학교 충무로 영상센터"
    assert building["dong_name"] is None
    assert (building["above_ground_floors"], building["ride_elevators"],
            building["emergency_elevators"]) == (11, 3, 3)
    assert building["checked_at"]
    assert register.calls == [LotCode("1114013800", False, 82, 1)]


def test_same_lot_is_not_fetched_again(client, register, move):
    save(client, move, origin=TOEGYE_PLACE)
    again = save(client, move, origin={**TOEGYE_PLACE, "floor": 5, "detail_address": "502호"})
    assert again["origin"]["value"]["building"]["status"] == "found"
    assert again["origin"]["value"]["floor"] == 5
    save(client, move, schedule={"move_date": "2099-01-01", "start_time": "09:00"})
    assert len(register.calls) == 1


def test_apartment_needs_dong_then_matches_without_new_call(client, register, move):
    plain = save(client, move, origin=GAEPO_PLACE)["origin"]["value"]
    assert plain["building"]["status"] == "ambiguous"
    assert (plain["elevator"], plain["elevator_source"]) == ("unknown", None)

    with_dong = save(client, move, origin={**GAEPO_PLACE, "detail_address": "209동 1203호"})
    building = with_dong["origin"]["value"]["building"]
    assert building["status"] == "found"
    assert (building["building_name"], building["dong_name"]) == ("대치아파트209동", "209")
    # 15 floors recorded with no elevator: treated as a gap in the record, not "none".
    assert with_dong["origin"]["value"]["elevator"] == "unknown"
    assert len(register.calls) == 1


def test_user_answer_outranks_register(client, register, move):
    save(client, move, origin=TOEGYE_PLACE)
    save(client, move, origin={**TOEGYE_PLACE, "elevator": "unavailable"})
    kept = save(client, move, origin={**TOEGYE_PLACE, "floor": 4})["origin"]["value"]
    assert (kept["elevator"], kept["elevator_source"]) == ("unavailable", "user")
    assert kept["building"]["ride_elevators"] == 3


def test_failed_lookup_saves_and_retries_on_next_save(client, move):
    flaky = FakeRegister({("1114013800", 82, 1): TOEGYE}, fail_times=1)
    client.app.state.building_register = flaky
    first = save(client, move, origin=TOEGYE_PLACE)["origin"]
    assert first["status"] == "filled"
    assert first["value"]["building"]["status"] == "failed"
    assert first["value"]["elevator"] == "unknown"

    retried = save(client, move, origin=TOEGYE_PLACE)["origin"]["value"]
    assert retried["building"]["status"] == "found"
    assert retried["elevator"] == "available"
    assert len(flaky.calls) == 2


def test_typed_address_is_not_looked_up(client, register, move):
    origin = save(client, move, origin={"road_address": "서울 중구 퇴계로36길 2", "floor": 3})
    assert origin["origin"]["value"]["building"]["status"] == "not_requested"
    assert register.calls == []


def test_unknown_lot_is_not_found(client, register, move):
    origin = save(client, move, origin={**TOEGYE_PLACE, "lot_main": 999})["origin"]["value"]
    assert origin["building"]["status"] == "not_found"
    assert origin["elevator"] == "unknown"


def test_new_address_drops_old_building(client, register, move):
    save(client, move, origin=TOEGYE_PLACE)
    moved = save(client, move, origin=GAEPO_PLACE)["origin"]["value"]
    assert moved["building"]["status"] == "ambiguous"
    assert moved["elevator"] == "unknown"
    cleared = save(client, move, origin=None)
    assert cleared["origin"] == {"status": "empty", "value": None}


def test_each_end_is_looked_up_separately(client, register, move):
    data = save(client, move, origin=TOEGYE_PLACE,
                destination={**GAEPO_PLACE, "detail_address": "101동"})
    assert data["origin"]["value"]["building"]["ride_elevators"] == 3
    assert data["destination"]["value"]["building"]["dong_name"] == "101"
    assert len(register.calls) == 2


@pytest.mark.parametrize("partial", [
    {"adm_cd": "1114013800"},
    {"lot_main": 82},
    {"adm_cd": "11140", "lot_main": 82},
])
def test_codes_must_come_together(client, register, move, partial):
    url, headers = move
    body = {"origin": {"road_address": "서울 중구 퇴계로36길 2", **partial}}
    assert client.patch(url, headers=headers, json=body).status_code == 422


# --- matching and the elevator rule ---

def test_road_core_and_dong():
    core = road_core("서울특별시 강남구 개포로109길 5 (개포동, 대치아파트)")
    assert core == "서울특별시강남구개포로109길5"
    assert dong_from("209동 1203호") == "209"
    assert dong_from("A동 3층") == "A"
    assert dong_from("가동") == "가"
    assert dong_from("302호") is None


def test_dong_found_on_another_road_of_the_same_lot():
    # 101동 is on 개포로109길 5, but the user picked the complex's 109길 9 address.
    match = select_building(GAEPO, "서울특별시 강남구 개포로109길 9", "대치아파트", "101동 502호")
    assert match.status == "found" and match.row["dongNm"] == "101"
    # A 동 that does not exist anywhere stays ambiguous.
    missing = select_building(GAEPO, "서울특별시 강남구 개포로109길 9", "대치아파트", "999동")
    assert missing.status == "ambiguous"


def test_select_skips_annex_and_uses_road_then_name():
    road_5 = "서울특별시 강남구 개포로109길 5 (개포동)"
    on_road_5 = select_building(GAEPO, road_5, "대치아파트", None)
    assert on_road_5.status == "found" and on_road_5.row["dongNm"] == "101"
    shop = select_building(GAEPO, "서울특별시 강남구 개포로109길 9", "대치종합상가", None)
    assert shop.status == "found" and shop.row["bldNm"] == "대치종합상가"
    assert select_building(GAEPO, "서울특별시 강남구 개포로109길 9", "대치아파트", None).status \
        == "ambiguous"
    assert select_building([], "x", None, None).status == "not_found"


@pytest.mark.parametrize(("ride", "emergency", "floors", "expected"), [
    (3, 3, 11, "available"),
    (1, 0, 4, "available"),
    (0, 0, 3, "unavailable"),
    (0, 0, 5, "unavailable"),
    (0, 0, 6, "unknown"),     # tall building recorded with none: likely missing
    (0, 1, 4, "available"),   # emergency only: often the everyday lift
    (0, 2, 23, "available"),  # 해솔마을 7단지 717동 as recorded on 2026-11-24
    ("", 0, 3, "unknown"),    # not recorded
    (None, None, 3, "unknown"),
])
def test_elevator_rule(ride, emergency, floors, expected):
    assert decide_elevator({"rideUseElvtCnt": ride, "emgenUseElvtCnt": emergency,
                            "grndFlrCnt": floors}) == expected


# --- the real client, with the network replaced ---

def ok_body(items, total):
    return {"response": {"header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE"},
                         "body": {"items": {"item": items}, "totalCount": total}}}


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(br, "RETRY_PAUSE_SECONDS", 0)


def test_client_params_and_single_item(monkeypatch):
    seen = []

    def fake_get(url, params, timeout):
        seen.append(params)
        return httpx.Response(200, json=ok_body(TOEGYE[0], 1), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    rows = HubBuildingRegister("https://hub.example", "k", 2.0, 10.0).fetch(
        LotCode("1114013800", False, 82, 1))
    assert rows == TOEGYE
    assert {k: seen[0][k] for k in ("sigunguCd", "bjdongCd", "platGbCd", "bun", "ji", "_type")} \
        == {"sigunguCd": "11140", "bjdongCd": "13800", "platGbCd": "0", "bun": "0082",
            "ji": "0001", "_type": "json"}


def test_client_retries_empty_and_timeout_bodies(monkeypatch):
    request = httpx.Request("GET", "https://hub.example")
    timeout_body = {"OpenAPI_ServiceResponse": {"cmmMsgHeader": {"returnReasonCode": "05"}}}
    responses = iter([httpx.Response(200, content=b"", request=request),
                      httpx.Response(503, json=timeout_body, request=request),
                      httpx.Response(200, json=ok_body(TOEGYE, 1), request=request)])
    monkeypatch.setattr(httpx, "get", lambda url, params, timeout: next(responses))
    assert HubBuildingRegister("https://hub.example", "k", 2.0, 10.0).fetch(
        LotCode("1114013800", False, 82, 1)) == TOEGYE


def test_client_gives_up_after_retries(monkeypatch):
    request = httpx.Request("GET", "https://hub.example")
    empty = httpx.Response(200, content=b"", request=request)
    monkeypatch.setattr(httpx, "get", lambda url, params, timeout: empty)
    with pytest.raises(BuildingLookupFailed):
        HubBuildingRegister("https://hub.example", "k", 2.0, 10.0).fetch(TOEGYE_LOT)


def test_client_reads_every_page(monkeypatch):
    pages = {1: GAEPO[:3], 2: GAEPO[3:]}

    def fake_get(url, params, timeout):
        return httpx.Response(200, json=ok_body(pages[params["pageNo"]], len(GAEPO)),
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert HubBuildingRegister("https://hub.example", "k", 2.0, 10.0).fetch(
        LotCode("1168010300", False, 12, 0)) == GAEPO


def test_client_without_key_never_calls_out(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("network called"))
    with pytest.raises(BuildingLookupFailed):
        HubBuildingRegister("https://hub.example", None, 2.0, 10.0).fetch(TOEGYE_LOT)


def test_parse_empty_and_error():
    assert parse_page({"response": {"header": {"resultCode": "00"},
                                    "body": {"items": "", "totalCount": 0}}}) == ([], 0)
    with pytest.raises(BuildingLookupFailed):
        parse_page({"response": {"header": {"resultCode": "30"}, "body": {}}})
    with pytest.raises(BuildingLookupFailed):
        parse_page({"OpenAPI_ServiceResponse": {}})


def test_client_stops_at_time_budget(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(br.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(br.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds))
    timeouts = []

    def slow_get(url, params, timeout):
        timeouts.append(timeout)
        clock["now"] += timeout  # every call runs until it times out
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx, "get", slow_get)
    with pytest.raises(BuildingLookupFailed):
        HubBuildingRegister("https://hub.example", "k", 5.0, 6.0).fetch(TOEGYE_LOT)
    # 5s, then only the 1s left: the whole lookup never runs past the 6s budget.
    assert timeouts == [5.0, 1.0]
    assert clock["now"] <= 6.0


def test_short_request_timeout_leaves_room_to_retry(monkeypatch):
    """A 503 that takes ~6 s must not use up the whole budget in one try."""
    clock = {"now": 0.0}
    monkeypatch.setattr(br.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(br.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds))
    monkeypatch.setattr(br, "RETRY_PAUSE_SECONDS", 0.5)
    request = httpx.Request("GET", "https://hub.example")
    timeouts = []

    def flaky_get(url, params, timeout):
        timeouts.append(timeout)
        if len(timeouts) < 3:
            clock["now"] += timeout  # the slow 503 is cut off at our timeout
            raise httpx.ReadTimeout("slow 503")
        clock["now"] += 0.5
        return httpx.Response(200, json=ok_body(TOEGYE, 1), request=request)

    monkeypatch.setattr(httpx, "get", flaky_get)
    rows = HubBuildingRegister("https://hub.example", "k", 2.0, 6.0).fetch(TOEGYE_LOT)
    assert rows == TOEGYE
    # Two cut-off tries and a third that succeeds, all inside the 6 s budget.
    assert timeouts == [2.0, 2.0, 1.0]
    assert clock["now"] <= 6.0


def test_app_uses_short_register_timeout(tmp_path):
    settings = Settings(auth_db_path=tmp_path / "app.sqlite3")
    with TestClient(create_app(settings)) as client:
        register = client.app.state.building_register
        assert register.timeout == 2.0
        assert register.budget == 6.0

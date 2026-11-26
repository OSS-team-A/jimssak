import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services import route as route_module
from backend.app.services.route import (
    AddressNotLocated,
    KakaoRouter,
    Point,
    Route,
    RouteLookupFailed,
    query_text,
)

MOVES = "/api/v1/moves"
GAEPO = "서울특별시 강남구 개포로109길 5 (개포동, 대치아파트)"
DONGGUK = "서울특별시 중구 필동로1길 30 (필동2가)"
TOEGYE = "서울특별시 중구 퇴계로36길 2 (필동2가)"
# The route the real API returned for 개포로109길 5 -> 필동로1길 30 (2026-11-24).
GAEPO_TO_DONGGUK = Route(12269, 1252, Point(127.0746, 37.4923), Point(126.9987, 37.5590))


class FakeRouter:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def route(self, origin, destination):
        self.calls.append((origin, destination))
        if self.error:
            error, self.error = self.error, None  # fail once, then work
            raise error
        return GAEPO_TO_DONGGUK


@pytest.fixture
def client(tmp_path):
    settings = Settings(media_dir=tmp_path / "media")
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def move(client):
    body = client.post(MOVES).json()
    return f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}


def use(client, router):
    client.app.state.router = router
    return router


def save(client, move, **sections):
    url, headers = move
    response = client.patch(url, headers=headers, json=sections)
    assert response.status_code == 200, response.text
    return response.json()["data"]["route"]


def place(address, floor=3, **extra):
    return {"road_address": address, "floor": floor, **extra}


def test_route_waits_for_both_ends(client, move):
    router = use(client, FakeRouter())
    assert save(client, move, origin=place(GAEPO)) == {
        "status": "not_ready", "distance_m": None, "duration_s": None,
        "basis": None, "checked_at": None,
    }
    assert router.calls == []


def test_route_is_computed_and_kept(client, move):
    router = use(client, FakeRouter())
    route = save(client, move, origin=place(GAEPO), destination=place(DONGGUK))
    assert route["status"] == "found"
    assert (route["distance_m"], route["duration_s"]) == (12269, 1252)
    assert route["basis"] == "kakao_car_recommend"
    assert route["checked_at"]
    assert router.calls == [(GAEPO, DONGGUK)]

    # Floors, detail addresses and the date do not move either end.
    save(client, move, origin=place(GAEPO, floor=5, detail_address="101동 502호"))
    save(client, move, schedule={"move_date": soon(), "start_time": "09:00"})
    url, headers = move
    assert client.get(url, headers=headers).json()["data"]["route"] == route
    assert len(router.calls) == 1


def test_changed_address_recomputes(client, move):
    router = use(client, FakeRouter())
    save(client, move, origin=place(GAEPO), destination=place(DONGGUK))
    save(client, move, destination=place(TOEGYE))
    assert router.calls[-1] == (GAEPO, TOEGYE)
    assert len(router.calls) == 2


def test_failed_route_saves_and_retries_on_next_save(client, move):
    router = use(client, FakeRouter(error=RouteLookupFailed("kakao down")))
    failed = save(client, move, origin=place(GAEPO), destination=place(DONGGUK))
    assert failed["status"] == "failed" and failed["distance_m"] is None
    retried = save(client, move, origin=place(GAEPO), destination=place(DONGGUK))
    assert retried["status"] == "found"
    assert len(router.calls) == 2


def test_unlocatable_address_is_not_retried(client, move):
    router = use(client, FakeRouter(error=AddressNotLocated("destination")))
    # The Figma sample address does not exist, so Kakao has no coordinates for it.
    missing = save(client, move, origin=place(GAEPO),
                   destination=place("서울 성동구 월계로44가길 8"))
    assert missing["status"] == "not_found"
    save(client, move, destination=place("서울 성동구 월계로44가길 8", floor=4))
    assert len(router.calls) == 1


def test_clearing_an_end_drops_the_route(client, move):
    use(client, FakeRouter())
    save(client, move, origin=place(GAEPO), destination=place(DONGGUK))
    assert save(client, move, destination=None)["status"] == "not_ready"


# --- the Kakao client, with the network replaced ---

GEOCODE = "https://dapi.kakao.com/v2/local/search/address.json"
DIRECTIONS = "https://apis-navi.kakaomobility.com/v1/directions"


def kakao(budget=10.0, key="test-key"):
    return KakaoRouter(GEOCODE, DIRECTIONS, key, 2.0, budget)


def geocode_body(x, y):
    return {"meta": {"total_count": 1}, "documents": [{"x": str(x), "y": str(y)}]}


def directions_body(code=0, distance=12269, duration=1252):
    best = {"result_code": code, "result_msg": "msg"}
    if code == 0:
        best["summary"] = {"distance": distance, "duration": duration}
    return {"routes": [best]}


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(route_module, "RETRY_PAUSE_SECONDS", 0)


def fake_kakao(monkeypatch, directions=None, geocodes=None):
    calls = []
    coords = geocodes or {"서울특별시 강남구 개포로109길 5": (127.0746, 37.4923),
                          "서울특별시 중구 필동로1길 30": (126.9987, 37.5590)}

    def fake_get(url, params, headers, timeout):
        calls.append((url, params, headers))
        request = httpx.Request("GET", url)
        if url == GEOCODE:
            found = coords.get(params["query"])
            body = geocode_body(*found) if found else {"meta": {"total_count": 0},
                                                         "documents": []}
            return httpx.Response(200, json=body, request=request)
        return httpx.Response(200, json=directions or directions_body(), request=request)

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def test_kakao_route(monkeypatch):
    calls = fake_kakao(monkeypatch)
    found = kakao().route(GAEPO, DONGGUK)
    assert (found.distance_m, found.duration_s) == (12269, 1252)
    assert found.origin == Point(127.0746, 37.4923)
    geocodes = [c for c in calls if c[0] == GEOCODE]
    # The "(개포동, 대치아파트)" part is dropped before geocoding.
    assert {c[1]["query"] for c in geocodes} == {
        "서울특별시 강남구 개포로109길 5", "서울특별시 중구 필동로1길 30"}
    assert all(c[2] == {"Authorization": "KakaoAK test-key"} for c in calls)
    directions = next(c for c in calls if c[0] == DIRECTIONS)[1]
    assert directions == {"origin": "127.0746,37.4923", "destination": "126.9987,37.559",
                          "priority": "RECOMMEND", "summary": "true"}


def test_same_place_is_zero_meters(monkeypatch):
    fake_kakao(monkeypatch, directions=directions_body(code=104))
    assert kakao().route(GAEPO, DONGGUK).distance_m == 0


def test_no_route_is_failure(monkeypatch):
    fake_kakao(monkeypatch, directions=directions_body(code=1))
    with pytest.raises(RouteLookupFailed):
        kakao().route(GAEPO, DONGGUK)


def test_unknown_address_is_not_located(monkeypatch):
    fake_kakao(monkeypatch)
    with pytest.raises(AddressNotLocated) as raised:
        kakao().route(GAEPO, "서울 성동구 월계로44가길 8")
    assert raised.value.role == "destination"


def test_http_errors_are_failure(monkeypatch):
    request = httpx.Request("GET", GEOCODE)
    monkeypatch.setattr(httpx, "get", lambda url, params, headers, timeout:
                        httpx.Response(401, request=request))
    with pytest.raises(RouteLookupFailed):
        kakao().route(GAEPO, DONGGUK)


def test_no_key_never_calls_out(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("network called"))
    with pytest.raises(RouteLookupFailed):
        kakao(key=None).route(GAEPO, DONGGUK)


def test_time_budget(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(route_module.time, "monotonic", lambda: clock["now"])

    def slow_get(url, params, headers, timeout):
        clock["now"] += timeout
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx, "get", slow_get)
    with pytest.raises(RouteLookupFailed):
        kakao(budget=3.0).route(GAEPO, DONGGUK)
    assert clock["now"] <= 3.0 + 2.0  # the two geocodes run side by side


def test_query_text():
    assert query_text(GAEPO) == "서울특별시 강남구 개포로109길 5"
    assert query_text("서울 중구  퇴계로36길 2") == "서울 중구 퇴계로36길 2"


def soon() -> str:
    """A move date the one-year limit accepts."""
    from datetime import datetime, timedelta

    from backend.app.schemas.moves import KST
    return (datetime.now(KST).date() + timedelta(days=30)).isoformat()

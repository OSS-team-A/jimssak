from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas.moves import KST

AUTH = "/api/v1/auth"
MOVES = "/api/v1/moves"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1000
TODAY = datetime.now(KST).date()
MOVE_DAY = (TODAY + timedelta(days=30)).isoformat()
# Addresses sent without search codes are not looked up in the building register.
NO_LOOKUP = {
    "status": "not_requested", "building_name": None, "dong_name": None,
    "above_ground_floors": None, "ride_elevators": None, "emergency_elevators": None,
    "checked_at": None,
}

# The SC-06 example from the Figma design.
ORIGIN = {
    "road_address": "서울 중구 퇴계로36길 2", "detail_address": "302호",
    "floor": 3, "elevator": "available",
}
DESTINATION = {"road_address": "서울 성동구 월계로44가길 8", "floor": 5}
SCHEDULE = {"move_date": MOVE_DAY, "start_time": "09:00"}


@pytest.fixture
def settings(tmp_path):
    return Settings(
        cors_origins=["http://localhost:5173"],
        auth_db_path=tmp_path / "app.sqlite3",
        media_dir=tmp_path / "media",
        media_max_bytes=4096,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def guest_move(client: TestClient) -> tuple[str, dict[str, str]]:
    body = client.post(MOVES).json()
    return f"{MOVES}/{body['move_id']}", {"X-Guest-Token": body["guest_token"]}


def member_headers(client: TestClient, email: str) -> dict[str, str]:
    account = {"email": email, "password": "a long passphrase"}
    client.post(f"{AUTH}/signup", json=account).raise_for_status()
    token = client.post(f"{AUTH}/login", json=account).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_sc06_form_completes_the_move(client):
    url, headers = guest_move(client)
    client.post(f"{url}/media", headers=headers, files={"file": ("room.mp4", MP4, "video/mp4")})

    response = client.patch(url, headers=headers, json={
        "origin": ORIGIN, "destination": DESTINATION, "schedule": SCHEDULE,
    })
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["origin"] == {"status": "filled", "value": {
        **ORIGIN, "elevator_source": "user", "building": NO_LOOKUP,
    }}
    # Elevator not sent: shown as 확인 필요, with no source yet.
    assert data["destination"] == {"status": "filled", "value": {
        **DESTINATION, "detail_address": None, "elevator": "unknown", "elevator_source": None,
        "building": NO_LOOKUP,
    }}
    assert data["schedule"]["status"] == "filled"
    assert data["schedule"]["value"] == {"move_date": MOVE_DAY, "start_time": "09:00:00"}
    # Date conditions are judged on save; without a key the KASI lookups are failed.
    assert data["schedule"]["day"]["holiday_status"] == "failed"
    assert response.json()["missing"] == []
    assert response.json()["next_step"] is None

    # Reloading shows the same values.
    assert client.get(url, headers=headers).json()["data"] == data


def test_only_sent_sections_change(client):
    url, headers = guest_move(client)
    client.patch(url, headers=headers, json={"origin": ORIGIN, "destination": DESTINATION})

    after = client.patch(url, headers=headers, json={"schedule": SCHEDULE}).json()
    assert after["data"]["origin"]["value"]["road_address"] == ORIGIN["road_address"]
    assert after["data"]["destination"]["value"]["floor"] == 5
    assert after["data"]["schedule"]["status"] == "filled"

    # A section is replaced as a whole: fields left out fall back to their defaults.
    moved = client.patch(url, headers=headers, json={
        "origin": {"road_address": "서울 중구 퇴계로36길 4", "floor": -1},
    }).json()["data"]["origin"]["value"]
    assert moved == {
        "road_address": "서울 중구 퇴계로36길 4", "detail_address": None, "floor": -1,
        "elevator": "unknown", "elevator_source": None, "building": NO_LOOKUP,
    }

    cleared = client.patch(url, headers=headers, json={"destination": None}).json()
    assert cleared["data"]["destination"] == {"status": "empty", "value": None}
    assert cleared["data"]["schedule"]["status"] == "filled"
    assert cleared["missing"] == ["room_video", "destination"]


def test_address_without_floor_is_partial(client):
    url, headers = guest_move(client)
    snapshot = client.patch(url, headers=headers, json={
        "origin": {"road_address": "서울 중구 퇴계로36길 2", "detail_address": "  "},
    }).json()
    assert snapshot["data"]["origin"]["status"] == "partial"
    assert snapshot["data"]["origin"]["value"]["detail_address"] is None
    assert "origin" in snapshot["missing"]


def test_elevator_unavailable_differs_from_unknown(client):
    url, headers = guest_move(client)
    data = client.patch(url, headers=headers, json={
        "origin": {**ORIGIN, "elevator": "unavailable"},
        "destination": {**DESTINATION, "elevator": "unknown"},
    }).json()["data"]
    assert data["origin"]["value"]["elevator"] == "unavailable"
    assert data["origin"]["value"]["elevator_source"] == "user"
    assert data["destination"]["value"]["elevator"] == "unknown"
    assert data["destination"]["value"]["elevator_source"] is None


@pytest.mark.parametrize("body", [
    {},
    {"origin": {**ORIGIN, "floor": 0}},
    {"origin": {**ORIGIN, "floor": 81}},
    {"origin": {**ORIGIN, "floor": -6}},
    {"origin": {**ORIGIN, "road_address": "   "}},
    {"origin": {**ORIGIN, "elevator": "maybe"}},
    {"origin": {**ORIGIN, "elevator_source": "building_register"}},
    {"schedule": {**SCHEDULE, "move_date": (TODAY - timedelta(days=1)).isoformat()}},
    {"schedule": {**SCHEDULE, "start_time": "09:15"}},
    {"schedule": {**SCHEDULE, "start_time": "09:30"}},
    {"schedule": {"move_date": MOVE_DAY}},
    {"moving_day": MOVE_DAY},
])
def test_invalid_input_is_rejected_by_field(client, body):
    url, headers = guest_move(client)
    response = client.patch(url, headers=headers, json=body)
    assert response.status_code == 422
    assert response.json()["detail"]
    # Nothing was saved.
    assert client.get(url, headers=headers).json()["data"]["origin"]["status"] == "empty"


def test_today_on_the_hour_is_allowed(client):
    url, headers = guest_move(client)
    response = client.patch(url, headers=headers, json={
        "schedule": {"move_date": TODAY.isoformat(), "start_time": "13:00"},
    })
    assert response.status_code == 200
    assert response.json()["data"]["schedule"]["value"]["start_time"] == "13:00:00"


def test_only_the_owner_can_save(client):
    url, headers = guest_move(client)
    assert client.patch(url, json={"schedule": SCHEDULE}).status_code == 401
    _, other_guest = guest_move(client)
    assert client.patch(url, headers=other_guest, json={"schedule": SCHEDULE}).status_code == 404

    owner = member_headers(client, "mover@example.com")
    member_url = f"{MOVES}/{client.post(MOVES, headers=owner).json()['move_id']}"
    assert client.patch(member_url, headers=owner, json={"schedule": SCHEDULE}).status_code == 200
    other = member_headers(client, "other@example.com")
    assert client.patch(member_url, headers=other, json={"schedule": SCHEDULE}).status_code == 404
    assert client.get(url, headers=headers).json()["data"]["schedule"]["status"] == "empty"


def test_details_survive_restart(settings):
    with TestClient(create_app(settings)) as client:
        url, headers = guest_move(client)
        client.patch(url, headers=headers, json={"origin": ORIGIN, "schedule": SCHEDULE})
    with TestClient(create_app(settings)) as client:
        data = client.get(url, headers=headers).json()["data"]
        assert data["origin"]["value"]["detail_address"] == "302호"
        assert data["schedule"]["value"]["move_date"] == MOVE_DAY


def test_cors_allows_patch(client):
    response = client.options(MOVES + "/x", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "PATCH",
        "Access-Control-Request-Headers": "X-Guest-Token, Content-Type",
    })
    assert response.status_code == 200


def test_elevator_kept_unless_building_changes(client):
    url, headers = guest_move(client)

    def origin(**fields):
        response = client.patch(url, headers=headers, json={"origin": fields})
        assert response.status_code == 200
        return response.json()["data"]["origin"]["value"]

    # First save from the SC-06 form: no elevator field on that screen.
    first = origin(road_address="서울 중구 퇴계로36길 2", floor=3)
    assert (first["elevator"], first["elevator_source"]) == ("unknown", None)

    # Confirmation screen: the user says there is an elevator.
    confirmed = origin(road_address="서울 중구 퇴계로36길 2", floor=3, elevator="available")
    assert (confirmed["elevator"], confirmed["elevator_source"]) == ("available", "user")

    # Only the floor changes: same building, so the elevator answer stays.
    refloored = origin(road_address="서울 중구 퇴계로36길 2", floor=4, detail_address="402호")
    assert refloored["floor"] == 4
    assert (refloored["elevator"], refloored["elevator_source"]) == ("available", "user")

    # A different building: the old answer no longer applies.
    moved = origin(road_address="서울 중구 퇴계로36길 4", floor=2)
    assert (moved["elevator"], moved["elevator_source"]) == ("unknown", None)

    # The user corrects it again for the new building.
    corrected = origin(road_address="서울 중구 퇴계로36길 4", floor=2, elevator="unavailable")
    assert (corrected["elevator"], corrected["elevator_source"]) == ("unavailable", "user")


def test_elevator_kept_per_end(client):
    url, headers = guest_move(client)
    client.patch(url, headers=headers, json={
        "origin": {**ORIGIN, "elevator": "available"},
        "destination": {**DESTINATION, "elevator": "unavailable"},
    })
    # Re-saving only the destination's floor keeps both answers.
    data = client.patch(url, headers=headers, json={
        "destination": {**DESTINATION, "floor": 6},
    }).json()["data"]
    assert data["origin"]["value"]["elevator"] == "available"
    assert data["destination"]["value"]["elevator"] == "unavailable"
    assert data["destination"]["value"]["floor"] == 6

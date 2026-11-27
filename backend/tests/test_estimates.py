from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas.estimates import EstimateOptions
from backend.app.schemas.inventory import InventoryItem
from backend.app.schemas.moves import DayInfoOut
from backend.app.services.estimates import EstimateCapacityExceeded, calculate_estimate
from backend.app.services.moves import PlaceRecord, ScheduleRecord
from backend.app.services.route import Point, Route

MOVES = "/api/v1/moves"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1000


class DemoRouter:
    def route(self, origin, destination):
        return Route(18_000, 1_200, Point(127.0, 37.5), Point(126.9, 37.5))


def test_estimate_api_returns_card_prices_and_assumptions(tmp_path):
    settings = Settings(media_dir=tmp_path, analysis_demo=True, analysis_demo_seconds=1)
    with TestClient(create_app(settings)) as client:
        client.app.state.router = DemoRouter()
        now = 1_800_000_000
        client.app.state.analysis_store.clock = lambda: now
        move = client.post(MOVES).json()
        url = f"{MOVES}/{move['move_id']}"
        headers = {"X-Guest-Token": move["guest_token"]}
        route = f"{url}/estimate"
        assert client.post(route).status_code == 401
        assert client.post(route, headers={"X-Guest-Token": "wrong"}).status_code == 404
        assert client.post(route, headers=headers).status_code == 409
        assert client.post(
            f"{url}/media", headers=headers, files={"file": ("room.mp4", MP4, "video/mp4")},
        ).status_code == 201
        now += 2
        assert client.post(f"{url}/inventory/confirm", headers=headers).status_code == 204
        missing = client.post(route, headers=headers).json()
        assert missing["ajd"] == {"general": 124000, "semi_pack": 354000, "full_pack": 554000}
        assert missing["yes2404"] == {"general": 150000, "semi_pack": 380000, "full_pack": 400000}
        assert "자동차 경로 미확인으로 거리 18km 가정" in missing["assumptions"]
        assert "일정 미입력으로 일반 평일 10시 시작 가정" in missing["assumptions"]
        assert "출발지 층수 미확인으로 진입층에서 바로 운반 가정" in missing["assumptions"]
        update = {
            "origin": {"road_address": "서울특별시 강남구 개포로109길 5", "floor": 3},
            "destination": {"road_address": "서울특별시 마포구 월드컵로 1", "floor": 5},
        }
        assert client.patch(url, headers=headers, json=update).status_code == 200
        options = {"origin": {"method": "stairs"}, "destination": {"method": "ladder"}}
        expected = {
            "ajd": {"general": 304000, "semi_pack": 534000, "full_pack": 734000},
            "yes2404": {"general": 330000, "semi_pack": 560000, "full_pack": 580000},
        }
        estimated = client.post(route, headers=headers, json=options)
        assert estimated.status_code == 200, estimated.text
        body = estimated.json()
        assert set(body) == {"ajd", "yes2404", "assumptions"}
        assert {provider: body[provider] for provider in expected} == expected
        assert not any("거리 18km 가정" in assumption for assumption in body["assumptions"])

        options["destination"]["handling_fee_krw"] = 0
        options["destination"]["carry_distance_m"] = 25
        result = client.post(route, headers=headers, json=options)
        assert result.status_code == 200, result.text
        assert result.json()["yes2404"]["full_pack"] == 450000
        assert client.delete(f"{url}/inventory/confirm", headers=headers).status_code == 204
        assert client.post(route, headers=headers).status_code == 409
        added = client.post(
            f"{url}/inventory/items", headers=headers, json={
                "label_ko": "이사 상자", "quantity": 11,
                "dimensions_cm": {"width": 100, "height": 100, "depth": 100},
            },
        )
        assert added.status_code == 201, added.text
        assert client.post(f"{url}/inventory/confirm", headers=headers).status_code == 204
        assert client.post(route, headers=headers, json=options).json()["yes2404"]["full_pack"] == (
            650000
        )


def test_handless_saturday_uses_highest_date_rate_and_second_floor_uses_elevator():
    origin = PlaceRecord("서울특별시 중구 필동로1길 30", None, 2, "available", "user")
    day = DayInfoOut(
        weekend=True, friday=False, month_end=False, holiday=False, son_eomneun=True,
        pre_holiday=False, holiday_status="confirmed", lunar_status="found",
        checked_at=datetime(2026, 12, 5, tzinfo=timezone.utc),
    )
    result = calculate_estimate(
        {"origin": origin}, {}, None, ScheduleRecord("2026-12-05", "10:00:00"), day,
        EstimateOptions(), {},
    )
    assert result.yes2404.full_pack == 520_000  # 44만 + 18km 3만 + 승강기 5만
    assert "출발지 승강기 사용 가정" in result.assumptions


def test_inventory_volume_uses_quantity_and_adds_each_extra_vehicle_once():
    box = InventoryItem(
        item_id="box", source="user", label_ko="상자", quantity=10,
        dimensions_cm={"width": 100, "depth": 100, "height": 100},
    )
    missing = InventoryItem(
        item_id="unmeasured", source="user", label_ko="의자", quantity=2, dimensions_cm={},
    )
    items = {item.item_id: item for item in (box, missing)}
    one = calculate_estimate({}, {}, None, None, None, EstimateOptions(), items)
    assert one.ajd.general == 124_000
    assert any("물품 2개" in note for note in one.assumptions)

    box.quantity = 11
    two = calculate_estimate({}, {}, None, None, None, EstimateOptions(extra_vehicles=1), items)
    assert two.ajd.general == 324_000
    assert two.yes2404.full_pack == 600_000

    confirmed = calculate_estimate(
        {}, {}, None, None, None, EstimateOptions(load_volume_m3=10), items,
    )
    assert confirmed.ajd.general == one.ajd.general
    box.quantity = 61
    with pytest.raises(EstimateCapacityExceeded):
        calculate_estimate({}, {}, None, None, None, EstimateOptions(), items)


def test_waiting_terms_apply_to_yes2404_only_and_cap_the_delay_fee():
    options = EstimateOptions(deposit_krw=50_000, confirmed_wait_minutes=330)
    result = calculate_estimate({}, {}, None, None, None, options, {})
    assert result.yes2404.full_pack == 500_000  # 계약금 2배가 상한
    assert result.ajd.full_pack == 674_000  # 시작한 시간 기준, 6시간 × 2만 원

    options.confirmed_wait_minutes = 90
    result = calculate_estimate({}, {}, None, None, None, options, {})
    assert result.yes2404.full_pack == 425_000  # 1시간 미만 잔여 시간 제외
    assert result.ajd.full_pack == 594_000  # 시작한 시간 기준, 2시간 × 2만 원

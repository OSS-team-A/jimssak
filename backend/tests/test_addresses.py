import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.address_search import (
    AddressCandidate,
    AddressKeywordRejected,
    AddressPage,
    AddressSearchUnavailable,
    JusoAddressSearch,
    parse_page,
)

URL = "/api/v1/addresses"

# Shape of a real 행정안전부 response for "서울특별시 강남구 개포동 12" (checked 2026-11-24).
JUSO_BODY = {
    "common": {"errorCode": "0", "errorMessage": "정상", "totalCount": "5",
               "currentPage": "1", "countPerPage": "5"},
    "juso": [
        {"roadAddr": "서울특별시 강남구 개포로109길 5 (개포동, 대치아파트)",
         "jibunAddr": "서울특별시 강남구 개포동 12 대치아파트", "bdNm": "대치아파트",
         "admCd": "1168010300", "mtYn": "0", "lnbrMnnm": "12", "lnbrSlno": "0",
         "zipNo": "06306"},
        {"roadAddr": "서울특별시 강남구 개포로 603 (개포동)",
         "jibunAddr": "서울특별시 강남구 개포동 12-7", "bdNm": "",
         "admCd": "1168010300", "mtYn": "0", "lnbrMnnm": "12", "lnbrSlno": "7",
         "zipNo": "06306"},
    ],
}


class FakeSearch:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def search(self, keyword, page, per_page):
        self.calls.append((keyword, page, per_page))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(tmp_path):
    settings = Settings(auth_db_path=tmp_path / "app.sqlite3", address_search_api_key=None)
    with TestClient(create_app(settings)) as client:
        yield client


def use(client, fake):
    client.app.state.address_search = fake
    return fake


def test_search_returns_candidates_with_building_lookup_codes(client):
    fake = use(client, FakeSearch(parse_page(JUSO_BODY, 1, 10)))
    response = client.get(URL, params={"keyword": "  서울특별시 강남구 개포동 12  "})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 5 and body["page"] == 1 and body["per_page"] == 10
    assert body["results"][0] == {
        "road_address": "서울특별시 강남구 개포로109길 5 (개포동, 대치아파트)",
        "jibun_address": "서울특별시 강남구 개포동 12 대치아파트",
        "building_name": "대치아파트",
        "adm_cd": "1168010300",
        "mountain": False,
        "lot_main": 12,
        "lot_sub": 0,
        "zip_code": "06306",
    }
    # Different lot on the same block: the sub number tells them apart.
    assert body["results"][1]["lot_sub"] == 7
    assert body["results"][1]["building_name"] is None
    # The keyword reaches the provider trimmed, with the default page size.
    assert fake.calls == [("서울특별시 강남구 개포동 12", 1, 10)]


def test_paging_is_passed_through(client):
    fake = use(client, FakeSearch(AddressPage([], 0, 3, 5)))
    body = client.get(URL, params={"keyword": "퇴계로36길", "page": 3, "per_page": 5}).json()
    assert body == {"results": [], "total": 0, "page": 3, "per_page": 5}
    assert fake.calls == [("퇴계로36길", 3, 5)]


def test_no_login_needed(client):
    use(client, FakeSearch(AddressPage([], 0, 1, 10)))
    assert client.get(URL, params={"keyword": "퇴계로36길"}).status_code == 200


@pytest.mark.parametrize("params", [
    {},
    {"keyword": ""},
    {"keyword": "퇴"},
    {"keyword": "   퇴  "},
    {"keyword": "가" * 81},
    {"keyword": "퇴계로", "page": 0},
    {"keyword": "퇴계로", "per_page": 21},
])
def test_bad_keyword_or_paging_is_422(client, params):
    fake = use(client, FakeSearch(AddressPage([], 0, 1, 10)))
    assert client.get(URL, params=params).status_code == 422
    assert fake.calls == []


def test_keyword_rejected_by_provider_is_422_with_its_reason(client):
    use(client, FakeSearch(error=AddressKeywordRejected("검색어에 특수문자를 사용할 수 없습니다.")))
    response = client.get(URL, params={"keyword": "퇴계로;--"})
    assert response.status_code == 422
    assert response.json()["detail"] == "검색어에 특수문자를 사용할 수 없습니다."


def test_provider_failure_is_503_without_details(client):
    use(client, FakeSearch(error=AddressSearchUnavailable("error code E0001 key=secret")))
    response = client.get(URL, params={"keyword": "퇴계로36길"})
    assert response.status_code == 503
    assert "secret" not in response.text and "E0001" not in response.text
    assert "직접 입력" in response.json()["detail"]


def test_missing_key_is_503(client):
    # The app built from settings without a key; no network call happens.
    response = client.get(URL, params={"keyword": "퇴계로36길"})
    assert response.status_code == 503


# --- parsing the provider response ---

def test_parse_empty_result():
    page = parse_page({"common": {"errorCode": "0", "totalCount": "0"}, "juso": None}, 1, 10)
    assert page == AddressPage([], 0, 1, 10)


def test_parse_mountain_lot():
    item = {**JUSO_BODY["juso"][0], "mtYn": "1"}
    page = parse_page({"common": JUSO_BODY["common"], "juso": [item]}, 1, 10)
    assert page.results[0].mountain is True


@pytest.mark.parametrize(("code", "error"), [
    ("E0001", AddressSearchUnavailable),  # 승인되지 않은 키
    ("E0014", AddressSearchUnavailable),  # 개발 승인키 기간 만료
    ("-999", AddressSearchUnavailable),   # 시스템 오류
    ("E0005", AddressKeywordRejected),    # 검색어 없음
    ("E0013", AddressKeywordRejected),    # 특수문자·SQL 예약어
])
def test_parse_error_codes(code, error):
    with pytest.raises(error):
        parse_page({"common": {"errorCode": code, "errorMessage": "x"}, "juso": None}, 1, 10)


def test_parse_unexpected_shape_is_unavailable():
    with pytest.raises(AddressSearchUnavailable):
        parse_page({"common": {"errorCode": "0", "totalCount": "1"}, "juso": [{"roadAddr": "x"}]},
                   1, 10)


# --- the real client, with the network replaced ---

def test_client_sends_key_and_parses(monkeypatch):
    sent = {}

    def fake_post(url, data, timeout):
        sent.update(url=url, data=data, timeout=timeout)
        return httpx.Response(200, json={"results": JUSO_BODY}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    page = JusoAddressSearch("https://juso.example/api", "test-key", 3.0).search("개포동 12", 2, 5)
    assert sent["data"] == {"confmKey": "test-key", "keyword": "개포동 12", "currentPage": 2,
                            "countPerPage": 5, "resultType": "json"}
    assert sent["timeout"] == 3.0
    assert page.results[0] == AddressCandidate(
        "서울특별시 강남구 개포로109길 5 (개포동, 대치아파트)",
        "서울특별시 강남구 개포동 12 대치아파트",
        "대치아파트", "1168010300", False, 12, 0, "06306",
    )


@pytest.mark.parametrize("failure", [
    httpx.ConnectTimeout("timed out"),
    httpx.ConnectError("no route"),
])
def test_client_network_failure_is_unavailable(monkeypatch, failure):
    def fake_post(url, data, timeout):
        raise failure

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(AddressSearchUnavailable):
        JusoAddressSearch("https://juso.example/api", "test-key", 3.0).search("개포동 12", 1, 10)


def test_client_bad_status_or_body_is_unavailable(monkeypatch):
    request = httpx.Request("POST", "https://juso.example/api")
    responses = iter([httpx.Response(500, request=request),
                      httpx.Response(200, text="<html>", request=request)])
    monkeypatch.setattr(httpx, "post", lambda url, data, timeout: next(responses))
    client = JusoAddressSearch("https://juso.example/api", "test-key", 3.0)
    for _ in range(2):
        with pytest.raises(AddressSearchUnavailable):
            client.search("개포동 12", 1, 10)


def test_client_without_key_never_calls_out(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("network called"))
    with pytest.raises(AddressSearchUnavailable):
        JusoAddressSearch("https://juso.example/api", None, 3.0).search("개포동 12", 1, 10)

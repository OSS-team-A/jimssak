"""Address search through 행정안전부 실시간 주소정보 조회(검색API).

The browser never calls the government API directly: the key stays on the server,
and only the fields the app needs are passed on.
"""

from dataclasses import dataclass
from typing import Protocol

import httpx

# juso error codes that mean the keyword itself was rejected (the user can fix it).
# Other codes (unapproved or expired key, system errors) are our problem, not theirs.
KEYWORD_ERRORS = {
    "E0005",  # 검색어 없음
    "E0006",  # 시도명만 입력
    "E0008",  # 한 글자 검색어
    "E0009",  # 숫자만 입력
    "E0010",  # 검색어가 너무 김
    "E0011",  # 너무 긴 숫자
    "E0012",  # 특수문자와 숫자만
    "E0013",  # SQL 예약어 또는 특수문자
    "E0015",  # 검색 범위 초과
}


@dataclass(frozen=True)
class AddressCandidate:
    road_address: str
    jibun_address: str
    building_name: str | None
    # 행정구역코드, 10 digits: first 5 = 시군구코드, last 5 = 법정동코드.
    adm_cd: str
    # True when the lot is on a mountain (산 지번).
    mountain: bool
    lot_main: int
    lot_sub: int
    zip_code: str | None


@dataclass(frozen=True)
class AddressPage:
    results: list[AddressCandidate]
    total: int
    page: int
    per_page: int


class AddressSearchUnavailable(Exception):
    """The search could not be done right now (no key, network, provider error)."""


class AddressKeywordRejected(ValueError):
    """The provider refused the keyword; the message can be shown to the user."""


class AddressSearch(Protocol):
    def search(self, keyword: str, page: int, per_page: int) -> AddressPage: ...


class JusoAddressSearch:
    def __init__(self, url: str, api_key: str | None, timeout: float):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout

    def search(self, keyword: str, page: int, per_page: int) -> AddressPage:
        if not self.api_key:
            raise AddressSearchUnavailable("ADDRESS_SEARCH_API_KEY is not set")
        try:
            response = httpx.post(
                self.url,
                data={
                    "confmKey": self.api_key,
                    "keyword": keyword,
                    "currentPage": page,
                    "countPerPage": per_page,
                    "resultType": "json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()["results"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise AddressSearchUnavailable(f"address search failed: {type(exc).__name__}") from exc
        return parse_page(body, page, per_page)


def parse_page(body: dict, page: int, per_page: int) -> AddressPage:
    common = body.get("common") or {}
    code = str(common.get("errorCode", ""))
    if code in KEYWORD_ERRORS:
        raise AddressKeywordRejected(common.get("errorMessage") or "검색어를 확인해 주세요.")
    if code != "0":
        raise AddressSearchUnavailable(f"address search error code {code or 'missing'}")
    try:
        results = [_candidate(item) for item in body.get("juso") or []]
        total = int(common.get("totalCount") or 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise AddressSearchUnavailable("unexpected address search response") from exc
    return AddressPage(results=results, total=total, page=page, per_page=per_page)


def _candidate(item: dict) -> AddressCandidate:
    return AddressCandidate(
        road_address=item["roadAddr"],
        jibun_address=item["jibunAddr"],
        building_name=(item.get("bdNm") or "").strip() or None,
        adm_cd=item["admCd"],
        mountain=str(item.get("mtYn", "0")) == "1",
        lot_main=int(item["lnbrMnnm"]),
        lot_sub=int(item.get("lnbrSlno") or 0),
        zip_code=item.get("zipNo") or None,
    )

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..schemas.addresses import AddressSearchResult
from ..services.address_search import (
    AddressKeywordRejected,
    AddressSearch,
    AddressSearchUnavailable,
)

router = APIRouter(prefix="/api/v1/addresses", tags=["addresses"])


def get_address_search(request: Request) -> AddressSearch:
    return request.app.state.address_search


@router.get(
    "",
    summary="주소 검색",
    description=(
        "SC-06 주소 검색 시트용. 서버가 행정안전부 주소 검색 API를 대신 호출해 후보를 반환한다. "
        "후보의 `adm_cd`, `mountain`, `lot_main`, `lot_sub`는 이후 건물 정보 조회에 쓰므로, "
        "사용자가 고른 후보를 그대로 보관해 둔다. 로그인하지 않아도 사용할 수 있다."
    ),
    responses={
        422: {"description": "검색어가 비었거나 너무 짧음·긺, 또는 주소 검색이 거부한 검색어"},
        503: {"description": "주소 검색을 잠시 사용할 수 없음(키 미설정, 외부 API 장애·시간 초과)"},
    },
)
def search_addresses(
    search: Annotated[AddressSearch, Depends(get_address_search)],
    keyword: Annotated[
        str,
        Query(min_length=2, max_length=80, description="도로명, 건물명, 지번 등. 예: 퇴계로36길"),
    ],
    page: Annotated[int, Query(ge=1, le=100)] = 1,
    per_page: Annotated[int, Query(ge=1, le=20)] = 10,
) -> AddressSearchResult:
    keyword = keyword.strip()
    if len(keyword) < 2:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "검색어를 두 글자 이상 입력해 주세요."
        )
    try:
        found = search.search(keyword, page, per_page)
    except AddressKeywordRejected as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except AddressSearchUnavailable as exc:
        # The cause is for server logs only; never show keys or provider details to users.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "주소 검색을 잠시 사용할 수 없습니다. "
            "주소를 직접 입력하거나 잠시 후 다시 시도해 주세요.",
        ) from exc
    return AddressSearchResult(
        results=[asdict(item) for item in found.results],
        total=found.total, page=found.page, per_page=found.per_page,
    )

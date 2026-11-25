from pydantic import BaseModel, Field


class AddressCandidateOut(BaseModel):
    """주소 후보 하나. 사용자가 고르면 이사 저장(`PATCH /moves/{move_id}`)의 출발지·도착지에
    `road_address`, `building_name`, `adm_cd`, `mountain`, `lot_main`, `lot_sub`를 넣는다."""

    road_address: str = Field(description="도로명주소, 예: 서울특별시 중구 퇴계로36길 2 (필동3가)")
    jibun_address: str = Field(description="지번주소")
    building_name: str | None = Field(description="건물명. 없으면 null")
    adm_cd: str = Field(
        description=(
            "행정구역코드 10자리. 앞 5자리 시군구코드, 뒤 5자리 법정동코드(건축물대장 조회용)"
        )
    )
    mountain: bool = Field(description="산 지번이면 true")
    lot_main: int = Field(description="지번 본번")
    lot_sub: int = Field(description="지번 부번. 없으면 0")
    zip_code: str | None = Field(description="우편번호")


class AddressSearchResult(BaseModel):
    """주소 검색 결과."""

    results: list[AddressCandidateOut] = Field(
        description="이번 페이지의 후보. 사용자가 고른 후보는 이사 저장에 그대로 쓴다."
    )
    total: int = Field(description="검색된 전체 주소 수")
    page: int = Field(description="현재 페이지(1부터)")
    per_page: int = Field(description="한 페이지 후보 수")

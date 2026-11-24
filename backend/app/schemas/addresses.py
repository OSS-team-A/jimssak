from pydantic import BaseModel, Field


class AddressCandidateOut(BaseModel):
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
    results: list[AddressCandidateOut]
    total: int = Field(description="검색된 전체 주소 수")
    page: int
    per_page: int

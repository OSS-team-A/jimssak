from pathlib import Path

from pydantic import PositiveFloat, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    auth_db_path: Path = BACKEND_DIR / "data" / "auth.sqlite3"
    auth_session_ttl_seconds: PositiveInt = 86400
    auth_refresh_ttl_seconds: PositiveInt = 14 * 24 * 60 * 60
    # A guest move stays open for this long after its last request.
    guest_move_ttl_seconds: PositiveInt = 24 * 60 * 60
    media_dir: Path = BACKEND_DIR / "data" / "media"
    media_max_bytes: PositiveInt = 500 * 1024 * 1024
    # 행정안전부 실시간 주소정보 조회(검색API) 승인키. Without it, address search returns 503.
    address_search_api_key: SecretStr | None = None
    address_search_url: str = "https://business.juso.go.kr/addrlink/addrLinkApi.do"
    external_api_timeout_seconds: PositiveFloat = 5.0
    # 공공데이터포털 일반 인증키(Decoding): 건축물대장 등. Without it, building lookups are skipped.
    data_go_kr_service_key: SecretStr | None = None
    # Total time one building lookup may take, retries included.
    building_lookup_budget_seconds: PositiveFloat = 6.0
    # Per request: good answers come back in under a second, while the register's 503
    # SERVICETIMEOUT_ERROR takes about 6 s. Giving up early leaves time to retry.
    building_register_request_timeout_seconds: PositiveFloat = 2.0
    building_register_url: str = (
        "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
    )
    # 카카오 디벨로퍼스 REST API 키: 주소 좌표(로컬)와 자동차 길찾기(카카오모빌리티).
    kakao_rest_api_key: SecretStr | None = None
    kakao_geocode_url: str = "https://dapi.kakao.com/v2/local/search/address.json"
    kakao_directions_url: str = "https://apis-navi.kakaomobility.com/v1/directions"
    # Total time one route calculation may take (two geocodes + directions, retries included).
    route_lookup_budget_seconds: PositiveFloat = 6.0
    # 한국천문연구원 특일(공휴일)·음양력 정보, also with DATA_GO_KR_SERVICE_KEY.
    kasi_holiday_url: str = (
        "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
    )
    kasi_lunar_url: str = (
        "https://apis.data.go.kr/B090041/openapi/service/LrsrCldInfoService/getLunCalInfo"
    )
    calendar_request_timeout_seconds: PositiveFloat = 3.0
    calendar_lookup_budget_seconds: PositiveFloat = 6.0
    # Video analysis. The AI analyzer authenticates with this key on the internal API;
    # without it the internal API is off.
    analysis_worker_key: SecretStr | None = None
    # A job with no result after this long is failed as timed out.
    analysis_timeout_seconds: PositiveInt = 600
    # Until the AI analyzer exists: finish every upload with a fixed sample result
    # (marked demo) so the front end can build its screens. Off by default.
    analysis_demo: bool = False
    analysis_demo_seconds: PositiveFloat = 6.0

    @field_validator("auth_db_path", "media_dir")
    @classmethod
    def resolve_backend_path(cls, path: Path) -> Path:
        return BACKEND_DIR / path

    @field_validator(
        "address_search_api_key", "data_go_kr_service_key", "kakao_rest_api_key",
        "analysis_worker_key", mode="before"
    )
    @classmethod
    def blank_key_is_missing(cls, key: object) -> object:
        # `KEY=` left empty in .env means "not configured", not an empty key.
        return None if isinstance(key, str) and not key.strip() else key

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
    building_register_url: str = (
        "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
    )

    @field_validator("auth_db_path", "media_dir")
    @classmethod
    def resolve_backend_path(cls, path: Path) -> Path:
        return BACKEND_DIR / path

    @field_validator("address_search_api_key", "data_go_kr_service_key", mode="before")
    @classmethod
    def blank_key_is_missing(cls, key: object) -> object:
        # `KEY=` left empty in .env means "not configured", not an empty key.
        return None if isinstance(key, str) and not key.strip() else key

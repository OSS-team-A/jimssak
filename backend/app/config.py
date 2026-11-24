from pathlib import Path

from pydantic import PositiveInt, field_validator
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

    @field_validator("auth_db_path")
    @classmethod
    def resolve_db_path(cls, path: Path) -> Path:
        return BACKEND_DIR / path

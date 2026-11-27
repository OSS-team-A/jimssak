import os
import secrets

import httpx
import psycopg
import pytest
from psycopg.conninfo import make_conninfo

# Tests need a PostgreSQL server; each test gets its own empty schema in this database.
# Locally: `docker compose up -d db` (same default as the app). CI sets TEST_DATABASE_URL.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://jimssak:jimssak@127.0.0.1:5432/jimssak"
)

# Settings also read backend/.env, where developers keep real API keys. Environment
# variables win over that file, so blanking them here keeps every test off the network.
EXTERNAL_KEYS = (
    "ADDRESS_SEARCH_API_KEY", "DATA_GO_KR_SERVICE_KEY", "KAKAO_REST_API_KEY", "LLM_API_KEY",
)


@pytest.fixture(autouse=True)
def no_real_api_keys(monkeypatch):
    for name in EXTERNAL_KEYS:
        monkeypatch.setenv(name, "")
    # Developers turn on ANALYSIS_DEMO in backend/.env for frontend work; tests that need the
    # demo analyzer pass analysis_demo=True themselves.
    monkeypatch.setenv("ANALYSIS_DEMO", "false")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if a test reaches a real external API; tests patch httpx themselves."""

    def blocked(*args, **kwargs):
        raise AssertionError("a test tried to call a real external API")

    for name in ("get", "post"):
        monkeypatch.setattr(httpx, name, blocked)


@pytest.fixture(autouse=True)
def database_url(monkeypatch):
    """A fresh schema per test; Settings picks it up through DATABASE_URL."""
    schema = f"test_{secrets.token_hex(6)}"
    try:
        admin = psycopg.connect(TEST_DATABASE_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as error:
        pytest.exit(f"Tests need PostgreSQL at TEST_DATABASE_URL ({error}). "
                    "Start one with `docker compose up -d db`.", returncode=1)
    with admin:
        admin.execute(f"CREATE SCHEMA {schema}")
        url = make_conninfo(TEST_DATABASE_URL, options=f"-c search_path={schema}")
        monkeypatch.setenv("DATABASE_URL", url)
        yield url
        admin.execute(f"DROP SCHEMA {schema} CASCADE")

import httpx
import pytest

# Settings also read backend/.env, where developers keep real API keys. Environment
# variables win over that file, so blanking them here keeps every test off the network.
EXTERNAL_KEYS = ("ADDRESS_SEARCH_API_KEY", "DATA_GO_KR_SERVICE_KEY", "KAKAO_REST_API_KEY")


@pytest.fixture(autouse=True)
def no_real_api_keys(monkeypatch):
    for name in EXTERNAL_KEYS:
        monkeypatch.setenv(name, "")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if a test reaches a real external API; tests patch httpx themselves."""

    def blocked(*args, **kwargs):
        raise AssertionError("a test tried to call a real external API")

    for name in ("get", "post"):
        monkeypatch.setattr(httpx, name, blocked)

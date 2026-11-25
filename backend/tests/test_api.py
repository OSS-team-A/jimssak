import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[2]
client = TestClient(create_app(Settings(cors_origins=["http://localhost:5173"])))


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_cors_preflight():
    response = client.options("/health", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unknown_origin_rejected():
    response = client.options("/health", headers={
        "Origin": "https://unknown.example", "Access-Control-Request-Method": "GET",
    })
    assert response.status_code == 400


def test_backend_runs_without_the_ai_package():
    # The API and the AI analyzer ship as separate containers; they share only HTTP.
    code = "import sys; sys.modules['ai'] = None; import backend.app.main"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)

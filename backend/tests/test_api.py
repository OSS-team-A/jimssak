from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_demo_contract_and_missing_calibration():
    response = client.post("/api/v1/analysis/demo")
    assert response.status_code == 200
    result = response.json()
    assert result["schema_version"] == "1.0"
    assert result["mode"] == "demo"
    assert result["detections"][0]["dimensions"] is None
    assert result["detections"][0]["measurement_status"] == "needs_calibration"
    assert result["detections"][0]["object_id"] is None


def test_cors_preflight():
    response = client.options("/api/v1/analysis/demo", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unknown_origin_rejected():
    response = client.options("/api/v1/analysis/demo", headers={
        "Origin": "https://unknown.example", "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 400

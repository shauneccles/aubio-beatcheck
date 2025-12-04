from litestar.status_codes import HTTP_200_OK
from litestar.testing import TestClient


def test_health_check(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_list_suites(test_client: TestClient):
    response = test_client.get("/api/suites")
    assert response.status_code == HTTP_200_OK
    suites = response.json()
    assert len(suites) > 0
    assert suites[0]["id"] == "tempo"


def test_run_analysis_start(test_client: TestClient):
    response = test_client.post("/api/analyze/tempo", json={"duration": 1.0})
    assert response.status_code == 201  # Litestar returns 201 for POST by default
    data = response.json()
    assert data["status"] == "started"
    assert data["suite_id"] == "tempo"

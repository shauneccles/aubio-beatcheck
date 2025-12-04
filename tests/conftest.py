import pytest
from litestar.testing import TestClient

from web_api.main import app


@pytest.fixture
def test_client() -> TestClient:
    return TestClient(app=app)

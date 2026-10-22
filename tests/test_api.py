from fastapi.testclient import TestClient

from src.web_service.main import app

client = TestClient(app)


def test_health_check() -> None:
    """The home route answers."""
    assert client.get("/").json() == {"health_check": "App up and running!"}


def test_predict_rejects_bad_input() -> None:
    """A wrong sex and a negative length are refused with a 422."""
    response = client.post("/predict", json={"sex": "X", "length": -1})
    assert response.status_code == 422

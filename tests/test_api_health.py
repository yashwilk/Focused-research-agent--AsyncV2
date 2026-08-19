from fastapi.testclient import TestClient
from focused_research_agent.api.app import create_app

app = create_app()
client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

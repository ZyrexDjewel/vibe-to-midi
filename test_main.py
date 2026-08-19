import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check_or_docs():
    """Verify that the OpenAPI docs endpoint loads successfully."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_missing_prompt_validation():
    """Verify that posting an empty request returns a 422 Unprocessable Entity."""
    response = client.post("/api/v1/generate", json={})
    assert response.status_code == 422
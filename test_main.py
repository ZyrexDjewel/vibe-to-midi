import os
from unittest.mock import patch, MagicMock
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

def test_root_404():
    """Verify that accessing root path returns 404 Not Found as expected."""
    response = client.get("/")
    assert response.status_code == 404

@patch("main.genai.Client")
def test_generate_midi_success_mocked(mock_genai_client):
    """Verify full end-to-end MIDI generation pipeline with mocked Gemini API."""
    # 1. Set up fake API key in environment
    os.environ["GEMINI_API_KEY"] = "fake_test_api_key"

    # 2. Build mock JSON response matching Pydantic SongStructure schema
    mock_json_response = """{
        "bpm": 120,
        "notes": [
            {"pitch": 60, "start_time": 0.0, "end_time": 0.5, "velocity": 90},
            {"pitch": 64, "start_time": 0.5, "end_time": 1.0, "velocity": 95}
        ]
    }"""

    # 3. Configure mock object hierarchy
    mock_response_obj = MagicMock()
    mock_response_obj.text = mock_json_response

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response_obj
    mock_genai_client.return_value = mock_client_instance

    # 4. Perform POST request to API endpoint
    payload = {"prompt": "Upbeat synth arpeggio in C major"}
    response = client.post("/api/v1/generate", json=payload)

    # 5. Assert response status and MIDI headers
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/midi"
    assert "attachment" in response.headers.get("content-disposition", "")
    assert len(response.content) > 0  # Ensure non-empty binary MIDI stream
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app, limiter
from google.genai.errors import APIError

client = TestClient(app)


def setup_function():
    """Reset rate limiter before each test run."""
    limiter.reset()


def test_health_check_or_docs():
    """Verify that the OpenAPI docs endpoint loads successfully."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_health_check_endpoint():
    """Verify that the health check endpoint returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "vibe-to-midi"


def test_missing_prompt_validation():
    """Verify that posting an empty request returns a 422 Unprocessable Entity."""
    response = client.post("/api/v1/generate", json={})
    assert response.status_code == 422


def test_root_404():
    """Verify that accessing root path returns 404 Not Found as expected."""
    response = client.get("/")
    assert response.status_code == 404


def test_prompt_length_validation():
    """Verify that empty strings or excessively long prompts trigger a 422 error."""
    res_short = client.post("/api/v1/generate", json={"prompt": "hi"})
    assert res_short.status_code == 422

    res_long = client.post("/api/v1/generate", json={"prompt": "a" * 501})
    assert res_long.status_code == 422


def test_out_of_bounds_target_bpm():
    """Verify target_bpm outside 40-240 returns 422 Unprocessable Entity."""
    res_low = client.post(
        "/api/v1/generate",
        json={"prompt": "Slow jam", "target_bpm": 30},
    )
    assert res_low.status_code == 422

    res_high = client.post(
        "/api/v1/generate",
        json={"prompt": "Fast techno", "target_bpm": 300},
    )
    assert res_high.status_code == 422


@patch("main.genai.Client")
def test_generate_midi_api_error(mock_genai_client):
    """Verify that upstream Gemini API failures return 502 Bad Gateway."""
    os.environ["GEMINI_API_KEY"] = "fake_test_api_key"

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.side_effect = APIError(
        429, {"message": "Quota exceeded"}
    )
    mock_genai_client.return_value = mock_client_instance

    response = client.post("/api/v1/generate", json={"prompt": "Synth loop"})

    assert response.status_code == 502
    assert "Gemini API provider error" in response.json()["detail"]


@patch("main.genai.Client")
def test_generate_midi_success_mocked(mock_genai_client):
    """Verify full end-to-end MIDI generation pipeline with mocked Gemini API."""
    os.environ["GEMINI_API_KEY"] = "fake_test_api_key"

    mock_json_response = """{
        "bpm": 120,
        "tracks": [
            {
                "name": "Bass",
                "instrument_program": 38,
                "is_drum": false,
                "notes": [
                    {"pitch": 60, "start_time": 0.0, "end_time": 0.5, "velocity": 90},
                    {"pitch": 64, "start_time": 0.5, "end_time": 1.0, "velocity": 95}
                ]
            }
        ]
    }"""

    mock_response_obj = MagicMock()
    mock_response_obj.text = mock_json_response

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response_obj
    mock_genai_client.return_value = mock_client_instance

    payload = {"prompt": "Upbeat synth arpeggio in C major"}
    response = client.post("/api/v1/generate", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/midi"
    assert "attachment" in response.headers.get("content-disposition", "")
    assert len(response.content) > 0


@patch("main.genai.Client")
def test_generate_midi_with_target_bpm(mock_genai_client):
    """Verify target_bpm parameter is accepted and processed cleanly."""
    os.environ["GEMINI_API_KEY"] = "fake_test_api_key"

    mock_json_response = """{
        "bpm": 140,
        "tracks": [
            {
                "name": "Synth",
                "instrument_program": 80,
                "is_drum": false,
                "notes": [{"pitch": 60, "start_time": 0.0, "end_time": 0.5, "velocity": 100}]
            }
        ]
    }"""

    mock_response_obj = MagicMock()
    mock_response_obj.text = mock_json_response

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response_obj
    mock_genai_client.return_value = mock_client_instance

    payload = {"prompt": "Fast trance arpeggio", "target_bpm": 140}
    response = client.post("/api/v1/generate", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/midi"


@patch("main.genai.Client")
def test_generate_midi_with_key_signature(mock_genai_client):
    """Verify key_signature constraint parameter is accepted."""
    os.environ["GEMINI_API_KEY"] = "fake_test_api_key"

    mock_json_response = """{
        "bpm": 110,
        "tracks": [
            {
                "name": "Pad",
                "instrument_program": 89,
                "is_drum": false,
                "notes": [{"pitch": 57, "start_time": 0.0, "end_time": 2.0, "velocity": 80}]
            }
        ]
    }"""

    mock_response_obj = MagicMock()
    mock_response_obj.text = mock_json_response

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response_obj
    mock_genai_client.return_value = mock_client_instance

    payload = {"prompt": "Ambient ambient progression", "key_signature": "A Minor"}
    response = client.post("/api/v1/generate", json=payload)

    assert response.status_code == 200


@patch("main.genai.Client")
def test_global_exception_handler(mock_genai_client):
    """Verify that unhandled server errors return formatted 500 JSON payloads."""
    client_no_raise = TestClient(app, raise_server_exceptions=False)

    @app.get("/test-unhandled-error")
    def trigger_crash():
        raise RuntimeError("Unhandled crash")

    response = client_no_raise.get("/test-unhandled-error")
    assert response.status_code == 500
    assert response.json() == {"detail": "An unexpected internal server error occurred."}


def test_generate_midi_rate_limit(monkeypatch):
    """Verify that exceeding 5 requests per minute returns HTTP 429 Too Many Requests."""
    limiter.reset()

    mock_response = MagicMock()
    mock_response.text = """{
        "bpm": 120,
        "tracks": [
            {
                "name": "Lead",
                "instrument_program": 81,
                "is_drum": false,
                "notes": [{"pitch": 60, "start_time": 0.0, "end_time": 1.0, "velocity": 100}]
            }
        ]
    }"""

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    monkeypatch.setattr("main.genai.Client", lambda api_key: mock_client_instance)
    monkeypatch.setenv("GEMINI_API_KEY", "fake_test_key")

    payload = {"prompt": "Upbeat synth arpeggio in C major"}

    # Fire 5 valid requests (within limit)
    for _ in range(5):
        response = client.post("/api/v1/generate", json=payload)
        assert response.status_code == 200

    # 6th request should trigger HTTP 429 Rate Limit Exceeded
    rate_limited_response = client.post("/api/v1/generate", json=payload)
    assert rate_limited_response.status_code == 429


def test_cors_preflight_headers():
    """Verify that CORS middleware returns correct Access-Control headers for preflight requests."""
    response = client.options(
        "/api/v1/generate",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

from main import sanitize_and_quantize_tracks, TrackStructure, MIDINote

def test_midi_sanitization_and_quantization():
    """Verify that pitch/velocity clamping and time quantization work correctly."""
    raw_tracks = [
        TrackStructure(
            name="Test Track",
            instrument_program=0,
            is_drum=False,
            notes=[
                # Pitch > 127, velocity < 1, end_time <= start_time
                MIDINote(pitch=150, start_time=-0.02, end_time=0.0, velocity=-10),
                # Valid note requiring grid snapping
                MIDINote(pitch=60, start_time=0.123, end_time=0.498, velocity=90),
            ]
        )
    ]

    sanitized = sanitize_and_quantize_tracks(tracks=raw_tracks, bpm=120)
    notes = sanitized[0].notes

    # Assert Note 1 was clamped and given valid duration
    assert notes[0].pitch == 127
    assert notes[0].velocity == 1
    assert notes[0].start_time == 0.0
    assert notes[0].end_time > notes[0].start_time

    # Assert Note 2 was snapped to grid steps
    assert notes[1].start_time == 0.125  # 1/16th note step at 120 BPM
    assert notes[1].end_time == 0.5
# Vibe-to-MIDI

> An AI-powered microservice translating natural language text prompts into downloadable, binary MIDI files using Google Gemini structured output and FastAPI.

## Features
- **FastAPI Endpoint:** Streams `.mid` files directly to clients via `POST /api/v1/generate`.
- **Health & Uptime Monitoring:** Dedicated `/health` route for status checks.
- **Structured AI Output:** Enforces strict JSON song schema parsing via Pydantic V2 and `pretty_midi`.
- **General MIDI Support:** Customizable instrument program selection (0–127).
- **Observability & Resilience:** HTTP execution timing middleware, structured logging, and unified global error handlers.
- **Dockerized:** Fully containerized via `docker-compose`.
- **CI/CD:** Automated testing pipeline using GitHub Actions and `pytest`.

## Quickstart

```powershell
# Setup environment variables
Copy-Item .env.example .env

# Run with Docker
docker compose up --build

API Documentation
POST /api/v1/generate

Generates and downloads a .mid file based on a natural language text prompt.

Request Payload:
JSON

{
  "prompt": "Dark synthwave arpeggiated bassline loop at 110 BPM in A minor",
  "instrument_program": 38
}

Parameters & Validation:

    prompt (string, required): Vibe description. Must be 3 to 500 characters (min_length=3, max_length=500).

    instrument_program (integer, optional): General MIDI program number from 0 to 127 (defaults to 38 - Synth Bass 1).

Response Codes:

    200 OK: Returns the binary .mid file stream (audio/midi).

    422 Unprocessable Entity: Validation failure (empty prompt, prompt length out of bounds, or instrument_program outside 0–127).

    502 Bad Gateway: Upstream Google Gemini API error or provider rate limit.

    500 Internal Server Error: Server configuration issue or unexpected internal runtime exception ({"detail": "An unexpected internal server error occurred."}).

GET /health

Endpoint for uptime monitors, container orchestrators, and load balancers.

Response (200 OK):
JSON

{
  "status": "healthy",
  "service": "vibe-to-midi",
  "version": "1.0.0"
}

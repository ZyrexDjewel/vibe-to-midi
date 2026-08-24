# Vibe-to-MIDI

> An AI-powered microservice translating natural language text prompts into downloadable, binary MIDI files using Google Gemini and FastAPI.

## Features
- **FastAPI Endpoint:** Streams `.mid` files via `POST /api/v1/generate`.
- **Schema Enforcement:** Uses Pydantic for structured Gemini output.
- **Dockerized:** Fully containerized via `docker-compose`.
- **CI/CD:** Automated testing pipeline using GitHub Actions.

## Quickstart

```powershell
# Run with Docker
docker compose up --build

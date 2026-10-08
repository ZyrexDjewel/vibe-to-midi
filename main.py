import os
import time
import uuid
import logging
import tempfile
import pretty_midi
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from config import get_settings

load_dotenv()  # Automatically loads variables from .env into os.environ

# Load centralized settings instance
settings = get_settings()

# Configure structured logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vibe-to-midi")

app = FastAPI(
    title=settings.app_name,
    description="Generate MIDI files from text prompts using Gemini structured output.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,   # Uses list from config
    allow_credentials=True,
    allow_methods=["*"],                   # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global exception handler for unexpected 500 runtime errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred."}
    )

@app.get("/health", tags=["System"])
async def health_check():
    """Endpoint for uptime monitors and container health checks."""
    return {
        "status": "healthy",
        "service": "vibe-to-midi",
        "version": "1.0.0",
        "environment": settings.environment
    }

# Request execution timer & status logger middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    
    logger.info(
        f"Method={request.method} Path={request.url.path} "
        f"Status={response.status_code} Duration={duration}ms"
    )
    return response

# 1. Pydantic Schemas for Request & Response
class VibeRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=500,
        json_schema_extra={"example": "Dark synthwave bassline"}
    )
    target_bpm: Optional[int] = Field(
        default=None,
        ge=40,
        le=240,
        description="Optional target BPM (40-240)"
    )
    key_signature: Optional[str] = Field(
        default=None,
        description="Optional key constraint (e.g., 'C Major', 'A Minor')"
    )

# Updated Pydantic Schemas for Multi-Track Support
class MIDINote(BaseModel):
    pitch: int = Field(description="MIDI pitch from 0 to 127")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    velocity: int = Field(description="Note volume from 0 to 127")

class TrackStructure(BaseModel):
    name: str = Field(description="Track name (e.g., 'Bass', 'Lead', 'Drums')")
    instrument_program: int = Field(
        default=0, 
        ge=0, 
        le=127, 
        description="General MIDI program number (0-127). For drums, program is ignored if is_drum is True."
    )
    is_drum: bool = Field(default=False, description="True if this track is a percussion/drum track (Channel 10).")
    notes: list[MIDINote] = Field(description="List of notes for this track")

class SongStructure(BaseModel):
    bpm: int = Field(description="Tempo in BPM")
    tracks: list[TrackStructure] = Field(description="List of tracks forming the composition")

class JobStatus(BaseModel):
    job_id: str
    status: str  # "PENDING", "PROCESSING", "COMPLETED", "FAILED"
    error: Optional[str] = None
    created_at: float
    updated_at: float

# In-memory storage for async generation jobs
jobs_db: Dict[str, Dict[str, Any]] = {}

def remove_file(path: str):
    """Utility to remove temporary files after response streaming."""
    if os.path.exists(path):
        os.remove(path)


# 2. Sanitization & Quantization Utility
def sanitize_and_quantize_tracks(
    tracks: list[TrackStructure], bpm: int, grid_division: int = 16
) -> list[TrackStructure]:
    """
    Sanitizes raw AI output and quantizes note start/end times to a grid.
    - Clamps pitch (0-127) and velocity (1-127)
    - Fixes zero/negative duration notes
    - Quantizes start and end times to the nearest grid step (default: 16th notes)
    """
    seconds_per_beat = 60.0 / bpm
    grid_step_seconds = (seconds_per_beat * 4) / grid_division
    min_note_duration = max(0.05, grid_step_seconds / 2)

    for track in tracks:
        for note in track.notes:
            # Clamp pitch and velocity to safe General MIDI ranges
            note.pitch = max(0, min(127, note.pitch))
            note.velocity = max(1, min(127, note.velocity))

            # Prevent negative start times
            note.start_time = max(0.0, note.start_time)

            # Quantize times to nearest grid step
            quantized_start = round(note.start_time / grid_step_seconds) * grid_step_seconds
            quantized_end = round(note.end_time / grid_step_seconds) * grid_step_seconds

            # Ensure end time strictly follows start time with minimum duration
            if quantized_end <= quantized_start:
                quantized_end = quantized_start + min_note_duration

            note.start_time = round(quantized_start, 4)
            note.end_time = round(quantized_end, 4)

    return tracks


# 3. Async Background Job Worker
async def process_midi_job(job_id: str, payload: VibeRequest):
    """Executes heavy Gemini generation and MIDI writing in the background."""
    jobs_db[job_id]["status"] = "PROCESSING"
    jobs_db[job_id]["updated_at"] = time.time()

    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        jobs_db[job_id]["status"] = "FAILED"
        jobs_db[job_id]["error"] = "GEMINI_API_KEY environment variable missing"
        return

    try:
        full_prompt = payload.prompt
        if payload.target_bpm:
            full_prompt += f" Target tempo: {payload.target_bpm} BPM."
        if payload.key_signature:
            full_prompt += f" Target key signature: {payload.key_signature}."

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SongStructure,
                temperature=0.7,
            ),
        )

        song_data = SongStructure.model_validate_json(response.text)
        sanitized_tracks = sanitize_and_quantize_tracks(
            tracks=song_data.tracks, bpm=song_data.bpm
        )

        midi = pretty_midi.PrettyMIDI(initial_tempo=song_data.bpm)
        for track_data in sanitized_tracks:
            instrument = pretty_midi.Instrument(
                program=track_data.instrument_program,
                is_drum=track_data.is_drum,
                name=track_data.name
            )
            for n in track_data.notes:
                instrument.notes.append(
                    pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start_time, end=n.end_time)
                )
            midi.instruments.append(instrument)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        midi.write(temp_file.name)
        temp_file.close()

        jobs_db[job_id]["status"] = "COMPLETED"
        jobs_db[job_id]["file_path"] = temp_file.name
        jobs_db[job_id]["updated_at"] = time.time()

    except Exception as e:
        logger.error(f"Async job {job_id} failed: {str(e)}")
        jobs_db[job_id]["status"] = "FAILED"
        jobs_db[job_id]["error"] = str(e)
        jobs_db[job_id]["updated_at"] = time.time()


# 4. API Endpoints

# Synchronous generation endpoint
@app.post("/api/v1/generate")
@limiter.limit(settings.rate_limit_per_minute)
async def generate_midi(request: Request, payload: VibeRequest, background_tasks: BackgroundTasks):
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is missing from environment variables.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GEMINI_API_KEY environment variable missing"
        )

    try:
        # Build prompt with optional user constraints
        full_prompt = payload.prompt
        if payload.target_bpm:
            full_prompt += f" Target tempo: {payload.target_bpm} BPM."
        if payload.key_signature:
            full_prompt += f" Target key signature: {payload.key_signature}."

        # Initialize Gemini Client
        client = genai.Client(api_key=api_key)

        # Generate structured note array
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SongStructure,
                temperature=0.7,
            ),
        )

        song_data: SongStructure = SongStructure.model_validate_json(response.text)

    except APIError as e:
        # Handles upstream Gemini API errors (rate limits, bad keys, service outages)
        logger.warning(f"Gemini API returned error: {e.message}")
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API provider error: {e.message}"
        )
    except ValueError as e:
        # Handles cases where returned output fails Pydantic schema parsing
        logger.warning(f"Pydantic validation error: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse AI output into valid MIDI schema: {str(e)}"
        )
    except Exception as e:
        # Fallback catch-all for unexpected internal runtime failures
        logger.error(f"Unexpected error during generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal service error during generation: {str(e)}"
        )

    try:
        # Sanitize and quantize raw notes before building MIDI
        sanitized_tracks = sanitize_and_quantize_tracks(
            tracks=song_data.tracks, bpm=song_data.bpm
        )

        # Build multi-track MIDI file
        midi = pretty_midi.PrettyMIDI(initial_tempo=song_data.bpm)

        for track_data in sanitized_tracks:
            instrument = pretty_midi.Instrument(
                program=track_data.instrument_program,
                is_drum=track_data.is_drum,
                name=track_data.name
            )
            
            for n in track_data.notes:
                instrument.notes.append(
                    pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start_time, end=n.end_time)
                )
            midi.instruments.append(instrument)

        # Save to a unique temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        midi.write(temp_file.name)
        temp_file.close()

        background_tasks.add_task(remove_file, temp_file.name)

        return FileResponse(
            path=temp_file.name,
            filename="vibe.mid",
            media_type="audio/midi"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Async generation endpoint
@app.post("/api/v1/jobs", status_code=202)
@limiter.limit(settings.rate_limit_per_minute)
async def create_midi_job(request: Request, payload: VibeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    now = time.time()

    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "PENDING",
        "error": None,
        "file_path": None,
        "created_at": now,
        "updated_at": now
    }

    background_tasks.add_task(process_midi_job, job_id, payload)

    return {
        "job_id": job_id,
        "status": "PENDING",
        "status_url": f"/api/v1/jobs/{job_id}",
        "download_url": f"/api/v1/jobs/{job_id}/download"
    }


# Job status polling endpoint
@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        error=job["error"],
        created_at=job["created_at"],
        updated_at=job["updated_at"]
    )


# Job download endpoint
@app.get("/api/v1/jobs/{job_id}/download")
async def download_job_midi(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    if job["status"] in ["PENDING", "PROCESSING"]:
        raise HTTPException(status_code=400, detail="Job is still processing")
    if job["status"] == "FAILED":
        raise HTTPException(status_code=500, detail=f"Job generation failed: {job['error']}")

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generated MIDI file no longer exists")

    background_tasks.add_task(remove_file, file_path)

    return FileResponse(
        path=file_path,
        filename=f"vibe_{job_id[:8]}.mid",
        media_type="audio/midi"
    )
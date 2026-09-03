import os
import time
import logging
import tempfile
import pretty_midi
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Automatically loads variables from .env into os.environ

# Configure structured logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("vibe-to-midi")

app = FastAPI(
    title="Vibe-to-MIDI API",
    description="Generate MIDI files from text prompts using Gemini structured output.",
    version="1.0.0"
)

@app.get("/health", tags=["System"])
async def health_check():
    """Endpoint for uptime monitors and container health checks."""
    return {
        "status": "healthy",
        "service": "vibe-to-midi",
        "version": "1.0.0"
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
        json_schema_extra={"example": "Dark synthwave arpeggiated bassline loop at 110 BPM in A minor"}
    )
    # Expand VibeRequest schema in main.py
    instrument_program: int = Field(
        default=38, 
        ge=0, 
        le=127, 
        description="General MIDI program number (0-127). Defaults to 38 (Synth Bass 1)."
    )

class MIDINote(BaseModel):
    pitch: int = Field(description="MIDI pitch from 0 to 127")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    velocity: int = Field(description="Note volume from 0 to 127")

class SongStructure(BaseModel):
    bpm: int = Field(description="Tempo in BPM")
    notes: list[MIDINote] = Field(description="List of notes")

def remove_file(path: str):
    """Utility to remove temporary files after response streaming."""
    if os.path.exists(path):
        os.remove(path)


# 2. API Endpoint
@app.post("/api/v1/generate")
async def generate_midi(request: VibeRequest, background_tasks: BackgroundTasks):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is missing from environment variables.")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GEMINI_API_KEY environment variable missing"
            )

    try:
        # Initialize Gemini Client
        client = genai.Client(api_key=api_key)

        # Generate structured note array
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=request.prompt,
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
        # Build MIDI file using the user-requested instrument (or default to 38)
        midi = pretty_midi.PrettyMIDI(initial_tempo=song_data.bpm)
        synth = pretty_midi.Instrument(program=request.instrument_program)  # Synth Bass

        for n in song_data.notes:
            note = pretty_midi.Note(
                velocity=n.velocity,
                pitch=n.pitch,
                start=n.start_time,
                end=n.end_time
            )
            synth.notes.append(note)

        midi.instruments.append(synth)

        # Save to a unique temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        midi.write(temp_file.name)
        temp_file.close()

        # Schedule automatic cleanup after response streaming completes
        background_tasks.add_task(remove_file, temp_file.name)

        # Return file as downloadable attachment
        return FileResponse(
            path=temp_file.name,
            filename="vibe.mid",
            media_type="audio/midi"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
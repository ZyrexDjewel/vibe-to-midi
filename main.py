import os
import tempfile
import pretty_midi
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Automatically loads variables from .env into os.environ

app = FastAPI(
    title="Vibe-to-MIDI API",
    description="Generate MIDI files from text prompts using Gemini structured output.",
    version="1.0.0"
)

# 1. Pydantic Schemas for Request & Response
class VibeRequest(BaseModel):
    prompt: str = Field(
        ...,
        json_schema_extra={"example": "Dark synthwave arpeggiated bassline loop at 110 BPM in A minor"}
    )

class MIDINote(BaseModel):
    pitch: int = Field(description="MIDI pitch from 0 to 127")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    velocity: int = Field(description="Note volume from 0 to 127")

class SongStructure(BaseModel):
    bpm: int = Field(description="Tempo in BPM")
    notes: list[MIDINote] = Field(description="List of notes")


# 2. API Endpoint
@app.post("/api/v1/generate")
async def generate_midi(request: VibeRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")

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

        # Build MIDI file in memory/temp location
        midi = pretty_midi.PrettyMIDI(initial_tempo=song_data.bpm)
        synth = pretty_midi.Instrument(program=38)  # Synth Bass

        for n in song_data.notes:
            note = pretty_midi.Note(
                velocity=n.velocity,
                pitch=n.pitch,
                start=n.start_time,
                end=n.end_time
            )
            synth.notes.append(note)

        midi.instruments.append(synth)

        # Save to a temporary file
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, "generated_vibe.mid")
        midi.write(file_path)

        # Return file as downloadable attachment
        return FileResponse(
            path=file_path,
            filename="vibe.mid",
            media_type="audio/midi"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import os
import pretty_midi
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Define the Pydantic Schema for structured LLM output
class MIDINote(BaseModel):
    pitch: int = Field(description="MIDI pitch value from 0 to 127 (e.g., 60 is C4)")
    start_time: float = Field(description="Start offset in seconds")
    end_time: float = Field(description="End offset in seconds")
    velocity: int = Field(description="Note loudness from 0 to 127")

class SongStructure(BaseModel):
    bpm: int = Field(description="Tempo in beats per minute, e.g., 120")
    notes: list[MIDINote] = Field(description="List of notes forming the melody or loop")

# 2. Call Gemini with Schema Enforcement
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

prompt = "Dark synthwave arpeggiated bassline loop at 110 BPM in A minor"

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SongStructure,
        temperature=0.7,
    ),
)

# 3. Parse typed JSON response directly into Pydantic model
song_data: SongStructure = SongStructure.model_validate_json(response.text)
print(f"Generated {len(song_data.notes)} notes at {song_data.bpm} BPM!")

# 4. Convert output to MIDI file
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
midi.write("vibe_output.mid")
print("Saved generated vibe to vibe_output.mid!")
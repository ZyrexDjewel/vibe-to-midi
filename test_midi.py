import pretty_midi

# 1. Initialize MIDI object at 120 BPM
midi = pretty_midi.PrettyMIDI(initial_tempo=120)

# 2. Create an instrument track (0 = Acoustic Grand Piano)
piano = pretty_midi.Instrument(program=0)

# 3. Add a simple C major chord (C4, E4, G4)
# Note pitch 60 = Middle C, 64 = E, 67 = G
# Timestamps are in seconds (start at 0.0s, end at 2.0s)
notes_to_add = [60, 64, 67]

for pitch in notes_to_add:
    note = pretty_midi.Note(
        velocity=100,  # Volume (0-127)
        pitch=pitch,
        start=0.0,
        end=2.0
    )
    piano.notes.append(note)

# 4. Add piano track to MIDI object
midi.instruments.append(piano)

# 5. Save file to disk
midi.write("output.mid")
print("Success! Created output.mid in your folder.")
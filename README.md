# synth_rs

A real-time physical-modeling synthesizer in Rust — a desktop re-imagining of a
2014 embedded "electric air guitar" that generated audio from a plucked-string
model (the Function Transformation Method).

Work lands in reviewable pieces:

1. **Base framework** — audio + MIDI + on-screen keyboard, the generic modal
   engine, and a basic triangle/saw oscillator plugin.
2. **Musical String** plugin — a music-friendly plucked string.
3. **Pure String** plugin — the firmware-faithful FTM string.

More to come (2-D membranes, 3-D solids).

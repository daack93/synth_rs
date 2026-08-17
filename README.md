# synth_rs

A real-time physical-modeling synthesizer in Rust — a desktop re-imagining of a
2014 embedded "electric air guitar" that generated audio from a plucked-string
model (the **Function Transformation Method**, FTM). The original derived pitch
and strike from an accelerometer; here a keyboard does.

## Architecture

The FTM represents any vibrating object — a string, a membrane, a solid — as a
finite sum of exponentially-decaying sinusoids (*modes*). That is the seam the
whole app is built on:

- **`synth`** — a generic polyphonic engine. It owns voices, phase accumulators,
  the amplitude envelope, voice-stealing, and live parameter rebuilds. It knows
  nothing about strings or drums.
- **`models`** — pluggable synthesis "modes". Each plugin implements one trait,
  [`FtmModel`], whose only job is to fill a `ModeBuffer` with the
  `(frequency, amplitude, decay)` of each mode for a struck note. The engine
  plays whatever bank the active plugin produced.
- **`audio`** / **`midi`** — `cpal` output and `midir` input.
- **`main`** — the `egui` UI: model picker, per-model parameters, an on-screen
  piano, and computer-keyboard / MIDI input.

Adding a new mode (a 2-D drum head, a 3-D solid, a different excitation) is just
a new file implementing `FtmModel` and one line in the registry.

## Playing it

```sh
cargo run --release
```

- **Computer keys:** `A W S E D F T G Y H U J K`; `Z` / `X` shift octave.
- **Mouse:** click the on-screen piano (vertical position sets velocity).
- **MIDI:** pick your device from the dropdown (Rescan if you plug in later).

## Models

- **Musical String** — a plucked string with music-friendly controls
  (inharmonicity, decay time, a pluck-position sweep from triangle to saw).
- **Pure String** — the firmware-faithful FTM string, driven by the exact
  `#define`s from the 2014 `main.h` (stiffness, propagation speed, damping,
  frequency-dependent damping, length, `DEPTH`, `DAMP_PERIOD`, `TIME_SCALE`),
  with triangle vs. saw pluck geometry and the accelerometer velocity mapping.
- **Basic Wave** — band-limited triangle / sawtooth; the framework's reference
  oscillator.

## Roadmap

- 2-D membrane (drum) and 3-D solid FTM models.

# synth_rs

A real-time modeling synthesizer and mini production studio in Rust. Play
pluggable instruments from an on-screen piano, the computer keyboard, or a MIDI
controller, then record and arrange them into songs.

## Architecture

Most instruments use **modal synthesis** — a vibrating object (a string, a
membrane, a solid) represented as a finite sum of exponentially-decaying
sinusoids, its *modes* — but the framework itself is method-agnostic: a model is
free to generate its output however it likes. That plugin seam is what the app
is built on:

- **`instrument`** — a generic polyphonic engine. It owns voices, phase
  accumulators, the amplitude envelope, voice-stealing, and live parameter
  rebuilds. It knows nothing about strings or drums.
- **`models`** — the pluggable synthesizer models. Each implements one trait,
  [`FtmModel`], whose only job is to fill a `ModeBuffer` with the
  `(frequency, amplitude, decay)` of each partial for a played note. The engine
  plays whatever bank the active model produced.
- **`audio`** / **`midi`** — `cpal` output and `midir` input.
- **`studio`** — the multi-track host: transport, recording, and the
  self-contained clips that make up an arrangement.
- **`main`** — the `egui` UI: model picker, per-model parameters, an on-screen
  piano, and computer-keyboard / MIDI input.

Adding a model (a 2-D drum head, a 3-D solid, a different excitation) is just a
new file implementing `FtmModel` and one line in the registry.

Models can be **struck** (an impulse rings and decays — strings, drums) or
**sustained/driven** (a blown wind holds while the note is played and its loss
shapes the steady-state spectrum — the horn), set by a flag on the modal bank.

## Playing it

```sh
cargo run --release
```

- **Computer keys:** `A W S E D F T G Y H U J K`; `Z` / `X` shift octave.
- **Mouse:** click the on-screen piano (vertical position sets velocity).
- **MIDI:** pick your device from the dropdown (Rescan if you plug in later).

## Tempo & grid

The transport has a **Tempo** row: BPM, beats-per-bar, a **Recording Duration**
in bars (or *Free*), a **Quantize** grid (Off / ¼ / ⅛ / ⅛T / 1⁄16), a metronome
**🔔 Click**, and **Count-in**. With a fixed recording length, a take auto-closes
exactly on the bar; quantize snaps recorded notes to the grid so parts lock
together. Tempo settings save with the project.

## Recording & arranging

Work is built up as **tracks** of **clips** shown below the keyboard. Each clip
owns its own notes and its own loop settings, so it records and plays back with
no per-event bookkeeping.

- **▶ Play / ⏺ Record** — Record before playing counts in, then captures a clip
  at the seek cursor; Record while playing punches in. **🔁 Repeat** loops the
  arrangement (off = play through once).
- It's **multi-timbral** — a track remembers the instrument it was recorded
  with, so you can lay a bass line, switch to a guitar preset for the next
  track, and they play back with their own sounds.
- Each track row has mute, a note timeline with a moving playhead, delete, and
  an **✎ edit** button — pick a track and the right-hand panel edits *that
  track's* instrument (model, parameters, engine, or load a preset onto it) live
  while playback continues.
- Select a region of a clip to **crop / delete / reverse / loop** it; every edit
  forks the clip so clips never share content.

## Projects

The **Project** bar saves your work: name it and **💾 Save** to a JSON file (in
`projects/`, or `$FTM_SYNTH_PROJECTS`); **Open…** reloads it; **Clear project**
starts fresh (with a confirmation). All positions are stored in seconds, so
projects are portable across sample rates. **⬇ Export song** renders the whole
arrangement to a WAV.

## Presets

Build a library of instruments. A **preset** is a model + its parameters +
engine settings + a name, saved as one JSON file per preset. Type a name and
**Save**; pick from the dropdown to **Load**; **Delete** removes the saved file.
A kit preset (a key-mapped set of instruments) saves and loads the same way.
Presets live in `presets/` under the working directory, or wherever
`$FTM_SYNTH_PRESETS` points. **★ Factory** restores/refreshes the built-ins.

## Models

- **Musical String** — a plucked string with music-friendly controls
  (inharmonicity, decay time, a pluck-position sweep from triangle to saw).
- **Pure String** — the modal string in its rawest form: stiffness, propagation
  speed, damping, frequency-dependent damping, length, mode count, and a
  continuous pluck position (a centred pluck gives a triangle-like spectrum, a
  near-end pluck a saw-like one — those are just the extremes of the sweep).
- **Drum (2D membrane)** — a circular drumhead: the same modal recipe with the
  Laplacian ∇², so the modes are the inharmonic Bessel-zero series (1 : 1.59 :
  2.14 : 2.30 …). Controls for wave speed, stiffness, damping, radius, and
  strike position (centre → rim).
- **Quadratic Webster Horn** — a flaring air column via Webster's horn equation.
  A quadratic bore `r(x) = r1 + r2·x + r3·x²` becomes a geometric potential
  `V(x) = 2r3/r(x)`; the synth numerically solves the eigenproblem
  `φ'' − V(x)φ = λφ` for the resonances (its own symmetric-tridiagonal
  eigensolver). Dial the three radius coefficients, length, and blow position,
  and choose **Open** ends or a **Brass** closed mouthpiece (odd-harmonic base
  the flare fills in). Physically-motivated losses: **Keefe** viscothermal wall
  loss (∝ √f, stronger in narrow bores — warm/stuffed tone) and **radiation**
  loss at the bell (∝ f² — highs escape, lows sustain). Wavefronts can be flat
  discs (**Planar**) or curved spherical caps (**Spherical**, `S = 2π r²/(1+cos θ)`)
  for more accurate high partials where the flare is steep.
- **Basic Wave** — band-limited triangle / sawtooth; the framework's reference
  oscillator.

## Roadmap

- 3-D solid FTM models.

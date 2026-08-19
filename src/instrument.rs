//! One polyphonic instrument: a voice pool driven by an [`FtmModel`] plugin.
//!
//! This is the former `Synth` engine, now a reusable building block. A
//! [`crate::studio::Studio`] hosts several instruments at once (the live one you
//! play plus one per loop track) and mixes them, so an instrument renders a
//! single frame at a time via [`Instrument::render_frame`] rather than owning
//! the output buffer.

use std::sync::Arc;

use crate::models::{default_model, FtmModel, ModeBuffer, MAX_MODES};

/// Number of simultaneously sounding notes per instrument.
pub const MAX_VOICES: usize = 16;
/// Ceiling on a mode's envelope, so a "swell" (negative decay) can't run away.
const ENV_CAP: f32 = 4.0;
/// For sustained (driven) modes, how strongly per-mode loss attenuates the
/// held amplitude: `amp /= 1 + decay·SUSTAIN_SHAPE`. Small = subtle high rolloff.
const SUSTAIN_SHAPE: f32 = 0.01;

const TABLE_SIZE: usize = 4096;
const TABLE_MASK: usize = TABLE_SIZE - 1;
const TWO_PI: f32 = std::f32::consts::TAU;

/// Build the shared sine wavetable. One table is made per [`Studio`] and shared
/// (via `Arc`) by every instrument, so adding tracks costs no extra table memory.
pub fn make_sine_table() -> Arc<[f32]> {
    let mut sine = vec![0.0f32; TABLE_SIZE + 1];
    for (i, s) in sine.iter_mut().enumerate() {
        *s = (TWO_PI * i as f32 / TABLE_SIZE as f32).sin();
    }
    sine.into()
}

/// Engine-wide (model-independent) parameters.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EngineParams {
    /// Master output gain (the firmware's SPEAKER_GAIN).
    pub gain: f32,
    /// Anti-click attack ramp (ms).
    pub attack_ms: f32,
    /// Release ramp on key-up (ms).
    pub release_ms: f32,
    /// Minimum time between strikes (the firmware's PLAY_PERIOD); 0 = off.
    pub retrigger_ms: f32,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            gain: 0.6,
            attack_ms: 3.0,
            release_ms: 150.0,
            retrigger_ms: 0.0,
        }
    }
}

struct Voice {
    active: bool,
    note: u8,
    age: u64,
    releasing: bool,
    f0: f32,
    vel: f32,
    elapsed: f32,
    gate: f32,
    atk_inc: f32,
    rel_mul: f32,
    n_modes: usize,
    phase: [f32; MAX_MODES],
    inc: [f32; MAX_MODES],
    amp: [f32; MAX_MODES],
    env: [f32; MAX_MODES],
    dmul: [f32; MAX_MODES],
    // Filtered-noise component.
    noise_level: f32,
    noise_env: f32,
    noise_dmul: f32,
    noise_hp_a: f32,
    noise_lp_a: f32,
    noise_hp_s: f32,
    noise_lp_s: f32,
    rng: u32,
}

impl Voice {
    fn silent() -> Self {
        Voice {
            active: false,
            note: 0,
            age: 0,
            releasing: false,
            f0: 0.0,
            vel: 0.0,
            elapsed: 0.0,
            gate: 0.0,
            atk_inc: 1.0,
            rel_mul: 0.0,
            n_modes: 0,
            phase: [0.0; MAX_MODES],
            inc: [0.0; MAX_MODES],
            amp: [0.0; MAX_MODES],
            env: [0.0; MAX_MODES],
            dmul: [0.0; MAX_MODES],
            noise_level: 0.0,
            noise_env: 0.0,
            noise_dmul: 0.0,
            noise_hp_a: 0.0,
            noise_lp_a: 0.0,
            noise_hp_s: 0.0,
            noise_lp_s: 0.0,
            rng: 1,
        }
    }
}

pub struct Instrument {
    sr: f32,
    engine: EngineParams,
    model: Box<dyn FtmModel>,
    /// Global pitch-bend as a frequency ratio (1.0 = no bend). Applied to every
    /// voice's phase increment at render — a whammy/pitch-wheel, plugin-agnostic.
    bend: f32,
    voices: Vec<Voice>,
    scratch: ModeBuffer,
    sine: Arc<[f32]>,
    age_counter: u64,
    now: u64,
    last_play: u64,
}

impl Instrument {
    pub fn new(sample_rate: f32, sine: Arc<[f32]>) -> Self {
        Self::with_config(sample_rate, sine, default_model(), EngineParams::default())
    }

    pub fn with_config(
        sample_rate: f32,
        sine: Arc<[f32]>,
        model: Box<dyn FtmModel>,
        engine: EngineParams,
    ) -> Self {
        Instrument {
            sr: sample_rate,
            engine,
            model,
            bend: 1.0,
            voices: (0..MAX_VOICES).map(|_| Voice::silent()).collect(),
            scratch: ModeBuffer::default(),
            sine,
            age_counter: 0,
            now: 0,
            last_play: 0,
        }
    }

    /// A fresh instrument with the same model + engine (independent voices).
    /// Used to bind a loop track to the instrument that recorded it.
    pub fn snapshot(&self) -> Instrument {
        Instrument::with_config(
            self.sr,
            self.sine.clone(),
            self.model.box_clone(),
            self.engine.clone(),
        )
    }

    pub fn model_name(&self) -> &'static str {
        self.model.display_name()
    }

    pub fn model_id(&self) -> &'static str {
        self.model.id()
    }

    pub fn model_json(&self) -> serde_json::Value {
        self.model.to_json()
    }

    pub fn engine_params(&self) -> EngineParams {
        self.engine.clone()
    }

    pub fn set_model(&mut self, model: Box<dyn FtmModel>) {
        self.model = model;
        self.rebuild_active();
    }

    pub fn set_engine(&mut self, engine: EngineParams) {
        self.engine = engine;
        self.rebuild_active();
    }

    /// Set the global pitch-bend ratio (`2^(semitones/12)`); 1.0 = no bend.
    /// Cheap — applied at render, no voice rebuild.
    pub fn set_bend(&mut self, ratio: f32) {
        self.bend = ratio.max(0.0);
    }

    #[inline]
    fn sine_at(&self, phase: f32) -> f32 {
        let x = phase * TABLE_SIZE as f32;
        let i = x as usize & TABLE_MASK;
        let frac = x - x.floor();
        let a = self.sine[i];
        let b = self.sine[i + 1];
        a + (b - a) * frac
    }

    fn rebuild_active(&mut self) {
        for vi in 0..self.voices.len() {
            if self.voices[vi].active {
                self.build_voice(vi, false);
            }
        }
    }

    fn alloc_voice(&mut self) -> usize {
        if let Some(i) = self.voices.iter().position(|v| !v.active) {
            return i;
        }
        let mut best = 0;
        let mut best_age = u64::MAX;
        for (i, v) in self.voices.iter().enumerate() {
            if v.age < best_age {
                best_age = v.age;
                best = i;
            }
        }
        best
    }

    pub fn note_on(&mut self, note: u8, vel: f32) {
        let lockout = (self.engine.retrigger_ms * 0.001 * self.sr) as u64;
        if lockout > 0 && self.now.saturating_sub(self.last_play) < lockout {
            return;
        }
        self.last_play = self.now;

        let idx = self.alloc_voice();
        self.age_counter += 1;
        let age = self.age_counter;
        {
            let v = &mut self.voices[idx];
            v.active = true;
            v.releasing = false;
            v.note = note;
            v.age = age;
            v.f0 = midi_to_freq(note);
            v.vel = vel;
        }
        self.build_voice(idx, true);
        if self.voices[idx].n_modes == 0 {
            self.voices[idx].active = false;
        }
    }

    pub fn note_off(&mut self, note: u8) {
        for v in &mut self.voices {
            if v.active && v.note == note && !v.releasing {
                v.releasing = true;
            }
        }
    }

    pub fn all_notes_off(&mut self) {
        for v in &mut self.voices {
            v.active = false;
        }
    }

    /// Release every sounding voice (as if each got a note-off) — used when the
    /// transport stops so a note held at the stop point fades instead of ringing
    /// forever. Struck voices keep decaying; sustained ones enter their release.
    pub fn release_all(&mut self) {
        for v in &mut self.voices {
            if v.active {
                v.releasing = true;
            }
        }
    }

    #[allow(dead_code)] // used in tests; handy for a future voice meter
    pub fn active_voices(&self) -> usize {
        self.voices.iter().filter(|v| v.active).count()
    }

    fn build_voice(&mut self, vi: usize, fresh: bool) {
        let sr = self.sr;
        let (f0, vel) = {
            let v = &self.voices[vi];
            (v.f0, v.vel)
        };
        let mut buf = std::mem::take(&mut self.scratch);
        self.model.excite(f0, vel, sr, &mut buf);

        let atk_ms = self.engine.attack_ms;
        let rel_ms = self.engine.release_ms;
        let elapsed = if fresh { 0.0 } else { self.voices[vi].elapsed };
        let old_n = if fresh { 0 } else { self.voices[vi].n_modes };

        let v = &mut self.voices[vi];
        if fresh {
            v.elapsed = 0.0;
            v.gate = 0.0;
            v.releasing = false;
        }
        v.atk_inc = 1.0 / (atk_ms * 0.001 * sr).max(1.0);
        v.rel_mul = (0.001f32).powf(1.0 / (rel_ms * 0.001 * sr).max(1.0));

        let n = buf.n.min(MAX_MODES);
        for i in 0..n {
            v.inc[i] = (buf.freq[i] / sr).max(0.0);
            if buf.sustain {
                // Driven/sustained (wind): hold the mode (no decay) while played;
                // its loss becomes steady-state attenuation instead — a lossier
                // mode is quieter, like a driven resonance. It fades on release
                // via the voice gate.
                v.amp[i] = buf.amp[i] / (1.0 + buf.decay[i].max(0.0) * SUSTAIN_SHAPE);
                v.dmul[i] = 1.0;
                if i >= old_n {
                    v.phase[i] = 0.0;
                    v.env[i] = 1.0;
                }
            } else {
                v.amp[i] = buf.amp[i];
                v.dmul[i] = (-buf.decay[i] / sr).exp();
                if i >= old_n {
                    v.phase[i] = 0.0;
                    v.env[i] = (-buf.decay[i] * elapsed).exp().min(ENV_CAP);
                }
            }
        }
        v.n_modes = n;

        // Filtered-noise component (snare wires, stick click, breath…).
        v.noise_level = buf.noise_level.max(0.0);
        if v.noise_level > 1e-6 {
            let cutoff_a = |hz: f32| 1.0 - (-2.0 * std::f32::consts::PI * hz.max(1.0) / sr).exp();
            v.noise_hp_a = cutoff_a(buf.noise_hp);
            v.noise_lp_a = cutoff_a(buf.noise_lp);
            v.noise_dmul = if buf.sustain {
                1.0
            } else {
                (-buf.noise_decay.max(0.0) / sr).exp()
            };
            if fresh {
                v.noise_env = 1.0;
                v.noise_hp_s = 0.0;
                v.noise_lp_s = 0.0;
                // Seed the per-voice noise RNG (never zero).
                v.rng = (self.age_counter as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(vi as u32 + 1)
                    | 1;
            }
        }

        self.scratch = buf;
    }

    /// Render exactly one (mono) sample, advancing every voice. The returned
    /// value already has this instrument's gain applied but is **not** clamped —
    /// the studio sums instruments and clamps the mix.
    #[inline]
    pub fn render_frame(&mut self) -> f32 {
        let mut s = 0.0f32;
        for vi in 0..self.voices.len() {
            if self.voices[vi].active {
                s += self.render_voice(vi);
            }
        }
        self.now += 1;
        s * self.engine.gain
    }

    #[inline]
    fn render_voice(&mut self, vi: usize) -> f32 {
        let n = self.voices[vi].n_modes;
        let mut acc = 0.0f32;
        for i in 0..n {
            let ph = self.voices[vi].phase[i];
            acc += self.sine_at(ph) * self.voices[vi].amp[i] * self.voices[vi].env[i];
        }

        let bend = self.bend;
        let v = &mut self.voices[vi];
        v.elapsed += 1.0 / self.sr;
        let mut alive = false;
        for i in 0..n {
            v.phase[i] += v.inc[i] * bend;
            while v.phase[i] >= 1.0 {
                v.phase[i] -= 1.0;
            }
            v.env[i] *= v.dmul[i];
            if v.env[i] > ENV_CAP {
                v.env[i] = ENV_CAP;
            }
            if v.env[i] > 1e-4 {
                alive = true;
            }
        }

        // Filtered-noise component: white noise → band-pass (one-pole HP then LP).
        if v.noise_level > 1e-6 {
            v.rng ^= v.rng << 13;
            v.rng ^= v.rng >> 17;
            v.rng ^= v.rng << 5;
            let white = (v.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
            v.noise_hp_s += v.noise_hp_a * (white - v.noise_hp_s);
            let hp = white - v.noise_hp_s;
            v.noise_lp_s += v.noise_lp_a * (hp - v.noise_lp_s);
            acc += v.noise_lp_s * v.noise_env * v.noise_level;
            v.noise_env *= v.noise_dmul;
            if v.noise_env > 1e-4 {
                alive = true;
            }
        }

        if v.releasing {
            v.gate *= v.rel_mul;
            if v.gate < 1e-4 {
                v.active = false;
            }
        } else {
            if v.gate < 1.0 {
                v.gate += v.atk_inc;
                if v.gate > 1.0 {
                    v.gate = 1.0;
                }
            }
            if !alive {
                v.active = false;
            }
        }
        acc * v.gate
    }
}

#[inline]
fn midi_to_freq(note: u8) -> f32 {
    440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::registry;

    fn render_rms(inst: &mut Instrument, secs: f32) -> f32 {
        let n = (inst.sr * secs) as usize;
        let mut sum = 0.0f32;
        for _ in 0..n {
            let s = inst.render_frame().clamp(-4.0, 4.0);
            assert!(s.is_finite(), "non-finite sample");
            sum += s * s;
        }
        (sum / n as f32).sqrt()
    }

    fn instrument() -> Instrument {
        Instrument::new(48_000.0, make_sine_table())
    }

    #[test]
    fn struck_note_decays() {
        let mut inst = instrument();
        inst.note_on(69, 1.0);
        let start = render_rms(&mut inst, 0.1);
        let _ = render_rms(&mut inst, 3.0);
        let end = render_rms(&mut inst, 0.1);
        assert!(start > 0.01, "attack too quiet: {start}");
        assert!(end < start, "expected decay: {start} -> {end}");
    }

    #[test]
    fn wind_sustains_while_string_decays() {
        use crate::models::pure_string::PureString;
        use crate::models::webster_horn::WebsterHorn;
        let hold = |model: Box<dyn FtmModel>| {
            let mut inst = Instrument::with_config(
                48_000.0,
                make_sine_table(),
                model,
                EngineParams::default(),
            );
            inst.note_on(60, 1.0);
            let start = render_rms(&mut inst, 0.05);
            let _ = render_rms(&mut inst, 2.0); // hold, no note-off
            let end = render_rms(&mut inst, 0.05);
            (start, end)
        };
        let (s0, s1) = hold(Box::new(PureString::default()));
        assert!(s1 < s0 * 0.5, "plucked string decays while held ({s0} -> {s1})");
        let (h0, h1) = hold(Box::new(WebsterHorn::default()));
        assert!(h1 > h0 * 0.7, "blown horn sustains while held ({h0} -> {h1})");
    }

    #[test]
    fn switching_models_live_is_stable() {
        let mut inst = instrument();
        inst.note_on(60, 0.9);
        render_rms(&mut inst, 0.05);
        for model in registry() {
            inst.set_model(model);
            render_rms(&mut inst, 0.05);
        }
    }

    #[test]
    fn note_off_frees_the_voice() {
        let mut inst = instrument();
        inst.note_on(60, 1.0);
        assert_eq!(inst.active_voices(), 1);
        inst.note_off(60);
        render_rms(&mut inst, 1.0);
        assert_eq!(inst.active_voices(), 0, "voice should free after release");
    }

    #[test]
    fn snapshot_is_independent() {
        let mut a = instrument();
        a.note_on(60, 1.0);
        let mut b = a.snapshot();
        assert_eq!(b.active_voices(), 0, "snapshot starts silent");
        b.note_on(64, 1.0);
        assert_eq!(a.active_voices(), 1);
        assert_eq!(b.active_voices(), 1);
    }
}

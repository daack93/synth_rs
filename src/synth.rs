//! Generic polyphonic engine.
//!
//! The engine knows nothing about strings, drums, or excitation shapes. It owns
//! the things every mode shares — a pool of voices, phase accumulators, the
//! amplitude envelope, voice-stealing, the retrigger lockout, and live parameter
//! rebuilds — and delegates the actual *sound* to whatever [`FtmModel`] plugin
//! is currently active (see [`crate::models`]). A voice is just a bank of
//! sinusoids whose frequencies/amplitudes/decays the model filled in.

use crate::models::{default_model, FtmModel, ModeBuffer, MAX_MODES};

/// Number of simultaneously sounding notes.
pub const MAX_VOICES: usize = 16;
/// Ceiling on a mode's envelope, so a "swell" (negative decay) can't run away.
const ENV_CAP: f32 = 4.0;

const TABLE_SIZE: usize = 4096;
const TABLE_MASK: usize = TABLE_SIZE - 1;
const TWO_PI: f32 = std::f32::consts::TAU;

/// Engine-wide (model-independent) parameters.
#[derive(Clone, PartialEq)]
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

/// Messages from the UI / MIDI threads to the audio thread.
pub enum Command {
    NoteOn { note: u8, vel: f32 },
    NoteOff { note: u8 },
    /// Swap the active synthesis model (also rebuilds sounding voices live).
    SetModel(Box<dyn FtmModel>),
    /// Update engine-wide parameters.
    SetEngine(EngineParams),
    AllNotesOff,
}

struct Voice {
    active: bool,
    note: u8,
    age: u64,
    releasing: bool,
    /// Fundamental of this note (Hz), kept so the model can re-excite it live.
    f0: f32,
    /// Strike velocity 0..1.
    vel: f32,
    /// Seconds since the note started (for placing newly-added modes on rebuild).
    elapsed: f32,
    gate: f32,
    atk_inc: f32,
    rel_mul: f32,
    n_modes: usize,
    phase: [f32; MAX_MODES], // cycles [0,1)
    inc: [f32; MAX_MODES],   // cycles per sample
    amp: [f32; MAX_MODES],
    env: [f32; MAX_MODES],
    dmul: [f32; MAX_MODES], // per-sample decay multiplier
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
        }
    }
}

pub struct Synth {
    sr: f32,
    engine: EngineParams,
    model: Box<dyn FtmModel>,
    voices: Vec<Voice>,
    /// Scratch bank the active model fills; reused to avoid per-note allocation.
    scratch: ModeBuffer,
    sine: Vec<f32>,
    age_counter: u64,
    now: u64,
    last_play: u64,
}

impl Synth {
    pub fn new(sample_rate: f32) -> Self {
        let mut sine = vec![0.0f32; TABLE_SIZE + 1];
        for (i, s) in sine.iter_mut().enumerate() {
            *s = (TWO_PI * i as f32 / TABLE_SIZE as f32).sin();
        }
        Synth {
            sr: sample_rate,
            engine: EngineParams::default(),
            model: default_model(),
            voices: (0..MAX_VOICES).map(|_| Voice::silent()).collect(),
            scratch: ModeBuffer::default(),
            sine,
            age_counter: 0,
            now: 0,
            last_play: 0,
        }
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

    pub fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::SetModel(m) => {
                self.model = m;
                self.rebuild_active();
            }
            Command::SetEngine(e) => {
                self.engine = e;
                self.rebuild_active();
            }
            Command::NoteOn { note, vel } => self.note_on(note, vel),
            Command::NoteOff { note } => self.note_off(note),
            Command::AllNotesOff => {
                for v in &mut self.voices {
                    v.active = false;
                }
            }
        }
    }

    /// Re-excite every sounding voice so parameter/model changes are heard live.
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
        // Steal the oldest.
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

    fn note_on(&mut self, note: u8, vel: f32) {
        // Retrigger lockout (PLAY_PERIOD).
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

        // A model may decline a strike (e.g. below PLAY_MAGNITUDE) => no modes.
        if self.voices[idx].n_modes == 0 {
            self.voices[idx].active = false;
        }
    }

    /// (Re)build a voice's oscillator bank from the active model.
    ///
    /// `fresh` = a new strike (envelopes reset). Otherwise it's a live rebuild:
    /// existing modes keep their phase and decayed level; a newly-added mode is
    /// placed at the level it would have reached had it rung since the strike.
    fn build_voice(&mut self, vi: usize, fresh: bool) {
        let sr = self.sr;
        let (f0, vel) = {
            let v = &self.voices[vi];
            (v.f0, v.vel)
        };

        // Fill the scratch bank on the audio thread (model is pure/allocation-free).
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
            v.amp[i] = buf.amp[i];
            v.dmul[i] = (-buf.decay[i] / sr).exp();
            if i >= old_n {
                v.phase[i] = 0.0;
                v.env[i] = (-buf.decay[i] * elapsed).exp().min(ENV_CAP);
            }
            // Existing modes keep their current phase[i] and env[i].
        }
        v.n_modes = n;

        self.scratch = buf; // return the scratch buffer
    }

    fn note_off(&mut self, note: u8) {
        for v in &mut self.voices {
            if v.active && v.note == note && !v.releasing {
                v.releasing = true;
            }
        }
    }

    /// Render `out` (interleaved by `channels`) mixing all active voices.
    pub fn render(&mut self, out: &mut [f32], channels: usize) {
        let gain = self.engine.gain;
        for frame in out.chunks_mut(channels) {
            let mut s = 0.0f32;
            for vi in 0..self.voices.len() {
                if self.voices[vi].active {
                    s += self.render_voice(vi);
                }
            }
            self.now += 1;
            let sample = (s * gain).clamp(-1.0, 1.0);
            for ch in frame.iter_mut() {
                *ch = sample;
            }
        }
    }

    #[inline]
    fn render_voice(&mut self, vi: usize) -> f32 {
        let n = self.voices[vi].n_modes;

        let mut acc = 0.0f32;
        for i in 0..n {
            let ph = self.voices[vi].phase[i];
            acc += self.sine_at(ph) * self.voices[vi].amp[i] * self.voices[vi].env[i];
        }

        let v = &mut self.voices[vi];
        v.elapsed += 1.0 / self.sr;
        let mut alive = false;
        for i in 0..n {
            v.phase[i] += v.inc[i];
            if v.phase[i] >= 1.0 {
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

        // Overall gate: attack ramp, then release ramp on key-up.
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
                v.active = false; // fully decayed
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

    fn render_rms(synth: &mut Synth, secs: f32) -> f32 {
        let n = (synth.sr * secs) as usize;
        let mut buf = vec![0.0f32; n];
        synth.render(&mut buf, 1);
        assert!(buf.iter().all(|s| s.is_finite()), "non-finite sample");
        assert!(buf.iter().all(|s| s.abs() <= 1.0001), "out of range");
        let sum: f32 = buf.iter().map(|s| s * s).sum();
        (sum / n as f32).sqrt()
    }

    #[test]
    fn struck_note_decays() {
        let mut synth = Synth::new(48_000.0);
        synth.handle(Command::NoteOn { note: 69, vel: 1.0 });
        let start = render_rms(&mut synth, 0.1);
        let _ = render_rms(&mut synth, 3.0);
        let end = render_rms(&mut synth, 0.1);
        assert!(start > 0.01, "attack too quiet: {start}");
        assert!(end < start, "expected decay: {start} -> {end}");
    }

    /// Swap through every registered model while a note is held; the engine must
    /// stay finite and in range across the live rebuilds. Grows as plugins are
    /// added, so it exercises whatever is registered.
    #[test]
    fn switching_models_live_is_stable() {
        let mut synth = Synth::new(48_000.0);
        synth.handle(Command::NoteOn { note: 60, vel: 0.9 });
        render_rms(&mut synth, 0.05);
        for model in registry() {
            synth.handle(Command::SetModel(model));
            render_rms(&mut synth, 0.05);
        }
    }

    #[test]
    fn big_gain_and_all_notes_off() {
        let mut synth = Synth::new(48_000.0);
        for n in [48, 55, 60, 64, 67] {
            synth.handle(Command::NoteOn { note: n, vel: 1.0 });
        }
        synth.handle(Command::SetEngine(EngineParams {
            gain: 4.0,
            ..EngineParams::default()
        }));
        render_rms(&mut synth, 0.2); // clamped, must stay in range
        synth.handle(Command::AllNotesOff);
        assert_eq!(synth.voices.iter().filter(|v| v.active).count(), 0);
    }

    #[test]
    fn note_off_frees_the_voice() {
        let mut synth = Synth::new(48_000.0);
        synth.handle(Command::NoteOn { note: 60, vel: 1.0 });
        assert_eq!(synth.voices.iter().filter(|v| v.active).count(), 1);
        synth.handle(Command::NoteOff { note: 60 });
        let mut buf = vec![0.0f32; 48_000];
        synth.render(&mut buf, 1);
        assert_eq!(
            synth.voices.iter().filter(|v| v.active).count(),
            0,
            "voice should free after release"
        );
    }
}

//! Exciter / Resonator decomposition (Phase 3 of the architecture roadmap).
//!
//! A synthesizer model can be expressed as two composable components with a
//! wave passed between them:
//!
//! * an **[`Exciter`]** emits an **[`Excitation`]** — a description of the input
//!   wave (how / where / how-hard the resonator is driven): a strike, a pluck,
//!   a bow;
//! * a **[`Resonator`]** receives that excitation and fills the whole
//!   [`ModeBuffer`] — frequencies, decays, sustain **and** amplitudes.
//!
//! Amplitude/coupling lives in the resonator on purpose: "how it responds to
//! being excited at position *x*" is a property of the resonator's own mode
//! shapes (a struck plate's shape, a plucked string's `sin(mπp)`), not of the
//! exciter. The exciter just hands over the excitation.
//!
//! Phase 3 proves the split by decomposing the plate and string models into a
//! `{Strike,Pluck}Exciter` driving a `{Plate,String}Resonator`, with no change
//! in output. The generic `Voiced` adapter that presents an exciter+resonator
//! pair *as* an [`super::FtmModel`], a noise-wash excitation, and preset
//! serialization arrive in the next phase, when they're wired into the registry
//! and UI (kept out of here so nothing lands unused).

use super::{strike_amplitude, ModeBuffer};

/// How a resonator is being driven.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExcitationKind {
    /// A one-shot impulse — the modes ring and decay (mallet, stick, hammer).
    Struck,
    /// A released initial displacement — a triangular pluck shape (string).
    Plucked,
    /// Continuously driven — the modes are held while the note sounds and only
    /// fade on release; loss shapes the steady-state spectrum (bow, breath).
    Bowed,
}

/// The wave an [`Exciter`] hands to a [`Resonator`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Excitation {
    pub kind: ExcitationKind,
    /// Where along the resonator it is driven, 0..1 (throat/centre → edge/end).
    pub position: f32,
    /// How hard, roughly velocity in 0..1 after any response curve.
    pub strength: f32,
}

/// Emits an [`Excitation`] for a played note. Stateless in Tier-1: the same
/// note+velocity always yields the same excitation.
pub trait Exciter {
    fn excite(&self, note_hz: f32, vel: f32) -> Excitation;
}

/// Receives an [`Excitation`] and fills the [`ModeBuffer`] for a note: its own
/// mode frequencies + intrinsic decays + sustain flag, and the per-mode
/// amplitudes (its mode shapes sampled at `exc.position`, scaled by
/// `exc.strength`).
pub trait Resonator {
    fn resonate(&self, note_hz: f32, sr: f32, exc: &Excitation, out: &mut ModeBuffer);
}

/// A percussive strike: raw velocity in, an impulse at a fixed position.
pub struct StrikeExciter {
    /// Strike position, 0 = centre → 1 = edge.
    pub position: f32,
}

impl Exciter for StrikeExciter {
    fn excite(&self, _note_hz: f32, vel: f32) -> Excitation {
        Excitation {
            kind: ExcitationKind::Struck,
            position: self.position.clamp(0.0, 1.0),
            strength: vel,
        }
    }
}

/// A string pluck/bow: a triangular initial displacement at a position, with a
/// velocity-response curve (play/max magnitude) and a plucked/bowed switch.
pub struct PluckExciter {
    /// Pluck position along the string, 0..1 (0.5 = centre → triangle).
    pub position: f32,
    pub play_magnitude: f32,
    pub max_magnitude: f32,
    /// Bowed (driven/sustained) instead of plucked.
    pub bowed: bool,
}

impl Exciter for PluckExciter {
    fn excite(&self, _note_hz: f32, vel: f32) -> Excitation {
        Excitation {
            kind: if self.bowed { ExcitationKind::Bowed } else { ExcitationKind::Plucked },
            position: self.position.clamp(0.0, 1.0),
            strength: strike_amplitude(vel, self.play_magnitude, self.max_magnitude),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::pure_plate::{PlateResonator, PurePlate};
    use crate::models::pure_string::{PureString, StringResonator};
    use crate::models::FtmModel;

    /// Same modes (freq/amp/decay), count and sustain flag.
    fn same(a: &ModeBuffer, b: &ModeBuffer) -> bool {
        a.n == b.n
            && a.sustain == b.sustain
            && a.freq[..a.n] == b.freq[..b.n]
            && a.amp[..a.n] == b.amp[..b.n]
            && a.decay[..a.n] == b.decay[..b.n]
    }

    #[test]
    fn strike_plus_plate_reproduces_pure_plate() {
        let m = PurePlate::default();
        let mut via_model = ModeBuffer::default();
        m.excite(220.0, 0.8, 48_000.0, &mut via_model);

        // Compose the parts by hand and confirm the same bank comes out.
        let exc = StrikeExciter { position: m.strike_pos }.excite(220.0, 0.8);
        let reso = PlateResonator {
            poisson: m.poisson,
            decay_time: m.decay_time,
            hf_damp: m.hf_damp,
            modes: m.modes,
            key_tracks_pitch: m.key_tracks_pitch,
        };
        let mut via_parts = ModeBuffer::default();
        reso.resonate(220.0, 48_000.0, &exc, &mut via_parts);

        assert!(via_model.n > 4, "plate produced modes");
        assert!(same(&via_model, &via_parts), "StrikeExciter+PlateResonator == PurePlate");
    }

    #[test]
    fn pluck_plus_string_reproduces_pure_string() {
        let m = PureString::default();
        let mut via_model = ModeBuffer::default();
        m.excite(196.0, 0.9, 48_000.0, &mut via_model);

        let exc = PluckExciter {
            position: m.pluck_pos,
            play_magnitude: m.play_magnitude,
            max_magnitude: m.max_magnitude,
            bowed: false,
        }
        .excite(196.0, 0.9);
        let reso = StringResonator {
            stiffness: m.stiffness,
            prop_speed: m.prop_speed,
            damping: m.damping,
            freq_dep_damping: m.freq_dep_damping,
            string_length: m.string_length,
            depth: m.depth,
            damp_period: m.damp_period,
            time_scale: m.time_scale,
            key_tracks_pitch: m.key_tracks_pitch,
            pitch_mode: m.pitch_mode,
        };
        let mut via_parts = ModeBuffer::default();
        reso.resonate(196.0, 48_000.0, &exc, &mut via_parts);

        assert!(via_model.n > 4, "string produced modes");
        assert!(same(&via_model, &via_parts), "PluckExciter+StringResonator == PureString");
    }
}

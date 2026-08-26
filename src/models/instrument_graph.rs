//! A data-driven, editable instrument graph.
//!
//! An instrument here is a small graph: a list of **components** (exciters and
//! resonators, each with its own natural parameters) wired by **edges** that
//! carry a signal scaled by a coupling **gain**. The whole thing is plain data,
//! so it serialises as a preset and is edited generically — the parameter panel
//! shows every component's controls plus a strength slider per edge, and any
//! edit rebuilds the per-voice [`crate::graph::Graph`] live.
//!
//! This is the generalisation of the hand-wired graph instruments: the same
//! `Strike → String → Body` sound is now `components + edges` you can retune.

use serde::{Deserialize, Serialize};

use super::cymbal::Cymbal;
use super::drum_membrane::DrumMembrane;
use super::metal_bell::MetalBell;
use super::musical_string::MusicalString;
use super::pure_plate::PurePlate;
use super::pure_string::PureString;
use super::webster_horn::{PlayMode, WebsterHorn};
use super::{freq_to_midi, midi_name, unbounded_slider, FtmModel, ModeBuffer};
use crate::graph::{
    BowExciter, CoupledDoubleReed, CoupledReed, DriveExciter, FormantResonator, Graph, HammerExciter, ImpulseExciter,
    JetPipe, LipReed, ModalResonator, Node, ReedExciter, SineExciter, SnareWires, Sum, VoiceExciter, WaveguideBore, WaveguideBow, WaveguideHammer, WaveguidePluck, WaveguideHorn, WaveguideReed,
};

const PI: f32 = std::f32::consts::PI;
const TWO_PI: f32 = std::f32::consts::TAU;
/// Speed of sound in air at ~20 °C, m/s — sets the body's Helmholtz pitch.
const C_AIR: f32 = 343.0;
/// ln(1000): a decay rate of `ln(1000)/T` reaches −60 dB at `t = T` seconds.
const LN_1000: f32 = 6.907_755;

/// Default reed-bore overblow ratio (a cylinder's twelfth) for presets predating
/// the configurable register break.
fn default_overblow() -> f32 {
    3.0
}

/// A double reed is always conical and overblows the octave — its serde defaults.
fn default_double_overblow() -> f32 {
    2.0
}
fn default_true() -> bool {
    true
}

/// Default air-jet convection-delay ratio (τ / acoustic period) for the flue
/// pipe — ~0.5 favours the fundamental register.
fn default_jet_ratio() -> f32 {
    0.5
}

/// Reed range floor: a single reed's lowest note is its full-length tube (the
/// anchor). Notes below `anchor · REED_FLOOR` (~a half-semitone of margin for
/// bend) don't sound — there is no longer tube. 2^(-0.5/12).
const REED_FLOOR: f32 = 0.9715;

/// One node in an instrument graph. Each variant is a physics component with
/// its own inherent parameters (kept as their natural types).
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Comp {
    /// Percussive strike (impulse); velocity comes from the key.
    Strike,
    /// A plucked/struck string resonator.
    String(PureString),
    /// A 2-D membrane resonator.
    Membrane(DrumMembrane),
    /// A free-plate resonator.
    Plate(PurePlate),
    /// A music-friendly plucked string (inharmonicity + decay controls).
    MusicalString(MusicalString),
    /// A struck metal bell / cowbell resonator.
    Bell(MetalBell),
    /// A struck cymbal / gong plate resonator (dense inharmonic modes).
    Cymbal(Cymbal),
    /// A resonant body: a Helmholtz air cavity (its pitch derived from
    /// `cavity_litres` + `soundhole_cm`) plus a top/soundboard plate resonance
    /// (`top_hz`), ringing for `decay_s` seconds. Set the cavity to 0 for a
    /// cavity-less soundboard (e.g. a piano); a small cavity models an oral tract.
    Body { cavity_litres: f32, soundhole_cm: f32, top_hz: f32, decay_s: f32 },
    /// Snare wires resting on a head: a rattle that buzzes with the head's motion.
    /// `level` = rattle amount, `tone` = band brightness.
    Wires { level: f32, tone: f32 },
    /// A continuous breath drive (band-passed noise) — sustains a resonator.
    Breath { level: f32, tone: f32 },
    /// A single-reed mouthpiece. `freq_hz > 0` makes it a *self-contained*
    /// mouthpiece buzzing at that one fixed pitch (a built-in bore — like a reed
    /// with the horn pulled off) that then drives a downstream resonator forward;
    /// `freq_hz = 0` makes it a bare valve that reads its resonant load back over
    /// a feedback edge (pitch comes from the bore it drives).
    Reed { pressure: f32, stiffness: f32, freq_hz: f32 },
    /// A **coupled reed + bore** — the physically-correct woodwind voice. The reed
    /// valve and a waveguide bore are one tightly-coupled loop solved *implicitly*
    /// each sample (no loop delay), so the pitch locks to the bore `length`
    /// (`f ≈ c/2L`) in tune. A register vent lets it overblow: the `overblow`
    /// ratio + `conical` flag define the register break per instrument — a
    /// cylinder (`conical:false`, `overblow:3`) overblows a twelfth on odd
    /// harmonics (clarinet), a cone (`conical:true`, `overblow:2`) overblows the
    /// octave on the full harmonic series (saxophone). `register` 0 = closed
    /// (low), open = overblown. Self-contained; drive a downstream Webster horn
    /// for bell colour. Key-map `length`.
    ReedBore {
        pressure: f32,
        stiffness: f32,
        length: f32,
        tone: f32,
        register: f32,
        #[serde(default = "default_overblow")]
        overblow: f32,
        #[serde(default)]
        conical: bool,
        /// Vocal-tract voicing: how strongly the airway resonance (tuned to the
        /// played note) loads the reed. 0 = off; raise it to voice the altissimo
        /// registers (biases the reed onto higher bore harmonics).
        #[serde(default)]
        tract_gain: f32,
        /// Sharpness (Q) of that tract resonance. 0 = a sensible default.
        #[serde(default)]
        tract_q: f32,
    },
    /// A **coupled double-reed + bore** — the oboe / bassoon / cor-anglais voice.
    /// Two stiff blades beat against each other (a high reed resonance) on a
    /// strongly conical bore, driving a very constricted, pinched flow through the
    /// same implicit reed↔bore solve as [`Comp::ReedBore`] — so the pitch still
    /// locks to `length` (`f ≈ c/2L`). It overblows the OCTAVE (full harmonic
    /// series) and carries a fixed nasal formant. Uses the same `OverblowTuned`
    /// key-map + calibration machinery as the single-reed family. Key-map `length`.
    DoubleReed {
        pressure: f32,
        stiffness: f32,
        length: f32,
        tone: f32,
        register: f32,
        /// Register-break ratio; a cone overblows the octave (2). Defaults to 2.
        #[serde(default = "default_double_overblow")]
        overblow: f32,
        /// Double reeds are conical; defaults true.
        #[serde(default = "default_true")]
        conical: bool,
        /// Vocal-tract voicing (as [`Comp::ReedBore`]): 0 = off, raise to voice the
        /// altissimo registers.
        #[serde(default)]
        tract_gain: f32,
        /// Sharpness (Q) of that tract resonance. 0 = a sensible default.
        #[serde(default)]
        tract_q: f32,
    },
    /// A **lip reed (brass) valve** implicitly coupled to a flaring bore — the
    /// trumpet / trombone / horn voice. The mirror of `ReedBore`: an
    /// *outward-striking* lip (higher mouth pressure blows the lips OPEN, not shut)
    /// solved against a waveguide bore each sample, so the pitch locks to `length`
    /// (`f ≈ c/2L`) and the lips overblow *up the harmonic series*. `pressure` =
    /// breath, `tension` = lip resonance as a multiple of the bore fundamental
    /// (≈ n biases the n-th partial; ~1 plays the fundamental), `tone` = bell
    /// brightness. Self-contained; key-map `length` for pitch. Drive a downstream
    /// Webster bell for colour.
    Lips {
        pressure: f32,
        tension: f32,
        length: f32,
        #[serde(default)]
        tone: f32,
    },
    /// A digital-waveguide reed pipe — a self-contained wind voice (bore delay +
    /// bell + reed) that self-oscillates into a clean reed tone. `pressure` =
    /// breath, `stiffness` = reed hardness, `tone` = bell brightness.
    ReedPipe { pressure: f32, stiffness: f32, tone: f32 },
    /// A **flue / air-jet pipe** — the flute / recorder voice. A blown air jet
    /// crosses the mouth edge (no reed) and drives an OPEN cylinder (both ends
    /// pressure-release → all harmonics, `f ≈ c/2L`) via a delayed, saturating jet
    /// deflection. `pressure` = blowing pressure / jet velocity (blow harder to
    /// overblow the octave), `jet_ratio` = the jet convection delay as a fraction
    /// of the acoustic period (register/timbre bias, ~0.5), `tone` = radiation
    /// brightness. Self-contained; drive a downstream Webster bell for air colour.
    /// Key-map `length` (`f ≈ c/2L`).
    AirJet {
        pressure: f32,
        #[serde(default = "default_jet_ratio")]
        jet_ratio: f32,
        tone: f32,
        length: f32,
    },
    /// A piano/dulcimer hammer: a nonlinear felt mass (needs a feedback edge from
    /// the string). `hardness` 0..1 sets the felt stiffness (dark→bright, long→
    /// short contact), `felt` is the compression nonlinearity.
    Hammer { hardness: f32, felt: f32 },
    /// A bow: stick-slip friction (needs a feedback edge from the string).
    /// `speed` = bow velocity, `force` = bow pressure.
    Bow { speed: f32, force: f32 },
    /// A digital-waveguide bowed string — a self-contained voice (string delays +
    /// bridge + friction junction) that locks into Helmholtz stick-slip.
    /// `speed` = bow velocity, `force` = bow pressure.
    /// A digital-waveguide bowed string (friction bow on the shared StringCore).
    /// `bow_pos` = beta (bow-to-bridge fraction; ~0.08 normal, smaller = brighter);
    /// `brightness` 0..1 keeps the sawtooth's upper partials; `speed`/`force` are
    /// the bow (both scale with note velocity -- dynamics). A bowed string is
    /// periodic, so its inharmonicity is ~0 (see WaveguideBow).
    BowedString {
        bow_pos: f32,
        brightness: f32,
        speed: f32,
        force: f32,
    },
    /// A digital-waveguide plucked/struck string — a self-contained voice
    /// (`StringCore`) excited by a note-on burst, then ringing and decaying.
    /// `pos` = pluck point (0 = nut, 1 = bridge); `decay` = the fundamental's
    /// -60 dB time (s); `damping` = HF damping 0..1 (darker tail); `stiffness` =
    /// dispersion coefficient (string stiffness → inharmonic partials).
    /// A digital-waveguide plucked/struck string (self-contained `StringCore`),
    /// grounded in REAL string physics: the inharmonicity is derived from the
    /// core (stiffness) diameter, tension, length and Young's modulus, and rises
    /// up the neck (fretted). `open_hz` is the open-string pitch; `core_mm` is the
    /// bending-resisting core (a wound string's thin steel core, NOT its overall
    /// gauge). `pos` = pluck point, `decay` = -60 dB time (s), `damping` = HF damp.
    PluckedString {
        length_m: f32,
        tension_n: f32,
        core_mm: f32,
        youngs_gpa: f32,
        open_hz: f32,
        pos: f32,
        decay: f32,
        damping: f32,
    },
    /// A digital-waveguide HAMMERED string (piano): a felt hammer on the shared
    /// StringCore, grounded in real string physics. `hardness`/`felt` shape the
    /// contact (key-map hardness for per-register voicing); the rest is the string.
    HammeredString {
        length_m: f32,
        tension_n: f32,
        core_mm: f32,
        youngs_gpa: f32,
        open_hz: f32,
        hardness: f32,
        felt: f32,
        pos: f32,
        decay: f32,
        damping: f32,
    },
    /// A vocal-fold (glottal) source, pitched at the played note. `open_quotient`
    /// = how long the folds stay open (breathy → pressed), `level` = drive.
    Voice { open_quotient: f32, level: f32 },
    /// A flaring air column (Webster horn) resonator.
    Horn(WebsterHorn),
    /// A waveguide air column (delay line + bell reflection): a wind bore that a
    /// reed/lip exciter drives (via a feedback edge) into self-oscillation. Its
    /// pitch comes from `length` (metres, `f = c/2L`) when `length > 0` — the
    /// key-map drives that length, and the reed *follows* the bore (the coupled
    /// reed↔bore loop that makes pitch track length). `length = 0` falls back to
    /// tracking the played key directly.
    Bore { tone: f32, length: f32 },
    /// A **traveling-wave flaring horn** — the geometry (`r(x) = r1 + r2·x +
    /// r3·x²`) built as segmented Kelly–Lochbaum waveguide, so a reed drives it
    /// into oscillation and its pitch tracks the bore `length` (a cylinder →
    /// odd harmonics/clarinet, a cone → all harmonics/sax). `segments` is the
    /// flare resolution. Needs a feedback edge from a reed, like `Bore`.
    WaveguideHorn { r1: f32, r2: f32, r3: f32, length: f32, segments: usize, tone: f32 },
    /// A pure sinusoid generator at the played pitch. `level` sets its amplitude.
    /// Doubles as the audition probe the graph editor feeds a component under
    /// test (a generator ignores it; a resonator resonates its pure tone).
    Sine { level: f32 },
    /// A mixer / output node: the (edge-scaled) sum of its inputs.
    Mix,
}

impl Comp {
    fn label(&self) -> &'static str {
        match self {
            Comp::Strike => "Strike (exciter)",
            Comp::String(_) => "String (resonator)",
            Comp::Membrane(_) => "Membrane (resonator)",
            Comp::Plate(_) => "Plate (resonator)",
            Comp::MusicalString(_) => "Musical string (resonator)",
            Comp::Bell(_) => "Bell / cowbell (resonator)",
            Comp::Cymbal(_) => "Cymbal / gong (resonator)",
            Comp::Body { .. } => "Body / cavity (resonator)",
            Comp::Wires { .. } => "Snare wires",
            Comp::Breath { .. } => "Breath (exciter)",
            Comp::Reed { .. } => "Reed / lip (exciter)",
            Comp::Lips { .. } => "Lips / brass (coupled)",
            Comp::ReedBore { .. } => "Reed + bore (coupled)",
            Comp::DoubleReed { .. } => "Double reed + bore (coupled)",
            Comp::ReedPipe { .. } => "Reed pipe (waveguide)",
            Comp::AirJet { .. } => "Air jet / flute (flue)",
            Comp::BowedString { .. } => "Bowed string (waveguide)",
            Comp::PluckedString { .. } => "Plucked string (waveguide)",
            Comp::HammeredString { .. } => "Hammered string (piano)",
            Comp::Hammer { .. } => "Hammer (exciter)",
            Comp::Bow { .. } => "Bow (exciter)",
            Comp::Voice { .. } => "Voice / glottis (exciter)",
            Comp::Horn(_) => "Air column (resonator)",
            Comp::Bore { .. } => "Air column (waveguide)",
            Comp::WaveguideHorn { .. } => "Waveguide horn (bore)",
            Comp::Sine { .. } => "Sine (tone exciter)",
            Comp::Mix => "Mix / output",
        }
    }

    /// True for the exciters (energy sources). A resonator fed by an exciter is
    /// the instrument's *voice* and resonates (struck normalization); one fed by
    /// another resonator is a coupling *filter* (so its Q colours the drive
    /// instead of amplifying it into a blow-up).
    pub(crate) fn is_exciter(&self) -> bool {
        matches!(
            self,
            Comp::Strike
                | Comp::Hammer { .. }
                | Comp::Breath { .. }
                | Comp::Reed { .. }
                | Comp::ReedBore { .. }
                | Comp::Lips { .. }
                | Comp::DoubleReed { .. }
                | Comp::Bow { .. }
                | Comp::Voice { .. }
                | Comp::ReedPipe { .. }
                | Comp::AirJet { .. }
                | Comp::BowedString { .. }
                | Comp::PluckedString { .. }
                | Comp::HammeredString { .. }
                | Comp::Sine { .. }
        )
    }

    /// The resting (fixed-geometry) pitch of a geometric resonator, Hz — the
    /// pitch it sounds at when it is *not* tracking the key (a coupling body/pot,
    /// a sympathetic string). `None` for components that don't set an absolute
    /// pitch from geometry (exciters, fixed-formant bodies, note-tracked models).
    fn natural_pitch(&self) -> Option<f32> {
        match self {
            Comp::String(m) => Some(m.open_pitch_hz()),
            Comp::Membrane(m) => Some(m.open_pitch_hz()),
            _ => None,
        }
    }

    /// Instantiate this component's per-voice DSP node for a played note.
    /// `driven` selects the resonator normalization: `false` = struck (a unit
    /// impulse rings out at the modal amplitudes), `true` = a constant-peak-gain
    /// filter (colours a continuous drive instead of amplifying it by its Q).
    fn instantiate(&self, freq_hz: f32, vel: f32, sr: f32, driven: bool) -> Box<dyn Node> {
        // A resonator gets its modes from the wrapped model's `excite`.
        let bank_of = |m: &dyn FtmModel| {
            let mut b = ModeBuffer::default();
            m.excite(freq_hz, vel, sr, &mut b);
            b
        };
        // Build a modal resonator struck or as a driven filter, per `driven`.
        let reso = |b: &ModeBuffer| -> Box<dyn Node> {
            if driven {
                Box::new(ModalResonator::from_bank_filter(b, sr))
            } else {
                Box::new(ModalResonator::from_bank(b, sr))
            }
        };
        match self {
            Comp::Strike => Box::new(ImpulseExciter::new(vel)),
            Comp::String(m) => reso(&bank_of(m)),
            Comp::Membrane(m) => reso(&bank_of(m)),
            Comp::Plate(m) => reso(&bank_of(m)),
            Comp::MusicalString(m) => reso(&bank_of(m)),
            Comp::Bell(m) => reso(&bank_of(m)),
            Comp::Cymbal(m) => reso(&bank_of(m)),
            Comp::Body { cavity_litres, soundhole_cm, top_hz, decay_s } => {
                Box::new(FormantResonator::new(
                    &body_bank(*cavity_litres, *soundhole_cm, *top_hz, *decay_s),
                    sr,
                ))
            }
            Comp::Wires { level, tone } => {
                let t = tone.clamp(0.3, 3.0);
                Box::new(SnareWires::new(*level, 600.0 * t, 8_000.0, sr))
            }
            Comp::Breath { level, tone } => {
                let t = tone.clamp(0.3, 3.0);
                Box::new(DriveExciter::new(*level, 300.0 * t, 3_000.0 * t, sr))
            }
            Comp::Horn(m) => reso(&bank_of(m)),
            Comp::Bore { tone, length } => {
                // Pitch from the (key-mapped) bore length when set; else the key.
                let f = if *length > 0.0 { C_AIR / (2.0 * length.max(0.02)) } else { freq_hz };
                Box::new(WaveguideBore::new(f, *tone, sr))
            }
            Comp::WaveguideHorn { r1, r2, r3, length, segments, tone } => {
                Box::new(WaveguideHorn::new(*r1, *r2, *r3, *length, *segments, *tone, sr))
            }
            Comp::Reed { pressure, stiffness, freq_hz } => {
                Box::new(ReedExciter::new(*pressure, *stiffness, *freq_hz, sr))
            }
            Comp::ReedBore {
                pressure, stiffness, length, tone, register, overblow, conical, tract_gain, tract_q,
            } => {
                let mut reed = CoupledReed::new(
                    *pressure, *stiffness, *length, *tone, *register, *overblow, *conical, sr,
                );
                // The vocal tract tunes to the played note, inside the reed loop.
                // Engage it ONLY in the altissimo registers (harmonic above the
                // first overblow) — there it enables the lock; on the normal two
                // registers it would just detune the calibrated tone. (`overblow`
                // now carries the played note's harmonic, set by the key-map.)
                let nat_h = if *conical { 2.0 } else { 3.0 };
                let voiced = *tract_gain > 0.0 && *overblow > nat_h + 0.5;
                let q = if *tract_q > 0.0 { *tract_q } else { 12.0 };
                reed.set_tract(freq_hz, q, if voiced { *tract_gain } else { 0.0 }, sr);
                Box::new(reed)
            }
            Comp::DoubleReed {
                pressure, stiffness, length, tone, register, overblow, conical, tract_gain, tract_q,
            } => {
                let mut reed = CoupledDoubleReed::new(
                    *pressure, *stiffness, *length, *tone, *register, *overblow, *conical, sr,
                );
                // Engage the tract voicing only in the altissimo (above the first
                // overblow), exactly as the single reed — it detunes the calibrated
                // low/overblown registers otherwise.
                let nat_h = if *conical { 2.0 } else { 3.0 };
                let voiced = *tract_gain > 0.0 && *overblow > nat_h + 0.5;
                let q = if *tract_q > 0.0 { *tract_q } else { 12.0 };
                reed.set_tract(freq_hz, q, if voiced { *tract_gain } else { 0.0 }, sr);
                Box::new(reed)
            }
            Comp::Lips { pressure, tension, length, tone } => {
                Box::new(LipReed::new(*pressure, *tension, *length, *tone, sr))
            }
            Comp::ReedPipe { pressure, stiffness, tone } => {
                Box::new(WaveguideReed::new(freq_hz, *pressure, *stiffness, *tone, sr))
            }
            Comp::AirJet { pressure, jet_ratio, tone, length } => {
                // Pitch is set by the (key-mapped) bore length, f ≈ c/2L; falls
                // back to the played note if the length isn't set.
                let l = if *length > 0.0 { *length } else { C_AIR / (2.0 * freq_hz.max(1.0)) };
                Box::new(JetPipe::new(l, *pressure, *jet_ratio, *tone, sr))
            }
            Comp::BowedString { bow_pos, brightness, speed, force } => {
                Box::new(WaveguideBow::new(freq_hz, *bow_pos, *brightness, *speed, *force, vel, sr))
            }
            Comp::PluckedString { length_m, tension_n, core_mm, youngs_gpa, open_hz, pos, decay, damping } => {
                Box::new(WaveguidePluck::from_physical(
                    freq_hz, *length_m, *tension_n, *core_mm, *youngs_gpa, *open_hz,
                    *pos, *decay, *damping, vel, sr,
                ))
            }
            Comp::HammeredString { length_m, tension_n, core_mm, youngs_gpa, open_hz, hardness, felt, pos, decay, damping } => {
                Box::new(WaveguideHammer::from_physical(
                    freq_hz, *length_m, *tension_n, *core_mm, *youngs_gpa, *open_hz,
                    *hardness, *felt, vel, *pos, *decay, *damping, sr,
                ))
            }
            Comp::Hammer { hardness, felt } => {
                Box::new(HammerExciter::new(vel, *hardness, *felt, sr))
            }
            Comp::Bow { speed, force } => Box::new(BowExciter::new(*speed, *force, sr)),
            Comp::Voice { open_quotient, level } => {
                Box::new(VoiceExciter::new(freq_hz, *open_quotient, *level, sr))
            }
            Comp::Sine { level } => Box::new(SineExciter::new(freq_hz, *level, sr)),
            Comp::Mix => Box::new(Sum),
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        match self {
            Comp::Strike => {
                ui.label(
                    egui::RichText::new("Impulse at note-on; velocity from the key.")
                        .weak()
                        .small(),
                );
                false
            }
            Comp::String(m) => m.params_ui(ui),
            Comp::Membrane(m) => m.params_ui(ui),
            Comp::Plate(m) => m.params_ui(ui),
            Comp::MusicalString(m) => m.params_ui(ui),
            Comp::Bell(m) => m.params_ui(ui),
            Comp::Cymbal(m) => m.params_ui(ui),
            Comp::Body { cavity_litres, soundhole_cm, top_hz, decay_s } => {
                let mut c = false;
                let f_h = helmholtz_hz(*cavity_litres, *soundhole_cm);
                if f_h > 0.0 {
                    ui.label(
                        egui::RichText::new(format!(
                            "air resonance ≈ {f_h:.0} Hz ({})",
                            midi_name(freq_to_midi(f_h))
                        ))
                        .weak()
                        .small(),
                    );
                }
                c |= ui
                    .add(unbounded_slider(cavity_litres, 0.0..=60.0, "Cavity volume").suffix(" L"))
                    .on_hover_text("Enclosed air volume. With the soundhole it sets the Helmholtz 'boom'. 0 = no cavity (a soundboard).")
                    .changed();
                c |= ui
                    .add(unbounded_slider(soundhole_cm, 0.0..=15.0, "Soundhole").suffix(" cm"))
                    .on_hover_text("Soundhole diameter — bigger raises the air resonance.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(top_hz, 40.0..=1500.0, "Top resonance").suffix(" Hz"))
                    .on_hover_text("Main top/soundboard plate resonance (a guitar top ≈ 195 Hz).")
                    .changed();
                c |= ui
                    .add(unbounded_slider(decay_s, 0.02..=1.0, "Body decay").suffix(" s"))
                    .on_hover_text("How long the body rings — bodies are well damped (~0.15 s).")
                    .changed();
                c
            }
            Comp::Wires { level, tone } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(level, 0.0..=2.0, "Rattle level")).changed();
                c |= ui.add(unbounded_slider(tone, 0.3..=3.0, "Rattle tone")).changed();
                c
            }
            Comp::Breath { level, tone } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(level, 0.0..=1.0, "Breath level")).changed();
                c |= ui.add(unbounded_slider(tone, 0.3..=3.0, "Breath tone")).changed();
                c
            }
            Comp::Horn(m) => m.params_ui(ui),
            Comp::Bore { tone, length } => {
                let mut c = ui.add(unbounded_slider(tone, 0.0..=1.5, "Bell brightness")).changed();
                c |= ui
                    .add(unbounded_slider(length, 0.0..=2.0, "Bore length (m, 0 = track key)"))
                    .on_hover_text("f = c/2L. Usually key-mapped so the reed follows the bore.")
                    .changed();
                c
            }
            Comp::WaveguideHorn { r1, r2, r3, length, segments, tone } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(length, 0.05..=2.0, "Bore length (m)"))
                    .on_hover_text("Sets the pitch (usually key-mapped): cylinder f≈c/4L, cone f≈c/2L.")
                    .changed();
                c |= ui.add(unbounded_slider(r1, 0.002..=0.03, "Throat radius (m)")).changed();
                c |= ui.add(unbounded_slider(r2, 0.0..=0.1, "Taper (cone → all harmonics)")).changed();
                c |= ui.add(unbounded_slider(r3, 0.0..=0.05, "Flare (bell)")).changed();
                c |= ui.add(unbounded_slider(tone, 0.0..=1.5, "Bell brightness")).changed();
                let mut seg = *segments as f32;
                if ui.add(unbounded_slider(&mut seg, 2.0..=32.0, "Segments (flare detail)")).changed() {
                    *segments = seg.round().clamp(2.0, 64.0) as usize;
                    c = true;
                }
                c
            }
            Comp::Reed { pressure, stiffness, freq_hz } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(pressure, 0.0..=2.0, "Mouth pressure")).changed();
                c |= ui.add(unbounded_slider(stiffness, 0.0..=3.0, "Reed stiffness")).changed();
                c |= ui
                    .add(unbounded_slider(freq_hz, 0.0..=600.0, "Fixed pitch (Hz)"))
                    .on_hover_text(
                        "0 = a bare valve driven by a feedback edge (pitch from the bore it drives). \
                         >0 = a self-contained mouthpiece buzzing at this one fixed pitch, to drive a resonator forward.",
                    )
                    .changed();
                c
            }
            Comp::ReedBore {
                pressure, stiffness, length, tone, register, overblow, conical, tract_gain, tract_q,
            } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(pressure, 0.1..=2.0, "Mouth pressure")).changed();
                c |= ui.add(unbounded_slider(stiffness, 0.0..=3.0, "Reed stiffness")).changed();
                c |= ui
                    .add(unbounded_slider(length, 0.05..=2.0, "Bore length (m)"))
                    .on_hover_text("Sets the pitch (usually key-mapped): f ≈ c/2L.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(register, 0.0..=0.6, "Register key"))
                    .on_hover_text("0 = closed (low register); open lifts the register vent → overblows.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(overblow, 2.0..=3.0, "Overblow ratio"))
                    .on_hover_text("Register break: 3 = a cylinder's twelfth (clarinet), 2 = a cone's octave (sax).")
                    .changed();
                c |= ui
                    .checkbox(conical, "Conical bore")
                    .on_hover_text("Cone: full harmonic series, octave overblow (sax/oboe). Off = cylinder: odd harmonics, twelfth (clarinet).")
                    .changed();
                c |= ui.add(unbounded_slider(tone, 0.0..=1.5, "Bell brightness")).changed();
                c |= ui
                    .add(unbounded_slider(tract_gain, 0.0..=8.0, "Vocal-tract voicing"))
                    .on_hover_text("Airway resonance tuned to the played note, inside the reed loop. 0 = off; raise it to voice the altissimo (upper) registers — biases the reed onto higher bore harmonics. Finicky: a narrow sweet spot.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(tract_q, 0.0..=40.0, "Tract Q"))
                    .on_hover_text("Sharpness of the tract resonance. 0 = default (~12). Higher = a tighter, more selective altissimo lock.")
                    .changed();
                c
            }
            Comp::DoubleReed {
                pressure, stiffness, length, tone, register, overblow, conical, tract_gain, tract_q,
            } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(pressure, 0.1..=2.0, "Mouth pressure")).changed();
                c |= ui
                    .add(unbounded_slider(stiffness, 0.0..=3.0, "Reed stiffness"))
                    .on_hover_text("Stiff double-reed blades resonate high (bright, buzzy).")
                    .changed();
                c |= ui
                    .add(unbounded_slider(length, 0.05..=2.0, "Bore length (m)"))
                    .on_hover_text("Sets the pitch (usually key-mapped): f ≈ c/2L.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(register, 0.0..=0.6, "Register key"))
                    .on_hover_text("0 = closed (low register); open lifts the vent → overblows the octave.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(overblow, 2.0..=3.0, "Overblow ratio"))
                    .on_hover_text("A cone overblows the octave (2). Kept for parity with the single reed.")
                    .changed();
                c |= ui
                    .checkbox(conical, "Conical bore")
                    .on_hover_text("Double reeds are conical: full harmonic series, octave overblow.")
                    .changed();
                c |= ui.add(unbounded_slider(tone, 0.0..=1.5, "Bell brightness")).changed();
                c |= ui
                    .add(unbounded_slider(tract_gain, 0.0..=8.0, "Vocal-tract voicing"))
                    .on_hover_text("Airway resonance inside the reed loop. 0 = off; raise to voice the altissimo registers.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(tract_q, 0.0..=40.0, "Tract Q"))
                    .on_hover_text("Sharpness of the tract resonance. 0 = default (~12).")
                    .changed();
                c
            }
            Comp::Lips { pressure, tension, length, tone } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(pressure, 0.3..=2.0, "Mouth pressure")).changed();
                c |= ui
                    .add(unbounded_slider(tension, 0.5..=6.0, "Lip tension (partial)"))
                    .on_hover_text("Embouchure: the lip resonance as a multiple of the bore fundamental. ≈1 plays the fundamental, ≈2 the octave, ≈3 the twelfth — brass overblow up the harmonic series on one tube.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(length, 0.05..=2.0, "Bore length (m)"))
                    .on_hover_text("Sets the pitch (usually key-mapped): f ≈ c/2L.")
                    .changed();
                c |= ui.add(unbounded_slider(tone, 0.0..=1.5, "Bell brightness")).changed();
                c
            }
            Comp::ReedPipe { pressure, stiffness, tone } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(pressure, 0.1..=1.5, "Breath pressure")).changed();
                c |= ui.add(unbounded_slider(stiffness, 0.2..=3.0, "Reed stiffness")).changed();
                c |= ui.add(unbounded_slider(tone, 0.0..=1.5, "Bell brightness")).changed();
                c
            }
            Comp::AirJet { pressure, jet_ratio, tone, length } => {
                let mut c = false;
                c |= ui
                    .add(unbounded_slider(pressure, 0.2..=1.5, "Blowing pressure"))
                    .on_hover_text("Jet velocity. Blow harder to overblow the octave (a faster jet).")
                    .changed();
                c |= ui
                    .add(unbounded_slider(jet_ratio, 0.2..=0.8, "Jet ratio"))
                    .on_hover_text("Jet convection delay ÷ acoustic period. ~0.5 = fundamental; smaller biases higher registers.")
                    .changed();
                c |= ui.add(unbounded_slider(tone, 0.0..=1.5, "Edge brightness")).changed();
                c |= ui
                    .add(unbounded_slider(length, 0.05..=2.0, "Bore length (m)"))
                    .on_hover_text("Sets the pitch (usually key-mapped): f ≈ c/2L.")
                    .changed();
                c
            }
            Comp::BowedString { bow_pos, brightness, speed, force } => {
                let mut c = false;
                c |= ui
                    .add(unbounded_slider(bow_pos, 0.04..=0.25, "Bow position (bridge -> tasto)"))
                    .on_hover_text("Bow-to-bridge distance as a fraction of the string. ~0.08 normal; smaller = near the bridge (brighter/ponticello).")
                    .changed();
                c |= ui.add(unbounded_slider(brightness, 0.0..=1.0, "Brightness")).changed();
                c |= ui.add(unbounded_slider(speed, 0.2..=3.0, "Bow speed")).changed();
                c |= ui.add(unbounded_slider(force, 0.1..=2.0, "Bow force")).changed();
                c
            }
            Comp::PluckedString { length_m, tension_n, core_mm, youngs_gpa, open_hz, pos, decay, damping } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(length_m, 0.1..=2.0, "Length (m)")).changed();
                c |= ui.add(unbounded_slider(tension_n, 20.0..=1000.0, "Tension (N)")).changed();
                c |= ui
                    .add(unbounded_slider(core_mm, 0.1..=2.0, "Core / stiffness gauge (mm)"))
                    .on_hover_text("The bending-resisting core — a wound string's thin steel core, not its overall diameter.")
                    .changed();
                c |= ui.add(unbounded_slider(youngs_gpa, 4.0..=220.0, "Young's modulus (GPa)")).changed();
                c |= ui.add(unbounded_slider(open_hz, 20.0..=440.0, "Open-string pitch (Hz)")).changed();
                c |= ui.add(unbounded_slider(pos, 0.02..=0.5, "Pluck position")).changed();
                c |= ui.add(unbounded_slider(decay, 0.2..=12.0, "Decay time (s)")).changed();
                c |= ui.add(unbounded_slider(damping, 0.0..=0.9, "HF damping")).changed();
                c
            }
            Comp::HammeredString { length_m, tension_n, core_mm, youngs_gpa, open_hz, hardness, felt, pos, decay, damping } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(length_m, 0.1..=2.0, "Length (m)")).changed();
                c |= ui.add(unbounded_slider(tension_n, 20.0..=1200.0, "Tension (N)")).changed();
                c |= ui.add(unbounded_slider(core_mm, 0.1..=2.0, "Core / stiffness gauge (mm)")).changed();
                c |= ui.add(unbounded_slider(youngs_gpa, 4.0..=220.0, "Young's modulus (GPa)")).changed();
                c |= ui.add(unbounded_slider(open_hz, 20.0..=440.0, "Open-string pitch (Hz)")).changed();
                c |= ui.add(unbounded_slider(hardness, 0.0..=1.0, "Felt hardness")).changed();
                c |= ui.add(unbounded_slider(felt, 1.0..=4.0, "Felt nonlinearity")).changed();
                c |= ui.add(unbounded_slider(pos, 0.02..=0.5, "Strike position")).changed();
                c |= ui.add(unbounded_slider(decay, 0.2..=12.0, "Decay time (s)")).changed();
                c |= ui.add(unbounded_slider(damping, 0.0..=0.9, "HF damping")).changed();
                c
            }
            Comp::Hammer { hardness, felt } => {
                let mut c = false;
                c |= ui
                    .add(unbounded_slider(hardness, 0.0..=1.0, "Felt hardness"))
                    .on_hover_text("Soft (0) = dark, long contact; hard (1) = bright, ~1 ms contact.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(felt, 1.8..=3.5, "Felt nonlinearity"))
                    .on_hover_text("Compression exponent p (piano felt ≈ 2.2–3.5). Higher = more velocity-dependent brightness.")
                    .changed();
                c
            }
            Comp::Bow { speed, force } => {
                let mut c = false;
                c |= ui
                    .add(unbounded_slider(speed, 0.0..=2.0, "Bow speed"))
                    .on_hover_text("How fast the bow travels — louder/brighter with more.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(force, 0.0..=3.0, "Bow force"))
                    .on_hover_text("Bow pressure. Too little slips (whistle), too much chokes.")
                    .changed();
                c
            }
            Comp::Voice { open_quotient, level } => {
                let mut c = false;
                c |= ui
                    .add(unbounded_slider(open_quotient, 0.1..=0.95, "Open quotient"))
                    .on_hover_text("How long the vocal folds stay open — low = pressed/buzzy, high = breathy.")
                    .changed();
                c |= ui.add(unbounded_slider(level, 0.0..=1.0, "Voice level")).changed();
                c
            }
            Comp::Sine { level } => {
                ui.add(unbounded_slider(level, 0.0..=1.0, "Sine level")).changed()
            }
            Comp::Mix => {
                ui.label(egui::RichText::new("Sums its inputs (see edge strengths).").weak().small());
                false
            }
        }
    }
}

/// A body's formant bank (base freq, gain), scaled by `tone`, damped by `ring`.
/// Helmholtz air-resonance frequency (Hz) of a cavity of `cavity_litres` with a
/// circular soundhole of diameter `soundhole_cm`: `f = (c/2π)·√(A/(V·L))`, with
/// neck-end correction `L ≈ 1.7·r`. Returns 0 if there is no cavity/hole.
fn helmholtz_hz(cavity_litres: f32, soundhole_cm: f32) -> f32 {
    let v = (cavity_litres * 1e-3).max(0.0); // m³
    let d = (soundhole_cm * 1e-2).max(0.0); // m
    if v <= 1e-6 || d <= 1e-4 {
        return 0.0;
    }
    let r = 0.5 * d;
    let area = PI * r * r; // m²
    let l_eff = 1.7 * r; // end-corrected neck length, m
    (C_AIR / TWO_PI) * (area / (v * l_eff)).sqrt()
}

/// A resonant body: a Helmholtz air cavity (from real volume + soundhole) plus a
/// top/soundboard plate resonance and two higher body modes above it. The
/// frequencies are note-independent — a real box rings at its own resonances,
/// not the played pitch. The higher body modes (×1.9, ×3.1 of the top) are
/// approximate stand-ins for the plate's inharmonic panel modes.
fn body_bank(cavity_litres: f32, soundhole_cm: f32, top_hz: f32, decay_s: f32) -> ModeBuffer {
    let mut b = ModeBuffer::default();
    let a0 = LN_1000 / decay_s.max(0.02); // base decay rate, 1/s
    let f_h = helmholtz_hz(cavity_litres, soundhole_cm);
    if f_h > 0.0 {
        b.push(f_h, 0.6, a0); // the "main air" resonance (the low boom)
    }
    let top = top_hz.max(0.0);
    if top > 1.0 {
        b.push(top, 0.45, a0 * 1.3); // main top / soundboard plate resonance
        b.push(top * 1.9, 0.28, a0 * 1.7); // higher body panel modes (approx)
        b.push(top * 3.1, 0.16, a0 * 2.2);
    }
    b
}

impl Comp {
    /// The numeric parameters of this component that a key can drive, by name.
    fn mappable(&self) -> &'static [&'static str] {
        match self {
            Comp::String(_) => &["length", "tension", "decay"],
            Comp::Membrane(_) => &["tension", "radius", "decay"],
            Comp::Plate(_) => &["ring"],
            Comp::Body { .. } => &["top_hz", "decay"],
            Comp::Horn(_) => &["length"],
            // The traveling-wave bore whose length the key drives to set pitch.
            Comp::WaveguideHorn { .. } => &["length"],
            // The waveguide bore's length — the coupled reed↔bore pitch control.
            Comp::Bore { .. } => &["length"],
            // The coupled reed+bore's length sets its (in-tune) pitch.
            Comp::ReedBore { .. } | Comp::DoubleReed { .. } => &["length", "register", "pressure"],
            // The lip-brass bore length sets the pitch; tension selects the partial.
            Comp::Lips { .. } => &["length", "tension"],
            // The air-jet flue pipe's length sets its (in-tune) pitch, f ≈ c/2L.
            Comp::AirJet { .. } => &["length"],
            // The reed's fixed pitch — so a key can drive embouchure/pitch on a
            // self-contained mouthpiece (`freq_hz > 0`).
            Comp::Reed { .. } => &["freq"],
            // The felt hammer's hardness/nonlinearity — key-mapped so the treble
            // gets harder, brighter hammers and the bass softer ones, as a real
            // piano's hammers are graded across the keyboard.
            Comp::Hammer { .. } => &["hardness", "felt"],
            // The plucked string's decay is key-mapped so treble notes ring
            // shorter than the bass, as real strings do.
            Comp::PluckedString { .. } => &["decay"],
            Comp::HammeredString { .. } => &["hardness", "decay"],
            Comp::MusicalString(_) | Comp::Bell(_) | Comp::Cymbal(_) | Comp::Strike | Comp::Mix
            | Comp::Wires { .. } | Comp::Breath { .. }
            | Comp::Bow { .. } | Comp::Voice { .. } | Comp::ReedPipe { .. } | Comp::BowedString { .. }
            | Comp::Sine { .. } => &[],
        }
    }

    /// Read a mappable parameter's current (base) value.
    fn get_param(&self, name: &str) -> Option<f32> {
        match (self, name) {
            (Comp::String(m), "length") => Some(m.length_m),
            (Comp::String(m), "tension") => Some(m.tension_n),
            (Comp::String(m), "decay") => Some(m.decay_time),
            (Comp::Membrane(m), "tension") => Some(m.tension_nm),
            (Comp::Membrane(m), "radius") => Some(m.radius_m),
            (Comp::Membrane(m), "decay") => Some(m.decay_time),
            (Comp::Plate(m), "ring") => Some(m.decay_time),
            (Comp::Body { top_hz, .. }, "top_hz") => Some(*top_hz),
            (Comp::Body { decay_s, .. }, "decay") => Some(*decay_s),
            (Comp::Horn(m), "length") => Some(m.length),
            (Comp::WaveguideHorn { length, .. }, "length") => Some(*length),
            (Comp::Bore { length, .. }, "length") => Some(*length),
            (Comp::ReedBore { length, .. }, "length") => Some(*length),
            (Comp::ReedBore { pressure, .. }, "pressure") => Some(*pressure),
            (Comp::DoubleReed { pressure, .. }, "pressure") => Some(*pressure),
            (Comp::ReedBore { register, .. }, "register") => Some(*register),
            (Comp::DoubleReed { length, .. }, "length") => Some(*length),
            (Comp::DoubleReed { register, .. }, "register") => Some(*register),
            (Comp::Lips { length, .. }, "length") => Some(*length),
            (Comp::Lips { tension, .. }, "tension") => Some(*tension),
            (Comp::AirJet { length, .. }, "length") => Some(*length),
            (Comp::Reed { freq_hz, .. }, "freq") => Some(*freq_hz),
            (Comp::Hammer { hardness, .. }, "hardness") => Some(*hardness),
            (Comp::Hammer { felt, .. }, "felt") => Some(*felt),
            (Comp::PluckedString { decay, .. }, "decay") => Some(*decay),
            (Comp::HammeredString { hardness, .. }, "hardness") => Some(*hardness),
            (Comp::HammeredString { decay, .. }, "decay") => Some(*decay),
            _ => None,
        }
    }

    /// Set a mappable parameter (used by the key map at note-on).
    fn set_param(&mut self, name: &str, v: f32) {
        match (self, name) {
            (Comp::String(m), "length") => m.length_m = v,
            (Comp::String(m), "tension") => m.tension_n = v,
            (Comp::String(m), "decay") => m.decay_time = v,
            (Comp::Membrane(m), "tension") => m.tension_nm = v,
            (Comp::Membrane(m), "radius") => m.radius_m = v,
            (Comp::Membrane(m), "decay") => m.decay_time = v,
            (Comp::Plate(m), "ring") => m.decay_time = v,
            (Comp::Body { top_hz, .. }, "top_hz") => *top_hz = v,
            (Comp::Body { decay_s, .. }, "decay") => *decay_s = v,
            (Comp::Horn(m), "length") => m.length = v,
            (Comp::WaveguideHorn { length, .. }, "length") => *length = v,
            (Comp::Bore { length, .. }, "length") => *length = v,
            (Comp::ReedBore { length, .. }, "length") => *length = v,
            (Comp::ReedBore { pressure, .. }, "pressure") => *pressure = v,
            (Comp::DoubleReed { pressure, .. }, "pressure") => *pressure = v,
            (Comp::ReedBore { register, .. }, "register") => *register = v,
            (Comp::DoubleReed { length, .. }, "length") => *length = v,
            (Comp::DoubleReed { register, .. }, "register") => *register = v,
            (Comp::Lips { length, .. }, "length") => *length = v,
            (Comp::Lips { tension, .. }, "tension") => *tension = v,
            (Comp::AirJet { length, .. }, "length") => *length = v,
            (Comp::Reed { freq_hz, .. }, "freq") => *freq_hz = v,
            (Comp::Hammer { hardness, .. }, "hardness") => *hardness = v.clamp(0.0, 1.0),
            (Comp::Hammer { felt, .. }, "felt") => *felt = v,
            (Comp::PluckedString { decay, .. }, "decay") => *decay = v.max(0.05),
            (Comp::HammeredString { hardness, .. }, "hardness") => *hardness = v.clamp(0.0, 1.0),
            (Comp::HammeredString { decay, .. }, "decay") => *decay = v.max(0.05),
            _ => {}
        }
    }

    /// Configure this component to overblow-track the key (the `Overblow` key-map
    /// strategy). On a Webster horn it switches on the overblow-tracked render:
    /// the horn picks an overblown bore length per key so a harmonic lands on it.
    fn set_overblow(&mut self, freq_hz: f32, anchor_hz: f32, steps: u32, microtune: bool) {
        match self {
            Comp::Horn(m) => {
                m.play_mode = PlayMode::OverblowTracked;
                m.key_tracks_pitch = true;
                m.overblow_anchor_hz = anchor_hz;
                m.valve_steps = steps;
                m.overblow_microtune = microtune;
            }
            // A coupled reed bore does the register break, generalised by the
            // component's own `overblow` ratio: below the break (ratio× the lowest
            // note) it plays the fundamental (low register, vent closed); at/above
            // it lifts the register vent and plays the ratio-th harmonic (a twelfth
            // up for a cylinder/clarinet at ratio 3, an octave for a cone/sax at
            // ratio 2) over the *same* bore-length range.
            Comp::ReedBore { length, register, pressure, overblow, .. } => {
                let f = freq_hz.max(1.0);
                let ratio = overblow.max(1.5);
                let f_break = ratio * anchor_hz.max(1.0);
                // Position within the register (ratio above its lowest note) and
                // the per-register micro-tune gain — the bore flattens as you play
                // up, more so in the overblown upper register than the low one.
                let (rel, gain) = if f < f_break {
                    *register = 0.0;
                    *length = C_AIR / (2.0 * f);
                    (f / anchor_hz.max(1.0), 0.05)
                } else {
                    *register = 0.3;
                    *length = C_AIR / (2.0 * (f / ratio)); // bore fundamental = f/ratio
                    (f / f_break, 0.20)
                };
                // Pressure micro-tune, the way a player lips each note in tune:
                // blow a little harder as the note rises in its register to cancel
                // the bore's residual flatness (harder = sharper).
                if microtune {
                    *pressure *= 1.0 + gain * (rel - 1.0);
                }
            }
            _ => {}
        }
    }

    /// Set a coupled reed+bore's register + bore length for a played note, given
    /// the tune `anchor_hz` and the register span `steps` in semitones (0 = the
    /// bore's natural overblow interval — a twelfth for a cylinder, an octave for
    /// a cone). Register `r = floor(semitones-above-anchor / steps)` extends both
    /// ways from the anchor: below the anchor and within the first `steps`
    /// semitones it plays the fundamental; higher, it lifts the register vent and
    /// overblows to the register's harmonic, the bore length shortening to reach
    /// any note. Register `r`'s harmonic climbs the bore's series — a cylinder's
    /// odd harmonics 1, 3, 5, 7…, a cone's full series 1, 2, 3, 4… — but the reed
    /// reliably locks only to its FIRST overblow on its own, so without a voiced
    /// vocal tract (`tract_gain` 0) the harmonic is capped at that first overblow
    /// and the upper registers just reuse it with a shorter bore. With the tract
    /// engaged, the harmonic climbs freely, reaching the altissimo registers.
    /// No-op for non-reed components.
    fn set_reed_register(&mut self, freq: f32, anchor_hz: f32, steps: f32) {
        if let Comp::ReedBore { length, register, overblow, conical, tract_gain, .. }
        | Comp::DoubleReed { length, register, overblow, conical, tract_gain, .. } = self
        {
            let f = freq.max(1.0);
            let anchor = anchor_hz.max(1.0);
            let nat_h = if *conical { 2.0 } else { 3.0 }; // first overblow harmonic
            let interval = if steps > 0.5 { steps } else { 12.0 * (nat_h as f32).log2() };
            let n = 12.0 * (f / anchor).log2(); // semitones above the anchor
            let r = (n / interval.max(1.0)).floor().max(0.0);
            if r < 1.0 {
                *overblow = 1.0;
                *register = 0.0; // fundamental
                *length = C_AIR / (2.0 * f);
            } else {
                // Register r's harmonic: cone 2,3,4,… ; cylinder 3,5,7,…
                let voiced = *tract_gain > 0.0;
                let h = if *conical { r + 1.0 } else { 2.0 * r + 1.0 };
                let h = if voiced { h } else { nat_h }; // capped without the tract
                *overblow = h; // vent at 1/h
                *register = 0.3;
                *length = C_AIR / (2.0 * (f / h)); // bore fundamental = f / h
            }
        }
    }

    /// Set a pitched exciter's bore length for a played note. A coupled reed uses
    /// its register-break geometry (`set_reed_register`); every other length-tuned
    /// exciter (the lip reed, the air jet) just plays the chromatic fundamental
    /// f ≈ c/2L. This is what lets the calibrated `OverblowTuned` key-map — and the
    /// calibrator — work on the brass/flue exciters, not only the reeds.
    fn set_pitch_geometry(&mut self, freq: f32, anchor_hz: f32, steps: f32) {
        if matches!(self, Comp::ReedBore { .. } | Comp::DoubleReed { .. }) {
            self.set_reed_register(freq, anchor_hz, steps);
        } else if self.get_param("length").is_some() {
            self.set_param("length", C_AIR / (2.0 * freq.max(1.0)));
        }
    }
}

/// How the played key drives one component — a per-component *strategy*, so
/// different components can respond to the keyboard in different ways.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum KeyMapKind {
    /// Power law: `param = base · (f/C4)^amount`. `amount = 0` is fixed; `+1`
    /// scales up with pitch (a drum-head tension ∝ f²); `-1` inversely (a bore
    /// length ∝ 1/f — chromatic transpose). The everyday geometric mapping.
    Power { param: String, amount: f32 },
    /// Overblow + steps: choose an overblown bore length so a harmonic lands on
    /// the key — a brass player picking a fingering and register. Drives a
    /// Webster horn's overblow-tracked rendering. `steps` = tube-length steps.
    Overblow { anchor_hz: f32, steps: u32, microtune: bool },
    /// Like `Overblow`, but for a coupled reed+bore: `anchor_hz` sets the tune and
    /// `steps` the register span in semitones (0 = the bore's natural overblow
    /// interval — a cylinder's twelfth, a cone's octave). The register is inferred
    /// per note (fundamental below the anchor + first span, overblown above), and
    /// a **calibrated** per-MIDI-note bore-length `table` (solved once) lands every
    /// note exactly in tune. `table[note]` is the bore-length multiplier — length
    /// has direct pitch authority for cones and cylinders alike.
    OverblowTuned {
        anchor_hz: f32,
        #[serde(default)]
        steps: f32,
        table: Vec<f32>,
    },
}

impl KeyMapKind {
    /// A short human label for the strategy (for the editor's dropdown).
    pub fn label(&self) -> &'static str {
        match self {
            KeyMapKind::Power { .. } => "Chromatic / power",
            KeyMapKind::Overblow { .. } => "Overblow + steps",
            KeyMapKind::OverblowTuned { .. } => "Overblow + calibrated tuning",
        }
    }
}

/// One key→component binding: which component the key drives, and by what
/// strategy. A key can carry several bindings (usually one per component).
#[derive(Clone, Serialize, Deserialize)]
pub struct KeyBinding {
    pub component: usize,
    #[serde(flatten)]
    pub map: KeyMapKind,
}

/// A directed, gained edge: `from`'s output feeds `to`, scaled by `gain` (the
/// coupling strength between the two components).
#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub gain: f32,
}

/// An instrument as a graph: components + edges + which node is the output.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InstrumentGraph {
    pub components: Vec<Comp>,
    pub edges: Vec<Edge>,
    pub output: usize,
    /// How the played key drives component parameters. Empty ⇒ resonators just
    /// track pitch on their own (their `key_tracks_pitch`).
    #[serde(default)]
    pub key_map: Vec<KeyBinding>,
}

impl InstrumentGraph {
    /// Human label for each component (for the graph editor).
    pub fn labels(&self) -> Vec<&'static str> {
        self.components.iter().map(|c| c.label()).collect()
    }

    /// Calibrate a per-MIDI-note bore-**length** correction table for a coupled-
    /// reed component `comp` (the `OverblowTuned` strategy). For each note it
    /// applies the register-break geometry, then *solves* (bisection over short
    /// renders) for the length multiplier that lands the note exactly in tune,
    /// capturing the model's whole tuning residual — physical + numerical — at
    /// once. Length has direct, full pitch authority (`f = c/2L`), so this works
    /// for both cylinders and cones (a cone's pitch barely responds to mouth
    /// pressure, so pressure calibration can't tune saxes). Run once (edit-time);
    /// the result is looked up per note-on at no cost.
    pub fn calibrate_tuning(&self, comp: usize, anchor_hz: f32, steps: f32, sr: f32) -> Vec<f32> {
        // Autocorrelation pitch with parabolic interpolation of the peak lag, so
        // the pitch is resolved to a fraction of a cent (integer lags alone are
        // only ~±16 cents — too coarse to null the tuning we're solving for).
        let acf = |y: &[f32], f0: f32| -> f32 {
            let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
            let s: Vec<f32> = y.iter().map(|v| v - mean).collect();
            let corr = |lag: usize| -> f32 { (0..s.len() - lag).map(|i| s[i] * s[i + lag]).sum() };
            let lo = (sr / (f0 * 1.6)) as usize;
            let hi = ((sr / (f0 * 0.5)) as usize).min(s.len() / 2 - 2);
            let (mut best, mut bc) = (lo.max(2), f32::MIN);
            for lag in lo.max(2)..hi {
                let c = corr(lag);
                if c > bc {
                    bc = c;
                    best = lag;
                }
            }
            let (a, b, c) = (corr(best - 1), corr(best), corr(best + 1));
            let denom = a - 2.0 * b + c;
            let frac = if denom.abs() > 1e-12 { 0.5 * (a - c) / denom } else { 0.0 };
            sr / (best as f32 + frac.clamp(-1.0, 1.0))
        };
        // Render one note at a given bore-length multiplier and return its pitch.
        // Calibrate on the reed component ALONE — the downstream bell colours the
        // tone but doesn't set pitch, and rebuilding a modal horn per solve step
        // is ~100× the cost of the reed. So strip to a one-node graph.
        let reed = self.components.get(comp).cloned();
        // Returns (pitch, rms). A long build-up lets even the slow, weak altissimo
        // (tract-voiced) locks settle before the tail is measured.
        let measure = |mult: f32, f: f32| -> (f32, f32) {
            let mut c = match &reed {
                Some(c) => c.clone(),
                None => return (0.0, 0.0),
            };
            c.set_pitch_geometry(f, anchor_hz, steps); // register/chromatic length
            if let Some(l) = c.get_param("length") {
                c.set_param("length", l * mult);
            }
            let g = InstrumentGraph {
                components: vec![c],
                edges: Vec::new(),
                output: 0,
                key_map: Vec::new(),
            };
            match g.build_graph(f, 1.0, sr) {
                Some(mut n) => {
                    let warm = (0.3 * sr) as usize;
                    let tail_len = ((24.0 * sr / f) as usize).clamp(8_000, 24_000);
                    let y: Vec<f32> = (0..warm + tail_len).map(|_| n.tick(&[])).collect();
                    let tail = &y[warm..];
                    let rms = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
                    if rms > 1e-3 {
                        (acf(tail, f), rms)
                    } else {
                        (0.0, rms)
                    }
                }
                None => (0.0, 0.0),
            }
        };
        let cents = |p: f32, f: f32| 1200.0 * (p / f).log2();
        let mut table = vec![1.0f32; 128];
        for note in 24u8..=108 {
            let f = super::REF_PITCH_HZ * 2f32.powf((note as f32 - 60.0) / 12.0);
            // Baseline (no correction) — the calibration must beat this to be used.
            let (p_raw, rms_raw) = measure(1.0, f);
            if p_raw <= 0.0 {
                continue; // didn't oscillate; leave 1.0
            }
            // Bisect the length multiplier so the rendered pitch = f. Shorter bore
            // → sharper (f = c/2L), so flat pitch means shorten (smaller mult).
            let (mut lo, mut hi) = (0.82f32, 1.18f32);
            for _ in 0..9 {
                let m = (lo + hi) * 0.5;
                let (p, _) = measure(m, f);
                if p <= 0.0 {
                    break;
                }
                if p < f {
                    hi = m; // flat → shorten the bore
                } else {
                    lo = m;
                }
            }
            let m = (lo + hi) * 0.5;
            // Accept the correction only if it genuinely improves tuning AND keeps
            // the note sounding. The finicky altissimo registers can otherwise be
            // detuned or choked by a length the pitch-solver liked in isolation;
            // there we keep the raw geometry (audible, if not perfectly in tune)
            // rather than make it worse.
            let (p_cal, rms_cal) = measure(m, f);
            if p_cal > 0.0
                && cents(p_cal, f).abs() < cents(p_raw, f).abs()
                && rms_cal > 0.6 * rms_raw
            {
                table[note as usize] = m;
            }
        }
        table
    }

    /// True if component `i` is an exciter (energy source) — for the editor.
    pub fn is_exciter_at(&self, i: usize) -> bool {
        self.components.get(i).map(|c| c.is_exciter()).unwrap_or(false)
    }

    /// Draw one component's parameter editor (used by the node editor).
    pub fn component_params_ui(&mut self, i: usize, ui: &mut egui::Ui) -> bool {
        self.components.get_mut(i).map(|c| c.params_ui(ui)).unwrap_or(false)
    }

    /// Auto-lay-out the nodes left-to-right by longest path from an exciter, so
    /// signal flows left→right. Returns one (x, y) per component. The editor
    /// keeps these in UI state and lets the user drag them.
    ///
    /// Feedback edges (a bore feeding its reed back, coupled membranes) would make
    /// a naive longest-path count diverge — the columns grow every pass and the
    /// nodes fly off the canvas. So we first find the back-edges with a DFS and
    /// layer using only the forward edges (a DAG), which stays bounded.
    pub fn auto_layout(&self) -> Vec<[f32; 2]> {
        let n = self.components.len();
        if n == 0 {
            return Vec::new();
        }
        // Outgoing adjacency (target node per outgoing edge of each node).
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.edges {
            if e.from < n && e.to < n {
                adj[e.from].push(e.to);
            }
        }
        // DFS marking back-edges: an edge to a node still on the DFS stack closes
        // a cycle. (state: 0 = unvisited, 1 = on stack, 2 = done.)
        let mut state = vec![0u8; n];
        let mut is_back: Vec<Vec<bool>> = adj.iter().map(|a| vec![false; a.len()]).collect();
        for s in 0..n {
            if state[s] != 0 {
                continue;
            }
            state[s] = 1;
            let mut stack: Vec<(usize, usize)> = vec![(s, 0)];
            while let Some((u, ci)) = stack.pop() {
                if ci < adj[u].len() {
                    stack.push((u, ci + 1));
                    let v = adj[u][ci];
                    match state[v] {
                        1 => is_back[u][ci] = true, // v is an ancestor → back edge
                        0 => {
                            state[v] = 1;
                            stack.push((v, 0));
                        }
                        _ => {}
                    }
                } else {
                    state[u] = 2;
                }
            }
        }
        // Longest-path columns over the forward edges only (now a DAG → bounded).
        let mut col = vec![0usize; n];
        for _ in 0..n {
            for u in 0..n {
                for (k, &v) in adj[u].iter().enumerate() {
                    if !is_back[u][k] {
                        col[v] = col[v].max(col[u] + 1);
                    }
                }
            }
        }
        let mut per_col: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        (0..n)
            .map(|i| {
                let c = col[i];
                let r = *per_col.entry(c).or_insert(0);
                per_col.entry(c).and_modify(|v| *v += 1);
                [40.0 + c as f32 * 175.0, 40.0 + r as f32 * 95.0]
            })
            .collect()
    }
}

impl Default for InstrumentGraph {
    /// A struck string coloured by a body: `Strike → String → Body`, mixed dry
    /// (string) + a light wet (body) at the output.
    fn default() -> Self {
        InstrumentGraph {
            components: vec![
                Comp::Strike,
                Comp::String(PureString::default()),
                Comp::Body { cavity_litres: 15.0, soundhole_cm: 9.0, top_hz: 195.0, decay_s: 0.18 },
                Comp::Mix,
            ],
            edges: vec![
                Edge { from: 0, to: 1, gain: 1.0 }, // strike drives the string
                Edge { from: 1, to: 2, gain: 1.0 }, // string drives the body
                Edge { from: 1, to: 3, gain: 1.0 }, // dry string → out
                Edge { from: 2, to: 3, gain: 0.05 }, // wet body → out (coupling strength)
            ],
            output: 3,
            key_map: Vec::new(),
        }
    }
}

impl InstrumentGraph {
    /// Remove component `r`, fixing up every index that referenced it: edges
    /// touching it are dropped and higher indices shifted down; key-map targets
    /// and the output node are adjusted the same way.
    pub(crate) fn remove_component(&mut self, r: usize) {
        if r >= self.components.len() {
            return;
        }
        self.components.remove(r);
        self.edges.retain(|e| e.from != r && e.to != r);
        for e in &mut self.edges {
            if e.from > r {
                e.from -= 1;
            }
            if e.to > r {
                e.to -= 1;
            }
        }
        self.key_map.retain(|k| k.component != r);
        for k in &mut self.key_map {
            if k.component > r {
                k.component -= 1;
            }
        }
        if self.output == r || self.output >= self.components.len() {
            self.output = self.components.len().saturating_sub(1);
        } else if self.output > r {
            self.output -= 1;
        }
    }
}

impl FtmModel for InstrumentGraph {
    fn id(&self) -> &'static str {
        "instrument_graph"
    }
    fn display_name(&self) -> &'static str {
        "Instrument Graph"
    }
    fn description(&self) -> &'static str {
        "A configurable graph of exciter/resonator components wired by coupling edges."
    }
    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        // Rendered via the graph, not the mode bank — but fill the bank from the
        // primary resonator so the classic path (and preset validation) has
        // something representative.
        out.clear();
        for c in &self.components {
            match c {
                Comp::String(m) => return m.excite(freq_hz, vel, sr, out),
                Comp::Membrane(m) => return m.excite(freq_hz, vel, sr, out),
                Comp::Plate(m) => return m.excite(freq_hz, vel, sr, out),
                Comp::MusicalString(m) => return m.excite(freq_hz, vel, sr, out),
                Comp::Bell(m) => return m.excite(freq_hz, vel, sr, out),
                Comp::Cymbal(m) => return m.excite(freq_hz, vel, sr, out),
                Comp::Horn(m) => return m.excite(freq_hz, vel, sr, out),
                _ => {}
            }
        }
        // Waveguide/self-contained voices (e.g. ReedPipe) have no mode bank; give
        // the classic-path validation a single representative partial at the note.
        if out.n == 0 {
            out.push(freq_hz.max(1.0), 1.0, 4.0);
        }
    }
    fn build_graph(&self, freq_hz: f32, vel: f32, sr: f32) -> Option<Box<dyn Node>> {
        if self.components.is_empty() || self.output >= self.components.len() {
            return None;
        }
        // Apply the key map: each binding drives its component by its strategy.
        let mut comps = self.components.clone();
        if !self.key_map.is_empty() {
            let ratio = (freq_hz / super::REF_PITCH_HZ).max(1e-4);
            for kb in &self.key_map {
                if let Some(c) = comps.get_mut(kb.component) {
                    match &kb.map {
                        KeyMapKind::Power { param, amount } => {
                            if let Some(base) = c.get_param(param) {
                                c.set_param(param, base * ratio.powf(*amount));
                            }
                            // A key-mapped horn is driven by the map, so switch off
                            // its own internal key-tracking (no competing control).
                            if let Comp::Horn(m) = c {
                                m.key_tracks_pitch = false;
                            }
                        }
                        KeyMapKind::Overblow { anchor_hz, steps, microtune } => {
                            c.set_overblow(freq_hz, *anchor_hz, *steps, *microtune);
                        }
                        KeyMapKind::OverblowTuned { anchor_hz, steps, table } => {
                            // Range floor (reeds only): a single reed can't sound
                            // below its longest tube (the anchor is its lowest note).
                            let is_reed = matches!(
                                c,
                                Comp::ReedBore { .. } | Comp::DoubleReed { .. }
                            );
                            if is_reed && freq_hz < anchor_hz * REED_FLOOR {
                                if let Comp::ReedBore { pressure, .. }
                                | Comp::DoubleReed { pressure, .. } = c
                                {
                                    *pressure = 0.0;
                                }
                            } else {
                                // Register/chromatic length from the anchor/steps,
                                // then the exact calibrated length correction — works
                                // for the reeds and the lip/jet exciters alike.
                                c.set_pitch_geometry(freq_hz, *anchor_hz, *steps);
                                let note = super::freq_to_midi(freq_hz).clamp(0, 127) as usize;
                                if let Some(&mult) = table.get(note) {
                                    if let Some(l) = c.get_param("length") {
                                        c.set_param("length", l * mult);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Choose each resonator's normalization from the topology:
        //  * fed by an exciter (Strike/Hammer/Breath/Reed/Bow/Voice) → struck:
        //    it is the instrument's voice and resonates (a plucked string, a
        //    blown bore);
        //  * fed by another *resonator* → filter, so its Q colours that drive
        //    instead of amplifying it into a blow-up (a body, a coupled head).
        // (Self-oscillating reed/bow loops fall out of the first case — their
        // bore is exciter-fed — so they keep the high Q the oscillation needs.)
        let n = comps.len();
        let driven: Vec<bool> = (0..n)
            .map(|i| {
                self.edges
                    .iter()
                    .any(|e| e.to == i && e.from < n && !comps[e.from].is_exciter())
            })
            .collect();
        // Pitch mapping: a *voice* resonator (exciter-fed, not `driven`) tracks the
        // played key; a *coupling* resonator (`driven` — a body, a banjo pot, a
        // shell) stays at its own fixed geometry and colours whatever drives it.
        // So the tom head tracks the key but the banjo pot does not, purely from
        // how each is wired. (A key_map can still override a coupling resonator's
        // param to make it track.)
        let nodes: Vec<Box<dyn Node>> = comps
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let fq = if driven[i] { c.natural_pitch().unwrap_or(freq_hz) } else { freq_hz };
                c.instantiate(fq, vel, sr, driven[i])
            })
            .collect();
        let mut inputs: Vec<Vec<(usize, f32)>> = vec![Vec::new(); nodes.len()];
        for e in &self.edges {
            if e.from < nodes.len() && e.to < nodes.len() {
                inputs[e.to].push((e.from, e.gain));
            }
        }
        Some(Box::new(Graph::new(nodes, inputs, self.output)))
    }
    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let labels: Vec<&'static str> = self.components.iter().map(|c| c.label()).collect();

        ui.label(egui::RichText::new("Components").strong());
        let ncomp = labels.len();
        let mut remove_comp: Option<usize> = None;
        for (i, c) in self.components.iter_mut().enumerate() {
            ui.separator();
            // Namespace each component's widgets so repeated types (e.g. two
            // Strings) or identical slider labels don't collide on egui ids.
            ui.push_id(i, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(format!("{i}. {}", labels[i]));
                    if ncomp > 1 && ui.small_button("✕").on_hover_text("remove component").clicked() {
                        remove_comp = Some(i);
                    }
                });
                changed |= c.params_ui(ui);
            });
        }
        if let Some(r) = remove_comp {
            self.remove_component(r);
            changed = true;
        }
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label("Add:");
            if ui.small_button("Strike").clicked() {
                self.components.push(Comp::Strike);
                changed = true;
            }
            if ui.small_button("String").clicked() {
                self.components.push(Comp::String(PureString::default()));
                changed = true;
            }
            if ui.small_button("Membrane").clicked() {
                self.components.push(Comp::Membrane(DrumMembrane::default()));
                changed = true;
            }
            if ui.small_button("Plate").clicked() {
                self.components.push(Comp::Plate(PurePlate::default()));
                changed = true;
            }
            if ui.small_button("Musical string").clicked() {
                self.components.push(Comp::MusicalString(MusicalString::default()));
                changed = true;
            }
            if ui.small_button("Bell").clicked() {
                self.components.push(Comp::Bell(MetalBell::default()));
                changed = true;
            }
            if ui.small_button("Cymbal").clicked() {
                self.components.push(Comp::Cymbal(Cymbal::default()));
                changed = true;
            }
            if ui.small_button("Body").clicked() {
                self.components.push(Comp::Body { cavity_litres: 15.0, soundhole_cm: 9.0, top_hz: 195.0, decay_s: 0.18 });
                changed = true;
            }
            if ui.small_button("Wires").clicked() {
                self.components.push(Comp::Wires { level: 0.6, tone: 1.0 });
                changed = true;
            }
            if ui.small_button("Breath").clicked() {
                self.components.push(Comp::Breath { level: 0.15, tone: 1.0 });
                changed = true;
            }
            if ui.small_button("Reed / lip").clicked() {
                self.components.push(Comp::Reed { pressure: 0.6, stiffness: 1.5, freq_hz: 0.0 });
                changed = true;
            }
            if ui.small_button("Reed pipe").clicked() {
                self.components.push(Comp::ReedPipe { pressure: 0.9, stiffness: 1.0, tone: 1.0 });
                changed = true;
            }
            if ui.small_button("Lips / brass").clicked() {
                self.components.push(Comp::Lips { pressure: 1.0, tension: 1.0, length: 0.6555, tone: 1.0 });
                changed = true;
            }
            if ui.small_button("Air jet / flute").clicked() {
                self.components.push(Comp::AirJet { pressure: 0.55, jet_ratio: 0.5, tone: 1.0, length: 0.66 });
                changed = true;
            }
            if ui.small_button("Bowed string").clicked() {
                self.components.push(Comp::BowedString { bow_pos: 0.09, brightness: 0.6, speed: 0.6, force: 0.4 });
                changed = true;
            }
            if ui.small_button("Plucked string").clicked() {
                self.components.push(Comp::PluckedString { length_m: 0.648, tension_n: 90.0, core_mm: 0.4, youngs_gpa: 200.0, open_hz: 82.4, pos: 0.13, decay: 4.0, damping: 0.15 });
                changed = true;
            }
            if ui.small_button("Hammered string").clicked() {
                self.components.push(Comp::HammeredString { length_m: 1.0, tension_n: 700.0, core_mm: 1.0, youngs_gpa: 200.0, open_hz: 27.5, hardness: 0.5, felt: 2.0, pos: 0.13, decay: 5.0, damping: 0.12 });
                changed = true;
            }
            if ui.small_button("Hammer").clicked() {
                self.components.push(Comp::Hammer { hardness: 0.6, felt: 2.5 });
                changed = true;
            }
            if ui.small_button("Bow").clicked() {
                self.components.push(Comp::Bow { speed: 0.6, force: 1.0 });
                changed = true;
            }
            if ui.small_button("Voice").clicked() {
                self.components.push(Comp::Voice { open_quotient: 0.6, level: 0.4 });
                changed = true;
            }
            if ui.small_button("Air column").clicked() {
                self.components.push(Comp::Horn(WebsterHorn::default()));
                changed = true;
            }
            if ui.small_button("Bore (waveguide)").clicked() {
                self.components.push(Comp::Bore { tone: 1.0, length: 0.0 });
                changed = true;
            }
            if ui.small_button("Mix").clicked() {
                self.components.push(Comp::Mix);
                changed = true;
            }
        });

        ui.separator();
        ui.label(egui::RichText::new("Edges — coupling strength").strong());
        let mut remove_edge: Option<usize> = None;
        for (ei, e) in self.edges.iter_mut().enumerate() {
            ui.push_id(("edge", ei), |ui| {
            ui.horizontal(|ui| {
                let f = egui::ComboBox::from_id_salt(("e_from", ei))
                    .width(96.0)
                    .selected_text(labels.get(e.from).copied().unwrap_or("?"))
                    .show_ui(ui, |ui| {
                        let mut ch = false;
                        for (i, lbl) in labels.iter().enumerate() {
                            ch |= ui.selectable_value(&mut e.from, i, *lbl).changed();
                        }
                        ch
                    });
                changed |= f.inner.unwrap_or(false);
                ui.label("→");
                let t = egui::ComboBox::from_id_salt(("e_to", ei))
                    .width(96.0)
                    .selected_text(labels.get(e.to).copied().unwrap_or("?"))
                    .show_ui(ui, |ui| {
                        let mut ch = false;
                        for (i, lbl) in labels.iter().enumerate() {
                            ch |= ui.selectable_value(&mut e.to, i, *lbl).changed();
                        }
                        ch
                    });
                changed |= t.inner.unwrap_or(false);
                changed |= ui.add(unbounded_slider(&mut e.gain, 0.0..=1.0, "gain")).changed();
                if ui.small_button("✕").clicked() {
                    remove_edge = Some(ei);
                }
            });
            });
        }
        if let Some(ei) = remove_edge {
            self.edges.remove(ei);
            changed = true;
        }
        ui.horizontal(|ui| {
            if ui.small_button("➕ Add edge").clicked() {
                self.edges.push(Edge { from: 0, to: labels.len().saturating_sub(1), gain: 0.5 });
                changed = true;
            }
            ui.separator();
            ui.label("Output:");
            let o = egui::ComboBox::from_id_salt("graph_out")
                .selected_text(labels.get(self.output).copied().unwrap_or("?"))
                .show_ui(ui, |ui| {
                    let mut ch = false;
                    for (i, lbl) in labels.iter().enumerate() {
                        ch |= ui.selectable_value(&mut self.output, i, *lbl).changed();
                    }
                    ch
                });
            changed |= o.inner.unwrap_or(false);
        });

        // Key map: one key can drive several component params.
        ui.separator();
        ui.label(egui::RichText::new("Key map — a key drives these params").strong());
        let params_of: Vec<&'static [&'static str]> =
            self.components.iter().map(|c| c.mappable()).collect();
        let mut remove: Option<usize> = None;
        // (binding index, component, anchor_hz) if the user clicked Calibrate —
        // run after the loop, since calibration borrows the whole graph.
        let mut calibrate_req: Option<(usize, usize, f32, f32)> = None;
        for (t, kb) in self.key_map.iter_mut().enumerate() {
            ui.push_id(("kb", t), |ui| {
            ui.horizontal(|ui| {
                let c = egui::ComboBox::from_id_salt(("kb_c", t))
                    .selected_text(labels.get(kb.component).copied().unwrap_or("?"))
                    .show_ui(ui, |ui| {
                        let mut ch = false;
                        for (i, lbl) in labels.iter().enumerate() {
                            ch |= ui.selectable_value(&mut kb.component, i, *lbl).changed();
                        }
                        ch
                    });
                changed |= c.inner.unwrap_or(false);

                // Strategy: how this key drives the component.
                let s = egui::ComboBox::from_id_salt(("kb_s", t))
                    .selected_text(kb.map.label())
                    .show_ui(ui, |ui| {
                        let mut ch = false;
                        if ui.selectable_label(matches!(kb.map, KeyMapKind::Power { .. }), "Chromatic / power").clicked()
                            && !matches!(kb.map, KeyMapKind::Power { .. })
                        {
                            kb.map = KeyMapKind::Power { param: String::new(), amount: -1.0 };
                            ch = true;
                        }
                        if ui.selectable_label(matches!(kb.map, KeyMapKind::Overblow { .. }), "Overblow + steps").clicked()
                            && !matches!(kb.map, KeyMapKind::Overblow { .. })
                        {
                            kb.map = KeyMapKind::Overblow { anchor_hz: 82.41, steps: 6, microtune: true };
                            ch = true;
                        }
                        if ui.selectable_label(matches!(kb.map, KeyMapKind::OverblowTuned { .. }), "Overblow + calibrated tuning").clicked()
                            && !matches!(kb.map, KeyMapKind::OverblowTuned { .. })
                        {
                            kb.map = KeyMapKind::OverblowTuned { anchor_hz: 146.83, steps: 0.0, table: Vec::new() };
                            ch = true;
                        }
                        ch
                    });
                changed |= s.inner.unwrap_or(false);

                match &mut kb.map {
                    KeyMapKind::Power { param, amount } => {
                        let opts = params_of.get(kb.component).copied().unwrap_or(&[]);
                        let p = egui::ComboBox::from_id_salt(("kb_p", t))
                            .selected_text(if param.is_empty() { "—" } else { param.as_str() })
                            .show_ui(ui, |ui| {
                                let mut ch = false;
                                for name in opts {
                                    ch |= ui.selectable_value(param, name.to_string(), *name).changed();
                                }
                                ch
                            });
                        changed |= p.inner.unwrap_or(false);
                        changed |= ui.add(unbounded_slider(amount, -2.0..=2.0, "amount")).changed();
                    }
                    KeyMapKind::Overblow { anchor_hz, steps, microtune } => {
                        changed |= ui.add(unbounded_slider(anchor_hz, 30.0..=440.0, "anchor Hz")).changed();
                        let mut s = *steps as f32;
                        if ui.add(unbounded_slider(&mut s, 0.0..=12.0, "steps")).changed() {
                            *steps = s.round() as u32;
                            changed = true;
                        }
                        changed |= ui.checkbox(microtune, "in-tune").changed();
                    }
                    KeyMapKind::OverblowTuned { anchor_hz, steps, table } => {
                        changed |= ui.add(unbounded_slider(anchor_hz, 30.0..=440.0, "anchor Hz")).changed();
                        changed |= ui
                            .add(unbounded_slider(steps, 0.0..=24.0, "register steps"))
                            .on_hover_text("Semitones per register before overblowing. 0 = the bore's natural interval (a cylinder's twelfth, a cone's octave). Recalibrate after changing.")
                            .changed();
                        let n = table.iter().filter(|&&v| (v - 1.0).abs() > 1e-4).count();
                        ui.label(
                            egui::RichText::new(if table.is_empty() {
                                "uncalibrated".into()
                            } else {
                                format!("{n} notes tuned")
                            })
                            .weak()
                            .small(),
                        );
                        if ui.button("Calibrate").clicked() {
                            calibrate_req = Some((t, kb.component, *anchor_hz, *steps));
                        }
                    }
                }
                if ui.button("✕").clicked() {
                    remove = Some(t);
                }
            });
            });
        }
        if let Some(t) = remove {
            self.key_map.remove(t);
            changed = true;
        }
        if let Some((t, comp, anchor, steps)) = calibrate_req {
            // Solve the per-note length-tuning table (a one-off; a moment of compute).
            let tbl = self.calibrate_tuning(comp, anchor, steps, 48_000.0);
            if let Some(kb) = self.key_map.get_mut(t) {
                if let KeyMapKind::OverblowTuned { table, .. } = &mut kb.map {
                    *table = tbl;
                }
            }
            changed = true;
        }
        if ui.button("➕ Add key mapping").clicked() {
            self.key_map.push(KeyBinding { component: 0, map: KeyMapKind::Power { param: String::new(), amount: -1.0 } });
            changed = true;
        }
        changed
    }
    fn box_clone(&self) -> Box<dyn FtmModel> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
    fn as_instrument_graph_mut(&mut self) -> Option<&mut InstrumentGraph> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::model_from_id;

    #[test]
    fn default_graph_serializes_rebuilds_and_renders() {
        let g = InstrumentGraph::default();
        // Round-trip through JSON + the registry.
        let json = g.to_json();
        let back = model_from_id("instrument_graph", &json).expect("rebuilds from json");
        assert_eq!(back.id(), "instrument_graph");
        // The per-voice graph renders non-silent audio.
        let mut node = back.build_graph(220.0, 1.0, 48_000.0).expect("builds a graph");
        let mut acc = 0.0f32;
        for _ in 0..24_000 {
            let s = node.tick(&[]);
            acc += s * s;
        }
        let rms = (acc / 24_000.0).sqrt();
        assert!(rms > 1e-4 && rms.is_finite(), "graph renders sound (rms={rms})");
    }

    #[test]
    fn body_edge_gain_controls_its_contribution() {
        // Turning the body→mix edge to 0 should make it quieter than at 0.5.
        let sr = 48_000.0;
        let render = |mix: f32| -> f32 {
            let mut g = InstrumentGraph::default();
            // edge index 3 is body(2) → mix(3)
            g.edges[3].gain = mix;
            let mut node = g.build_graph(110.0, 1.0, sr).unwrap();
            let mut acc = 0.0f32;
            for _ in 0..24_000 {
                let s = node.tick(&[]);
                acc += s * s;
            }
            (acc / 24_000.0).sqrt()
        };
        assert!(render(0.5) > render(0.0), "more body mix = more energy");
    }

    #[test]
    fn body_helmholtz_matches_real_geometry() {
        // A ~15 L guitar box with a 9 cm soundhole rings near its measured
        // "main air" resonance (~100–130 Hz), and the bank carries that mode.
        let f_h = helmholtz_hz(15.0, 9.0);
        assert!((100.0..=140.0).contains(&f_h), "guitar air resonance {f_h} Hz off");
        // No cavity → no air mode (a soundboard is just its plate resonance).
        assert_eq!(helmholtz_hz(0.0, 9.0), 0.0);
        let bank = body_bank(15.0, 9.0, 195.0, 0.18);
        assert!(bank.n >= 2, "cavity body has an air mode + top modes");
        assert!(bank.freq[..bank.n].iter().any(|f| (*f - f_h).abs() < 1.0), "air mode present");
        assert!(bank.freq[..bank.n].iter().any(|f| (*f - 195.0).abs() < 1.0), "top mode present");
    }
    #[test]
    fn repointing_output_auditions_that_component() {
        // Repointing `output` taps a component's signal where it sits in the
        // graph. Default graph: Strike(0) → String(1) → Body(2) → Mix(3).
        // Tapping the String rings; tapping the Strike is a one-shot.
        let sr = 48_000.0;
        let base = InstrumentGraph::default();
        let render = |out: usize| -> Vec<f32> {
            let mut g = base.clone();
            g.output = out;
            let mut n = g.build_graph(220.0, 1.0, sr).unwrap();
            (0..24_000).map(|_| n.tick(&[])).collect()
        };
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        let string = render(1); // the String node
        let strike = render(0); // the Strike exciter (impulse then silence)
        assert!(rms(&string[2000..]) > 1e-4, "auditioning the string rings");
        assert!(rms(&strike[2000..]) < rms(&string[2000..]), "the strike is a one-shot, quieter tail");
    }

    #[test]
    fn sine_probe_isolates_a_component() {
        // The editor auditions a component in isolation via `Sine → comp → Mix`.
        // A pure sine holds a steady tone; a resonator fed that sine at its own
        // pitch resonates it — audibly louder than the bare probe off-resonance.
        let sr = 48_000.0;
        let mk = |comp: Comp| InstrumentGraph {
            components: vec![Comp::Sine { level: 1.0 }, comp, Comp::Mix],
            edges: vec![
                Edge { from: 0, to: 1, gain: 1.0 },
                Edge { from: 1, to: 2, gain: 1.0 },
            ],
            output: 2,
            key_map: vec![],
        };
        let render = |g: InstrumentGraph| -> Vec<f32> {
            let mut n = g.build_graph(220.0, 1.0, sr).unwrap();
            (0..24_000).map(|_| n.tick(&[])).collect()
        };
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        // A pure sine sustains (its late-window energy stays up).
        let sine = render(mk(Comp::Mix)); // Sine → Mix → Mix: just the tone
        assert!(rms(&sine[12_000..]) > 1e-3, "sine probe sustains a tone");
        // A string tuned to the played pitch resonates the probe into sound.
        let strung = render(mk(Comp::String(PureString::default())));
        assert!(
            rms(&strung[2000..]) > 1e-4 && rms(&strung[2000..]).is_finite(),
            "resonator resonates the pure probe tone"
        );
    }

    #[test]
    fn key_map_drives_a_param() {
        // The grounded string tracks pitch itself, so the key map drives a
        // non-pitch parameter: map decay time inversely to pitch (higher notes
        // decay faster), then a high note must carry less late-window energy.
        let sr = 48_000.0;
        let g = InstrumentGraph {
            components: vec![
                Comp::Strike,
                Comp::String(PureString { decay_time: 3.0, hf_damping: 0.0, ..Default::default() }),
                Comp::Mix,
            ],
            edges: vec![
                Edge { from: 0, to: 1, gain: 1.0 },
                Edge { from: 1, to: 2, gain: 1.0 },
            ],
            output: 2,
            key_map: vec![KeyBinding { component: 1, map: KeyMapKind::Power { param: "decay".into(), amount: -2.0 } }],
        };
        // Energy in a late window (0.3–0.5 s) — a proxy for how long it rings.
        let late_energy = |note: f32| -> f32 {
            let mut n = g.build_graph(note, 1.0, sr).unwrap();
            let y: Vec<f32> = (0..24_000).map(|_| n.tick(&[])).collect();
            y[14_400..].iter().map(|s| s * s).sum::<f32>()
        };
        let low = late_energy(220.0);
        let high = late_energy(880.0); // amount=-2 → much shorter decay up high
        assert!(
            low > high * 2.0,
            "the decay key-map makes high notes ring shorter (low={low} high={high})"
        );
    }
    #[test]
    fn remove_component_reindexes_everything() {
        // default: [Strike(0), String(1), Body(2), Mix(3)],
        // edges (0→1),(1→2),(1→3),(2→3), output 3.
        let mut g = InstrumentGraph::default();
        g.key_map.push(KeyBinding { component: 2, map: KeyMapKind::Power { param: "top_hz".into(), amount: 1.0 } });
        g.remove_component(1); // drop the String
        assert_eq!(g.components.len(), 3, "one fewer component");
        // edges touching 1 dropped; only old (2→3) survives, shifted to (1→2).
        assert!(g.edges.iter().all(|e| e.from < 3 && e.to < 3), "no dangling indices");
        assert!(g.edges.iter().any(|e| e.from == 1 && e.to == 2), "(2→3) became (1→2)");
        assert_eq!(g.output, 2, "output 3 shifted to 2");
        assert_eq!(g.key_map[0].component, 1, "key target 2 shifted to 1");
        assert!(g.build_graph(220.0, 1.0, 48_000.0).is_some(), "still builds");
    }
    #[test]
    fn coupled_snare_is_stable_and_uses_feedback() {
        let sr = 48_000.0;
        // Two coupled heads + wires, with the wires re-exciting the bottom head
        // (a feedback edge). `wires` scales the rattle so we can A/B it.
        let snare = |wires: f32| -> Vec<f32> {
            let top = DrumMembrane { radius_m: 0.165, tension_nm: 2000.0, decay_time: 0.18, num_modes: 24, ..DrumMembrane::default() };
            let bottom = DrumMembrane { radius_m: 0.165, tension_nm: 2600.0, decay_time: 0.12, num_modes: 20, ..DrumMembrane::default() };
            let g = InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Membrane(top),
                    Comp::Membrane(bottom),
                    Comp::Wires { level: wires, tone: 1.0 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 0.5 },
                    Edge { from: 2, to: 3, gain: 1.0 },
                    Edge { from: 3, to: 2, gain: 0.3 }, // wires re-excite the bottom (feedback)
                    Edge { from: 1, to: 4, gain: 1.0 },
                    Edge { from: 2, to: 4, gain: 0.5 },
                    Edge { from: 3, to: 4, gain: 0.6 },
                ],
                output: 4,
                key_map: Vec::new(),
            };
            let mut n = g.build_graph(180.0, 1.0, sr).unwrap();
            (0..24_000).map(|_| n.tick(&[])).collect()
        };
        let with_wires = snare(0.6);
        let no_wires = snare(0.0);
        // Stable: the two-membrane + wires feedback graph stays finite and bounded.
        assert!(with_wires.iter().all(|v| v.is_finite() && v.abs() < 50.0), "coupled loop is stable");
        // Makes sound, and the wires (rattle + their re-excitation of the bottom
        // head) audibly change it vs a wireless drum — same seed, so the
        // difference is purely the wire coupling.
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        assert!(rms(&with_wires) > 1e-4, "non-silent");
        let diff: f32 = with_wires.iter().zip(&no_wires).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / with_wires.len() as f32;
        assert!(diff > 1e-5, "the snare wires change the sound (diff={diff:.6})");
    }
    #[test]
    fn breath_driven_voice_sustains() {
        // A continuous breath drive should make a resonator sustain: its late
        // energy stays comparable to its early energy (a struck voice would have
        // decayed away by then).
        let sr = 48_000.0;
        let g = InstrumentGraph {
            components: vec![
                Comp::Breath { level: 0.1, tone: 1.0 },
                Comp::String(PureString::default()),
                Comp::Mix,
            ],
            edges: vec![
                Edge { from: 0, to: 1, gain: 1.0 },
                Edge { from: 1, to: 2, gain: 1.0 },
            ],
            output: 2,
            key_map: Vec::new(),
        };
        let mut n = g.build_graph(220.0, 1.0, sr).unwrap();
        let y: Vec<f32> = (0..48_000).map(|_| n.tick(&[])).collect();
        assert!(y.iter().all(|v| v.is_finite()), "stable");
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        let early = rms(&y[4_800..9_600]); // 0.1–0.2 s
        let late = rms(&y[38_400..43_200]); // 0.8–0.9 s
        assert!(late > early * 0.5, "driven voice sustains (early={early:.4} late={late:.4})");
    }
    #[test]
    fn reed_feedback_is_stable_and_sounds() {
        // A reed exciter driving a waveguide bore over a feedback edge — the
        // graph-decomposed wind. The traveling wave in the bore reflects, so the
        // reed self-oscillates into a clean, sustained tone.
        let sr = 48_000.0;
        let g = InstrumentGraph {
            components: vec![
                Comp::Reed { pressure: 0.9, stiffness: 1.0, freq_hz: 0.0 },
                Comp::Bore { tone: 1.0, length: 0.0 },
                Comp::Mix,
            ],
            edges: vec![
                Edge { from: 0, to: 1, gain: 1.0 }, // reed → bore
                Edge { from: 1, to: 0, gain: 1.0 }, // bore → reed (feedback)
                Edge { from: 1, to: 2, gain: 1.0 }, // bore → out
            ],
            output: 2,
            key_map: Vec::new(),
        };
        let mut n = g.build_graph(220.0, 1.0, sr).unwrap();
        let y: Vec<f32> = (0..48_000).map(|_| n.tick(&[])).collect();
        assert!(y.iter().all(|v| v.is_finite() && v.abs() < 10.0), "reed loop is stable");
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        assert!(rms(&y[24_000..48_000]) > 0.05, "reed self-oscillates into a strong tone");
    }

    #[test]
    fn auto_layout_stays_on_canvas_with_feedback_loops() {
        // A feedback edge (bore → reed) must not make the longest-path layout
        // diverge and fling every node off the right of the canvas (the blank-
        // editor bug). Columns are bounded by the node count.
        let g = InstrumentGraph {
            components: vec![
                Comp::Reed { pressure: 0.9, stiffness: 1.0, freq_hz: 0.0 },
                Comp::Bore { tone: 1.0, length: 0.0 },
                Comp::Mix,
            ],
            edges: vec![
                Edge { from: 0, to: 1, gain: 1.0 }, // reed → bore
                Edge { from: 1, to: 0, gain: 1.0 }, // bore → reed (feedback)
                Edge { from: 1, to: 2, gain: 1.0 }, // bore → out
            ],
            output: 2,
            key_map: Vec::new(),
        };
        let layout = g.auto_layout();
        assert_eq!(layout.len(), 3);
        // Every node's column index stays < n, so x < 40 + n·175 — on-canvas.
        let max_x = 40.0 + g.components.len() as f32 * 175.0;
        for (i, p) in layout.iter().enumerate() {
            assert!(p[0] >= 0.0 && p[0] < max_x, "node {i} x={} off canvas", p[0]);
        }
        // Forward flow is preserved: the reed sits left of its bore.
        assert!(layout[0][0] < layout[1][0], "reed is left of the bore");
    }

    /// Autocorrelation pitch (integer-lag) — robust to the flute's harmonic tone.
    fn acf_freq(y: &[f32], sr: f32, f0: f32) -> f32 {
        let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
        let s: Vec<f32> = y.iter().map(|v| v - mean).collect();
        let lo = (sr / (f0 * 3.0)) as usize;
        let hi = ((sr / (f0 * 0.4)) as usize).min(s.len() / 2);
        let (mut best, mut bc) = (lo.max(2), f32::MIN);
        for lag in lo.max(2)..hi {
            let c: f32 = (0..s.len() - lag).map(|i| s[i] * s[i + lag]).sum();
            if c > bc {
                bc = c;
                best = lag;
            }
        }
        // Parabolic interpolation of the peak lag → sub-sample (sub-cent) pitch.
        let corr = |lag: usize| -> f32 { (0..s.len() - lag).map(|i| s[i] * s[i + lag]).sum() };
        let (a, b, c) = (corr(best - 1), corr(best), corr(best + 1));
        let denom = a - 2.0 * b + c;
        let frac = if denom.abs() > 1e-12 { 0.5 * (a - c) / denom } else { 0.0 };
        sr / (best as f32 + frac.clamp(-1.0, 1.0))
    }
    /// AC RMS — RMS after removing the mean, so DC (which a mis-built open bore
    /// accumulates) doesn't masquerade as sound.
    fn ac_rms(y: &[f32]) -> f32 {
        let m: f32 = y.iter().sum::<f32>() / y.len() as f32;
        (y.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / y.len() as f32).sqrt()
    }

    #[test]
    fn air_jet_flute_sounds_in_tune_and_overblows_the_octave() {
        // The air-jet flue pipe: a delayed, saturating jet driving an OPEN bore.
        // It must (a) make real AC sound (not DC), (b) lock to f ≈ c/2L in tune
        // across octaves, and (c) overblow the octave when the jet velocity rises.
        let sr = 48_000.0;
        let c = C_AIR;
        // One AirJet → Mix voice at bore length L, blowing pressure `pr`.
        let render = |length: f32, pr: f32| -> Vec<f32> {
            let g = InstrumentGraph {
                components: vec![
                    Comp::AirJet { pressure: pr, jet_ratio: 0.5, tone: 1.0, length },
                    Comp::Mix,
                ],
                edges: vec![Edge { from: 0, to: 1, gain: 1.0 }],
                output: 1,
                key_map: Vec::new(),
            };
            let mut n = g.build_graph(c / (2.0 * length), 1.0, sr).unwrap();
            (0..44_000).map(|_| n.tick(&[])).collect()
        };

        // (a)+(b): in tune at f ≈ c/2L across three octaves, on real AC amplitude.
        for &f0 in &[262.0f32, 392.0, 523.0, 784.0, 1047.0] {
            let l = c / (2.0 * f0);
            let y = render(l, 0.55);
            assert!(y.iter().all(|v| v.is_finite() && v.abs() < 20.0), "jet pipe stable at {f0} Hz");
            let tail = &y[32_000..];
            let ac = ac_rms(tail);
            let mean: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
            // The sound is genuine oscillation, not a DC pedestal.
            assert!(ac > 0.05, "flute makes real AC sound at {f0} Hz (ac_rms {ac})");
            assert!(mean.abs() < ac, "output is AC, not DC-dominated ({f0} Hz: mean {mean}, ac {ac})");
            let f = acf_freq(tail, sr, f0);
            let cents = 1200.0 * (f / f0).log2();
            assert!(cents.abs() < 30.0, "plays c/2L in tune at {f0} Hz (got {cents:+.0} cents)");
        }

        // (c): on a fixed bore, a faster jet (higher blowing pressure) overblows —
        // the pitch jumps roughly an octave up (the flute's octave register break).
        let f0 = 392.0;
        let l = c / (2.0 * f0);
        let low = acf_freq(&render(l, 0.55)[32_000..], sr, f0);
        let over = acf_freq(&render(l, 1.0)[32_000..], sr, f0 * 2.0);
        assert!((low / f0 - 1.0).abs() < 0.05, "low register is the fundamental (got {low})");
        assert!((over / low / 2.0 - 1.0).abs() < 0.08, "blowing harder overblows the octave: {low} → {over}");
    }
}

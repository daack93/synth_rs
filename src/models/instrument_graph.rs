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
use super::webster_horn::WebsterHorn;
use super::{freq_to_midi, midi_name, unbounded_slider, FtmModel, ModeBuffer};
use crate::graph::{
    BowExciter, DriveExciter, FormantResonator, Graph, HammerExciter, ImpulseExciter,
    ModalResonator, Node, ReedExciter, SnareWires, Sum, VoiceExciter, WaveguideBow, WaveguideReed,
};

const PI: f32 = std::f32::consts::PI;
const TWO_PI: f32 = std::f32::consts::TAU;
/// Speed of sound in air at ~20 °C, m/s — sets the body's Helmholtz pitch.
const C_AIR: f32 = 343.0;
/// ln(1000): a decay rate of `ln(1000)/T` reaches −60 dB at `t = T` seconds.
const LN_1000: f32 = 6.907_755;

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
    /// A self-oscillating reed/lip (needs a feedback edge from its resonator).
    Reed { pressure: f32, stiffness: f32 },
    /// A digital-waveguide reed pipe — a self-contained wind voice (bore delay +
    /// bell + reed) that self-oscillates into a clean reed tone. `pressure` =
    /// breath, `stiffness` = reed hardness, `tone` = bell brightness.
    ReedPipe { pressure: f32, stiffness: f32, tone: f32 },
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
    BowedString { speed: f32, force: f32 },
    /// A vocal-fold (glottal) source, pitched at the played note. `open_quotient`
    /// = how long the folds stay open (breathy → pressed), `level` = drive.
    Voice { open_quotient: f32, level: f32 },
    /// A flaring air column (Webster horn) resonator.
    Horn(WebsterHorn),
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
            Comp::ReedPipe { .. } => "Reed pipe (waveguide)",
            Comp::BowedString { .. } => "Bowed string (waveguide)",
            Comp::Hammer { .. } => "Hammer (exciter)",
            Comp::Bow { .. } => "Bow (exciter)",
            Comp::Voice { .. } => "Voice / glottis (exciter)",
            Comp::Horn(_) => "Air column (resonator)",
            Comp::Mix => "Mix / output",
        }
    }

    /// True for the exciters (energy sources). A resonator fed by an exciter is
    /// the instrument's *voice* and resonates (struck normalization); one fed by
    /// another resonator is a coupling *filter* (so its Q colours the drive
    /// instead of amplifying it into a blow-up).
    fn is_exciter(&self) -> bool {
        matches!(
            self,
            Comp::Strike
                | Comp::Hammer { .. }
                | Comp::Breath { .. }
                | Comp::Reed { .. }
                | Comp::Bow { .. }
                | Comp::Voice { .. }
                | Comp::ReedPipe { .. }
                | Comp::BowedString { .. }
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
            Comp::Reed { pressure, stiffness } => Box::new(ReedExciter::new(*pressure, *stiffness, sr)),
            Comp::ReedPipe { pressure, stiffness, tone } => {
                Box::new(WaveguideReed::new(freq_hz, *pressure, *stiffness, *tone, sr))
            }
            Comp::BowedString { speed, force } => {
                Box::new(WaveguideBow::new(freq_hz, *speed, *force, sr))
            }
            Comp::Hammer { hardness, felt } => {
                Box::new(HammerExciter::new(vel, *hardness, *felt, sr))
            }
            Comp::Bow { speed, force } => Box::new(BowExciter::new(*speed, *force, sr)),
            Comp::Voice { open_quotient, level } => {
                Box::new(VoiceExciter::new(freq_hz, *open_quotient, *level, sr))
            }
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
            Comp::Reed { pressure, stiffness } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(pressure, 0.0..=2.0, "Mouth pressure")).changed();
                c |= ui.add(unbounded_slider(stiffness, 0.0..=3.0, "Reed stiffness")).changed();
                c
            }
            Comp::ReedPipe { pressure, stiffness, tone } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(pressure, 0.1..=1.5, "Breath pressure")).changed();
                c |= ui.add(unbounded_slider(stiffness, 0.2..=3.0, "Reed stiffness")).changed();
                c |= ui.add(unbounded_slider(tone, 0.0..=1.5, "Bell brightness")).changed();
                c
            }
            Comp::BowedString { speed, force } => {
                let mut c = false;
                c |= ui.add(unbounded_slider(speed, 0.2..=3.0, "Bow speed")).changed();
                c |= ui.add(unbounded_slider(force, 0.1..=2.0, "Bow force")).changed();
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
            Comp::MusicalString(_) | Comp::Bell(_) | Comp::Cymbal(_) | Comp::Strike | Comp::Mix
            | Comp::Wires { .. } | Comp::Breath { .. } | Comp::Reed { .. } | Comp::Hammer { .. }
            | Comp::Bow { .. } | Comp::Voice { .. } | Comp::ReedPipe { .. } | Comp::BowedString { .. } => &[],
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
            _ => {}
        }
    }
}

/// One key→parameter mapping: the played note drives `component`'s `param` as
/// `base · (f / C4)^amount`. `amount = 0` is fixed; `+1` scales the param up with
/// pitch; `-1` inversely (e.g. a resonator that shortens as the note rises). One
/// key can carry several of these, across different components.
#[derive(Clone, Serialize, Deserialize)]
pub struct KeyTarget {
    pub component: usize,
    pub param: String,
    pub amount: f32,
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
    pub key_map: Vec<KeyTarget>,
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
    fn remove_component(&mut self, r: usize) {
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
        // Apply the key map: drive each targeted (component, param) from the note.
        let mut comps = self.components.clone();
        if !self.key_map.is_empty() {
            let ratio = (freq_hz / super::REF_PITCH_HZ).max(1e-4);
            for kt in &self.key_map {
                if let Some(c) = comps.get_mut(kt.component) {
                    if let Some(base) = c.get_param(&kt.param) {
                        c.set_param(&kt.param, base * ratio.powf(kt.amount));
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
                self.components.push(Comp::Reed { pressure: 0.6, stiffness: 1.5 });
                changed = true;
            }
            if ui.small_button("Reed pipe").clicked() {
                self.components.push(Comp::ReedPipe { pressure: 0.9, stiffness: 1.0, tone: 1.0 });
                changed = true;
            }
            if ui.small_button("Bowed string").clicked() {
                self.components.push(Comp::BowedString { speed: 1.2, force: 0.6 });
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
        for (t, kt) in self.key_map.iter_mut().enumerate() {
            ui.push_id(("kt", t), |ui| {
            ui.horizontal(|ui| {
                let c = egui::ComboBox::from_id_salt(("kt_c", t))
                    .selected_text(labels.get(kt.component).copied().unwrap_or("?"))
                    .show_ui(ui, |ui| {
                        let mut ch = false;
                        for (i, lbl) in labels.iter().enumerate() {
                            ch |= ui.selectable_value(&mut kt.component, i, *lbl).changed();
                        }
                        ch
                    });
                changed |= c.inner.unwrap_or(false);

                let opts = params_of.get(kt.component).copied().unwrap_or(&[]);
                let p = egui::ComboBox::from_id_salt(("kt_p", t))
                    .selected_text(if kt.param.is_empty() { "—" } else { kt.param.as_str() })
                    .show_ui(ui, |ui| {
                        let mut ch = false;
                        for name in opts {
                            ch |= ui.selectable_value(&mut kt.param, name.to_string(), *name).changed();
                        }
                        ch
                    });
                changed |= p.inner.unwrap_or(false);

                changed |= ui.add(unbounded_slider(&mut kt.amount, -2.0..=2.0, "amount")).changed();
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
        if ui.button("➕ Add key mapping").clicked() {
            self.key_map.push(KeyTarget { component: 0, param: String::new(), amount: 1.0 });
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
            key_map: vec![KeyTarget { component: 1, param: "decay".into(), amount: -2.0 }],
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
        g.key_map.push(KeyTarget { component: 2, param: "top_hz".into(), amount: 1.0 });
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
        use crate::models::webster_horn::{Boundary, WebsterHorn};
        let sr = 48_000.0;
        let g = InstrumentGraph {
            components: vec![
                Comp::Reed { pressure: 0.6, stiffness: 1.5 },
                Comp::Horn(WebsterHorn { boundary: Boundary::Brass, r1: 0.0073, r3: 0.002, length: 0.66, depth: 18, resolution: 300, damping: 4.0, ..WebsterHorn::default() }),
                Comp::Mix,
            ],
            edges: vec![
                Edge { from: 0, to: 1, gain: 0.12 },
                Edge { from: 1, to: 0, gain: 0.4 },
                Edge { from: 1, to: 2, gain: 1.0 },
            ],
            output: 2,
            key_map: Vec::new(),
        };
        let mut n = g.build_graph(220.0, 1.0, sr).unwrap();
        let y: Vec<f32> = (0..48_000).map(|_| n.tick(&[])).collect();
        // The reed's tanh nonlinearity must bound the feedback loop (a real reed clips).
        assert!(y.iter().all(|v| v.is_finite() && v.abs() < 10.0), "reed loop is stable");
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        eprintln!("REED rms(settled) = {}", rms(&y[24_000..48_000]));
        assert!(rms(&y[24_000..48_000]) > 1e-4, "reed drives the bore (makes sound)");
    }
}

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

use super::drum_membrane::DrumMembrane;
use super::pure_plate::PurePlate;
use super::pure_string::PureString;
use super::{unbounded_slider, FtmModel, ModeBuffer};
use crate::graph::{
    FormantResonator, Graph, ImpulseExciter, ModalResonator, Node, Sum,
};

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
    /// A body / oral-cavity resonator: fixed formants, `ring` = how long it
    /// rings, `tone` = formant-frequency scale.
    Body { ring: f32, tone: f32 },
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
            Comp::Body { .. } => "Body (resonator)",
            Comp::Mix => "Mix / output",
        }
    }

    /// Instantiate this component's per-voice DSP node for a played note.
    fn instantiate(&self, freq_hz: f32, vel: f32, sr: f32) -> Box<dyn Node> {
        // A resonator gets its modes from the wrapped model's `excite`.
        let bank_of = |m: &dyn FtmModel| {
            let mut b = ModeBuffer::default();
            m.excite(freq_hz, vel, sr, &mut b);
            b
        };
        match self {
            Comp::Strike => Box::new(ImpulseExciter::new(vel)),
            Comp::String(m) => Box::new(ModalResonator::from_bank(&bank_of(m), sr)),
            Comp::Membrane(m) => Box::new(ModalResonator::from_bank(&bank_of(m), sr)),
            Comp::Plate(m) => Box::new(ModalResonator::from_bank(&bank_of(m), sr)),
            Comp::Body { ring, tone } => {
                Box::new(FormantResonator::new(&body_bank(*ring, *tone), sr))
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
            Comp::Body { ring, tone } => {
                let mut c = false;
                c |= ui
                    .add(unbounded_slider(ring, 0.1..=4.0, "Body ring"))
                    .on_hover_text("How long the body rings. Keep modest — a body is damped.")
                    .changed();
                c |= ui
                    .add(unbounded_slider(tone, 0.5..=2.0, "Body tone"))
                    .on_hover_text("Shifts the body's resonant frequencies (brightness).")
                    .changed();
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
fn body_bank(ring: f32, tone: f32) -> ModeBuffer {
    let base = [(100.0f32, 0.5f32), (210.0, 0.4), (300.0, 0.3), (450.0, 0.2)];
    let ring = ring.clamp(0.1, 6.0);
    let tone = tone.clamp(0.4, 2.5);
    let mut b = ModeBuffer::default();
    for (f, g) in base {
        b.push(f * tone, g, 45.0 / ring);
    }
    b
}

impl Comp {
    /// The numeric parameters of this component that a key can drive, by name.
    fn mappable(&self) -> &'static [&'static str] {
        match self {
            Comp::String(_) => &["length", "stiffness"],
            Comp::Membrane(_) => &["radius"],
            Comp::Plate(_) => &["ring"],
            Comp::Body { .. } => &["ring", "tone"],
            Comp::Strike | Comp::Mix => &[],
        }
    }

    /// Read a mappable parameter's current (base) value.
    fn get_param(&self, name: &str) -> Option<f32> {
        match (self, name) {
            (Comp::String(m), "length") => Some(m.string_length),
            (Comp::String(m), "stiffness") => Some(m.stiffness),
            (Comp::Membrane(m), "radius") => Some(m.radius),
            (Comp::Plate(m), "ring") => Some(m.decay_time),
            (Comp::Body { ring, .. }, "ring") => Some(*ring),
            (Comp::Body { tone, .. }, "tone") => Some(*tone),
            _ => None,
        }
    }

    /// Set a mappable parameter (used by the key map at note-on).
    fn set_param(&mut self, name: &str, v: f32) {
        match (self, name) {
            (Comp::String(m), "length") => m.string_length = v,
            (Comp::String(m), "stiffness") => m.stiffness = v,
            (Comp::Membrane(m), "radius") => m.radius = v,
            (Comp::Plate(m), "ring") => m.decay_time = v,
            (Comp::Body { ring, .. }, "ring") => *ring = v,
            (Comp::Body { tone, .. }, "tone") => *tone = v,
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
                Comp::Body { ring: 1.0, tone: 1.0 },
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
                _ => {}
            }
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
        let nodes: Vec<Box<dyn Node>> =
            comps.iter().map(|c| c.instantiate(freq_hz, vel, sr)).collect();
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
        for (i, c) in self.components.iter_mut().enumerate() {
            ui.separator();
            ui.strong(format!("{i}. {}", labels[i]));
            changed |= c.params_ui(ui);
        }

        ui.separator();
        ui.label(egui::RichText::new("Edges — coupling strength").strong());
        for e in &mut self.edges {
            let name = format!(
                "{} → {}",
                labels.get(e.from).copied().unwrap_or("?"),
                labels.get(e.to).copied().unwrap_or("?"),
            );
            changed |= ui.add(unbounded_slider(&mut e.gain, 0.0..=1.0, &name)).changed();
        }

        // Key map: one key can drive several component params.
        ui.separator();
        ui.label(egui::RichText::new("Key map — a key drives these params").strong());
        let params_of: Vec<&'static [&'static str]> =
            self.components.iter().map(|c| c.mappable()).collect();
        let mut remove: Option<usize> = None;
        for (t, kt) in self.key_map.iter_mut().enumerate() {
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
    fn key_map_drives_a_param() {
        // Map the string's length inversely to pitch (shorter = higher). With the
        // string's own pitch-tracking off, the key map is the only thing setting
        // pitch, so a higher note must land more energy in a high band.
        let sr = 48_000.0;
        let g = InstrumentGraph {
            components: vec![
                Comp::Strike,
                Comp::String(PureString { key_tracks_pitch: false, ..Default::default() }),
                Comp::Mix,
            ],
            edges: vec![
                Edge { from: 0, to: 1, gain: 1.0 },
                Edge { from: 1, to: 2, gain: 1.0 },
            ],
            output: 2,
            key_map: vec![KeyTarget { component: 1, param: "length".into(), amount: -1.0 }],
        };
        // Zero-crossing rate as a renderer-agnostic pitch proxy.
        let zcr = |note: f32| -> usize {
            let mut n = g.build_graph(note, 1.0, sr).unwrap();
            let y: Vec<f32> = (0..24_000).map(|_| n.tick(&[])).collect();
            y.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count()
        };
        let low = zcr(220.0);
        let high = zcr(660.0); // amount=-1 → shorter string → higher pitch
        assert!(
            high > low * 3 / 2,
            "the length key-map raises pitch on higher notes (high={high} low={low})"
        );
    }
}

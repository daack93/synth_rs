//! Graph-rendered versions of the struck instruments, for A/B comparison.
//!
//! Each of these wraps an existing struck model unchanged and reuses its exact
//! mode computation (`excite`), but renders the note through the **per-sample
//! voice graph** ([`crate::graph`]) — a strike driving a bank of driven
//! resonators — instead of the classic free-oscillator mode bank. That's the
//! only difference, so you can load one next to its original in the picker and
//! compare the two renderers by ear. The originals are left completely intact.

use serde::{Deserialize, Serialize};

use super::drum_membrane::DrumMembrane;
use super::pure_plate::PurePlate;
use super::musical_string::MusicalString;
use super::pure_string::PureString;
use super::{unbounded_slider, FtmModel, ModeBuffer};
use crate::graph::{BodyResonator, Graph, ImpulseExciter, ModalResonator, Node, StruckVoice};

/// Generate a graph-rendered wrapper model around a struck inner model.
macro_rules! graph_model {
    ($name:ident, $inner:ty, $id:literal, $display:literal, $desc:literal) => {
        #[derive(Clone, Serialize, Deserialize, Default)]
        #[serde(default)]
        pub struct $name {
            /// The wrapped model — its parameters and physics, unchanged.
            pub inner: $inner,
        }

        impl FtmModel for $name {
            fn id(&self) -> &'static str {
                $id
            }
            fn display_name(&self) -> &'static str {
                $display
            }
            fn description(&self) -> &'static str {
                $desc
            }
            fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
                // Same modes as the original; the graph is built from this bank.
                self.inner.excite(freq_hz, vel, sr, out);
            }
            fn build_graph(&self, bank: &ModeBuffer, sr: f32) -> Option<Box<dyn Node>> {
                Some(Box::new(StruckVoice::new(bank, sr)))
            }
            fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
                ui.label(
                    egui::RichText::new("Per-sample graph renderer (A/B vs the original)")
                        .weak()
                        .small(),
                );
                self.inner.params_ui(ui)
            }
            fn box_clone(&self) -> Box<dyn FtmModel> {
                Box::new(self.clone())
            }
            fn to_json(&self) -> serde_json::Value {
                serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
            }
        }
    };
}

graph_model!(
    GraphString,
    PureString,
    "graph_string",
    "String (graph)",
    "The Pure String rendered through the per-sample voice graph — A/B vs Pure String."
);
graph_model!(
    GraphPlate,
    PurePlate,
    "graph_plate",
    "Plate (graph)",
    "The Pure Plate rendered through the per-sample voice graph — A/B vs Pure Plate."
);
graph_model!(
    GraphDrum,
    DrumMembrane,
    "graph_drum",
    "Drum (graph)",
    "The drum membrane rendered through the per-sample voice graph — A/B vs Drum."
);
graph_model!(
    GraphMusicalString,
    MusicalString,
    "graph_musical_string",
    "Musical String (graph)",
    "The Musical String rendered through the per-sample voice graph — A/B vs Musical String."
);

/// The first genuinely multi-component instrument: a plucked string fed into a
/// **body** resonator — a two-resonator graph `Impulse → String → Body`. It
/// demonstrates a secondary resonator (like a guitar body or an oral cavity)
/// colouring a primary, wired through the general [`Graph`]. Its parameter panel
/// shows the graph node-by-node (exciter, each resonator) plus the edge mix.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphBodiedString {
    /// Primary resonator: the string (its own params + physics, unchanged).
    pub inner: PureString,
    /// Body resonator: how long it rings (larger = longer). A body is fairly
    /// damped, so keep this modest or it turns into a ringing filter.
    pub body_ring: f32,
    /// Body resonator: shifts its formant frequencies (brightness).
    pub body_tone: f32,
    /// Edge String → Body: wet mix. 0 = bypass the body entirely; the dry string
    /// is always kept, so this only *adds* body colour.
    pub body_mix: f32,
}

impl Default for GraphBodiedString {
    fn default() -> Self {
        GraphBodiedString {
            inner: PureString::default(),
            body_ring: 1.0,
            body_tone: 1.0,
            body_mix: 0.2,
        }
    }
}

impl GraphBodiedString {
    /// The body's formants (base freq, gain), scaled by `body_tone`. Fairly
    /// damped so the driven resonator can't build up runaway resonant gain.
    fn body_bank(&self) -> ModeBuffer {
        let base = [(100.0f32, 0.5f32), (210.0, 0.4), (300.0, 0.3), (450.0, 0.2)];
        let ring = self.body_ring.clamp(0.1, 6.0);
        let tone = self.body_tone.clamp(0.4, 2.5);
        let mut body = ModeBuffer::default();
        for (f, g) in base {
            body.push(f * tone, g, 45.0 / ring); // decay ∝ 1/ring; well damped
        }
        body
    }
}

impl FtmModel for GraphBodiedString {
    fn id(&self) -> &'static str {
        "bodied_string"
    }
    fn display_name(&self) -> &'static str {
        "String + Body (graph)"
    }
    fn description(&self) -> &'static str {
        "A plucked string fed into a body resonator — a two-resonator voice graph."
    }
    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        self.inner.excite(freq_hz, vel, sr, out);
    }
    fn build_graph(&self, bank: &ModeBuffer, sr: f32) -> Option<Box<dyn Node>> {
        let body = self.body_bank();
        // Impulse(0) → String(1) → Body(2); the body node is the output.
        let nodes: Vec<Box<dyn Node>> = vec![
            Box::new(ImpulseExciter::new(1.0)),
            Box::new(ModalResonator::from_bank(bank, sr)),
            Box::new(BodyResonator::new(&body, sr, 1.0, self.body_mix.clamp(0.0, 1.0))),
        ];
        let inputs = vec![vec![], vec![0], vec![1]];
        Some(Box::new(Graph::new(nodes, inputs, 2)))
    }
    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label(egui::RichText::new("Voice graph:  Strike → String → Body").weak().small());

        ui.separator();
        ui.strong("① Exciter — Strike");
        ui.label(egui::RichText::new("Impulse at note-on; velocity from the key.").weak().small());

        ui.separator();
        ui.strong("② Resonator — String");
        changed |= self.inner.params_ui(ui);

        ui.separator();
        ui.strong("③ Resonator — Body");
        changed |= ui
            .add(unbounded_slider(&mut self.body_ring, 0.1..=4.0, "Body ring"))
            .on_hover_text("How long the body rings. Keep modest — a body is damped.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.body_tone, 0.5..=2.0, "Body tone"))
            .on_hover_text("Shifts the body's resonant frequencies (brightness).")
            .changed();

        ui.separator();
        ui.strong("Edge — String → Body");
        changed |= ui
            .add(unbounded_slider(&mut self.body_mix, 0.0..=1.0, "Body mix (wet)"))
            .on_hover_text("How much body colour to add. 0 = dry string only.")
            .changed();
        changed
    }
    fn box_clone(&self) -> Box<dyn FtmModel> {
        Box::new(self.clone())
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

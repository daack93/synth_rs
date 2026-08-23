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
use super::pure_string::PureString;
use super::{FtmModel, ModeBuffer};
use crate::graph::{Node, StruckVoice};

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

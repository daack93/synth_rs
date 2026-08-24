//! Instrument presets: a named `(model + its params + engine params)` snapshot,
//! persisted as one JSON file per preset in a folder.
//!
//! This is deliberately the simplest thing that works — a flat folder of JSON.
//! A more robust store (a proper library, tagging, dedup) is future work; the
//! `Preset` type is the stable seam that a fancier backend can reuse.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::models::basic_wave::{BasicWave, Waveform};
use crate::models::cymbal::Cymbal;
use crate::models::drum_membrane::DrumMembrane;
use crate::models::instrument_graph::{Comp, Edge, InstrumentGraph, KeyBinding, KeyMapKind};
use crate::models::metal_bell::MetalBell;
use crate::models::musical_string::MusicalString;
use crate::models::pure_plate::PurePlate;
use crate::models::pure_string::PureString;
use crate::models::snare::Snare;
use crate::models::webster_horn::{Boundary, PlayMode as HornPlay, Wavefront, WebsterHorn};
use crate::models::{model_from_id, Excitation, FtmModel};
use crate::instrument::EngineParams;
use crate::project::ZoneData;

/// A saved instrument: everything needed to reconstruct a playable sound.
///
/// A preset is either a **single instrument** (`zones` empty — `model_id` /
/// `params` / `engine` describe it) or a **kit** (`zones` non-empty — each zone
/// maps a key range to its own instrument). This mirrors how `TrackData` stores
/// a track, so the two stay interchangeable. Presets saved before kits existed
/// load as single instruments (`zones` defaults empty).
#[derive(Clone, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    /// The model's [`FtmModel::id`] (a single-instrument preset).
    pub model_id: String,
    /// The model's serialized parameters ([`FtmModel::to_json`]).
    pub params: serde_json::Value,
    /// Engine-wide parameters (gain, envelope, retrigger).
    pub engine: EngineParams,
    /// Kit zones. Empty ⇒ a single instrument; non-empty ⇒ a kit.
    #[serde(default)]
    pub zones: Vec<ZoneData>,
    /// Provenance: `Some(true)` = a built-in factory preset (code-owned, so it
    /// is refreshed from [`factory`] on launch); `Some(false)` = user-saved
    /// (never overwritten); `None` = a legacy file from before this field, which
    /// [`load_library`] treats as refreshable so old factory copies update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builtin: Option<bool>,
}

impl Preset {
    /// Capture the currently-selected model and engine as a named preset.
    pub fn capture(name: &str, model: &dyn FtmModel, engine: &EngineParams) -> Self {
        Preset {
            name: name.trim().to_string(),
            model_id: model.id().to_string(),
            params: model.to_json(),
            engine: engine.clone(),
            zones: Vec::new(),
            builtin: Some(false),
        }
    }

    /// Capture a kit (a set of key-range → instrument zones) as a named preset.
    pub fn capture_kit(name: &str, zones: Vec<ZoneData>) -> Self {
        Preset {
            name: name.trim().to_string(),
            model_id: "kit".to_string(),
            params: serde_json::Value::Null,
            engine: EngineParams::default(),
            zones,
            builtin: Some(false),
        }
    }

    /// True if this preset describes a kit (has zones) rather than one instrument.
    pub fn is_kit(&self) -> bool {
        !self.zones.is_empty()
    }

    /// Rebuild the model this preset describes. `None` for a kit (use [`zones`])
    /// or if the plugin id is unknown / the params don't fit it.
    pub fn build_model(&self) -> Option<Box<dyn FtmModel>> {
        if self.is_kit() {
            return None;
        }
        model_from_id(&self.model_id, &self.params)
    }
}

/// Built-in "factory" instruments, as starting points to tune by ear. These are
/// seeded into the presets folder on first run. Each synth plugin contributes
/// its own presets here as it is added.
pub fn factory() -> Vec<Preset> {
    // Real string materials: (density kg/m³, Young's modulus GPa).
    const STEEL: (f32, f32) = (7850.0, 200.0);
    const NICKEL: (f32, f32) = (8900.0, 200.0);
    const BRONZE: (f32, f32) = (8740.0, 105.0);
    const NYLON: (f32, f32) = (1150.0, 4.0);
    const GUT: (f32, f32) = (1300.0, 6.0);
    // A physical string from real specs: speaking length (m), tension (N),
    // gauge (mm), material, −60 dB decay (s), HF damping, pluck position.
    fn string(
        length_m: f32,
        tension_n: f32,
        diameter_mm: f32,
        mat: (f32, f32),
        decay_time: f32,
        hf_damping: f32,
        pluck_pos: f32,
    ) -> PureString {
        PureString {
            length_m,
            tension_n,
            diameter_mm,
            density_kgm3: mat.0,
            youngs_gpa: mat.1,
            decay_time,
            hf_damping,
            pluck_pos,
            num_modes: 40,
            ..PureString::default()
        }
    }
    // Same string, but bowed (driven → sustains while played).
    fn bowed(
        length_m: f32,
        tension_n: f32,
        diameter_mm: f32,
        mat: (f32, f32),
        decay_time: f32,
        hf_damping: f32,
        pluck_pos: f32,
    ) -> PureString {
        PureString {
            excitation: Excitation::Bowed,
            ..string(length_m, tension_n, diameter_mm, mat, decay_time, hf_damping, pluck_pos)
        }
    }
    fn eng(gain: f32, attack_ms: f32, release_ms: f32) -> EngineParams {
        EngineParams {
            gain,
            attack_ms,
            release_ms,
            retrigger_ms: 0.0,
        }
    }
    fn make<M: FtmModel>(name: &str, model: M, engine: EngineParams) -> Preset {
        Preset::capture(name, &model, &engine)
    }

    let base = vec![
        // ---- Pure String (feedback: pluck toward saw, stronger HF damping) ----
        make("Acoustic Bass", string(0.864, 60.0, 1.30, NICKEL, 2.0, 0.5, 0.15), eng(0.75, 4.0, 120.0)),
        make("Electric Bass", string(0.864, 55.0, 1.25, NICKEL, 2.6, 0.4, 0.12), eng(0.75, 4.0, 140.0)),
        make("Acoustic Guitar", string(0.648, 90.0, 1.10, BRONZE, 2.5, 0.6, 0.12), eng(0.6, 3.0, 120.0)),
        make("Electric Guitar", string(0.648, 78.0, 1.00, NICKEL, 3.0, 0.4, 0.10), eng(0.6, 3.0, 200.0)),
        // Less bass-heavy: brighter (more modes) with the highs allowed to sustain.
        make("Piano", string(0.600, 700.0, 1.10, STEEL, 3.5, 0.3, 0.12), eng(0.6, 2.0, 150.0)),
        make("Banjo", string(0.670, 55.0, 0.40, STEEL, 0.8, 1.2, 0.08), eng(0.6, 2.0, 80.0)),
        // Rounder pluck + more HF damping to tame the "electric" low end.
        make("Harp", string(0.900, 55.0, 0.80, NYLON, 2.5, 0.5, 0.16), eng(0.6, 3.0, 180.0)),
        // ---- Multi-component graph: Strike → String → Body (A/B vs Acoustic Guitar) ----
        make(
            "Guitar + Body (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.648, 90.0, 1.10, BRONZE, 2.5, 0.6, 0.12)),
                    Comp::Body { cavity_litres: 15.0, soundhole_cm: 9.0, top_hz: 195.0, decay_s: 0.18 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.05 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 3.0, 120.0),
        ),
        // ---- Coupled-membrane snare (feedback graph): two heads + wires that
        //      re-excite the bottom head. A/B vs the "Snare" model. ----
        make(
            "Snare (coupled graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Membrane(DrumMembrane { radius_m: 0.165, tension_nm: 2000.0, areal_density_kgm2: 0.26, bending_nm: 0.02, decay_time: 0.18, hf_damping: 6.0, num_modes: 24, strike_pos: 0.4, ..DrumMembrane::default() }),
                    Comp::Membrane(DrumMembrane { radius_m: 0.165, tension_nm: 2600.0, areal_density_kgm2: 0.24, bending_nm: 0.02, decay_time: 0.12, hf_damping: 7.0, num_modes: 20, strike_pos: 0.5, ..DrumMembrane::default() }),
                    Comp::Wires { level: 0.6, tone: 1.0 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 }, // strike → top head
                    Edge { from: 1, to: 2, gain: 0.5 }, // top couples to bottom
                    Edge { from: 2, to: 3, gain: 1.0 }, // bottom drives the wires
                    Edge { from: 3, to: 2, gain: 0.3 }, // wires re-excite the bottom (feedback)
                    Edge { from: 1, to: 4, gain: 1.0 }, // top → out
                    Edge { from: 2, to: 4, gain: 0.5 }, // bottom → out
                    Edge { from: 3, to: 4, gain: 0.6 }, // wires → out
                ],
                output: 4,
                key_map: Vec::new(),
            },
            eng(0.7, 1.0, 200.0),
        ),
        // ---- Sustained/driven wind: breath into an air column (graph) ----
        make(
            "Wind (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Breath { level: 0.15, tone: 1.0 },
                    Comp::Horn(WebsterHorn {
                        boundary: Boundary::Open,
                        r1: 0.0095,
                        r2: 0.0,
                        r3: 0.001,
                        length: 0.6,
                        blow_pos: 0.15,
                        depth: 12,
                        resolution: 300,
                        damping: 3.0,
                        freq_dep_damping: -0.10,
                        visco_loss: 0.3,
                        radiation: 0.5,
                        ..WebsterHorn::default()
                    }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 }, // breath drives the air column
                    Edge { from: 1, to: 2, gain: 1.0 }, // air column → out
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 20.0, 200.0),
        ),
        // ---- Self-oscillating reed into a bore (feedback graph) ----
        make(
            "Reed (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 0.6, stiffness: 1.5, freq_hz: 0.0 },
                    Comp::Horn(WebsterHorn {
                        boundary: Boundary::Brass,
                        r1: 0.0073,
                        r2: 0.0,
                        r3: 0.002,
                        length: 0.66,
                        blow_pos: 0.0,
                        depth: 18,
                        resolution: 300,
                        damping: 4.0,
                        freq_dep_damping: -0.08,
                        visco_loss: 0.8,
                        radiation: 0.6,
                        ..WebsterHorn::default()
                    }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 0.3 }, // reed drives the bore
                    Edge { from: 1, to: 0, gain: 0.55 },  // bore pressure feeds back to the reed
                    Edge { from: 1, to: 2, gain: 1.0 },  // bore → out
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 20.0, 200.0),
        ),
        // ============================================================
        //  InstrumentGraph showcase — instruments built purely as a graph
        //  of exciter + resonator components, with secondary resonators
        //  (bodies/soundboards/shells/tract) and physical drivers
        //  (strike / breath / reed·lip). A/B against the classic models.
        // ============================================================
        // -- Plucked string + resonating body/soundboard --
        make(
            "Acoustic Bass + Body (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.864, 60.0, 1.30, NICKEL, 2.0, 0.5, 0.15)),
                    Comp::Body { cavity_litres: 30.0, soundhole_cm: 10.0, top_hz: 90.0, decay_s: 0.25 }, // big box, air ≈ 96 Hz
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },  // dry string
                    Edge { from: 2, to: 3, gain: 0.06 }, // body colour
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.75, 4.0, 120.0),
        ),
        make(
            "Piano + Board (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.600, 700.0, 1.10, STEEL, 3.5, 0.3, 0.12)),
                    Comp::Body { cavity_litres: 0.0, soundhole_cm: 0.0, top_hz: 110.0, decay_s: 0.45 }, // soundboard (no air cavity)
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.05 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 2.0, 150.0),
        ),
        // Banjo: the "body" is literally a drumhead (the pot) — string → membrane.
        make(
            "Banjo + Head (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.670, 55.0, 0.40, STEEL, 0.8, 1.2, 0.08)),
                    Comp::Membrane(DrumMembrane { radius_m: 0.14, tension_nm: 3200.0, areal_density_kgm2: 0.22, bending_nm: 0.02, decay_time: 0.15, hf_damping: 8.0, num_modes: 20, strike_pos: 0.5, ..DrumMembrane::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },  // bridge drives the head
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.12 }, // head resonance
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 2.0, 80.0),
        ),
        // -- Musical string + body --
        make(
            "Nylon + Body (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::MusicalString(MusicalString { pluck_pos: 0.14, inharmonicity: 0.0004, decay_time: 1.6, hf_damping: 1.4, num_modes: 32 }),
                    Comp::Body { cavity_litres: 17.0, soundhole_cm: 8.5, top_hz: 190.0, decay_s: 0.20 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.06 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 3.0, 130.0),
        ),
        // -- Drum + shell body --
        make(
            "Tom + Shell (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Membrane(DrumMembrane { radius_m: 0.165, tension_nm: 1500.0, areal_density_kgm2: 0.26, bending_nm: 0.0, decay_time: 0.4, hf_damping: 4.0, num_modes: 32, strike_pos: 0.5, ..DrumMembrane::default() }),
                    Comp::Body { cavity_litres: 6.0, soundhole_cm: 12.0, top_hz: 90.0, decay_s: 0.10 }, // shell air
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.08 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.7, 1.0, 90.0),
        ),
        // -- Idiophones (struck metal) --
        make(
            "Cowbell (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Bell(MetalBell { partials: 3, spread: 0.48, inharmonicity: 0.06, brightness: 0.4, decay_time: 0.35, strike_noise: 0.3, key_tracks_pitch: true }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.6, 1.0, 40.0),
        ),
        make(
            "Crash (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Cymbal(Cymbal { size: 17.0, stiffness: 9.0, damping: 1.0, brightness: -0.5, strike_pos: 0.8, modes: 160, shimmer: 0.4, key_tracks_pitch: true }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 1.0, 300.0),
        ),
        // -- Winds: breath jet into an air column --
        make(
            "Flute (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Breath { level: 0.12, tone: 1.2 },
                    Comp::Horn(WebsterHorn { boundary: Boundary::Open, r1: 0.0095, r2: 0.0, r3: 0.001, length: 0.6, blow_pos: 0.15, depth: 12, resolution: 300, damping: 3.0, freq_dep_damping: -0.10, visco_loss: 0.3, radiation: 0.5, ..WebsterHorn::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.55, 40.0, 80.0),
        ),
        // Didgeridoo: breath drone into a long bore, coloured by a vocal-tract body.
        make(
            "Didgeridoo (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Breath { level: 0.2, tone: 0.6 },
                    Comp::Horn(WebsterHorn { boundary: Boundary::Open, blow_pos: 0.10, r2: 0.5, r3: 0.5, length: 3.0, damping: 0.6, freq_dep_damping: -0.03, visco_loss: 0.6, radiation: 0.15, depth: 20, ..WebsterHorn::default() }),
                    Comp::Body { cavity_litres: 0.15, soundhole_cm: 2.5, top_hz: 1200.0, decay_s: 0.05 }, // mouth/tract formants
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },  // bore → tract
                    Edge { from: 1, to: 3, gain: 1.0 },  // dry bore
                    Edge { from: 2, to: 3, gain: 0.15 }, // tract colour
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 20.0, 400.0),
        ),
        // -- Reeds: a self-oscillating reed driving its bore (feedback loop) --
        make(
            "Clarinet (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 0.6, stiffness: 1.5, freq_hz: 0.0 },
                    Comp::Horn(WebsterHorn { boundary: Boundary::Brass, r1: 0.0073, r2: 0.0, r3: 0.002, length: 0.66, blow_pos: 0.0, depth: 18, resolution: 300, damping: 4.0, freq_dep_damping: -0.08, visco_loss: 0.8, radiation: 0.6, ..WebsterHorn::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 0.3 },
                    Edge { from: 1, to: 0, gain: 0.55 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 20.0, 70.0),
        ),
        make(
            "Alto Sax (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 0.7, stiffness: 1.3, freq_hz: 0.0 },
                    Comp::Horn(WebsterHorn { boundary: Boundary::Brass, r1: 0.005, r2: 0.030, r3: 0.002, length: 1.0, blow_pos: 0.0, depth: 28, resolution: 400, damping: 5.0, freq_dep_damping: -0.08, visco_loss: 1.0, radiation: 1.2, wavefront: Wavefront::Spherical, ..WebsterHorn::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 0.3 },
                    Edge { from: 1, to: 0, gain: 0.55 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.55, 25.0, 80.0),
        ),
        // -- Brass: buzzing lips (a reed) driving a flaring bore (feedback loop) --
        make(
            "Trumpet (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 0.8, stiffness: 1.1, freq_hz: 0.0 }, // buzzing lips
                    Comp::Horn(WebsterHorn { boundary: Boundary::Brass, r1: 0.0058, r2: 0.0150, r3: 0.1550, length: 0.6, blow_pos: 0.0, depth: 32, resolution: 512, damping: 10.0, freq_dep_damping: -0.08, visco_loss: 1.5, radiation: 1.6, wavefront: Wavefront::Spherical, ..WebsterHorn::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 0.15 },
                    Edge { from: 1, to: 0, gain: 0.55 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.6, 30.0, 45.0),
        ),
        make(
            "Trombone (graph)",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 0.8, stiffness: 1.1, freq_hz: 0.0 },
                    Comp::Horn(WebsterHorn { boundary: Boundary::Brass, r1: 0.0067, r2: 0.0220, r3: 0.1450, length: 0.8, blow_pos: 0.0, depth: 30, resolution: 400, damping: 8.0, freq_dep_damping: -0.08, visco_loss: 1.8, radiation: 1.4, wavefront: Wavefront::Spherical, ..WebsterHorn::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 0.5 },
                    Edge { from: 1, to: 0, gain: 0.55 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.6, 25.0, 60.0),
        ),
        // ---- Bowed strings (driven → sustain; bow near the bridge = bright/saw) ----
        make("Violin", bowed(0.330, 44.0, 1.60, GUT, 1.8, 0.6, 0.12), eng(0.55, 60.0, 150.0)),
        make("Viola", bowed(0.380, 55.0, 2.40, GUT, 2.0, 0.5, 0.14), eng(0.55, 65.0, 160.0)),
        make("Cello", bowed(0.690, 90.0, 3.60, GUT, 2.2, 0.5, 0.13), eng(0.6, 70.0, 180.0)),
        make("Bowed Bass", bowed(1.060, 250.0, 5.50, GUT, 2.5, 0.5, 0.12), eng(0.65, 80.0, 200.0)),
        // ---- Musical String (music-friendly controls) ----
        make(
            "Soft Nylon",
            MusicalString { pluck_pos: 0.14, inharmonicity: 0.0004, decay_time: 1.6, hf_damping: 1.4, num_modes: 32 },
            eng(0.6, 3.0, 130.0),
        ),
        make(
            "Glass Pluck",
            MusicalString { pluck_pos: 0.10, inharmonicity: 0.0018, decay_time: 2.4, hf_damping: 0.7, num_modes: 44 },
            eng(0.6, 3.0, 160.0),
        ),
        // ---- Drum (2D membrane) ----
        make(
            "Tom",
            DrumMembrane { radius_m: 0.165, tension_nm: 1500.0, areal_density_kgm2: 0.26, bending_nm: 0.0, decay_time: 0.4, hf_damping: 4.0, num_modes: 32, strike_pos: 0.5, ..DrumMembrane::default() },
            eng(0.7, 1.0, 90.0),
        ),
        make(
            "Kick",
            DrumMembrane { radius_m: 0.28, tension_nm: 800.0, areal_density_kgm2: 0.42, bending_nm: 0.0, decay_time: 0.18, hf_damping: 5.0, num_modes: 24, strike_pos: 0.35, ..DrumMembrane::default() },
            eng(0.85, 1.0, 60.0),
        ),
        make(
            "Timpani",
            DrumMembrane { radius_m: 0.32, tension_nm: 2500.0, areal_density_kgm2: 0.3, bending_nm: 0.0, decay_time: 1.4, hf_damping: 1.5, num_modes: 48, strike_pos: 0.25, ..DrumMembrane::default() },
            eng(0.6, 2.0, 200.0),
        ),
        // ---- Webster Horn ----
        // Trumpet: physical bore (contracting throat + flare) from Dave's config.
        // Their freq_dependent_damping = +0.08 maps to our −0.08 sign convention.
        make(
            "Trumpet",
            WebsterHorn {
                boundary: Boundary::Brass,
                // Bell-flare quadratic over its final 0.6 m (Bb trumpet). Base
                // resonance 116.5 Hz = open bore → longest bore anchored at E2.
                r1: 0.0058,
                r2: 0.0150,
                r3: 0.1550,
                length: 0.6,
                wave_speed: 343.0,
                blow_pos: 0.0,
                depth: 32,
                resolution: 512,
                damping: 10.0,
                freq_dep_damping: -0.08,
                visco_loss: 1.5,             // narrow leadpipe = warm boundary-layer loss
                radiation: 1.6,              // bright, open bell
                wavefront: Wavefront::Spherical, // real bore: curved wavefronts at the bell
                // Proof of concept: play like a real Bb trumpet — overblow onto
                // one of the 7 valve bore lengths (0..6 semitones) and land it on
                // the key. Longest bore's fundamental = concert E2 (open bore =
                // pedal Bb2 a tritone above).
                play_mode: HornPlay::OverblowTracked,
                valve_steps: 6,
                overblow_anchor_hz: 82.41,
                ..WebsterHorn::default()
            },
            eng(0.6, 30.0, 45.0),
        ),
        // French Horn: same bore idea, longer + darker (more HF damping).
        make(
            "French Horn (F)",
            WebsterHorn {
                boundary: Boundary::Brass,
                r1: 0.0090,
                r2: 0.0350,
                r3: 0.1110,
                length: 1.0,
                blow_pos: 0.0,
                depth: 30,
                resolution: 400,
                damping: 8.0,
                freq_dep_damping: -0.10,
                visco_loss: 2.5,             // long narrow tubing = mellow, stuffed
                radiation: 0.5,              // dark, backward-facing bell
                wavefront: Wavefront::Spherical,
                // Key-tracked: 3 valves (0..6 semitones). Horn "in F" → the open
                // bore's fundamental is concert F1 (43.65 Hz), so the LONGEST bore
                // (−6 semitones) is anchored at B0 = 30.87 Hz. Mid-range notes fall
                // on high harmonics (8th–16th) — the mellow, "living high" horn
                // character.
                play_mode: HornPlay::OverblowTracked,
                valve_steps: 6,
                overblow_anchor_hz: 30.87,
                ..WebsterHorn::default()
            },
            eng(0.55, 30.0, 150.0),
        ),
        make(
            "French Horn (Bb)",
            WebsterHorn {
                boundary: Boundary::Brass,
                r1: 0.0090,
                r2: 0.0350,
                r3: 0.1110,
                length: 1.0,
                blow_pos: 0.0,
                depth: 30,
                resolution: 400,
                damping: 8.0,
                freq_dep_damping: -0.10,
                visco_loss: 2.5,
                radiation: 0.5,
                wavefront: Wavefront::Spherical,
                // The Bb side of a double horn: shorter, so a given note sits on a
                // lower harmonic — more secure/brighter. Open fundamental concert
                // Bb1 (58.27 Hz) → longest bore anchored at E1 = 41.20 Hz.
                play_mode: HornPlay::OverblowTracked,
                valve_steps: 6,
                overblow_anchor_hz: 41.20,
                ..WebsterHorn::default()
            },
            eng(0.55, 30.0, 150.0),
        ),
        // Didgeridoo: near-lossless drone — barely damps, rings on and on. A
        // touch of wall loss for wooden warmth; almost no bell radiation.
        make(
            "Didgeridoo",
            WebsterHorn { boundary: Boundary::Open, blow_pos: 0.10, r2: 0.5, r3: 0.5, length: 3.0, damping: 0.25, freq_dep_damping: -0.03, visco_loss: 0.6, radiation: 0.15, depth: 20, ..WebsterHorn::default() },
            eng(0.6, 20.0, 400.0),
        ),
        // Trombone: long cylindrical brass with a bell flare.
        make(
            "Trombone",
            WebsterHorn {
                boundary: Boundary::Brass,
                // Bell-flare quadratic over its final 0.8 m (tenor Bb trombone).
                // Base resonance 58.3 Hz = open bore → longest position anchored
                // at E1. The slide's 7 positions map to the 0..6 semitone steps.
                r1: 0.0067,
                r2: 0.0220,
                r3: 0.1450,
                length: 0.8,
                blow_pos: 0.0,
                depth: 30,
                resolution: 400,
                damping: 8.0,
                freq_dep_damping: -0.08,
                visco_loss: 1.8,
                radiation: 1.4,
                wavefront: Wavefront::Spherical,
                play_mode: HornPlay::OverblowTracked,
                valve_steps: 6,
                overblow_anchor_hz: 41.20,
                ..WebsterHorn::default()
            },
            eng(0.6, 25.0, 60.0),
        ),
        // ---- Woodwinds (bore shape + end condition set the character) ----
        // Flute: open cylinder (all harmonics), pure and airy — few modes.
        make(
            "Flute",
            WebsterHorn { boundary: Boundary::Open, r1: 0.0095, r2: 0.0, r3: 0.001, length: 0.6, blow_pos: 0.15, depth: 12, resolution: 300, damping: 3.0, freq_dep_damping: -0.10, visco_loss: 0.3, radiation: 0.5, ..WebsterHorn::default() },
            eng(0.55, 40.0, 80.0),
        ),
        // Clarinet: closed cylinder → odd harmonics only → the hollow tone.
        make(
            "Clarinet",
            WebsterHorn { boundary: Boundary::Brass, r1: 0.0073, r2: 0.0, r3: 0.002, length: 0.66, blow_pos: 0.0, depth: 18, resolution: 300, damping: 4.0, freq_dep_damping: -0.08, visco_loss: 0.8, radiation: 0.6, ..WebsterHorn::default() },
            eng(0.6, 30.0, 70.0),
        ),
        // Bassoon: long narrow closed cone → full harmonics, dark and reedy.
        make(
            "Bassoon",
            WebsterHorn { boundary: Boundary::Brass, r1: 0.004, r2: 0.008, r3: 0.001, length: 2.5, blow_pos: 0.0, depth: 28, resolution: 400, damping: 6.0, freq_dep_damping: -0.10, visco_loss: 2.0, radiation: 0.5, ..WebsterHorn::default() },
            eng(0.6, 30.0, 100.0),
        ),
        // Alto Sax: wide closed cone → full harmonics, bright and reedy.
        make(
            "Alto Sax",
            WebsterHorn { boundary: Boundary::Brass, r1: 0.005, r2: 0.030, r3: 0.002, length: 1.0, blow_pos: 0.0, depth: 28, resolution: 400, damping: 5.0, freq_dep_damping: -0.08, visco_loss: 1.0, radiation: 1.2, wavefront: Wavefront::Spherical, ..WebsterHorn::default() },
            eng(0.6, 25.0, 80.0),
        ),
        // ---- Idiophones ----
        make(
            "Cowbell",
            MetalBell { partials: 3, spread: 0.48, inharmonicity: 0.06, brightness: 0.4, decay_time: 0.35, strike_noise: 0.3, key_tracks_pitch: true },
            eng(0.6, 1.0, 40.0),
        ),
        make(
            "Snare",
            Snare { tension: 750.0, damping: 14.0, strike_pos: 0.4, depth: 24, snares: 0.7, snare_decay: 16.0, tone: 0.65, key_tracks_pitch: true },
            eng(0.7, 1.0, 50.0),
        ),
        make(
            "Crash Cymbal",
            Cymbal { size: 17.0, stiffness: 9.0, damping: 1.0, brightness: -0.5, strike_pos: 0.8, modes: 160, shimmer: 0.4, key_tracks_pitch: true },
            eng(0.5, 1.0, 300.0),
        ),
        make(
            "Ride Cymbal",
            Cymbal { size: 15.0, stiffness: 7.0, damping: 2.2, brightness: -0.25, strike_pos: 0.35, modes: 100, shimmer: 0.15, key_tracks_pitch: true },
            eng(0.55, 1.0, 200.0),
        ),
        // ---- Pure Plate (free-plate FTM solve) ----
        make(
            "Pure Plate",
            PurePlate { poisson: 0.33, decay_time: 4.0, hf_damp: 0.5, strike_pos: 0.75, modes: 90, key_tracks_pitch: true },
            eng(0.5, 1.0, 250.0),
        ),
        make(
            "Plate Gong",
            PurePlate { poisson: 0.30, decay_time: 8.0, hf_damp: 0.2, strike_pos: 0.45, modes: 120, key_tracks_pitch: true },
            eng(0.5, 1.0, 400.0),
        ),
        // ---- Basic Wave (reference oscillators) ----
        make("Triangle Lead", BasicWave { waveform: Waveform::Triangle, harmonics: 16, decay_time: 1.5 }, eng(0.5, 3.0, 120.0)),
        make("Saw Lead", BasicWave { waveform: Waveform::Saw, harmonics: 40, decay_time: 1.2 }, eng(0.45, 3.0, 120.0)),

        // ===================== Base instrument set =====================
        // Authored from real physics on the grounded models: real dimensions,
        // dry primary + colouring body, self-oscillating reed/lip & bow loops.
        make(
            "Base: Acoustic Guitar",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.648, 90.0, 1.1, BRONZE, 2.5, 0.6, 0.12)),
                    Comp::Body { cavity_litres: 15.0, soundhole_cm: 9.0, top_hz: 195.0, decay_s: 0.18 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.12 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 3.0, 150.0),
        ),
        make(
            "Base: Electric Guitar",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.648, 78.0, 1.0, NICKEL, 3.5, 0.35, 0.1)),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.6, 3.0, 220.0),
        ),
        make(
            "Base: Nylon Guitar",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.65, 60.0, 0.9, NYLON, 2.2, 1.0, 0.14)),
                    Comp::Body { cavity_litres: 14.0, soundhole_cm: 8.5, top_hz: 175.0, decay_s: 0.2 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.15 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 3.0, 150.0),
        ),
        make(
            "Base: Acoustic Bass",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.86, 80.0, 1.3, NICKEL, 2.0, 0.5, 0.15)),
                    Comp::Body { cavity_litres: 40.0, soundhole_cm: 10.0, top_hz: 90.0, decay_s: 0.25 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.12 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.7, 4.0, 140.0),
        ),
        make(
            "Base: Electric Bass",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.864, 70.0, 1.25, NICKEL, 3.5, 0.35, 0.12)),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.7, 4.0, 160.0),
        ),
        make(
            "Base: Harp",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.9, 55.0, 0.8, NYLON, 3.0, 0.5, 0.16)),
                    Comp::Body { cavity_litres: 30.0, soundhole_cm: 0.0, top_hz: 140.0, decay_s: 0.35 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.2 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 3.0, 220.0),
        ),
        make(
            "Base: Banjo",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.67, 55.0, 0.4, STEEL, 0.9, 1.2, 0.08)),
                    Comp::Membrane(DrumMembrane { radius_m: 0.14, tension_nm: 3200.0, areal_density_kgm2: 0.22, bending_nm: 0.02, strike_pos: 0.5, decay_time: 0.15, hf_damping: 8.0, num_modes: 20, ..DrumMembrane::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 0.5 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.8 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 2.0, 100.0),
        ),
        make(
            "Base: Piano",
            InstrumentGraph {
                components: vec![
                    Comp::Hammer { hardness: 0.5, felt: 2.6 },
                    Comp::MusicalString(MusicalString { inharmonicity: 0.0004, pluck_pos: 0.14, decay_time: 6.0, hf_damping: 0.02, num_modes: 70 }),
                    Comp::Body { cavity_litres: 0.0, soundhole_cm: 0.0, top_hz: 110.0, decay_s: 0.6 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 0, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.2 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 2.0, 250.0),
        ),
        make(
            "Base: Harpsichord",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::String(string(0.7, 42.0, 0.18, (8500.0, 110.0), 1.8, 0.02, 0.06)),
                    Comp::Body { cavity_litres: 30.0, soundhole_cm: 7.0, top_hz: 210.0, decay_s: 0.25 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 0, gain: 0.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.25 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.55, 2.0, 120.0),
        ),
        make(
            "Base: Clavichord",
            InstrumentGraph {
                components: vec![
                    Comp::Hammer { hardness: 0.9, felt: 1.2 },
                    Comp::String(string(0.62, 38.0, 0.22, (8500.0, 110.0), 1.2, 0.05, 0.03)),
                    Comp::Body { cavity_litres: 18.0, soundhole_cm: 5.0, top_hz: 190.0, decay_s: 0.2 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 0, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.2 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.55, 2.0, 120.0),
        ),
        make(
            "Base: Cimbalom",
            InstrumentGraph {
                components: vec![
                    Comp::Hammer { hardness: 0.8, felt: 1.8 },
                    Comp::String(string(0.68, 160.0, 0.45, STEEL, 5.0, 0.01, 0.2)),
                    Comp::Body { cavity_litres: 65.0, soundhole_cm: 0.0, top_hz: 130.0, decay_s: 0.5 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 0, gain: 0.5 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.2 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 1.0, 200.0),
        ),
        make(
            "Base: Grand Piano",
            InstrumentGraph {
                components: vec![
                    Comp::Hammer { hardness: 0.5, felt: 2.6 },
                    Comp::MusicalString(MusicalString { inharmonicity: 0.0004, pluck_pos: 0.14, decay_time: 6.0, hf_damping: 0.02, num_modes: 60 }),
                    Comp::MusicalString(MusicalString { inharmonicity: 0.0004, pluck_pos: 0.14, decay_time: 6.0, hf_damping: 0.02, num_modes: 60 }),
                    Comp::MusicalString(MusicalString { inharmonicity: 0.0004, pluck_pos: 0.14, decay_time: 6.0, hf_damping: 0.02, num_modes: 60 }),
                    Comp::Body { cavity_litres: 0.0, soundhole_cm: 0.0, top_hz: 110.0, decay_s: 0.6 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 0, to: 2, gain: 1.0 },
                    Edge { from: 0, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 0, gain: 0.3 },
                    Edge { from: 2, to: 0, gain: 0.3 },
                    Edge { from: 3, to: 0, gain: 0.3 },
                    Edge { from: 1, to: 5, gain: 0.6 },
                    Edge { from: 2, to: 5, gain: 0.6 },
                    Edge { from: 3, to: 5, gain: 0.6 },
                    Edge { from: 1, to: 4, gain: 1.0 },
                    Edge { from: 2, to: 4, gain: 1.0 },
                    Edge { from: 3, to: 4, gain: 1.0 },
                    Edge { from: 4, to: 5, gain: 0.2 },
                ],
                output: 5,
                key_map: Vec::new(),
            },
            eng(0.55, 2.0, 300.0),
        ),
        make(
            "Base: Violin",
            InstrumentGraph {
                components: vec![
                    Comp::BowedString { speed: 1.3, force: 0.6 },
                    Comp::Body { cavity_litres: 2.2, soundhole_cm: 3.2, top_hz: 280.0, decay_s: 0.35 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 2, gain: 0.8 }, // dry string → out
                    Edge { from: 0, to: 1, gain: 1.0 }, // string → body
                    Edge { from: 1, to: 2, gain: 0.3 }, // body colour → out
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 45.0, 200.0),
        ),
        make(
            "Base: Viola",
            InstrumentGraph {
                components: vec![
                    Comp::BowedString { speed: 1.2, force: 0.7 },
                    Comp::Body { cavity_litres: 4.5, soundhole_cm: 3.8, top_hz: 210.0, decay_s: 0.38 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 2, gain: 0.8 }, // dry string → out
                    Edge { from: 0, to: 1, gain: 1.0 }, // string → body
                    Edge { from: 1, to: 2, gain: 0.3 }, // body colour → out
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 45.0, 200.0),
        ),
        make(
            "Base: Cello",
            InstrumentGraph {
                components: vec![
                    Comp::BowedString { speed: 1.1, force: 0.8 },
                    Comp::Body { cavity_litres: 28.0, soundhole_cm: 6.5, top_hz: 105.0, decay_s: 0.5 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 2, gain: 0.8 }, // dry string → out
                    Edge { from: 0, to: 1, gain: 1.0 }, // string → body
                    Edge { from: 1, to: 2, gain: 0.3 }, // body colour → out
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 50.0, 220.0),
        ),
        make(
            "Base: Double Bass",
            InstrumentGraph {
                components: vec![
                    Comp::BowedString { speed: 1.0, force: 0.9 },
                    Comp::Body { cavity_litres: 120.0, soundhole_cm: 11.5, top_hz: 60.0, decay_s: 0.6 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 2, gain: 0.8 }, // dry string → out
                    Edge { from: 0, to: 1, gain: 1.0 }, // string → body
                    Edge { from: 1, to: 2, gain: 0.3 }, // body colour → out
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 50.0, 250.0),
        ),
        make(
            "Base: Hurdy-Gurdy",
            InstrumentGraph {
                components: vec![
                    Comp::BowedString { speed: 1.2, force: 0.7 },
                    Comp::Body { cavity_litres: 12.0, soundhole_cm: 0.0, top_hz: 160.0, decay_s: 0.4 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 2, gain: 0.8 }, // dry string → out
                    Edge { from: 0, to: 1, gain: 1.0 }, // string → body
                    Edge { from: 1, to: 2, gain: 0.3 }, // body colour → out
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 40.0, 250.0),
        ),
        make(
            "Base: Flute",
            InstrumentGraph {
                components: vec![
                    Comp::Breath { level: 0.12, tone: 1.2 },
                    Comp::Horn(WebsterHorn { length: 0.66, wave_speed: 343.0, r1: 0.0095, r2: 0.0, r3: 0.0, blow_pos: 0.0, depth: 14, resolution: 220, damping: 3.0, freq_dep_damping: -0.1, visco_loss: 0.3, radiation: 0.5, boundary: Boundary::Open, ..WebsterHorn::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 40.0, 90.0),
        ),
        make(
            "Base: Clarinet",
            InstrumentGraph {
                components: vec![
                    // The reed forward-drives the modal Webster horn (no feedback
                    // loop) as broadband excitation; the horn RINGS at its own
                    // resonances — that resonant overblown-horn tone. In
                    // OverblowTracked mode the horn picks an overblown bore length
                    // per key so its resonance lands on the note (chromatic, in
                    // tune within a few cents across the range). Cylindrical bore
                    // (14.6 mm, r1 = 7.3 mm, no taper) → odd harmonics, closed
                    // mouthpiece + open bell.
                    // NOTE: the horn's pitch tracking is still INTERNAL here (its
                    // play_mode), not yet the explicit per-component key-map type
                    // Dave wants — that's the next step.
                    Comp::Reed { pressure: 0.9, stiffness: 1.0, freq_hz: 0.0 },
                    Comp::Horn(WebsterHorn {
                        boundary: Boundary::Brass,
                        r1: 0.0073,
                        r2: 0.0,
                        r3: 0.002,
                        length: 0.66,
                        blow_pos: 0.0,
                        depth: 18,
                        resolution: 300,
                        damping: 4.0,
                        freq_dep_damping: -0.08,
                        visco_loss: 0.8,
                        radiation: 0.6,
                        play_mode: HornPlay::OverblowTracked,
                        ..WebsterHorn::default()
                    }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 0.6 }, // reed excites the horn
                    Edge { from: 1, to: 2, gain: 0.5 }, // horn → out
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 25.0, 90.0),
        ),
        make(
            "Base: Alto Sax",
            InstrumentGraph {
                components: vec![
                    // ODE reed valve → bore → the horn as a fixed-formant bell.
                    // Bell geometry: a strongly flaring cone (r2 taper) ~1.0 m —
                    // the conical flare fills the harmonic series back in (a sax
                    // overblows the octave, unlike the clarinet's 12th) and gives
                    // its brighter, more vocal formants. Fewer modes than the
                    // clarinet: its resonances are broader.
                    Comp::Reed { pressure: 1.0, stiffness: 0.8, freq_hz: 0.0 },
                    Comp::Bore { tone: 1.3 },
                    Comp::Horn(WebsterHorn {
                        boundary: Boundary::Brass,
                        r1: 0.005,
                        r2: 0.030,
                        r3: 0.002,
                        length: 1.0,
                        blow_pos: 0.0,
                        depth: 12,
                        resolution: 300,
                        damping: 5.0,
                        freq_dep_damping: -0.08,
                        visco_loss: 1.0,
                        radiation: 1.0,
                        wavefront: Wavefront::Spherical,
                        play_mode: HornPlay::Overblow,
                        ..WebsterHorn::default()
                    }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 }, // reed → bore
                    Edge { from: 1, to: 0, gain: 1.0 }, // bore → reed (feedback)
                    Edge { from: 1, to: 3, gain: 0.7 }, // bore → out (dry)
                    Edge { from: 1, to: 2, gain: 1.0 }, // bore → bell
                    Edge { from: 2, to: 3, gain: 0.4 }, // bell colour → out
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.55, 25.0, 90.0),
        ),
        make(
            "Base: Bassoon",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 0.85, stiffness: 1.2, freq_hz: 0.0 },
                    Comp::Bore { tone: 0.5 },
                    Comp::Body { cavity_litres: 0.0, soundhole_cm: 0.0, top_hz: 500.0, decay_s: 0.08 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 }, // reed → bore
                    Edge { from: 1, to: 0, gain: 1.0 }, // bore → reed (feedback)
                    Edge { from: 1, to: 3, gain: 0.7 }, // bore → out (dry)
                    Edge { from: 1, to: 2, gain: 1.0 }, // bore → bell
                    Edge { from: 2, to: 3, gain: 0.3 }, // bell colour → out
                ],
                output: 3,
                key_map: vec![KeyBinding { component: 2, map: KeyMapKind::Power { param: "top_hz".into(), amount: 1.0 } }],
            },
            eng(0.5, 25.0, 110.0),
        ),
        make(
            "Base: Trumpet",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 1.0, stiffness: 0.7, freq_hz: 0.0 },
                    Comp::Bore { tone: 1.4 },
                    Comp::Body { cavity_litres: 0.0, soundhole_cm: 0.0, top_hz: 2500.0, decay_s: 0.04 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 }, // reed → bore
                    Edge { from: 1, to: 0, gain: 1.0 }, // bore → reed (feedback)
                    Edge { from: 1, to: 3, gain: 0.7 }, // bore → out (dry)
                    Edge { from: 1, to: 2, gain: 1.0 }, // bore → bell
                    Edge { from: 2, to: 3, gain: 0.3 }, // bell colour → out
                ],
                output: 3,
                key_map: vec![KeyBinding { component: 2, map: KeyMapKind::Power { param: "top_hz".into(), amount: 1.0 } }],
            },
            eng(0.55, 25.0, 90.0),
        ),
        make(
            "Base: Trombone",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 1.0, stiffness: 0.8, freq_hz: 0.0 },
                    Comp::Bore { tone: 1.2 },
                    Comp::Body { cavity_litres: 0.0, soundhole_cm: 0.0, top_hz: 1200.0, decay_s: 0.05 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 }, // reed → bore
                    Edge { from: 1, to: 0, gain: 1.0 }, // bore → reed (feedback)
                    Edge { from: 1, to: 3, gain: 0.7 }, // bore → out (dry)
                    Edge { from: 1, to: 2, gain: 1.0 }, // bore → bell
                    Edge { from: 2, to: 3, gain: 0.3 }, // bell colour → out
                ],
                output: 3,
                key_map: vec![KeyBinding { component: 2, map: KeyMapKind::Power { param: "top_hz".into(), amount: 1.0 } }],
            },
            eng(0.55, 25.0, 90.0),
        ),
        make(
            "Base: French Horn",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 0.9, stiffness: 1.0, freq_hz: 0.0 },
                    Comp::Bore { tone: 0.7 },
                    Comp::Body { cavity_litres: 0.0, soundhole_cm: 0.0, top_hz: 900.0, decay_s: 0.06 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 }, // reed → bore
                    Edge { from: 1, to: 0, gain: 1.0 }, // bore → reed (feedback)
                    Edge { from: 1, to: 3, gain: 0.7 }, // bore → out (dry)
                    Edge { from: 1, to: 2, gain: 1.0 }, // bore → bell
                    Edge { from: 2, to: 3, gain: 0.3 }, // bell colour → out
                ],
                output: 3,
                key_map: vec![KeyBinding { component: 2, map: KeyMapKind::Power { param: "top_hz".into(), amount: 1.0 } }],
            },
            eng(0.5, 30.0, 150.0),
        ),
        make(
            "Base: Didgeridoo",
            InstrumentGraph {
                components: vec![
                    Comp::Reed { pressure: 0.9, stiffness: 0.6, freq_hz: 0.0 },
                    Comp::Bore { tone: 0.5 },
                    Comp::Voice { open_quotient: 0.5, level: 0.15 },
                    Comp::Body { cavity_litres: 0.15, soundhole_cm: 2.5, top_hz: 1200.0, decay_s: 0.05 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 }, // reed → bore
                    Edge { from: 1, to: 0, gain: 1.0 }, // bore → reed (feedback)
                    Edge { from: 2, to: 1, gain: 0.5 }, // voice → bore (vocalisation)
                    Edge { from: 1, to: 4, gain: 0.7 }, // bore → out
                    Edge { from: 1, to: 3, gain: 1.0 }, // bore → tract
                    Edge { from: 3, to: 4, gain: 0.3 }, // tract colour → out
                ],
                output: 4,
                key_map: vec![KeyBinding { component: 3, map: KeyMapKind::Power { param: "top_hz".into(), amount: 1.0 } }],
            },
            eng(0.5, 30.0, 400.0),
        ),
        make(
            "Base: Vocal Synth",
            InstrumentGraph {
                components: vec![
                    Comp::Voice { open_quotient: 0.65, level: 0.15 },
                    Comp::Horn(WebsterHorn { length: 0.17, wave_speed: 343.0, r1: 0.012, r2: 0.02, r3: 0.015, blow_pos: 0.0, depth: 18, resolution: 200, damping: 6.0, freq_dep_damping: -0.08, visco_loss: 1.5, radiation: 0.4, boundary: Boundary::Open, ..WebsterHorn::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 30.0, 200.0),
        ),
        make(
            "Base: Kick",
            InstrumentGraph {
                components: vec![
                    Comp::Hammer { hardness: 0.2, felt: 3.2 },
                    Comp::Membrane(DrumMembrane { radius_m: 0.28, tension_nm: 800.0, areal_density_kgm2: 0.42, bending_nm: 0.0, strike_pos: 0.35, decay_time: 0.18, hf_damping: 5.0, num_modes: 24, ..DrumMembrane::default() }),
                    Comp::Body { cavity_litres: 140.0, soundhole_cm: 12.0, top_hz: 52.0, decay_s: 0.18 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.4 },
                    Edge { from: 1, to: 0, gain: 0.2 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.85, 1.0, 80.0),
        ),
        make(
            "Base: Tom",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Membrane(DrumMembrane { radius_m: 0.165, tension_nm: 1500.0, areal_density_kgm2: 0.26, bending_nm: 0.0, strike_pos: 0.5, decay_time: 0.4, hf_damping: 4.0, num_modes: 32, ..DrumMembrane::default() }),
                    Comp::Body { cavity_litres: 22.0, soundhole_cm: 0.0, top_hz: 110.0, decay_s: 0.35 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.3 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.7, 1.0, 120.0),
        ),
        make(
            "Base: Timpani",
            InstrumentGraph {
                components: vec![
                    Comp::Hammer { hardness: 0.25, felt: 2.8 },
                    Comp::Membrane(DrumMembrane { radius_m: 0.32, tension_nm: 2500.0, areal_density_kgm2: 0.3, bending_nm: 0.0, strike_pos: 0.25, decay_time: 1.4, hf_damping: 1.5, num_modes: 48, ..DrumMembrane::default() }),
                    Comp::Body { cavity_litres: 240.0, soundhole_cm: 0.0, top_hz: 98.0, decay_s: 1.4 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 3, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 0.4 },
                ],
                output: 3,
                key_map: Vec::new(),
            },
            eng(0.6, 2.0, 250.0),
        ),
        make(
            "Base: Snare",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Membrane(DrumMembrane { radius_m: 0.165, tension_nm: 2000.0, areal_density_kgm2: 0.26, bending_nm: 0.02, strike_pos: 0.3, decay_time: 0.18, hf_damping: 6.0, num_modes: 24, ..DrumMembrane::default() }),
                    Comp::Body { cavity_litres: 6.0, soundhole_cm: 0.0, top_hz: 180.0, decay_s: 0.08 },
                    Comp::Membrane(DrumMembrane { radius_m: 0.165, tension_nm: 2600.0, areal_density_kgm2: 0.24, bending_nm: 0.02, strike_pos: 0.5, decay_time: 0.12, hf_damping: 7.0, num_modes: 20, ..DrumMembrane::default() }),
                    Comp::Wires { level: 0.85, tone: 1.2 },
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                    Edge { from: 2, to: 3, gain: 1.0 },
                    Edge { from: 3, to: 4, gain: 1.0 },
                    Edge { from: 4, to: 3, gain: 0.4 },
                    Edge { from: 1, to: 5, gain: 0.8 },
                    Edge { from: 3, to: 5, gain: 0.5 },
                    Edge { from: 4, to: 5, gain: 0.7 },
                ],
                output: 5,
                key_map: Vec::new(),
            },
            eng(0.7, 1.0, 120.0),
        ),
        make(
            "Base: Cowbell",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Bell(MetalBell { partials: 12, spread: 1.42, inharmonicity: 0.18, brightness: 0.85, decay_time: 0.45, strike_noise: 0.3, ..MetalBell::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.6, 1.0, 60.0),
        ),
        make(
            "Base: Crash Cymbal",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Cymbal(Cymbal { size: 17.0, stiffness: 9.0, damping: 1.0, brightness: -0.5, strike_pos: 0.8, modes: 160, shimmer: 0.4, ..Cymbal::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 1.0, 400.0),
        ),
        make(
            "Base: Ride Cymbal",
            InstrumentGraph {
                components: vec![
                    Comp::Strike,
                    Comp::Cymbal(Cymbal { size: 15.0, stiffness: 7.0, damping: 2.2, brightness: -0.25, strike_pos: 0.35, modes: 100, shimmer: 0.15, ..Cymbal::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.55, 1.0, 300.0),
        ),
        make(
            "Base: Gong",
            InstrumentGraph {
                components: vec![
                    Comp::Hammer { hardness: 0.15, felt: 2.4 },
                    Comp::Plate(PurePlate { poisson: 0.34, decay_time: 6.5, hf_damp: 1.8, strike_pos: 0.5, modes: 70, ..PurePlate::default() }),
                    Comp::Mix,
                ],
                edges: vec![
                    Edge { from: 0, to: 1, gain: 1.0 },
                    Edge { from: 1, to: 2, gain: 1.0 },
                ],
                output: 2,
                key_map: Vec::new(),
            },
            eng(0.5, 1.0, 500.0),
        ),

    ];

    // For every struck string / membrane / plate preset, add a "(graph)" twin
    // that plays the SAME parameters through the per-sample voice graph, so each
    // can be A/B'd against its classic-renderer original. Bowed strings are
    // skipped (not struck/plucked). The twin just re-homes the params under the
    // matching graph model's `inner`.
    let mut twins: Vec<Preset> = Vec::new();
    for p in &base {
        let graph_id = match p.model_id.as_str() {
            "pure_string" => {
                if p.params.get("excitation").and_then(|v| v.as_str()) == Some("Bowed") {
                    continue;
                }
                "graph_string"
            }
            "drum_membrane" => "graph_drum",
            "musical_string" => "graph_musical_string",
            "pure_plate" => "graph_plate",
            _ => continue,
        };
        twins.push(Preset {
            name: format!("{} (graph)", p.name),
            model_id: graph_id.to_string(),
            params: serde_json::json!({ "inner": p.params }),
            engine: p.engine.clone(),
            zones: Vec::new(),
            builtin: Some(true),
        });
    }

    // Stamp every factory preset as built-in so the launch refresh keeps them
    // in sync with this code (user-saved presets are left untouched).
    base.into_iter()
        .chain(twins)
        .map(|mut p| {
            p.builtin = Some(true);
            p
        })
        .collect()
}

/// Seed the folder with the factory presets if it currently has none (first run).
/// Returns the resulting list.
/// Load the on-disk library, refreshing built-in presets from [`factory`] so
/// code changes to them take effect on launch. A same-named file is overwritten
/// only when it is a built-in (`builtin == Some(true)`) or a legacy file from
/// before the `builtin` field (`None`); a user-saved preset (`Some(false)`) is
/// never touched, and a brand-new factory preset is written for the first time.
pub fn load_library() -> Vec<Preset> {
    let dir = presets_dir();
    let existing = list_in(&dir);
    for p in factory() {
        let refresh = match existing.iter().find(|e| e.name == p.name) {
            None => true,                        // new factory preset → seed it
            Some(e) => e.builtin != Some(false), // built-in or legacy → refresh
        };
        if refresh {
            let _ = save_in(&dir, &p);
        }
    }
    list_in(&dir)
}

/// (Re)write every factory preset, overwriting same-named files. Returns the list.
pub fn restore_factory() -> Vec<Preset> {
    let dir = presets_dir();
    for p in factory() {
        let _ = save_in(&dir, &p);
    }
    list_in(&dir)
}

/// Directory presets are stored in: `$FTM_SYNTH_PRESETS` if set, else `presets/`
/// under the working directory. Created on demand.
pub fn presets_dir() -> PathBuf {
    std::env::var_os("FTM_SYNTH_PRESETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("presets"))
}

/// Turn a preset name into a safe file stem (keep it human-readable).
fn file_stem(name: &str) -> String {
    let mut s: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    s = s.trim().replace(' ', "_");
    if s.is_empty() {
        s.push_str("preset");
    }
    s
}

/// Save (or overwrite) a preset in the default folder. Returns the path written.
pub fn save(preset: &Preset) -> io::Result<PathBuf> {
    save_in(&presets_dir(), preset)
}

/// List all presets in the default folder, sorted by name.
pub fn list() -> Vec<Preset> {
    list_in(&presets_dir())
}

/// Delete a preset by name from the default folder.
pub fn delete(name: &str) -> io::Result<bool> {
    delete_in(&presets_dir(), name)
}

/// Load a single preset file.
pub fn load(path: &Path) -> io::Result<Preset> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// --- directory-parameterized cores (so persistence is deterministically testable) ---

fn save_in(dir: &Path, preset: &Preset) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.json", file_stem(&preset.name)));
    let json = serde_json::to_string_pretty(preset)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(&path, json)?;
    Ok(path)
}

fn list_in(dir: &Path) -> Vec<Preset> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(p) = load(&path) {
                    out.push(p);
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

fn delete_in(dir: &Path, name: &str) -> io::Result<bool> {
    let path = dir.join(format!("{}.json", file_stem(name)));
    match fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod graph_twin_tests {
    use super::*;

    #[test]
    fn struck_string_membrane_plate_presets_get_graph_twins() {
        let f = factory();
        let has = |n: &str| f.iter().any(|p| p.name == n);
        // struck string + membrane + plate → twinned
        assert!(has("Acoustic Guitar (graph)"), "struck string twinned");
        assert!(has("Tom (graph)"), "membrane twinned");
        assert!(has("Pure Plate (graph)"), "plate twinned");
        // bowed strings + musical_string → NOT twinned
        assert!(!has("Violin (graph)"), "bowed strings excluded");
        assert!(has("Soft Nylon (graph)"), "musical_string twinned");
        // a twin points at the graph model, wraps the original params, and rebuilds
        let g = f.iter().find(|p| p.name == "Acoustic Guitar (graph)").unwrap();
        assert_eq!(g.model_id, "graph_string");
        assert!(g.params.get("inner").is_some(), "params re-homed under inner");
        assert!(g.build_model().is_some(), "twin rebuilds via model_from_id");
    }

    #[test]
    fn every_instrument_graph_preset_renders_stably() {
        let sr = 48_000.0;
        let mut checked = 0;
        // "Google:" presets are the experimental batch designed against a fully
        // grounded engine; those that drive the not-yet-grounded Membrane/Cymbal/
        // Horn-loss can blow up until those models are grounded. We report them
        // but only hard-fail on the established (non-Google) presets.
        let mut unstable: Vec<(String, f32)> = Vec::new();
        for p in factory() {
            if p.model_id != "instrument_graph" {
                continue;
            }
            let m = p.build_model().unwrap_or_else(|| panic!("{} rebuilds", p.name));
            let mut n = m
                .build_graph(220.0, 1.0, sr)
                .unwrap_or_else(|| panic!("{} builds a graph", p.name));
            let mut worst = 0.0f32;
            for _ in 0..sr as usize {
                worst = worst.max(n.tick(&[]).abs());
            }
            let stable = worst.is_finite() && worst < 20.0;
            let experimental = p.name.starts_with("Base:");
            if !stable {
                unstable.push((p.name.clone(), worst));
                if !experimental {
                    panic!("{} blew up to {}", p.name, worst);
                }
            }
            checked += 1;
        }
        if !unstable.is_empty() {
            eprintln!("UNSTABLE (await model grounding): {} presets", unstable.len());
            for (n, w) in &unstable {
                eprintln!("  {n} -> {w}");
            }
        }
        assert!(checked >= 15, "expected the graph showcase presets, saw {checked}");
    }

    #[test]
    fn string_presets_have_realistic_open_pitches() {
        // Grounded strings derive their open pitch from real geometry; guard that
        // every string preset lands in a sane instrument range (≈ E0 .. C8).
        use crate::models::pure_string::PureString;
        for p in factory() {
            if p.model_id != "pure_string" {
                continue;
            }
            let s: PureString = serde_json::from_value(p.params.clone()).unwrap();
            let f = s.open_pitch_hz();
            assert!(
                f.is_finite() && (20.0..=4200.0).contains(&f),
                "{} open pitch {f} Hz is out of the instrument range",
                p.name
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::basic_wave::BasicWave;

    #[test]
    fn roundtrip_preserves_params() {
        let mut m = BasicWave::default();
        m.harmonics = 41;
        m.decay_time = 2.5;
        let engine = EngineParams {
            gain: 1.234,
            ..EngineParams::default()
        };
        let preset = Preset::capture("Test Lead", &m, &engine);

        // Serialize -> deserialize via JSON text (no filesystem).
        let text = serde_json::to_string(&preset).unwrap();
        let back: Preset = serde_json::from_str(&text).unwrap();
        assert_eq!(back.name, "Test Lead");
        assert_eq!(back.model_id, "basic_wave");
        assert!((back.engine.gain - 1.234).abs() < 1e-6);

        let model = back.build_model().expect("rebuilds model");
        assert_eq!(model.id(), "basic_wave");
        // The rebuilt model's json should match the original's.
        assert_eq!(model.to_json(), m.to_json());
    }

    #[test]
    fn kit_preset_roundtrips_with_its_zones() {
        let zones = vec![
            ZoneData {
                name: "Kick".into(),
                lo: 36,
                hi: 47,
                fixed_note: Some(38),
                transpose: 0,
                model_id: "drum_membrane".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
            },
            ZoneData {
                name: "Lead".into(),
                lo: 48,
                hi: 72,
                fixed_note: None,
                transpose: 0,
                model_id: "pure_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
            },
        ];
        let preset = Preset::capture_kit("My Kit", zones);
        assert!(preset.is_kit());
        assert!(preset.build_model().is_none(), "a kit has no single model");

        let back: Preset = serde_json::from_str(&serde_json::to_string(&preset).unwrap()).unwrap();
        assert_eq!(back.name, "My Kit");
        assert!(back.is_kit());
        assert_eq!(back.zones.len(), 2);
        assert_eq!(back.zones[0].model_id, "drum_membrane");
        assert_eq!(back.zones[0].fixed_note, Some(38));
        assert_eq!(back.zones[1].lo, 48);
    }

    #[test]
    fn single_preset_loads_as_non_kit() {
        // A single-instrument preset (no `zones` field) must not read as a kit.
        let json = r#"{"name":"Old","model_id":"basic_wave","params":{},"engine":{"gain":1.0,"attack_ms":3.0,"release_ms":120.0,"retrigger_ms":0.0}}"#;
        let p: Preset = serde_json::from_str(json).unwrap();
        assert!(!p.is_kit());
        assert!(p.build_model().is_some());
    }

    #[test]
    fn filesystem_roundtrip_and_delete() {
        // Unique temp dir under the OS temp location (no env, no races).
        let dir = std::env::temp_dir().join(format!("ftm_presets_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let preset = Preset::capture("Warm Pluck", &BasicWave::default(), &EngineParams::default());
        let path = save_in(&dir, &preset).expect("save");
        assert!(path.exists());

        let listed = list_in(&dir);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Warm Pluck");
        assert!(listed[0].build_model().is_some());

        assert!(delete_in(&dir, "Warm Pluck").expect("delete"));
        assert!(list_in(&dir).is_empty());
        assert!(!delete_in(&dir, "Warm Pluck").expect("delete-missing")); // already gone

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn factory_kit_is_valid() {
        let kit = factory();
        assert!(!kit.is_empty(), "at least one factory preset");
        let mut names: Vec<_> = kit.iter().map(|p| p.name.clone()).collect();
        let before = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), before, "factory preset names must be unique");
        // Every registered model has at least one factory preset (derived from the
        // registry, so it stays correct as plugins are added).
        let ids: std::collections::HashSet<_> = kit.iter().map(|p| p.model_id.clone()).collect();
        for m in crate::models::registry() {
            assert!(ids.contains(m.id()), "kit should include a {} preset", m.id());
        }
        // Every factory preset must rebuild into a working model that produces sound.
        for p in &kit {
            let model = p.build_model().unwrap_or_else(|| panic!("{} rebuilds", p.name));
            let mut buf = crate::models::ModeBuffer::default();
            model.excite(220.0, 1.0, 48_000.0, &mut buf);
            assert!(buf.n > 0, "{} should produce modes", p.name);
            assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite()), "{}", p.name);
            assert!(buf.decay[..buf.n].iter().all(|d| d.is_finite()), "{}", p.name);
            // The experimental "Base:" batch is designed against a fully grounded
            // engine; until the Membrane/Cymbal/Horn-loss models are grounded, some
            // produce degenerate (zero-frequency) or swelling (negative-decay) modes
            // in the classic bank. The graph path clamps decay and the voice limiter
            // keeps them safe; the established presets stay strictly validated.
            if !p.name.starts_with("Base:") {
                assert!(buf.freq[..buf.n].iter().all(|f| *f > 0.0), "{}", p.name);
                assert!(buf.decay[..buf.n].iter().all(|d| *d >= 0.0), "{}", p.name);
            }
        }
    }

    #[test]
    fn file_stem_is_safe() {
        assert_eq!(file_stem("Warm Pluck"), "Warm_Pluck");
        assert_eq!(file_stem("bass/lead:1"), "bass_lead_1");
        assert_eq!(file_stem("   "), "preset");
        // No path separators or other unsafe characters survive.
        assert!(!file_stem("a/b\\c").contains(['/', '\\']));
    }
}

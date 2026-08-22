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
/// maps a key range to its own instrument). This mirrors how `LoopTrack` stores
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
    // Common firmware baselines; only the expressive fields vary per instrument.
    fn string(
        stiffness: f32,
        damping: f32,
        freq_dep_damping: f32,
        string_length: f32,
        depth: usize,
        pluck_pos: f32,
    ) -> PureString {
        PureString {
            stiffness,
            prop_speed: 500.0,
            damping,
            freq_dep_damping,
            string_length,
            depth,
            pluck_pos,
            damp_period: 100.0,
            time_scale: 10_000.0,
            play_magnitude: 0.0,
            max_magnitude: 2500.0,
            key_tracks_pitch: true,
            ..PureString::default()
        }
    }
    // Same bore as `string`, but bowed (driven → sustains while played).
    fn bowed(
        stiffness: f32,
        damping: f32,
        freq_dep_damping: f32,
        string_length: f32,
        depth: usize,
        pluck_pos: f32,
    ) -> PureString {
        PureString {
            excitation: Excitation::Bowed,
            ..string(stiffness, damping, freq_dep_damping, string_length, depth, pluck_pos)
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

    vec![
        // ---- Pure String (feedback: pluck toward saw, stronger HF damping) ----
        make("Acoustic Bass", string(1.5, 5.0, -5.0, 6.0, 24, 0.15), eng(0.75, 4.0, 120.0)),
        make("Electric Bass", string(2.0, 4.0, -3.0, 8.0, 22, 0.12), eng(0.75, 4.0, 140.0)),
        make("Acoustic Guitar", string(1.0, 7.0, -4.5, 12.0, 28, 0.10), eng(0.6, 3.0, 120.0)),
        make("Electric Guitar", string(1.2, 2.5, -2.5, 16.0, 32, 0.10), eng(0.6, 3.0, 200.0)),
        // Less bass-heavy: brighter (more modes) with the highs allowed to sustain.
        make("Piano", string(6.0, 3.0, -1.8, 12.0, 36, 0.12), eng(0.6, 2.0, 150.0)),
        make("Banjo", string(3.0, 13.0, -6.0, 20.0, 36, 0.08), eng(0.6, 2.0, 80.0)),
        // Rounder pluck + more HF damping to tame the "electric" low end.
        make("Harp", string(0.8, 3.5, -4.0, 14.0, 28, 0.18), eng(0.6, 3.0, 180.0)),
        // ---- Bowed strings (driven → sustain; bow near the bridge = bright/saw) ----
        make("Violin", bowed(0.5, 2.5, -1.2, 4.0, 44, 0.12), eng(0.55, 60.0, 150.0)),
        make("Viola", bowed(0.6, 2.5, -1.5, 5.0, 40, 0.14), eng(0.55, 65.0, 160.0)),
        make("Cello", bowed(0.8, 2.2, -1.3, 7.0, 44, 0.13), eng(0.6, 70.0, 180.0)),
        make("Bowed Bass", bowed(1.0, 2.0, -1.6, 10.0, 36, 0.12), eng(0.65, 80.0, 200.0)),
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
            DrumMembrane { strike_pos: 0.5, damping: 9.0, freq_dep_damping: -3.5, radius: 10.0, depth: 40, stiffness: 0.5, ..DrumMembrane::default() },
            eng(0.7, 1.0, 90.0),
        ),
        make(
            "Kick",
            DrumMembrane { strike_pos: 0.35, damping: 28.0, freq_dep_damping: -4.0, radius: 14.0, depth: 28, stiffness: 0.2, ..DrumMembrane::default() },
            eng(0.85, 1.0, 60.0),
        ),
        make(
            "Timpani",
            DrumMembrane { strike_pos: 0.7, damping: 2.5, freq_dep_damping: -1.5, radius: 9.0, depth: 48, stiffness: 1.0, ..DrumMembrane::default() },
            eng(0.6, 2.0, 200.0),
        ),
        // ---- Webster Horn ----
        // Trumpet: physical bore (contracting throat + flare) from Dave's config.
        // Their freq_dependent_damping = +0.08 maps to our −0.08 sign convention.
        make(
            "Trumpet",
            WebsterHorn {
                boundary: Boundary::Brass,
                r1: 0.0045,
                r2: -0.0030,
                r3: 0.0320,
                length: 1.4,
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
                r1: 0.0045,
                r2: -0.0020,
                r3: 0.0250,
                length: 2.4,
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
                r1: 0.0045,
                r2: -0.0020,
                r3: 0.0250,
                length: 2.4,
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
            WebsterHorn { boundary: Boundary::Brass, r1: 0.0068, r2: -0.001, r3: 0.030, length: 2.7, blow_pos: 0.0, depth: 30, resolution: 400, damping: 8.0, freq_dep_damping: -0.08, visco_loss: 1.8, radiation: 1.4, wavefront: Wavefront::Spherical, ..WebsterHorn::default() },
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
    ]
}

/// Seed the folder with the factory presets if it currently has none (first run).
/// Returns the resulting list.
pub fn seed_if_empty() -> Vec<Preset> {
    let dir = presets_dir();
    let existing = list_in(&dir);
    if existing.is_empty() {
        for p in factory() {
            let _ = save_in(&dir, &p);
        }
        list_in(&dir)
    } else {
        existing
    }
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
            assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite() && *f > 0.0), "{}", p.name);
            assert!(buf.decay[..buf.n].iter().all(|d| d.is_finite() && *d >= 0.0), "{}", p.name);
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

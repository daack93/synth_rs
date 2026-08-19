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
use crate::models::{model_from_id, FtmModel};
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

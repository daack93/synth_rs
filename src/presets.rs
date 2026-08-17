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

use crate::models::pure_string::PureString;
use crate::models::{model_from_id, FtmModel};
use crate::instrument::EngineParams;

/// A saved instrument: everything needed to reconstruct a playable sound.
#[derive(Clone, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    /// The model's [`FtmModel::id`].
    pub model_id: String,
    /// The model's serialized parameters ([`FtmModel::to_json`]).
    pub params: serde_json::Value,
    /// Engine-wide parameters (gain, envelope, retrigger).
    pub engine: EngineParams,
}

impl Preset {
    /// Capture the currently-selected model and engine as a named preset.
    pub fn capture(name: &str, model: &dyn FtmModel, engine: &EngineParams) -> Self {
        Preset {
            name: name.trim().to_string(),
            model_id: model.id().to_string(),
            params: model.to_json(),
            engine: engine.clone(),
        }
    }

    /// Rebuild the model this preset describes. `None` if the plugin id is
    /// unknown (e.g. a preset from a newer build) or the params don't fit it.
    pub fn build_model(&self) -> Option<Box<dyn FtmModel>> {
        model_from_id(&self.model_id, &self.params)
    }
}

/// Built-in "factory" instruments (all Pure String), as starting points to tune
/// by ear. These are seeded into the presets folder on first run.
///
/// The parameters are chosen from the physical levers: `damping` (d1) sets
/// sustain (T60 ≈ 13.8/d1 s), `freq_dep_damping` (d3) sets how fast the tone
/// darkens, `stiffness` (S) sets inharmonicity, and pluck/`depth`/`string_length`
/// shape brightness.
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
    fn make(name: &str, ps: PureString, engine: EngineParams) -> Preset {
        Preset::capture(name, &ps, &engine)
    }

    vec![
        // Round upright pizz: near-center pluck (hollow), dark, medium sustain.
        make(
            "Acoustic Bass",
            string(1.5, 5.0, -3.0, 6.0, 24, 0.5),
            eng(0.75, 4.0, 120.0),
        ),
        // Growly finger bass: fuller off-center pluck, a touch brighter, longer.
        make(
            "Electric Bass",
            string(2.0, 4.0, -2.0, 8.0, 20, 0.15),
            eng(0.75, 4.0, 140.0),
        ),
        // Steel-string body: moderate brightness and sustain.
        make(
            "Acoustic Guitar",
            string(1.0, 7.0, -1.8, 12.0, 24, 0.13),
            eng(0.6, 3.0, 120.0),
        ),
        // Clean electric: bright, long sustain, slow tone decay.
        make(
            "Electric Guitar",
            string(1.2, 2.0, -0.6, 16.0, 28, 0.12),
            eng(0.6, 3.0, 200.0),
        ),
        // Hammered piano: strong inharmonicity (high stiffness), long ring.
        make(
            "Piano",
            string(6.0, 3.0, -1.2, 12.0, 24, 0.13),
            eng(0.6, 2.0, 150.0),
        ),
        // Banjo: very bright, quick "plink" (fast HF + short sustain).
        make(
            "Banjo",
            string(3.0, 13.0, -4.0, 20.0, 32, 0.1),
            eng(0.6, 2.0, 80.0),
        ),
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
    use crate::models::musical_string::MusicalString;

    #[test]
    fn roundtrip_preserves_params() {
        let mut m = MusicalString::default();
        m.inharmonicity = 0.01234;
        m.num_modes = 41;
        let engine = EngineParams {
            gain: 1.234,
            ..EngineParams::default()
        };
        let preset = Preset::capture("Test Lead", &m, &engine);

        // Serialize -> deserialize via JSON text (no filesystem).
        let text = serde_json::to_string(&preset).unwrap();
        let back: Preset = serde_json::from_str(&text).unwrap();
        assert_eq!(back.name, "Test Lead");
        assert_eq!(back.model_id, "musical_string");
        assert!((back.engine.gain - 1.234).abs() < 1e-6);

        let model = back.build_model().expect("rebuilds model");
        assert_eq!(model.id(), "musical_string");
        // The rebuilt model's json should match the original's.
        assert_eq!(model.to_json(), m.to_json());
    }

    #[test]
    fn filesystem_roundtrip_and_delete() {
        // Unique temp dir under the OS temp location (no env, no races).
        let dir = std::env::temp_dir().join(format!("ftm_presets_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let preset = Preset::capture("Warm Pluck", &MusicalString::default(), &EngineParams::default());
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
        assert_eq!(kit.len(), 6);
        let mut names: Vec<_> = kit.iter().map(|p| p.name.clone()).collect();
        let before = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), before, "factory preset names must be unique");
        // Every factory preset must rebuild into a working model that produces sound.
        for p in &kit {
            assert_eq!(p.model_id, "pure_string");
            let model = p.build_model().expect("factory preset rebuilds");
            let mut buf = crate::models::ModeBuffer::default();
            model.excite(110.0, 1.0, 48_000.0, &mut buf);
            assert!(buf.n > 0, "{} should produce modes", p.name);
            assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite() && *f > 0.0));
            assert!(buf.decay[..buf.n].iter().all(|d| d.is_finite() && *d >= 0.0));
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

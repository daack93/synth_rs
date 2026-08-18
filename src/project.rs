//! Projects: save a loop (its tracks — note events + instruments) to disk, and
//! collect several named loops into a project. A project is one JSON file per
//! project in a folder (like presets).
//!
//! Positions and lengths are stored in **seconds**, not samples, so a project is
//! portable across sample rates. The `Project` type has room to grow a song
//! arrangement (a timeline of these loops) in a later step.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::instrument::EngineParams;

/// One recorded note event, timed from the loop start (seconds).
#[derive(Clone, Serialize, Deserialize)]
pub struct LoopEvent {
    pub t: f32,
    pub on: bool,
    pub note: u8,
    pub vel: f32,
}

/// One zone of a kit: an instrument (model + params + engine) mapped to a key
/// range, optionally forced to a fixed pitch (a drum pad). Used by kit tracks.
#[derive(Clone, Serialize, Deserialize)]
pub struct ZoneData {
    pub name: String,
    /// Inclusive MIDI key range this zone responds to.
    pub lo: u8,
    pub hi: u8,
    /// If set, any key in range plays this fixed note (a percussion pad).
    #[serde(default)]
    pub fixed_note: Option<u8>,
    /// Semitone shift applied to chromatic (non-fixed) zones.
    #[serde(default)]
    pub transpose: i8,
    pub model_id: String,
    pub params: serde_json::Value,
    #[serde(default)]
    pub engine: EngineParams,
}

/// One recorded parameter move: at time `t` (seconds), `target` was set to
/// `value`. `target` is a model parameter id (e.g. `"damping"`) or an engine
/// parameter prefixed `eng:` (`"eng:gain"`, `"eng:attack"`, `"eng:release"`,
/// `"eng:retrigger"`). These are the per-knob deltas captured while recording.
#[derive(Clone, Serialize, Deserialize)]
pub struct AutoPoint {
    pub t: f32,
    pub target: String,
    pub value: f32,
}

fn default_volume() -> f32 {
    1.0
}

/// One track of a loop: its instrument plus the notes it plays.
///
/// A track is either a **single instrument** (`zones` empty — the top-level
/// `model_id`/`params`/`engine` describe it) or a **kit** (`zones` non-empty —
/// each zone routes a key range to its own instrument). Old projects predate
/// `zones` and load as single instruments.
#[derive(Clone, Serialize, Deserialize)]
pub struct LoopTrack {
    pub name: String,
    pub model_id: String,
    pub params: serde_json::Value,
    #[serde(default)]
    pub engine: EngineParams,
    #[serde(default)]
    pub muted: bool,
    /// Mixer level (linear, 1.0 = unity).
    #[serde(default = "default_volume")]
    pub volume: f32,
    /// Stereo pan, -1 (left) … 0 (centre) … +1 (right).
    #[serde(default)]
    pub pan: f32,
    /// Fade-in / fade-out lengths in seconds (0 = none).
    #[serde(default)]
    pub fade_in: f32,
    #[serde(default)]
    pub fade_out: f32,
    /// This track's own loop length in seconds. `None` (older projects) = the
    /// whole loop; otherwise the track repeats at this period independently.
    #[serde(default)]
    pub period: Option<f32>,
    #[serde(default)]
    pub zones: Vec<ZoneData>,
    /// Recorded parameter automation (per-knob moves over the loop).
    #[serde(default)]
    pub automation: Vec<AutoPoint>,
    pub events: Vec<LoopEvent>,
}

/// A complete loop: its length and all its tracks.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct LoopData {
    /// Loop length in seconds (0 = no loop).
    pub length: f32,
    pub tracks: Vec<LoopTrack>,
}

impl LoopData {
    pub fn is_empty(&self) -> bool {
        self.length <= 0.0 || self.tracks.is_empty()
    }
    pub fn note_count(&self) -> usize {
        self.tracks.iter().map(|t| t.events.iter().filter(|e| e.on).count()).sum()
    }
}

/// A named loop within a project.
#[derive(Clone, Serialize, Deserialize)]
pub struct NamedLoop {
    pub name: String,
    #[serde(rename = "loop")]
    pub data: LoopData,
}

/// One section of a song arrangement: a loop (by index into `Project.loops`)
/// played a number of times.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Section {
    pub loop_index: usize,
    pub repeats: u32,
}

/// Tempo + grid settings for recording in time.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct TempoGrid {
    pub bpm: f32,
    pub beats_per_bar: u32,
    /// Fixed loop length in bars (0 = free-length; the take sets the length).
    pub bars: u32,
    /// Quantize grid steps per beat (0 = off; 1 = 1/4, 2 = 1/8, 3 = 1/8T, 4 = 1/16).
    pub quantize: u32,
    pub metronome: bool,
    /// Play one bar of clicks before a fixed-bars recording starts.
    pub count_in: bool,
}

impl Default for TempoGrid {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            beats_per_bar: 4,
            bars: 0,
            quantize: 0,
            metronome: false,
            count_in: false,
        }
    }
}

/// A project: a named collection of loops plus a song arrangement over them.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    #[serde(default)]
    pub loops: Vec<NamedLoop>,
    /// The song: an ordered list of sections played straight through.
    #[serde(default)]
    pub arrangement: Vec<Section>,
    /// Tempo + grid settings (saved with the project).
    #[serde(default)]
    pub tempo: TempoGrid,
}

/// Directory projects are stored in: `$FTM_SYNTH_PROJECTS`, else `projects/`.
pub fn projects_dir() -> PathBuf {
    std::env::var_os("FTM_SYNTH_PROJECTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("projects"))
}

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
        s.push_str("project");
    }
    s
}

pub fn save(project: &Project) -> io::Result<PathBuf> {
    save_in(&projects_dir(), project)
}

pub fn list() -> Vec<String> {
    list_in(&projects_dir())
}

pub fn load_named(name: &str) -> io::Result<Project> {
    load(&projects_dir().join(format!("{}.json", file_stem(name))))
}

pub fn load(path: &Path) -> io::Result<Project> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn save_in(dir: &Path, project: &Project) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.json", file_stem(&project.name)));
    let json = serde_json::to_string_pretty(project)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(&path, json)?;
    Ok(path)
}

/// List project names (file stems) in a folder, sorted.
fn list_in(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(p) = load(&path) {
                    out.push(p.name);
                }
            }
        }
    }
    out.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_loop() -> LoopData {
        LoopData {
            length: 2.0,
            tracks: vec![LoopTrack {
                name: "Track 1".into(),
                model_id: "musical_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                fade_in: 0.0,
                fade_out: 0.0,
                period: None,
                zones: Vec::new(),
                automation: vec![
                    AutoPoint { t: 0.1, target: "damping".into(), value: 5.0 },
                    AutoPoint { t: 0.3, target: "eng:gain".into(), value: 0.8 },
                ],
                events: vec![
                    LoopEvent { t: 0.0, on: true, note: 60, vel: 0.9 },
                    LoopEvent { t: 0.5, on: false, note: 60, vel: 0.0 },
                ],
            }],
        }
    }

    #[test]
    fn project_roundtrip_on_disk() {
        let dir = std::env::temp_dir().join(format!("ftm_projects_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut project = Project {
            name: "My Song".into(),
            loops: Vec::new(),
            arrangement: Vec::new(),
            tempo: TempoGrid::default(),
        };
        project.loops.push(NamedLoop { name: "Groove A".into(), data: sample_loop() });
        let path = save_in(&dir, &project).unwrap();
        assert!(path.exists());

        let names = list_in(&dir);
        assert_eq!(names, vec!["My Song".to_string()]);

        let back = load(&path).unwrap();
        assert_eq!(back.name, "My Song");
        assert_eq!(back.loops.len(), 1);
        assert_eq!(back.loops[0].name, "Groove A");
        assert_eq!(back.loops[0].data.tracks.len(), 1);
        assert_eq!(back.loops[0].data.note_count(), 1);
        assert!((back.loops[0].data.length - 2.0).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }
}

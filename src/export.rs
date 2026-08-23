//! Offline WAV export.
//!
//! Because the studio is deterministic and times everything in seconds, a loop
//! or a song can be re-rendered off the audio thread — as fast as the CPU
//! allows and at any sample rate. A fresh [`Studio`] is built for the render so
//! live playback is untouched, the loop/song is installed, and the result is
//! written as a stereo WAV (per-track pan baked into the channels) at a chosen
//! bit depth (16 / 24 / 32-bit integer PCM).
//!
//! **Hi-res:** with no real-time deadline we can raise quality. Rendering at a
//! higher sample rate is the broad win — it lowers aliasing on the bright
//! inharmonic models *and* lets more modes fit under Nyquist for free. On top of
//! that, [`apply_hi_res`] maxes the Webster horn's eigensolve resolution, a pure
//! accuracy knob that doesn't change the intended timbre.

use std::io;
use std::path::{Path, PathBuf};

use crate::project::SongData;
use crate::studio::{Command, Studio};
use crate::wav;

/// Directory exports are written to: `$FTM_SYNTH_EXPORTS`, else `exports/`.
pub fn exports_dir() -> PathBuf {
    std::env::var_os("FTM_SYNTH_EXPORTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("exports"))
}

/// Turn a name into a `.wav` path in the exports directory.
pub fn export_path(name: &str) -> PathBuf {
    let stem: String = name
        .trim()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let stem = if stem.is_empty() { "export".to_string() } else { stem };
    exports_dir().join(format!("{stem}.wav"))
}

/// Raise render-only quality knobs on a loop's instruments. Currently: max out
/// the Webster horn eigensolve resolution (numerical accuracy only). Applied to
/// a clone, so the live sound is unaffected.
pub fn apply_hi_res(data: &mut SongData) {
    for t in &mut data.tracks {
        bump(&t.model_id, &mut t.params);
        for z in &mut t.zones {
            bump(&z.model_id, &mut z.params);
        }
    }
}

fn bump(model_id: &str, params: &mut serde_json::Value) {
    if model_id == "webster_horn" {
        if let Some(obj) = params.as_object_mut() {
            obj.insert("resolution".into(), serde_json::json!(512));
        }
    }
}

/// Render a single loop `repeats` times (plus a `tail_secs` ring-out) to a
/// stereo WAV at the given bit depth. Per-track pan is applied to the channels.
pub fn render_loop_to_wav(
    mut data: SongData,
    sr: f32,
    repeats: u32,
    tail_secs: f32,
    hi_res: bool,
    depth: wav::BitDepth,
    path: &Path,
) -> io::Result<()> {
    if data.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "nothing to export"));
    }
    if hi_res {
        apply_hi_res(&mut data);
    }
    let loop_frames = (data.length * sr).round() as usize;
    let total = loop_frames.saturating_mul(repeats.max(1) as usize);
    let tail = (tail_secs.max(0.0) * sr) as usize;

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut studio = Studio::new(sr);
    studio.handle(Command::LoadSong(data));
    let buf = studio.render_offline(total, tail); // interleaved stereo
    wav::write_pcm(path, sr as u32, 2, depth, &buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instrument::EngineParams;
    use crate::project::{NoteEvent, TrackData};

    fn one_note_loop() -> SongData {
        SongData {
            length: 0.1,
            arrangement: Vec::new(),
            tracks: vec![TrackData {
                name: "T".into(),
                model_id: "musical_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                fade_in: 0.0,
                fade_out: 0.0,
                period: None,
                start: None,
                span: None,
                zones: Vec::new(),
                automation: Vec::new(),
                events: vec![NoteEvent { t: 0.0, on: true, note: 60, vel: 1.0 }],
            }],
        }
    }

    #[test]
    fn renders_a_loop_with_the_expected_length() {
        let dir = std::env::temp_dir().join(format!("ftm_export_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("loop.wav");
        // 0.1s loop × 3 repeats + 0.2s tail @ 48k = (0.3 + 0.2)·48000 frames.
        render_loop_to_wav(one_note_loop(), 48_000.0, 3, 0.2, false, wav::BitDepth::Int16, &path)
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let expected_frames = ((0.1 * 3.0 + 0.2) * 48_000.0) as usize;
        // Stereo 16-bit: 2 channels × 2 bytes per frame.
        assert_eq!(bytes.len(), 44 + expected_frames * 2 * 2);
        assert!(&bytes[0..4] == b"RIFF");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hi_res_maxes_horn_resolution() {
        let mut data = SongData {
            length: 0.1,
            arrangement: Vec::new(),
            tracks: vec![TrackData {
                name: "H".into(),
                model_id: "webster_horn".into(),
                params: serde_json::json!({ "resolution": 64 }),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                fade_in: 0.0,
                fade_out: 0.0,
                period: None,
                start: None,
                span: None,
                zones: Vec::new(),
                automation: Vec::new(),
                events: Vec::new(),
            }],
        };
        apply_hi_res(&mut data);
        assert_eq!(data.tracks[0].params["resolution"], serde_json::json!(512));
    }
}

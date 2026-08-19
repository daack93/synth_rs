//! A minimal, dependency-free 16-bit PCM WAV writer (mono).
//!
//! The engine mixes to a single mono stream clamped to `[-1, 1]`, so a mono
//! 16-bit PCM file is the honest, universally-playable format for exports.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Write `samples` (mono, `[-1, 1]`) as a 16-bit PCM WAV at `sample_rate`.
pub fn write_pcm16_mono(path: &Path, sample_rate: u32, samples: &[f32]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);

    let channels: u16 = 1;
    let bits: u16 = 16;
    let block_align: u16 = channels * bits / 8;
    let byte_rate: u32 = sample_rate * block_align as u32;
    let data_bytes: u32 = (samples.len() as u32) * block_align as u32;

    // RIFF header
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_bytes).to_le_bytes())?;
    w.write_all(b"WAVE")?;
    // fmt chunk
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // PCM fmt chunk size
    w.write_all(&1u16.to_le_bytes())?; // audio format = PCM
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits.to_le_bytes())?;
    // data chunk
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_valid_wav_header() {
        let dir = std::env::temp_dir().join(format!("ftm_wav_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.wav");
        // 100 samples of a ramp.
        let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0 - 0.5).collect();
        write_pcm16_mono(&path, 48_000, &samples).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        // 44-byte header + 100 samples × 2 bytes.
        assert_eq!(bytes.len(), 44 + 100 * 2);
        // data chunk size field matches the sample bytes.
        let data_len = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(data_len, 200);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

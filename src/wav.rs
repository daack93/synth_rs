//! A minimal, dependency-free integer-PCM WAV writer.
//!
//! Handles mono or interleaved multi-channel float samples (`[-1, 1]`) at
//! 16-, 24-, or 32-bit integer PCM. Exports are stereo (the arrangement's pan
//! controls are baked into the two channels) and default to 16-bit, with 24-
//! and 32-bit available for higher-quality masters.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Output sample format.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    Int16,
    Int24,
    Int32,
}

impl BitDepth {
    pub fn bits(self) -> u16 {
        match self {
            BitDepth::Int16 => 16,
            BitDepth::Int24 => 24,
            BitDepth::Int32 => 32,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            BitDepth::Int16 => "16-bit",
            BitDepth::Int24 => "24-bit",
            BitDepth::Int32 => "32-bit",
        }
    }
}

/// Write interleaved `samples` (`[-1, 1]`, `channels`-interleaved) as integer
/// PCM at `sample_rate`.
pub fn write_pcm(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    depth: BitDepth,
    samples: &[f32],
) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);

    let bits = depth.bits();
    let bytes_per_sample = (bits / 8) as u32;
    let block_align: u16 = channels * bits / 8;
    let byte_rate: u32 = sample_rate * block_align as u32;
    let data_bytes: u32 = samples.len() as u32 * bytes_per_sample;

    // RIFF header
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_bytes).to_le_bytes())?;
    w.write_all(b"WAVE")?;
    // fmt chunk
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // PCM fmt chunk size
    w.write_all(&1u16.to_le_bytes())?; // audio format = integer PCM
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits.to_le_bytes())?;
    // data chunk
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    for &s in samples {
        let s = s.clamp(-1.0, 1.0);
        match depth {
            BitDepth::Int16 => {
                let v = (s * i16::MAX as f32).round() as i16;
                w.write_all(&v.to_le_bytes())?;
            }
            BitDepth::Int24 => {
                // 24-bit signed, little-endian: the low three bytes of the i32.
                let v = (s as f64 * 8_388_607.0).round() as i32; // 2^23 - 1
                let b = v.to_le_bytes();
                w.write_all(&b[0..3])?;
            }
            BitDepth::Int32 => {
                let v = (s as f64 * 2_147_483_647.0).round() as i32; // 2^31 - 1
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_valid_stereo_16bit_header() {
        let dir = std::env::temp_dir().join(format!("ftm_wav_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.wav");
        // 100 stereo frames = 200 interleaved samples.
        let samples: Vec<f32> = (0..200).map(|i| i as f32 / 200.0 - 0.5).collect();
        write_pcm(&path, 48_000, 2, BitDepth::Int16, &samples).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        // 44-byte header + 200 samples × 2 bytes.
        assert_eq!(bytes.len(), 44 + 200 * 2);
        let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
        assert_eq!(channels, 2);
        let data_len = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(data_len, 400);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bit_depth_sets_the_sample_size() {
        let dir = std::env::temp_dir().join(format!("ftm_wav_depth_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        for (depth, bytes_per) in [(BitDepth::Int16, 2), (BitDepth::Int24, 3), (BitDepth::Int32, 4)] {
            let path = dir.join(format!("d{}.wav", depth.bits()));
            let samples = vec![0.25f32; 10]; // 5 stereo frames
            write_pcm(&path, 48_000, 2, depth, &samples).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(bytes.len(), 44 + 10 * bytes_per, "{}", depth.label());
            let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
            assert_eq!(bits, depth.bits());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

//! The per-sample voice graph — the coupled, real-time synthesis path.
//!
//! Where the mode-bank engine ([`crate::instrument`]) precomputes each note as a
//! set of free-decaying sinusoids that never interact, this runs a small graph
//! of DSP blocks **one sample at a time**, so components can drive each other —
//! including in feedback loops. Every node is a [`Node`]: it `tick`s once per
//! sample, reads its input signal(s), and emits an output sample. Edges just
//! carry that one scalar signal.
//!
//! The bridge to the existing physics: a resonator still gets its modes
//! (`frequency, amplitude, decay`) from a model's [`ModeBuffer`] exactly as
//! today — but instead of playing each mode as a free oscillator with a fixed
//! decay envelope, it runs each as a **driven two-pole resonator**. Struck with
//! a unit impulse, a driven resonator rings out the *same* decaying sinusoid; fed
//! a continuous signal, it responds continuously; wired into a loop, it couples.
//! So the modal physics is unchanged — only how it's rendered, which is what
//! unlocks coupling.

use crate::models::ModeBuffer;

const TAU: f32 = std::f32::consts::TAU;

/// A per-voice, per-sample DSP block. `tick` advances one sample: it reads its
/// input signals (empty for a source like an exciter) and returns its output.
pub trait Node: Send {
    fn tick(&mut self, inputs: &[f32]) -> f32;
}

/// One resonant mode as a two-pole resonator. Its impulse response is
/// `amp · rⁿ · sin(θ(n+1))`, i.e. a sinusoid at the mode frequency decaying at
/// the mode's rate — the same thing the old free-oscillator bank produced —
/// but because it filters its input each sample it can be driven and coupled.
struct Mode {
    a1: f32,
    a2: f32,
    b0: f32,
    y1: f32,
    y2: f32,
}

impl Mode {
    fn new(freq: f32, decay: f32, amp: f32, sr: f32) -> Self {
        let theta = TAU * (freq / sr); // resonant angle
        let r = (-decay / sr).exp().clamp(0.0, 0.999_999); // per-sample pole radius
        Mode {
            a1: 2.0 * r * theta.cos(),
            a2: -(r * r),
            // Chosen so a unit impulse yields peak output ≈ `amp`.
            b0: amp * theta.sin(),
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.a1 * self.y1 + self.a2 * self.y2 + self.b0 * x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// A parallel bank of driven two-pole resonators — a resonator node built from a
/// model's mode bank. Each sample it sums every mode's response to the input.
pub struct ModalResonator {
    modes: Vec<Mode>,
}

impl ModalResonator {
    /// Build the resonator from a model's computed mode bank (freq/amp/decay).
    pub fn from_bank(bank: &ModeBuffer, sr: f32) -> Self {
        let modes = (0..bank.n)
            .map(|i| Mode::new(bank.freq[i], bank.decay[i].max(0.0), bank.amp[i], sr))
            .collect();
        ModalResonator { modes }
    }

    pub fn mode_count(&self) -> usize {
        self.modes.len()
    }
}

impl Node for ModalResonator {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let x = inputs.first().copied().unwrap_or(0.0);
        let mut s = 0.0;
        for m in &mut self.modes {
            s += m.tick(x);
        }
        s
    }
}

/// A one-shot strike: emits `amp` on the first sample, then silence. A struck
/// resonator rings it out. (Velocity is carried here as `amp`.)
pub struct ImpulseExciter {
    amp: f32,
    fired: bool,
}

impl ImpulseExciter {
    pub fn new(amp: f32) -> Self {
        ImpulseExciter { amp, fired: false }
    }
}

impl Node for ImpulseExciter {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        if self.fired {
            0.0
        } else {
            self.fired = true;
            self.amp
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Goertzel magnitude of `x` at frequency `f`.
    fn mag_at(x: &[f32], f: f32, sr: f32) -> f32 {
        let w = -TAU * f / sr;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (n, &v) in x.iter().enumerate() {
            re += v as f64 * (w as f64 * n as f64).cos();
            im += v as f64 * (w as f64 * n as f64).sin();
        }
        ((re * re + im * im).sqrt() / x.len() as f64) as f32
    }

    #[test]
    fn one_mode_rings_at_its_frequency_and_decays() {
        let sr = 48_000.0;
        let (f, decay, amp) = (440.0f32, 3.0f32, 1.0f32);
        let mut bank = ModeBuffer::default();
        bank.push(f, amp, decay);

        let mut exc = ImpulseExciter::new(1.0);
        let mut reso = ModalResonator::from_bank(&bank, sr);

        let n = sr as usize; // 1 second
        let mut y = vec![0.0f32; n];
        for s in y.iter_mut() {
            let e = exc.tick(&[]);
            *s = reso.tick(&[e]);
        }

        // Energy concentrated at 440 Hz, not at an unrelated frequency.
        let on = mag_at(&y, 440.0, sr);
        let off = mag_at(&y, 700.0, sr);
        assert!(on > 20.0 * off, "rings at 440 Hz (on={on:.4}, off={off:.4})");

        // Envelope decays as exp(-decay*t): compare early vs late RMS windows.
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        let early = rms(&y[0..4800]); // ~0..0.1 s
        let late = rms(&y[38400..43200]); // ~0.8..0.9 s
        // Over ~0.85 s at decay 3.0, amplitude drops by ~exp(-2.55) ≈ 0.078.
        let ratio = late / early;
        assert!(ratio < 0.2 && ratio > 0.01, "decays roughly as expected (ratio={ratio:.4})");
    }

    #[test]
    fn plate_bank_reproduces_the_decaying_sinusoid_sum() {
        // The per-sample resonator bank should render the same modal content as
        // the "ideal" sum of decaying sinusoids the old engine approximates.
        use crate::models::pure_plate::PurePlate;
        use crate::models::FtmModel;

        let sr = 48_000.0;
        let mut bank = ModeBuffer::default();
        PurePlate::default().excite(220.0, 1.0, sr, &mut bank);
        assert!(bank.n > 8);

        // Reference: sum_i amp_i * exp(-decay_i * t) * sin(2π f_i t).
        let n = 24_000usize; // 0.5 s
        let mut reference = vec![0.0f32; n];
        for (k, s) in reference.iter_mut().enumerate() {
            let t = k as f32 / sr;
            let mut acc = 0.0f32;
            for i in 0..bank.n {
                acc += bank.amp[i] * (-bank.decay[i] * t).exp() * (TAU * bank.freq[i] * t).sin();
            }
            *s = acc;
        }

        // Per-sample graph render.
        let mut exc = ImpulseExciter::new(1.0);
        let mut reso = ModalResonator::from_bank(&bank, sr);
        let mut got = vec![0.0f32; n];
        for s in got.iter_mut() {
            let e = exc.tick(&[]);
            *s = reso.tick(&[e]);
        }

        // Compare the two at each mode frequency: the per-mode magnitudes should
        // track (same freqs, amplitudes, decays → same spectral envelope).
        let mut worst = 0.0f32;
        for i in 0..bank.n {
            let f = bank.freq[i];
            if f < 40.0 || f > 18_000.0 {
                continue;
            }
            let a = mag_at(&reference, f, sr);
            let b = mag_at(&got, f, sr);
            if a > 1e-4 {
                let rel = (a - b).abs() / a;
                worst = worst.max(rel);
            }
        }
        assert!(worst < 0.25, "per-mode magnitudes match within 25% (worst={worst:.3})");
    }
}

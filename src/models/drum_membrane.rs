//! 2-D circular membrane — a drumhead — via the same FTM recipe as the string.
//!
//! The membrane obeys `u_tt = c²∇²u − S²∇⁴u − d1 u_t + d3 ∇²u_t`, the exact 2-D
//! analogue of the string's stiffness+damping wave equation (∇² replaces
//! ∂²/∂x²). Substituting a mode `u = φ·e^{st}` with `∇²φ = −k²φ` gives the same
//! characteristic equation as the string,
//!
//!     s² + (d1 + d3·k²)·s + (c²·k² + S²·k⁴) = 0,
//!
//! so `σ = (d1 + d3·k²)/2` and `ω = √(c²k² + S²k⁴ − σ²)` — identical to
//! `pure_string`, only the wavenumbers differ. For a circular head clamped at
//! the rim, the eigenfunctions are `J_ν(k r)·cos(νθ)` and the boundary forces
//! `k_{ν,j} = α_{ν,j} / R`, where `α_{ν,j}` is the j-th zero of the Bessel
//! function `J_ν`. Those zeros are *inharmonic* (1 : 1.59 : 2.14 : 2.30 …) —
//! the sound of a drum. Striking at radius fraction `ρ` weights each mode by
//! `J_ν(α_{ν,j}·ρ)`: a centre hit excites only the axisymmetric (ν=0) modes.

use serde::{Deserialize, Serialize};

use super::{strike_amplitude, unbounded_slider, FtmModel, ModeBuffer, PitchMode, TICK_RATE};

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f32 = std::f32::consts::TAU;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DrumMembrane {
    /// Wave speed c (tension / density). With radius, sets the pitch.
    pub prop_speed: f32,
    /// Head stiffness S (the k⁴ term): more = gong-like, more inharmonic.
    pub stiffness: f32,
    /// Uniform damping d1.
    pub damping: f32,
    /// Frequency-dependent damping d3 (negative = high modes decay faster).
    pub freq_dep_damping: f32,
    /// Membrane radius R.
    pub radius: f32,
    /// Number of modes summed.
    pub depth: usize,
    /// Strike position as a fraction of the radius, 0 = centre, 1 = rim.
    pub strike_pos: f32,
    pub damp_period: f32,
    pub time_scale: f32,
    pub play_magnitude: f32,
    pub max_magnitude: f32,
    /// If true the key sets the pitch (the fundamental (0,1) mode lands on it).
    pub key_tracks_pitch: bool,
    /// Transpose a fixed drum, or resize it per note (bigger = lower): higher
    /// notes become more inharmonic and decay faster.
    #[serde(default)]
    pub pitch_mode: PitchMode,
}

impl Default for DrumMembrane {
    fn default() -> Self {
        Self {
            prop_speed: 500.0,
            stiffness: 0.5,
            damping: 8.0,
            freq_dep_damping: -2.0,
            radius: 10.0,
            depth: 48,
            strike_pos: 0.6,
            damp_period: 100.0,
            time_scale: 10_000.0,
            play_magnitude: 0.0,
            max_magnitude: 2500.0,
            key_tracks_pitch: true,
            pitch_mode: PitchMode::Transpose,
        }
    }
}

impl FtmModel for DrumMembrane {
    fn id(&self) -> &'static str {
        "drum_membrane"
    }

    fn display_name(&self) -> &'static str {
        "Drum (2D membrane)"
    }

    fn description(&self) -> &'static str {
        "Circular drumhead: FTM membrane with inharmonic Bessel modes and a strike position."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        let amp_strike = strike_amplitude(vel, self.play_magnitude, self.max_magnitude);
        if amp_strike <= 0.0 {
            return;
        }

        let radius = self.radius.max(1e-3) as f64;
        let rho = self.strike_pos.clamp(0.0, 0.999) as f64;
        let c = self.prop_speed as f64;
        let s_stiff = self.stiffness as f64;
        let d1 = self.damping as f64;
        let d3 = self.freq_dep_damping as f64;
        let damp_per = self.damp_period.max(1e-3) as f64;
        let n_req = self.depth.clamp(1, super::MAX_MODES);

        // Physical mode resizes the head per note (bigger = lower), matching
        // Transpose at C4. The frequency terms use the note's radius; decay keeps
        // the fixed-radius shape and speeds up gently with pitch.
        let (radius_freq, decay_scale) =
            if self.key_tracks_pitch && self.pitch_mode == PitchMode::Physical {
                let f0 = freq_hz.max(1.0) as f64;
                let fref = super::REF_PITCH_HZ as f64;
                (radius * (fref / f0).clamp(0.02, 50.0), (f0 / fref).powf(0.6))
            } else {
                (radius, 1.0)
            };

        // Enumerate the `n_req` lowest-frequency modes. α_{ν,j} grows with both
        // ν and j, so a triangular grid of candidates covers the lowest ones.
        let span = ((2 * n_req) as f64).sqrt().ceil() as u32 + 6;
        let mut cands: Vec<(f64, u32)> = Vec::with_capacity((span * span) as usize);
        for nu in 0..=span {
            for j in 1..=span {
                cands.push((bessel_zero(nu, j), nu));
            }
        }
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cands.truncate(n_req);

        // First pass: modal frequency W and weight K per mode. W[0] (the (0,1)
        // mode) is the reference the key maps onto.
        struct Mode {
            w: f64,
            sigma: f64,
            k_weight: f64,
        }
        let mut modes: Vec<Mode> = Vec::with_capacity(cands.len());
        let mut amp_sum = 0.0f64;
        for &(alpha, nu) in &cands {
            let kf = alpha / radius_freq; // frequency wavenumber (note's radius)
            let kf2 = kf * kf;
            let kf4 = kf2 * kf2;
            let kd2 = (alpha / radius).powi(2); // fixed-radius wavenumber for decay
            // sigma < 0 for a decaying mode (d3 negative dominates at high k).
            let sigma = (d3 * kd2 - d1) / 2.0;
            let o_lin = (sigma / damp_per).exp(); // ~1, mirrors the firmware O[i]
            let w2 = (c * c * kf2 + s_stiff.powi(4) * kf4 - o_lin * o_lin).max(0.0);
            let k_weight = bessel_jn(nu, alpha * rho);
            amp_sum += k_weight.abs();
            modes.push(Mode {
                w: w2.sqrt(),
                sigma,
                k_weight,
            });
        }
        if modes.is_empty() {
            return;
        }

        let w0 = if modes[0].w > 1e-9 { modes[0].w } else { 1.0 };
        let ts = self.time_scale.max(1.0) as f64;
        let norm = if amp_sum > 1e-9 {
            amp_strike as f64 / amp_sum
        } else {
            amp_strike as f64
        };

        for m in &modes {
            let freq = if self.key_tracks_pitch {
                freq_hz as f64 * (m.w / w0)
            } else {
                m.w * TICK_RATE as f64 / (ts * TWO_PI as f64)
            };
            if freq >= sr as f64 * 0.45 {
                continue; // above Nyquist — skip (modes aren't strictly sorted by W)
            }
            // Same decay mapping as the string: sigma applied every DAMP_PERIOD
            // board-tick => per-second rate −sigma·TICK_RATE/DAMP_PERIOD².
            let decay = -m.sigma * TICK_RATE as f64 / (damp_per * damp_per) * decay_scale;
            out.push(freq as f32, (m.k_weight * norm) as f32, decay as f32);
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Membrane Parameters");
        changed |= ui
            .add(
                unbounded_slider(&mut self.strike_pos, 0.0..=1.0, "Strike position").custom_formatter(
                    |v, _| {
                        if v < 0.05 {
                            "centre".into()
                        } else if v > 0.9 {
                            "rim".into()
                        } else {
                            format!("{v:.2}")
                        }
                    },
                ),
            )
            .on_hover_text("0 = centre (only axisymmetric modes), toward the rim = more modes / brighter.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.prop_speed, 1.0..=2000.0, "Wave speed (c)"))
            .on_hover_text("With radius, sets the pitch (fundamental ≈ 2.405·c/2πR).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.stiffness, 0.0..=50.0, "Stiffness (S)"))
            .on_hover_text("The k⁴ term: more = gong/plate-like, more inharmonic.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.damping, -50.0..=200.0, "Damping (d1)"))
            .on_hover_text("Uniform decay of every mode.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.freq_dep_damping, -100.0..=20.0, "Freq-dep damping (d3)"))
            .on_hover_text("Extra decay on high modes (negative = darker, faster-decaying head).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.radius, 0.1..=100.0, "Radius (R)"))
            .on_hover_text("Head size. Larger = brighter and less inharmonic.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.depth, 1..=super::MAX_MODES, "Modes (DEPTH)"))
            .changed();

        ui.add_space(6.0);
        ui.strong("Timing / velocity");
        changed |= ui
            .add(unbounded_slider(&mut self.damp_period, 1.0..=1000.0, "DAMP_PERIOD"))
            .on_hover_text("Larger = longer sustain.")
            .changed();
        changed |= ui
            .add(
                unbounded_slider(&mut self.time_scale, 100.0..=100_000.0, "TIME_SCALE")
                    .logarithmic(true),
            )
            .on_hover_text("Divides modal frequency in physical-pitch mode.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.play_magnitude, 0.0..=2500.0, "PLAY_MAGNITUDE"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.max_magnitude, 1.0..=5000.0, "MAX_MAGNITUDE"))
            .changed();
        changed |= ui
            .checkbox(&mut self.key_tracks_pitch, "Key tracks pitch")
            .on_hover_text("On: the fundamental (0,1) mode lands on the played key.")
            .changed();
        ui.add_enabled_ui(self.key_tracks_pitch, |ui| {
            egui::ComboBox::from_label("Pitch mode")
                .selected_text(match self.pitch_mode {
                    PitchMode::Transpose => "Transpose",
                    PitchMode::Physical => "Physical size",
                })
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut self.pitch_mode, PitchMode::Transpose, "Transpose")
                        .on_hover_text("One drum stretched to each note — uniform timbre.")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.pitch_mode, PitchMode::Physical, "Physical size")
                        .on_hover_text("Resize the head per note (bigger = lower): more inharmonic, faster-decaying highs.")
                        .changed();
                });
        });
        changed
    }

    fn box_clone(&self) -> Box<dyn FtmModel> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// The j-th positive zero of the Bessel function `J_n`, via McMahon's asymptotic
/// expansion — accurate to a few parts in 10⁴, ample for audio and far cheaper
/// than root-finding.
fn bessel_zero(n: u32, j: u32) -> f64 {
    let nn = n as f64;
    let beta = (j as f64 + 0.5 * nn - 0.25) * PI;
    let mu = 4.0 * nn * nn;
    let b8 = 8.0 * beta;
    let t1 = (mu - 1.0) / b8;
    let t2 = 4.0 * (mu - 1.0) * (7.0 * mu - 31.0) / (3.0 * b8.powi(3));
    let t3 = 32.0 * (mu - 1.0) * (83.0 * mu * mu - 982.0 * mu + 3779.0) / (15.0 * b8.powi(5));
    beta - t1 - t2 - t3
}

/// Integer-order Bessel function `J_n(x)` for `n ≥ 0`, `x ≥ 0`, via Miller's
/// downward recurrence with the normalization `J_0 + 2(J_2 + J_4 + …) = 1`.
/// Stable across the argument range we use (unlike the power series).
fn bessel_jn(n: u32, x: f64) -> f64 {
    if x <= 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    let n = n as i64;
    let tox = 2.0 / x;
    // Start well above both n and x for accuracy, and make it even.
    let start = (n.max(x.ceil() as i64) + 15 + (2.0 * x.sqrt()) as i64) | 1;
    let start = start + 1; // even

    let mut bjp = 0.0f64; // J_{j+1}
    let mut bj = 1.0f64; // J_j (unnormalized seed)
    let mut ans = 0.0f64;
    let mut sum = 0.0f64;
    let mut jsum = false;
    let mut j = start;
    while j > 0 {
        let bjm = j as f64 * tox * bj - bjp; // J_{j-1}
        bjp = bj;
        bj = bjm;
        if bj.abs() > 1e10 {
            bj *= 1e-10;
            bjp *= 1e-10;
            ans *= 1e-10;
            sum *= 1e-10;
        }
        if jsum {
            sum += bj; // even-order terms J_2, J_4, …
        }
        jsum = !jsum;
        if j == n {
            ans = bjp; // = J_n
        }
        j -= 1;
    }
    // After the loop bj = J_0 (unnormalized).
    if n == 0 {
        ans = bj;
    }
    let norm = 2.0 * sum - bj; // == J_0 + 2(J_2 + J_4 + …), the normalization
    ans / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn bessel_values() {
        assert!(approx(bessel_jn(0, 0.0), 1.0, 1e-9));
        assert!(approx(bessel_jn(1, 0.0), 0.0, 1e-9));
        assert!(approx(bessel_jn(0, 1.0), 0.765_197_7, 1e-4), "{}", bessel_jn(0, 1.0));
        assert!(approx(bessel_jn(1, 1.0), 0.440_050_6, 1e-4), "{}", bessel_jn(1, 1.0));
        assert!(approx(bessel_jn(2, 1.0), 0.114_903_5, 1e-4), "{}", bessel_jn(2, 1.0));
        assert!(approx(bessel_jn(0, 5.0), -0.177_596_8, 1e-4), "{}", bessel_jn(0, 5.0));
        // J_0 is ~0 at its first zero.
        assert!(bessel_jn(0, 2.404_83).abs() < 1e-3);
    }

    #[test]
    fn bessel_zeros_are_close() {
        assert!(approx(bessel_zero(0, 1), 2.4048, 2e-3), "{}", bessel_zero(0, 1));
        assert!(approx(bessel_zero(1, 1), 3.8317, 2e-3), "{}", bessel_zero(1, 1));
        assert!(approx(bessel_zero(0, 2), 5.5201, 2e-3), "{}", bessel_zero(0, 2));
        assert!(approx(bessel_zero(2, 1), 5.1356, 5e-3), "{}", bessel_zero(2, 1));
    }

    #[test]
    fn produces_inharmonic_modes_and_tracks_key() {
        let mut buf = ModeBuffer::default();
        let d = DrumMembrane { key_tracks_pitch: true, depth: 12, ..DrumMembrane::default() };
        d.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 3, "drum should have several modes");
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite() && *f > 0.0));
        assert!(buf.decay[..buf.n].iter().all(|dd| dd.is_finite() && *dd >= 0.0));
        // Fundamental sits on the key.
        assert!((buf.freq[0] - 220.0).abs() < 2.0, "fundamental should track the key: {}", buf.freq[0]);
        // Second partial is inharmonic (not 2x) — Bessel ratio ~1.59.
        let ratio = buf.freq[1] / buf.freq[0];
        assert!(ratio > 1.4 && ratio < 1.8, "second mode ratio ~1.59, got {ratio}");
    }

    #[test]
    fn centre_strike_excites_only_axisymmetric_modes() {
        let mut buf = ModeBuffer::default();
        let d = DrumMembrane { strike_pos: 0.0, depth: 10, ..DrumMembrane::default() };
        d.excite(220.0, 1.0, 48_000.0, &mut buf);
        // At the centre, only (0,j) modes have nonzero weight; the (1,1) mode
        // (second lowest) must be ~silent.
        assert!(buf.amp[1].abs() < 1e-4, "off-axis mode should vanish at a centre strike");
    }

    #[test]
    fn stays_finite_with_degenerate_params() {
        let mut buf = ModeBuffer::default();
        let d = DrumMembrane {
            radius: 0.0,
            damping: -5.0,
            freq_dep_damping: 3.0,
            stiffness: 40.0,
            depth: 2000,
            damp_period: 0.0,
            ..DrumMembrane::default()
        };
        d.excite(110.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite()));
        assert!(buf.amp[..buf.n].iter().all(|a| a.is_finite()));
        assert!(buf.decay[..buf.n].iter().all(|dd| dd.is_finite()));
    }
}

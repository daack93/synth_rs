//! Quadratic Webster Horn — a flaring air column via the Webster horn equation.
//!
//! Acoustic pressure in a column of varying area `A(x) = π r(x)²` obeys Webster's
//! equation. With the substitution `ψ = p·√A` it becomes a 1-D wave equation with
//! a **geometric potential** `V(x)`:
//!
//!     ψ_xx − V(x) ψ − (1/c²) ψ_tt = 0,   V(x) = r''(x)/r(x).
//!
//! For a quadratic bore `r(x) = r1 + r2·x + r3·x²` (so `r'' = 2 r3`),
//!
//!     V(x) = 2 r3 / (r1 + r2 x + r3 x²).
//!
//! The resonances are the eigenvalues of the spatial operator
//! `φ'' − V(x) φ = λ φ` on `x ∈ [0, L]`. This is the FTM "Method A": solve the
//! Sturm–Liouville problem numerically once (a symmetric tridiagonal eigenproblem
//! on a grid), take modal wavenumbers `k_n = √(−λ_n)`, and play them as a bank of
//! damped sinusoids. `V(x)` acts as a high-pass barrier — the horn cutoff: low
//! modes are pushed up / evanescent, high modes propagate freely.
//!
//! Boundary conditions here are open (Dirichlet on ψ) at both ends, giving a full
//! harmonic-style series that the flare then reshapes. The three radius
//! coefficients `r1, r2, r3` are the scalars you dial in.

use serde::{Deserialize, Serialize};

use super::{strike_amplitude, unbounded_slider, FtmModel, ModeBuffer, TICK_RATE};

const TWO_PI: f64 = std::f64::consts::TAU;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebsterHorn {
    /// Bore radius polynomial r(x) = r1 + r2·x + r3·x²  (throat, taper, flare).
    pub r1: f32,
    pub r2: f32,
    pub r3: f32,
    /// Column length L.
    pub length: f32,
    /// Wave speed c (m/s); sets absolute pitch in physical-pitch mode.
    pub wave_speed: f32,
    /// Excitation point along the bore, 0 (throat) .. 1 (bell).
    pub blow_pos: f32,
    /// Number of modes summed.
    pub depth: usize,
    /// Grid resolution for the eigensolve (higher = more accurate, costlier).
    pub resolution: usize,
    /// Uniform damping d1.
    pub damping: f32,
    /// Frequency-dependent damping d3 (negative = high modes decay faster).
    pub freq_dep_damping: f32,
    pub damp_period: f32,
    pub play_magnitude: f32,
    pub max_magnitude: f32,
    /// If true the key sets the pitch (fundamental resonance on the played note).
    pub key_tracks_pitch: bool,
}

impl Default for WebsterHorn {
    fn default() -> Self {
        Self {
            r1: 1.0,
            r2: 0.0,
            r3: 3.0,
            length: 1.0,
            wave_speed: 343.0,
            blow_pos: 0.12,
            depth: 24,
            resolution: 80,
            damping: 2.0,
            freq_dep_damping: -0.05,
            damp_period: 100.0,
            play_magnitude: 0.0,
            max_magnitude: 2500.0,
            key_tracks_pitch: true,
        }
    }
}

impl FtmModel for WebsterHorn {
    fn id(&self) -> &'static str {
        "webster_horn"
    }

    fn display_name(&self) -> &'static str {
        "Quadratic Webster Horn"
    }

    fn description(&self) -> &'static str {
        "A flaring air column (Webster's horn equation); r1/r2/r3 shape the bore."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        let amp_strike = strike_amplitude(vel, self.play_magnitude, self.max_magnitude);
        if amp_strike <= 0.0 {
            return;
        }

        let l = self.length.max(1e-3) as f64;
        let (r1, r2, r3) = (self.r1 as f64, self.r2 as f64, self.r3 as f64);
        // Grid: n intervals, m = n-1 interior unknowns (Dirichlet at both ends).
        let n = self.resolution.clamp(16, 400);
        let m = n - 1;
        let h = l / n as f64;
        let inv_h2 = 1.0 / (h * h);

        let mut diag = vec![0.0f64; m];
        let mut off = vec![0.0f64; m];
        let mut z: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let mut row = vec![0.0f64; m];
                row[i] = 1.0;
                row
            })
            .collect();
        for j in 0..m {
            let x = (j + 1) as f64 * h;
            let r = r1 + r2 * x + r3 * x * x;
            let v = if r.abs() > 1e-9 { 2.0 * r3 / r } else { 0.0 };
            diag[j] = -2.0 * inv_h2 - v;
            off[j] = inv_h2;
        }

        if !tqli(&mut diag, &mut off, &mut z) {
            return; // solver failed to converge (pathological params)
        }

        // Collect propagating modes (λ < 0): wavenumber k = √(−λ) and the
        // eigenvector value at the blow point (excitation weight).
        let blow_idx = ((self.blow_pos.clamp(0.0, 1.0) as f64) * (m as f64 - 1.0)).round() as usize;
        let blow_idx = blow_idx.min(m - 1);
        let mut modes: Vec<(f64, f64)> = Vec::with_capacity(m);
        for j in 0..m {
            if diag[j] < -1e-9 {
                let k = (-diag[j]).sqrt();
                modes.push((k, z[blow_idx][j]));
            }
        }
        if modes.is_empty() {
            return;
        }
        modes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let k0 = modes[0].0;
        let c = self.wave_speed as f64;
        let damp_per = self.damp_period.max(1e-3) as f64;
        let d1 = self.damping as f64;
        let d3 = self.freq_dep_damping as f64;
        let nyq = sr as f64 * 0.45;

        // First pass: frequency + weight, tracking the amplitude sum for normalizing.
        let mut kept: Vec<(f64, f64, f64)> = Vec::new(); // (freq, weight, decay)
        let mut amp_sum = 0.0f64;
        for &(k, w) in modes.iter().take(self.depth.clamp(1, super::MAX_MODES)) {
            let kr = k / k0; // harmonic ratio (geometry, not scale, dependent)
            let freq = if self.key_tracks_pitch {
                freq_hz as f64 * kr
            } else {
                c * k / TWO_PI
            };
            if freq >= nyq || freq <= 0.0 {
                continue;
            }
            // Decay mirrors the other FTM models but keyed on the harmonic ratio
            // so d1/d3 behave the same regardless of bore length.
            let sigma = (d3 * kr * kr - d1) / 2.0;
            let decay = -sigma * TICK_RATE as f64 / (damp_per * damp_per);
            kept.push((freq, w, decay));
            amp_sum += w.abs();
        }
        if kept.is_empty() {
            return;
        }
        let norm = if amp_sum > 1e-9 {
            amp_strike as f64 / amp_sum
        } else {
            amp_strike as f64
        };
        for (freq, w, decay) in kept {
            out.push(freq as f32, (w * norm) as f32, decay as f32);
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Bore  r(x) = r1 + r2·x + r3·x²");
        changed |= ui
            .add(unbounded_slider(&mut self.r1, 0.01..=10.0, "r1  (throat)"))
            .on_hover_text("Throat radius (constant term).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.r2, -10.0..=10.0, "r2  (taper)"))
            .on_hover_text("Linear taper.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.r3, -5.0..=20.0, "r3  (flare)"))
            .on_hover_text("Quadratic flare — the horn's cutoff comes from this.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.length, 0.1..=8.0, "Length (L)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.blow_pos, 0.0..=1.0, "Blow position"))
            .on_hover_text("Excitation point along the bore (throat → bell); shapes the spectrum.")
            .changed();

        ui.add_space(6.0);
        ui.strong("Modes / damping");
        changed |= ui
            .add(unbounded_slider(&mut self.depth, 1..=super::MAX_MODES, "Modes (DEPTH)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.resolution, 16..=400, "Resolution"))
            .on_hover_text("Eigensolve grid size (accuracy vs. cost at note-on).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.damping, -50.0..=200.0, "Damping (d1)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.freq_dep_damping, -5.0..=2.0, "Freq-dep damping (d3)"))
            .changed();

        ui.add_space(6.0);
        ui.strong("Timing / velocity");
        changed |= ui
            .add(unbounded_slider(&mut self.wave_speed, 100.0..=1000.0, "Wave speed c (m/s)"))
            .on_hover_text("Sets absolute pitch when the key doesn't track.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.damp_period, 1.0..=1000.0, "DAMP_PERIOD"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.play_magnitude, 0.0..=2500.0, "PLAY_MAGNITUDE"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.max_magnitude, 1.0..=5000.0, "MAX_MAGNITUDE"))
            .changed();
        changed |= ui
            .checkbox(&mut self.key_tracks_pitch, "Key tracks pitch")
            .changed();
        changed
    }

    fn box_clone(&self) -> Box<dyn FtmModel> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

#[inline]
fn pythag(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs(), b.abs());
    if a > b {
        a * (1.0 + (b / a).powi(2)).sqrt()
    } else if b == 0.0 {
        0.0
    } else {
        b * (1.0 + (a / b).powi(2)).sqrt()
    }
}

/// Eigenvalues and eigenvectors of a symmetric tridiagonal matrix by the QL
/// algorithm with implicit shifts (Numerical Recipes `tqli`).
///
/// `d`: diagonal (length n) → eigenvalues on output.
/// `e`: off-diagonal (length n); `e[i]` connects `d[i-1]` and `d[i]` — destroyed.
/// `z`: n×n identity on input → column `j` is the eigenvector for `d[j]`.
/// Returns false if it fails to converge.
fn tqli(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>]) -> bool {
    let n = d.len();
    if n == 0 {
        return true;
    }
    if n == 1 {
        return true;
    }
    // Renumber so e[i] sits between d[i] and d[i+1].
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut iter = 0;
        loop {
            // Find a small off-diagonal element to split at.
            let mut mm = l;
            while mm < n - 1 {
                let dd = d[mm].abs() + d[mm + 1].abs();
                if e[mm].abs() <= f64::EPSILON * dd {
                    break;
                }
                mm += 1;
            }
            if mm == l {
                break;
            }
            iter += 1;
            if iter > 50 {
                return false;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = pythag(g, 1.0);
            g = d[mm] - d[l] + e[l] / (g + r.copysign(g));
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut broke_zero = false;
            let mut i = mm - 1;
            loop {
                let mut f = s * e[i];
                let b = c * e[i];
                r = pythag(f, g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[mm] = 0.0;
                    broke_zero = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for k in 0..n {
                    f = z[k][i + 1];
                    z[k][i + 1] = s * z[k][i] + c * f;
                    z[k][i] = c * z[k][i] - s * f;
                }
                if i == l {
                    break;
                }
                i -= 1;
            }
            if broke_zero {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[mm] = 0.0;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight tube (V = 0) with Dirichlet ends has k_j ≈ jπ/L; check the
    /// eigensolver recovers that.
    #[test]
    fn eigensolver_matches_straight_tube() {
        let n = 100usize;
        let l = 1.0f64;
        let h = l / n as f64;
        let m = n - 1;
        let mut d = vec![-2.0 / (h * h); m];
        let mut e = vec![1.0 / (h * h); m];
        let mut z: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let mut row = vec![0.0; m];
                row[i] = 1.0;
                row
            })
            .collect();
        assert!(tqli(&mut d, &mut e, &mut z));
        let mut ks: Vec<f64> = d.iter().filter(|&&x| x < 0.0).map(|&x| (-x).sqrt()).collect();
        ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Fundamental ~ π/L, second ~ 2π/L.
        let pi = std::f64::consts::PI;
        assert!((ks[0] - pi).abs() < 0.05, "k1 ~ pi, got {}", ks[0]);
        assert!((ks[1] - 2.0 * pi).abs() < 0.1, "k2 ~ 2pi, got {}", ks[1]);
    }

    #[test]
    fn horn_produces_finite_modes_and_tracks_key() {
        let mut buf = ModeBuffer::default();
        let h = WebsterHorn { depth: 12, ..WebsterHorn::default() };
        h.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 2, "horn should have several modes");
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite() && *f > 0.0));
        assert!(buf.decay[..buf.n].iter().all(|d| d.is_finite() && *d >= 0.0));
        assert!((buf.freq[0] - 220.0).abs() < 2.0, "fundamental tracks the key: {}", buf.freq[0]);
        // The flare pushes the second resonance off a pure 2:1 (it's a horn, not a tube).
        let ratio = buf.freq[1] / buf.freq[0];
        assert!(ratio > 1.5, "upper resonance above the fundamental: {ratio}");
    }

    #[test]
    fn stays_finite_with_degenerate_params() {
        let mut buf = ModeBuffer::default();
        let h = WebsterHorn {
            r1: 0.0,
            r2: 0.0,
            r3: 0.0,
            length: 0.0,
            depth: 500,
            resolution: 400,
            damping: -5.0,
            freq_dep_damping: 2.0,
            ..WebsterHorn::default()
        };
        h.excite(110.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite()));
        assert!(buf.amp[..buf.n].iter().all(|a| a.is_finite()));
    }
}

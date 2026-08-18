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
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Boundary conditions at the two ends of the bore.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Boundary {
    /// Open at both ends (Dirichlet) — a full harmonic-style series.
    Open,
    /// Closed throat (mouthpiece) + open bell — brass-like. A straight tube then
    /// gives odd harmonics; the flare fills the series back in.
    Brass,
}

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
    /// End conditions: open both ends, or a closed (brass) mouthpiece.
    pub boundary: Boundary,
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
            boundary: Boundary::Open,
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
        let n = self.resolution.clamp(16, 512);
        let h = l / n as f64;
        let inv_h2 = 1.0 / (h * h);
        let v_at = |x: f64| {
            let r = r1 + r2 * x + r3 * x * x;
            if r.abs() > 1e-9 {
                2.0 * r3 / r
            } else {
                0.0
            }
        };

        // Build the discretized operator φ'' − V(x)φ. The bell (x = L) is always
        // open (Dirichlet). The throat is either open (Open) or closed (Brass),
        // which changes the unknown set and the first row.
        //
        //  * Open:  unknowns are interior nodes i = 1..n-1 (x = i·h).
        //  * Brass: the throat node i = 0 is an unknown with a Robin condition
        //    ψ'(0) = (r'(0)/r(0))·ψ(0) = (r2/r1)·ψ(0). The one-sided second
        //    difference makes row 0 asymmetric; a diagonal similarity restores
        //    symmetry (off-diagonal → √2/h², eigenvector[0] scales by √2).
        let brass = self.boundary == Boundary::Brass;
        let (m, node_start) = if brass { (n, 0usize) } else { (n - 1, 1usize) };
        let mut diag = vec![0.0f64; m];
        let mut off = vec![inv_h2; m];
        for j in 0..m {
            let x = (j + node_start) as f64 * h;
            diag[j] = -2.0 * inv_h2 - v_at(x);
        }
        if brass {
            let beta = if r1.abs() > 1e-9 { r2 / r1 } else { 0.0 }; // r'(0)/r(0)
            diag[0] = -2.0 * (1.0 + h * beta) * inv_h2 - v_at(0.0);
            if m > 1 {
                off[1] = SQRT_2 * inv_h2; // symmetrized (0,1) off-diagonal
            }
        }

        // Eigenvalues only — O(m²), so high resolution stays cheap. Keep `diag`
        // and `off` intact for inverse iteration below.
        let mut ev_d = diag.clone();
        let mut ev_e = off.clone();
        if !tqli(&mut ev_d, &mut ev_e, None) {
            return; // failed to converge (pathological params)
        }

        // Excitation point → nearest unknown node (throat node gets the √2 scale
        // in the brass case, from the similarity transform above).
        let blow_x = (self.blow_pos.clamp(0.0, 1.0) as f64) * l;
        let blow_idx = ((blow_x / h).round() as i64 - node_start as i64)
            .clamp(0, m as i64 - 1) as usize;
        let throat_scale = if brass && blow_idx == 0 { SQRT_2 } else { 1.0 };

        // Propagating modes (λ < 0), lowest wavenumber first (λ nearest 0). For
        // each kept mode, recover just its blow-point weight by inverse iteration
        // (a couple of tridiagonal solves) — far cheaper than all eigenvectors.
        let mut lambdas: Vec<f64> = ev_d.into_iter().filter(|&x| x < -1e-9).collect();
        if lambdas.is_empty() {
            return;
        }
        lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending λ ⇒ ascending k
        let take = self.depth.clamp(1, super::MAX_MODES).min(lambdas.len());
        let mut modes: Vec<(f64, f64)> = Vec::with_capacity(take);
        for &lambda in lambdas.iter().take(take) {
            let k = (-lambda).sqrt();
            let w = eigenvector_at(&diag, &off, lambda, blow_idx) * throat_scale;
            modes.push((k, w));
        }

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
        egui::ComboBox::from_label("Ends")
            .selected_text(match self.boundary {
                Boundary::Open => "Open (both ends)",
                Boundary::Brass => "Brass (closed throat)",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.boundary, Boundary::Open, "Open (both ends)")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.boundary, Boundary::Brass, "Brass (closed throat)")
                    .changed();
            });
        ui.add_space(4.0);
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
            .add(unbounded_slider(&mut self.resolution, 16..=512, "Resolution"))
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

/// Solve a symmetric tridiagonal system `A·x = rhs`, where `A` has diagonal
/// `diag` and off-diagonals `off` (`off[j]` between rows `j-1` and `j`), via the
/// Thomas algorithm. Returns false on a zero pivot.
fn tridiag_solve(off: &[f64], diag: &[f64], rhs: &[f64], out: &mut [f64]) -> bool {
    let n = diag.len();
    if n == 0 {
        return true;
    }
    let mut cp = vec![0.0f64; n];
    let mut dp = vec![0.0f64; n];
    let mut b = diag[0];
    if b.abs() < 1e-300 {
        return false;
    }
    cp[0] = if n > 1 { off[1] / b } else { 0.0 };
    dp[0] = rhs[0] / b;
    for j in 1..n {
        b = diag[j] - off[j] * cp[j - 1];
        if b.abs() < 1e-300 {
            b = 1e-300;
        }
        cp[j] = if j < n - 1 { off[j + 1] / b } else { 0.0 };
        dp[j] = (rhs[j] - off[j] * dp[j - 1]) / b;
    }
    out[n - 1] = dp[n - 1];
    for j in (0..n - 1).rev() {
        out[j] = dp[j] - cp[j] * out[j + 1];
    }
    true
}

/// The component at `node` of the (unit-norm) eigenvector for eigenvalue
/// `lambda`, found by a couple of inverse-iteration steps on the shifted
/// tridiagonal `(A − λ')`. Cheap: O(m) per step vs. O(m²) for a full solve.
fn eigenvector_at(diag: &[f64], off: &[f64], lambda: f64, node: usize) -> f64 {
    let m = diag.len();
    if m == 0 {
        return 0.0;
    }
    let shift = lambda + lambda.abs().max(1.0) * 1e-9 + 1e-12;
    let dsh: Vec<f64> = diag.iter().map(|&d| d - shift).collect();
    let mut x = vec![1.0 / (m as f64).sqrt(); m];
    let mut y = vec![0.0f64; m];
    for _ in 0..2 {
        if !tridiag_solve(off, &dsh, &x, &mut y) {
            break;
        }
        let norm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-300 {
            break;
        }
        for (xi, yi) in x.iter_mut().zip(y.iter()) {
            *xi = yi / norm;
        }
    }
    x[node]
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
/// `z`: `Some(n×n identity)` → columns become eigenvectors; `None` computes
/// eigenvalues only (O(n²), used when eigenvectors are found separately).
/// Returns false if it fails to converge.
fn tqli(d: &mut [f64], e: &mut [f64], mut z: Option<&mut [Vec<f64>]>) -> bool {
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
                let f = s * e[i];
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
                if let Some(zz) = z.as_mut() {
                    for k in 0..n {
                        let f2 = zz[k][i + 1];
                        zz[k][i + 1] = s * zz[k][i] + c * f2;
                        zz[k][i] = c * zz[k][i] - s * f2;
                    }
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
        assert!(tqli(&mut d, &mut e, Some(&mut z)));
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
    fn brass_straight_tube_is_odd_harmonics() {
        // Closed throat + open bell, straight tube (r3 = 0 → V = 0, r2 = 0 →
        // Neumann throat) gives the odd-harmonic series 1 : 3 : 5.
        let mut buf = ModeBuffer::default();
        let h = WebsterHorn {
            boundary: Boundary::Brass,
            r1: 1.0,
            r2: 0.0,
            r3: 0.0,
            depth: 6,
            resolution: 200,
            blow_pos: 0.0,
            ..WebsterHorn::default()
        };
        h.excite(200.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n >= 3, "expected several modes");
        assert!((buf.freq[0] - 200.0).abs() < 2.0, "fundamental tracks key: {}", buf.freq[0]);
        let ratio = buf.freq[1] / buf.freq[0];
        assert!((ratio - 3.0).abs() < 0.15, "closed-open 2nd mode ~3×, got {ratio}");
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

//! Pure Plate — a genuine FTM solve of a free circular plate.
//!
//! Unlike the pragmatic `Cymbal` model (which borrows the drum membrane's modes
//! and cranks the stiffness), this solves the real physics from scratch. A thin
//! plate obeys the Kirchhoff–Love equation
//!
//! ```text
//!     ∂²w/∂t² = -(D/ρh) ∇⁴w        (bending stiffness, the biharmonic ∇⁴)
//! ```
//!
//! Separating time and looking for modes `w = W(r,θ)·e^{iωt}` turns this into
//! `∇⁴W = β⁴W`, with `ω = β²·√(D/ρh)`. The biharmonic factors as
//! `(∇²+β²)(∇²-β²)W = 0`, so on a solid disk (regular at the centre) the mode
//! shapes are
//!
//! ```text
//!     W(r,θ) = [A·Jₙ(βr) + B·Iₙ(βr)] · cos(nθ)
//! ```
//!
//! a Bessel `Jₙ` part plus a *modified* Bessel `Iₙ` part. A **free edge** imposes
//! zero bending moment `Mᵣ` and zero Kirchhoff shear `Vᵣ` at `r = a`; that 2×2
//! homogeneous system has a non-trivial solution only when its determinant
//! vanishes. Reducing the moment/shear operators against the Bessel ODEs gives
//! closed forms (see [`edge_terms`]); the roots `λ = βa` of the determinant are
//! the plate's eigenvalues, and every mode frequency is `∝ λ²`.
//!
//! Those roots are the famous inharmonic free-plate ratios `1 : 1.73 : 2.33 :
//! 3.9 : …` — the sound of a struck cymbal/gong, derived, not faked. The `λ`
//! ratios depend only on Poisson's ratio (not the size), so a bigger plate is
//! simply lower with the same timbre. Purely modal: no noise wash, so you hear
//! the plate solution on its own.

use serde::{Deserialize, Serialize};

use super::{FtmModel, ModeBuffer, MAX_MODES, REF_PITCH_HZ};

/// Nodal-diameter orders to search (0 = breathing, 1 = rocking, …).
const N_MAX: usize = 18;
/// Largest `λ = βa` to search — sets how high the mode series goes.
const LAM_MAX: f64 = 38.0;
/// Scan step for root bracketing (free-plate roots are ~π apart per order).
const SCAN_STEP: f64 = 0.05;
/// Skip everything below this `λ`; the lowest real plate mode is (2,0) at
/// λ≈2.29, so this only discards the rigid-body (λ→0) modes.
const LAM_MIN: f64 = 1.0;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PurePlate {
    /// Poisson's ratio of the metal — the only *timbre* control (sets the
    /// inharmonic ratios). ~0.3 for steel/bronze.
    pub poisson: f32,
    /// Overall ring time in seconds (the fundamental's decay).
    pub decay_time: f32,
    /// How much faster high modes decay than the fundamental (brightness fade).
    pub hf_damp: f32,
    /// Strike position, 0 = centre (only breathing modes) → 1 = edge (all modes).
    pub strike_pos: f32,
    /// How many modes to sum (lowest-frequency first).
    pub modes: usize,
    /// If true the key sets the pitch (fundamental → played note).
    pub key_tracks_pitch: bool,
}

impl Default for PurePlate {
    fn default() -> Self {
        Self {
            poisson: 0.33,
            decay_time: 4.0,
            hf_damp: 0.4,
            strike_pos: 0.7,
            modes: 90,
            key_tracks_pitch: true,
        }
    }
}

impl FtmModel for PurePlate {
    fn id(&self) -> &'static str {
        "pure_plate"
    }

    fn display_name(&self) -> &'static str {
        "Pure Plate"
    }

    fn description(&self) -> &'static str {
        "Free circular plate solved from the Kirchhoff biharmonic — real cymbal/gong modes."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        let nu = self.poisson.clamp(-0.9, 0.49) as f64;

        // Solve the free-plate eigenvalues λ (geometry-independent).
        let roots = plate_eigenvalues(nu);
        if roots.is_empty() {
            return;
        }
        let lam0 = roots[0].1; // fundamental = smallest λ

        // The fundamental maps to the played note (or a fixed reference).
        let played = if self.key_tracks_pitch { freq_hz } else { REF_PITCH_HZ };
        let nyquist = 0.45 * sr;
        let base_rate = 1.0 / self.decay_time.max(0.05);
        let want = self.modes.clamp(1, MAX_MODES);
        let rho_s = self.strike_pos.clamp(0.0, 1.0) as f64;

        let mut weights: Vec<f32> = Vec::with_capacity(want);
        let mut max_w = 0.0f32;

        for &(n, lam) in &roots {
            if out.n >= want {
                break;
            }
            let f = played * (lam * lam / (lam0 * lam0)) as f32;
            if f >= nyquist || !f.is_finite() {
                continue;
            }
            // Modal excitation weight: the mode shape sampled at the strike
            // radius, normalized by its own peak so each mode couples in [0,1].
            let w = strike_weight(nu, n, lam, rho_s) as f32;
            if !w.is_finite() {
                continue;
            }
            // Highs decay faster: decay grows with the frequency ratio.
            let decay = (base_rate * (1.0 + self.hf_damp * (f / played - 1.0))).max(0.02);
            out.push(f, w, decay);
            weights.push(w);
            max_w = max_w.max(w.abs());
        }

        // Normalize the loudest mode to unit amplitude, then apply velocity.
        if max_w > 1e-9 {
            out.scale_amps(vel / max_w);
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        use super::unbounded_slider;
        let mut changed = false;
        ui.strong("Plate (free circular, Kirchhoff)");
        changed |= ui
            .add(unbounded_slider(&mut self.poisson, -0.5..=0.49, "Poisson ratio (timbre)"))
            .on_hover_text("Sets the inharmonic mode ratios. ~0.3 for metal.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.decay_time, 0.1..=12.0, "Ring time (s)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.hf_damp, 0.0..=3.0, "HF damping"))
            .on_hover_text("How much faster the highs die (brightness fade).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.strike_pos, 0.0..=1.0, "Strike (centre → edge)"))
            .on_hover_text("Centre excites only the breathing modes; the edge excites all.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.modes, 1..=MAX_MODES, "Modes"))
            .changed();
        changed |= ui.checkbox(&mut self.key_tracks_pitch, "Key tracks pitch").changed();
        changed
    }

    fn box_clone(&self) -> Box<dyn FtmModel> {
        Box::new(self.clone())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

// ---------------------------------------------------------------------------
// The free-plate eigenproblem
// ---------------------------------------------------------------------------

/// The four reduced free-edge boundary terms at `λ` for nodal-diameter order
/// `n`: `(M_J, M_I, S_J, S_I)`, where `M` is the bending-moment row and `S` the
/// Kirchhoff-shear row, `_J`/`_I` the Bessel-`J` / modified-Bessel-`I` columns.
/// The `I` terms are returned **scaled by `e^{-λ}`** (via `Ĩ = e^{-λ}I`); since
/// the determinant is homogeneous this leaves its roots unchanged and avoids the
/// exponential overflow of `Iₙ`.
#[inline]
fn edge_terms(nu: f64, n: usize, lam: f64) -> (f64, f64, f64, f64) {
    let nn = (n * n) as f64;
    let a = 1.0 - nu;
    let nf = n as f64;

    let (jn, jd) = bessel_j_nd(n, lam);
    let (it, itd) = bessel_i_scaled_nd(n, lam);

    // Mᵣ = 0 :  M_J·A + M_I·B = 0
    let m_j = -a * lam * jd - (lam * lam - nn * a) * jn;
    let m_i = -a * lam * itd + (lam * lam + nn * a) * it; // ·e^{-λ}
    // Vᵣ = 0 :  S_J·A + S_I·B = 0
    let s_j = -lam * (lam * lam + nn * a) * jd + nn * a * jn;
    let s_i = lam * (lam * lam - nn * a) * itd + nn * a * it; // ·e^{-λ}
    let _ = nf;
    (m_j, m_i, s_j, s_i)
}

/// The free-plate characteristic determinant at `λ` (scaled by `e^{-λ}`; roots
/// preserved).
#[inline]
fn plate_det(nu: f64, n: usize, lam: f64) -> f64 {
    let (m_j, m_i, s_j, s_i) = edge_terms(nu, n, lam);
    m_j * s_i - m_i * s_j
}

/// All plate eigenvalues `(n, λ)` with `λ ≤ LAM_MAX`, sorted ascending by `λ`
/// (so the first is the fundamental).
fn plate_eigenvalues(nu: f64) -> Vec<(usize, f64)> {
    let mut roots: Vec<(usize, f64)> = Vec::new();
    for n in 0..=N_MAX {
        let mut lo = LAM_MIN;
        let mut d_lo = plate_det(nu, n, lo);
        let mut lam = LAM_MIN + SCAN_STEP;
        while lam <= LAM_MAX {
            let d_hi = plate_det(nu, n, lam);
            if d_lo == 0.0 {
                roots.push((n, lo));
            } else if (d_lo < 0.0) != (d_hi < 0.0) {
                roots.push((n, bisect(nu, n, lo, lam, d_lo, d_hi)));
            }
            lo = lam;
            d_lo = d_hi;
            lam += SCAN_STEP;
        }
    }
    roots.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    roots
}

/// Refine a bracketed root of `plate_det` by bisection.
fn bisect(nu: f64, n: usize, mut lo: f64, mut hi: f64, mut d_lo: f64, _d_hi: f64) -> f64 {
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let d_mid = plate_det(nu, n, mid);
        if (d_lo < 0.0) != (d_mid < 0.0) {
            hi = mid;
        } else {
            lo = mid;
            d_lo = d_mid;
        }
    }
    0.5 * (lo + hi)
}

/// Excitation weight for mode `(n, λ)` struck at normalized radius `rho_s`
/// (`r/a`): the mode's radial shape at the strike point, divided by its own peak
/// over the disk (so the result is in `[0, 1]`). A centre strike (`rho_s = 0`)
/// only couples the `n = 0` breathing modes, exactly as a real plate does.
fn strike_weight(nu: f64, n: usize, lam: f64, rho_s: f64) -> f64 {
    let (m_j, m_i, _s_j, _s_i) = edge_terms(nu, n, lam);
    // Radial shape (up to a global e^{λ} factor, cancelled by the peak
    // normalization): R̂(ρ) = M_I·Jₙ(λρ) − M_J·e^{-λ(1-ρ)}·Ĩₙ(λρ).
    let shape = |rho: f64| -> f64 {
        let x = lam * rho;
        let jn = bessel_j_nd(n, x).0;
        let it = bessel_i_scaled_nd(n, x).0;
        m_i * jn - m_j * (-lam * (1.0 - rho)).exp() * it
    };
    let mut peak = 0.0f64;
    let samples = 24;
    for i in 0..=samples {
        let rho = i as f64 / samples as f64;
        peak = peak.max(shape(rho).abs());
    }
    if peak < 1e-12 {
        return 0.0;
    }
    (shape(rho_s).abs() / peak).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Bessel helpers (self-contained, f64)
// ---------------------------------------------------------------------------

/// `Jₙ(x)` and `Jₙ'(x)`.
fn bessel_j_nd(n: usize, x: f64) -> (f64, f64) {
    if x.abs() < 1e-12 {
        // Jₙ(0)=0 except J₀(0)=1; J₀'(0)=0, J₁'(0)=½, else 0.
        let j = if n == 0 { 1.0 } else { 0.0 };
        let d = if n == 1 { 0.5 } else { 0.0 };
        return (j, d);
    }
    let hi = n.max(1);
    let j = bessel_j_all(hi, x);
    let jn = j[n];
    let jd = if n == 0 { -j[1] } else { j[n - 1] - (n as f64 / x) * j[n] };
    (jn, jd)
}

/// Scaled `Ĩₙ(x) = e^{-x}Iₙ(x)` and its derivative `e^{-x}Iₙ'(x)`.
fn bessel_i_scaled_nd(n: usize, x: f64) -> (f64, f64) {
    if x.abs() < 1e-12 {
        let i = if n == 0 { 1.0 } else { 0.0 };
        let d = if n == 1 { 0.5 } else { 0.0 };
        return (i, d);
    }
    let hi = n.max(1);
    let it = bessel_i_scaled_all(hi, x);
    let itn = it[n];
    let itd = if n == 0 { it[1] } else { it[n - 1] - (n as f64 / x) * it[n] };
    (itn, itd)
}

/// `J_0..=J_nmax` at `x` via Miller's downward recurrence, normalized by the
/// identity `J₀ + 2(J₂ + J₄ + …) = 1`.
fn bessel_j_all(nmax: usize, x: f64) -> Vec<f64> {
    let ax = x.abs();
    let start = nmax.max(ax.ceil() as usize) + 20;
    let mut jkp1 = 0.0f64; // J_{k+1}
    let mut jk = 1.0f64; // J_k, k = start
    let mut store = vec![0.0f64; nmax + 1];
    let mut norm = 0.0f64;
    let mut k = start as isize;
    loop {
        if (k as usize) <= nmax {
            store[k as usize] = jk;
        }
        if k == 0 {
            norm += jk;
        } else if k % 2 == 0 {
            norm += 2.0 * jk;
        }
        if k == 0 {
            break;
        }
        let jkm1 = (2.0 * k as f64 / x) * jk - jkp1;
        jkp1 = jk;
        jk = jkm1;
        k -= 1;
        // Guard against overflow far from the origin.
        if jk.abs() > 1e250 {
            let s = 1e250;
            jk /= s;
            jkp1 /= s;
            norm /= s;
            for v in store.iter_mut() {
                *v /= s;
            }
        }
    }
    if norm != 0.0 {
        for v in store.iter_mut() {
            *v /= norm;
        }
    }
    store
}

/// `Ĩ_0..=Ĩ_nmax` (scaled `e^{-x}Iₙ`) at `x` via downward recurrence, normalized
/// by the identity `I₀ + 2(I₁ + I₂ + …) = eˣ` (so the eˣ cancels the scaling).
fn bessel_i_scaled_all(nmax: usize, x: f64) -> Vec<f64> {
    let ax = x.abs();
    let start = nmax.max(ax.ceil() as usize) + 20;
    let mut ikp1 = 0.0f64; // î_{k+1}
    let mut ik = 1.0f64; // î_k, k = start
    let mut store = vec![0.0f64; nmax + 1];
    let mut norm = 0.0f64;
    let mut k = start as isize;
    loop {
        if (k as usize) <= nmax {
            store[k as usize] = ik;
        }
        if k == 0 {
            norm += ik;
        } else {
            norm += 2.0 * ik;
        }
        if k == 0 {
            break;
        }
        let ikm1 = (2.0 * k as f64 / x) * ik + ikp1;
        ikp1 = ik;
        ik = ikm1;
        k -= 1;
        if ik.abs() > 1e250 {
            let s = 1e250;
            ik /= s;
            ikp1 /= s;
            norm /= s;
            for v in store.iter_mut() {
                *v /= s;
            }
        }
    }
    if norm != 0.0 {
        for v in store.iter_mut() {
            *v /= norm;
        }
    }
    store
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n0_first_root_matches_leissa() {
        // Free plate, ν=0.33: the first breathing (n=0) root sits at λ≈3.01
        // (λ²≈9.08 in Leissa's table). This validates the whole determinant.
        let mut lo = 1.0;
        let mut d_lo = plate_det(0.33, 0, lo);
        let mut lam = 1.0 + SCAN_STEP;
        let mut root = None;
        while lam <= 8.0 {
            let d = plate_det(0.33, 0, lam);
            if (d_lo < 0.0) != (d < 0.0) {
                root = Some(bisect(0.33, 0, lo, lam, d_lo, d));
                break;
            }
            lo = lam;
            d_lo = d;
            lam += SCAN_STEP;
        }
        let r = root.expect("n=0 root found");
        assert!((r - 3.014).abs() < 0.03, "n=0 first root ≈3.01, got {r}");
    }

    #[test]
    fn fundamental_is_2_0_and_ratios_are_inharmonic() {
        let roots = plate_eigenvalues(0.33);
        assert!(roots.len() > 10, "found a decent mode set, got {}", roots.len());
        // The global fundamental is the (2,0) "potato-chip" mode, λ≈2.29.
        let (n0, lam0) = roots[0];
        assert_eq!(n0, 2, "fundamental has 2 nodal diameters");
        assert!((lam0 - 2.29).abs() < 0.05, "fundamental λ≈2.29, got {lam0}");
        // Second partial is the breathing (0,1) mode; freq ratio = (λ/λ0)² ≈ 1.73.
        let (_, lam1) = roots[1];
        let ratio = (lam1 * lam1) / (lam0 * lam0);
        assert!((ratio - 1.73).abs() < 0.08, "second partial ≈1.73×, got {ratio}");
    }

    #[test]
    fn excite_produces_inharmonic_modes() {
        let mut buf = ModeBuffer::default();
        PurePlate::default().excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 8, "several modes, got {}", buf.n);
        assert!(!buf.sustain, "struck, not sustained");
        assert_eq!(buf.noise_level, 0.0, "pure plate has no noise wash");
        assert!((buf.freq[0] - 220.0).abs() < 1.0, "fundamental tracks the key");
        // Not a harmonic series: the 2nd partial is far from 2×.
        let r = buf.freq[1] / buf.freq[0];
        assert!(r > 1.5 && r < 1.95, "inharmonic 2nd partial, got {r}");
        assert!(buf.decay[..buf.n].iter().all(|&d| d > 0.0), "all modes decay");
    }

    #[test]
    fn centre_strike_excites_only_breathing_modes() {
        // Struck dead centre, only the n=0 (axisymmetric) modes couple, so the
        // fundamental drops out and far fewer modes survive than an edge strike.
        let mut edge = ModeBuffer::default();
        PurePlate { strike_pos: 1.0, ..PurePlate::default() }.excite(220.0, 1.0, 48_000.0, &mut edge);
        let mut centre = ModeBuffer::default();
        PurePlate { strike_pos: 0.0, ..PurePlate::default() }.excite(220.0, 1.0, 48_000.0, &mut centre);
        let edge_loud = edge.amp[..edge.n].iter().filter(|&&a| a > 1e-3).count();
        let centre_loud = centre.amp[..centre.n].iter().filter(|&&a| a > 1e-3).count();
        assert!(centre_loud < edge_loud, "centre couples fewer modes ({centre_loud} < {edge_loud})");
    }
}

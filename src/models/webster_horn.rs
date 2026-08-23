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

use super::{strike_amplitude, unbounded_slider, FtmModel, ModeBuffer};

const TWO_PI: f64 = std::f64::consts::TAU;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// How the keyboard drives the horn.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayMode {
    /// The whole instrument resizes so its fundamental resonance lands on the
    /// played key — chromatic, like a valve/slide instrument.
    Chromatic,
    /// A fixed tube: the key selects (overblows) the nearest natural resonance,
    /// and the tone is that resonance's harmonic series filtered by the tube —
    /// a bugle/natural horn. Snaps pitch to the resonance ladder; the fixed
    /// cutoff makes low notes dark/resonant and high notes bright.
    Overblow,
    /// Key-tracked overblow: like a real brass player — overblow to the harmonic
    /// just above the key, then adjust the bore length (valve/slide) by the small
    /// amount that tunes that harmonic exactly onto the key. Chromatic tracking
    /// with the overblown, fixed-formant timbre (formant motion stays bounded to
    /// the gap between adjacent harmonics rather than sweeping the whole range).
    OverblowTracked,
}

/// Choose a fingering for a key-tracked overblow, like a real brass player.
///
/// There are `valve_steps + 1` bore lengths — the open tube plus one per added
/// semitone of valve/slide tubing (a Bb trumpet's valves give 0..6 semitones →
/// 7 lengths). `res` are the open bore's resonances (Hz), `tune` anchors them.
/// Among every (length, harmonic) pair we pick the pitch nearest the key
/// (preferring the shortest tube on ties — the conventional fingering), then
/// return the scale to apply to `res` so that harmonic lands *exactly* on the
/// key (the fingering choice plus a small lip/tuning-slide nudge).
fn overblow_fingering(target: f64, res: &[f64], tune: f64, valve_steps: u32, microtune: bool) -> f64 {
    let lt = target.max(1.0).ln();
    let mut best_dist = f64::INFINITY;
    let mut best_scale = tune;
    for k in 0..=valve_steps {
        let len = tune * 2f64.powf(-(k as f64) / 12.0); // added tubing lowers pitch
        for &r in res {
            let f = r * len;
            if f <= 0.0 {
                continue;
            }
            let d = (f.ln() - lt).abs();
            if d < best_dist - 1e-9 {
                best_dist = d;
                // Micro-tune: nudge this bore so the harmonic hits the key exactly.
                // Otherwise leave it at the natural harmonic pitch (authentic).
                best_scale = if microtune { len * (target / f) } else { len };
            }
        }
    }
    best_scale
}

/// Wavefront geometry used to build the horn potential.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Wavefront {
    /// Flat cross-sectional discs (classic Webster) — `V = r''/r`.
    Planar,
    /// Curved spherical-cap wavefronts — `V = (√S)''/√S`, cap area
    /// `S = 2π r²/(1 + cos θ)`, flare angle `θ = arctan r'`. More accurate high
    /// partials and bell behaviour, especially where the flare is steep.
    Spherical,
}

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
    /// Keefe viscothermal wall loss (boundary layer): adds ∝ √f · (bore
    /// narrowness) damping — the "warm, stuffed" tone of long narrow tubing.
    #[serde(default)]
    pub visco_loss: f32,
    /// Radiation loss at the bell: adds ∝ f² damping — highs escape the bell
    /// (open, brilliant), lows reflect and sustain.
    #[serde(default)]
    pub radiation: f32,
    pub damp_period: f32,
    pub play_magnitude: f32,
    pub max_magnitude: f32,
    /// If true the key sets the pitch (fundamental resonance on the played note).
    pub key_tracks_pitch: bool,
    /// End conditions: open both ends, or a closed (brass) mouthpiece.
    pub boundary: Boundary,
    /// Wavefront geometry: flat discs, or curved spherical caps.
    #[serde(default = "default_wavefront")]
    pub wavefront: Wavefront,
    /// Chromatic (resize per note) or Overblow (fixed tube, select a resonance).
    #[serde(default = "default_playmode")]
    pub play_mode: PlayMode,
    /// Key-tracked overblow: how many semitones of valve/slide tubing are
    /// available (bore lengths 0..=this). A Bb trumpet's three valves reach 6.
    pub valve_steps: u32,
    /// Key-tracked overblow: the fundamental (Hz) of the *longest* bore — the
    /// tuning anchor. For a Bb trumpet this is concert E2 (82.41), which makes
    /// the open bore's fundamental the pedal Bb2 a tritone above.
    pub overblow_anchor_hz: f32,
    /// Key-tracked overblow: nudge the chosen bore so the note is exactly in
    /// tune (true), or play the natural harmonic pitch — authentic, slightly
    /// off equal-temperament (false).
    pub overblow_microtune: bool,
}

fn default_wavefront() -> Wavefront {
    Wavefront::Planar
}
fn default_playmode() -> PlayMode {
    PlayMode::Chromatic
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
            visco_loss: 0.0,
            radiation: 0.0,
            damp_period: 100.0,
            play_magnitude: 0.0,
            max_magnitude: 2500.0,
            key_tracks_pitch: true,
            boundary: Boundary::Open,
            wavefront: Wavefront::Planar,
            play_mode: PlayMode::Chromatic,
            valve_steps: 6,
            overblow_anchor_hz: 82.41, // concert E2 — longest-bore fundamental
            overblow_microtune: true,
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
        out.sustain = true; // a horn is blown/driven — it holds while played
        let amp_strike = strike_amplitude(vel, self.play_magnitude, self.max_magnitude);
        if amp_strike <= 0.0 {
            return;
        }

        let l = self.length.max(1e-3) as f64;
        let (r1, r2, r3) = (self.r1 as f64, self.r2 as f64, self.r3 as f64);
        let n = self.resolution.clamp(16, 512);
        let h = l / n as f64;
        let inv_h2 = 1.0 / (h * h);

        // The horn potential V(x) and the throat coefficient β = (√area)'/(√area)|₀.
        // Planar: flat discs, V = r''/r = 2r3/r (analytic). Spherical: curved
        // wavefronts of cap area S = 2π r²/(1+cos θ), θ = arctan r', so
        // V = (√S)''/√S computed numerically on the grid.
        let (v_arr, beta): (Vec<f64>, f64) = match self.wavefront {
            Wavefront::Planar => {
                let v = (0..=n)
                    .map(|i| {
                        let x = i as f64 * h;
                        let r = r1 + r2 * x + r3 * x * x;
                        if r.abs() > 1e-9 {
                            2.0 * r3 / r
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let b = if r1.abs() > 1e-9 { r2 / r1 } else { 0.0 };
                (v, b)
            }
            Wavefront::Spherical => {
                // g = √S = √(2π)·r / √(1 + cos θ), cos θ = 1/√(1+r'²).
                let g = |x: f64| {
                    let r = (r1 + r2 * x + r3 * x * x).abs().max(1e-9);
                    let rp = r2 + 2.0 * r3 * x;
                    let cos_t = 1.0 / (1.0 + rp * rp).sqrt();
                    TWO_PI.sqrt() * r / (1.0 + cos_t).sqrt()
                };
                let gvals: Vec<f64> = (0..=n).map(|i| g(i as f64 * h)).collect();
                let v: Vec<f64> = (0..=n)
                    .map(|i| {
                        let gi = gvals[i].max(1e-12);
                        let gpp = if i == 0 {
                            (gvals[0] - 2.0 * gvals[1] + gvals[2]) * inv_h2
                        } else if i == n {
                            (gvals[n] - 2.0 * gvals[n - 1] + gvals[n - 2]) * inv_h2
                        } else {
                            (gvals[i - 1] - 2.0 * gvals[i] + gvals[i + 1]) * inv_h2
                        };
                        gpp / gi
                    })
                    .collect();
                let b = (gvals[1] - gvals[0]) / (h * gvals[0].max(1e-12)); // g'(0)/g(0)
                (v, b)
            }
        };

        // Build the discretized operator φ'' − V(x)φ. The bell (x = L) is always
        // open (Dirichlet). The throat is either open (Open) or closed (Brass),
        // which changes the unknown set and the first row.
        //
        //  * Open:  unknowns are interior nodes i = 1..n-1 (x = i·h).
        //  * Brass: the throat node i = 0 is an unknown with a Robin condition
        //    ψ'(0) = β·ψ(0). The one-sided second difference makes row 0
        //    asymmetric; a diagonal similarity restores symmetry (off-diagonal →
        //    √2/h², eigenvector[0] scales by √2).
        let brass = self.boundary == Boundary::Brass;
        let (m, node_start) = if brass { (n, 0usize) } else { (n - 1, 1usize) };
        let mut diag = vec![0.0f64; m];
        let mut off = vec![inv_h2; m];
        for j in 0..m {
            diag[j] = -2.0 * inv_h2 - v_arr[j + node_start];
        }
        if brass {
            diag[0] = -2.0 * (1.0 + h * beta) * inv_h2 - v_arr[0];
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
        // A finite-difference eigensolve on `n` grid points only resolves its
        // lowest ~n/3 eigenvalues accurately; higher ones saturate against the
        // operator's largest eigenvalue and pile up into a dense, spurious,
        // *buzzy* cluster (measured: on a 512 grid the mode ratios flatten near
        // index ~240). Cap the modes we keep to that resolvable count so a large
        // DEPTH can never sound that numerical garbage. Chromatic sounds these
        // directly; the Overblow paths only use them as a resonance ladder, so
        // for those this just trims an inaudible tail.
        let resolvable = (n / 3).max(1);
        let take = self
            .depth
            .clamp(1, super::MAX_MODES)
            .min(lambdas.len())
            .min(resolvable);
        let mut modes: Vec<(f64, f64)> = Vec::with_capacity(take);
        for &lambda in lambdas.iter().take(take) {
            let k = (-lambda).sqrt();
            let w = eigenvector_at(&diag, &off, lambda, blow_idx) * throat_scale;
            modes.push((k, w));
        }

        let k0 = modes[0].0;
        let c = self.wave_speed as f64;
        let d1 = self.damping as f64;
        let nyq = sr as f64 * 0.45;

        // Keefe viscothermal loss scales with bore narrowness. Use mean(r_bell /
        // r(x)) over the bore — scale-invariant (a cylinder = 1, a flared horn ≫ 1
        // because the throat is narrow), so it works whatever the r-coefficients'
        // absolute size.
        let visco = self.visco_loss as f64;
        let radiation = self.radiation as f64;
        let flare_loss = if visco > 0.0 {
            let r_of = |x: f64| (r1 + r2 * x + r3 * x * x).abs().max(1e-9);
            let r_bell = r_of(l);
            let samples = 32;
            let mut acc = 0.0;
            for i in 0..samples {
                let x = (i as f64 + 0.5) / samples as f64 * l;
                acc += r_bell / r_of(x);
            }
            (acc / samples as f64).max(0.0)
        } else {
            0.0
        };
        // Grounded loss magnitudes (in real 1/s): calibrated so a woodwind bore
        // (visco ≈ 1) rings at Q ≈ 50–100 at the fundamental — the range where a
        // reed self-oscillates cleanly rather than exploding or dying. Wall loss
        // ∝ √f (viscothermal boundary layer, Keefe), radiation loss ∝ f² (the
        // open bell). `damping` is a real per-second base loss (mouthpiece/player).
        const KEEFE_C: f64 = 0.5;
        const RAD_C: f64 = 5.0;
        let keefe_rad = |fh: f64| {
            d1.max(0.0) + KEEFE_C * visco * flare_loss * fh.sqrt() + RAD_C * radiation * (fh * 1e-3).powi(2)
        };

        // --- Overblow family: fixed-tube timbre; Overblow snaps to the resonance
        //     ladder, OverblowTracked tunes the bore so the chosen harmonic lands
        //     exactly on the key (chromatic tracking, overblown tone). ---
        if matches!(self.play_mode, PlayMode::Overblow | PlayMode::OverblowTracked) {
            // Resonances at absolute Hz (fixed geometry).
            let res: Vec<f64> = modes
                .iter()
                .map(|&(k, _)| c * k / TWO_PI)
                .filter(|&f| f > 0.0)
                .collect();
            if res.is_empty() {
                return;
            }
            // Choose the sounding pitch `f_sel` and the resonance ladder `boost`
            // that shapes its harmonics.
            let (f_sel, boost): (f64, Vec<f64>) = if self.play_mode == PlayMode::OverblowTracked
            {
                // Pick a valve fingering (one of the discrete bore lengths) whose
                // harmonic is nearest the key, then land it exactly on the key.
                // `tune` anchors the tuning so the LONGEST bore's fundamental is
                // `overblow_anchor_hz` (e.g. concert E2 for a Bb trumpet, making
                // the open bore a tritone higher — the pedal Bb2).
                let target = (freq_hz as f64).max(1.0);
                let steps = self.valve_steps as f64;
                let open_fundamental = self.overblow_anchor_hz as f64 * 2f64.powf(steps / 12.0);
                let tune = open_fundamental / res[0].max(1e-9);
                let scale = overblow_fingering(target, &res, tune, self.valve_steps, self.overblow_microtune);
                let scaled: Vec<f64> = res.iter().map(|&f| f * scale).collect();
                (target, scaled)
            } else {
                // Snap the played note to the nearest resonance (log distance).
                let ft = (freq_hz as f64).max(1.0).ln();
                let f_sel = *res
                    .iter()
                    .min_by(|a, b| (a.ln() - ft).abs().partial_cmp(&(b.ln() - ft).abs()).unwrap())
                    .unwrap();
                (f_sel, res.clone())
            };
            let res = boost; // resonance ladder used to shape the harmonic series
            // Synthesize that resonance's harmonic series, each harmonic boosted
            // when it lands on a tube resonance (so a harmonic tube sounds full,
            // an odd-resonance tube like a cylinder-closed clarinet loses its
            // even harmonics → hollow).
            let width = 0.4; // resonance capture width, in units of f_sel
            let mut amp_sum = 0.0f64;
            let mut kept: Vec<(f64, f64, f64)> = Vec::new();
            for h in 1..=self.depth.clamp(1, super::MAX_MODES) {
                let fh = f_sel * h as f64;
                if fh >= nyq {
                    break;
                }
                let dist = res
                    .iter()
                    .map(|&fr| (fr - fh).abs())
                    .fold(f64::INFINITY, f64::min)
                    / f_sel;
                let gain = (-(dist * dist) / (2.0 * width * width)).exp();
                let amp = gain / h as f64;
                if amp < 1e-3 {
                    continue;
                }
                // Frequency-dependent damping (same law as Chromatic), keyed on
                // the harmonic index so `damping` (overall) and `freq_dep_damping`
                // (how fast highs roll off) both shape the sustained tone; plus
                // Keefe wall + bell-radiation loss.
                let decay = keefe_rad(fh); // grounded wall + radiation loss, 1/s
                kept.push((fh, amp, decay.max(0.0)));
                amp_sum += amp;
            }
            if kept.is_empty() {
                return;
            }
            let norm = if amp_sum > 1e-9 { amp_strike as f64 / amp_sum } else { amp_strike as f64 };
            for (f, a, d) in kept {
                out.push(f as f32, (a * norm) as f32, d as f32);
            }
            return;
        }

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
            // Grounded loss: base + Keefe wall (∝√f) + bell radiation (∝f²), 1/s.
            let decay = keefe_rad(freq);
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
        egui::ComboBox::from_label("Wavefront")
            .selected_text(match self.wavefront {
                Wavefront::Planar => "Planar (flat discs)",
                Wavefront::Spherical => "Spherical (curved)",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.wavefront, Wavefront::Planar, "Planar (flat discs)")
                    .on_hover_text("Classic Webster: flat cross-sectional discs.")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.wavefront, Wavefront::Spherical, "Spherical (curved)")
                    .on_hover_text("Curved spherical-cap wavefronts — more accurate high partials where the flare is steep.")
                    .changed();
            });
        egui::ComboBox::from_label("Play")
            .selected_text(match self.play_mode {
                PlayMode::Chromatic => "Chromatic (resize)",
                PlayMode::Overblow => "Overblow (fixed tube)",
                PlayMode::OverblowTracked => "Overblow (key-tracked)",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.play_mode, PlayMode::Chromatic, "Chromatic (resize)")
                    .on_hover_text("Instrument resizes per note — plays any pitch.")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.play_mode, PlayMode::Overblow, "Overblow (fixed tube)")
                    .on_hover_text("Fixed tube: the key selects the nearest natural resonance (a bugle). Pitch snaps to the resonance ladder.")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.play_mode, PlayMode::OverblowTracked, "Overblow (key-tracked)")
                    .on_hover_text("Overblow to the harmonic above the key, then tune the bore length onto it — chromatic pitch with the overblown, fixed-formant tone.")
                    .changed();
            });
        if self.play_mode == PlayMode::OverblowTracked {
            ui.horizontal(|ui| {
                ui.label("Valve steps");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.valve_steps).range(0..=12).suffix(" st"))
                    .on_hover_text("Semitones of valve/slide tubing available → this many bore lengths beyond the open tube (a Bb trumpet = 6).")
                    .changed();
                ui.label("Anchor");
                // Anchor as a concert pitch (longest bore's fundamental).
                let mut note = super::freq_to_midi(self.overblow_anchor_hz);
                if super::note_field(ui, "horn_anchor", &mut note) {
                    self.overblow_anchor_hz = super::midi_freq(note);
                    changed = true;
                }
            });
            changed |= ui
                .checkbox(&mut self.overblow_microtune, "Micro-tune to key (in-tune)")
                .on_hover_text("On: nudge the bore so each note is exactly in tune. Off: play the natural harmonic pitch — authentic brass intonation (5th/7th harmonics sit flat).")
                .changed();
        }
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
        changed |= ui
            .add(unbounded_slider(&mut self.visco_loss, 0.0..=5.0, "Wall loss (Keefe)"))
            .on_hover_text("Viscothermal boundary-layer loss ∝ √f, stronger in narrow bores — warm/stuffed tone.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.radiation, 0.0..=5.0, "Radiation (bell)"))
            .on_hover_text("Loss ∝ f² at the bell: highs radiate out (open, brilliant), lows reflect and sustain.")
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

    #[test]
    fn overblow_fingering_lands_a_harmonic_on_the_key() {
        let res = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
        for &target in &[175.0_f64, 210.0, 250.0, 333.0, 512.0] {
            // Micro-tuned: a fingering's harmonic lands exactly on the key.
            let scale = overblow_fingering(target, &res, 1.0, 6, true);
            let landed = res.iter().any(|&r| (r * scale - target).abs() < 1e-6);
            assert!(landed, "micro-tuned harmonic lands on {target} (scale {scale})");
            // Authentic (no micro-tune): closest harmonic, within a semitone.
            let raw = overblow_fingering(target, &res, 1.0, 6, false);
            let nearest = res.iter().map(|&r| r * raw).fold(f64::INFINITY, |b, f| {
                if (f / target).ln().abs() < (b / target).ln().abs() { f } else { b }
            });
            let cents = 1200.0 * (nearest / target).log2();
            assert!(cents.abs() < 60.0, "authentic pitch within a semitone of {target}: {cents} cents");
        }
    }

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
    fn overblow_tracked_lands_the_fundamental_on_each_key() {
        let h = WebsterHorn { play_mode: PlayMode::OverblowTracked, ..WebsterHorn::default() };
        for &target in &[196.0_f32, 262.0, 330.0, 392.0, 523.0] {
            let mut buf = ModeBuffer::default();
            h.excite(target, 1.0, 48_000.0, &mut buf);
            assert!(buf.n > 0, "tracked overblow produced modes at {target} Hz");
            let lo = buf.freq[..buf.n].iter().cloned().fold(f32::INFINITY, f32::min);
            assert!(
                (lo - target).abs() / target < 0.02,
                "fundamental sits on the key: {lo} vs {target}"
            );
        }
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
    fn wall_and_radiation_losses_damp_highs_more() {
        let mut buf = ModeBuffer::default();
        let base = WebsterHorn { visco_loss: 0.0, radiation: 0.0, depth: 8, ..WebsterHorn::default() };
        base.excite(220.0, 1.0, 48_000.0, &mut buf);
        let (base_lo, base_hi) = (buf.decay[0], buf.decay[buf.n - 1]);

        // Keefe wall loss: the top mode gains more damping than the fundamental.
        let keefe = WebsterHorn { visco_loss: 2.0, depth: 8, ..WebsterHorn::default() };
        keefe.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.decay[buf.n - 1] > base_hi, "wall loss speeds up highs");
        assert!(
            buf.decay[buf.n - 1] - base_hi > buf.decay[0] - base_lo,
            "wall loss is frequency-dependent (more on highs)"
        );

        // Radiation loss (f²) also damps the top more than the fundamental.
        let rad = WebsterHorn { radiation: 2.0, depth: 8, ..WebsterHorn::default() };
        rad.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.decay[buf.n - 1] > base_hi, "radiation speeds up highs");
    }

    #[test]
    fn spherical_wavefront_shifts_flared_horn() {
        let mut buf = ModeBuffer::default();
        // Use a high partial — the curvature correction is largest up top.
        let ratio = |h: &WebsterHorn, buf: &mut ModeBuffer| {
            h.excite(220.0, 1.0, 48_000.0, buf);
            buf.freq[buf.n - 1] / buf.freq[0]
        };
        // Cylinder (no flare): spherical caps == flat discs.
        let cyl_p = WebsterHorn { r1: 1.0, r2: 0.0, r3: 0.0, depth: 6, wavefront: Wavefront::Planar, ..WebsterHorn::default() };
        let cyl_s = WebsterHorn { wavefront: Wavefront::Spherical, ..cyl_p.clone() };
        assert!((ratio(&cyl_p, &mut buf) - ratio(&cyl_s, &mut buf)).abs() < 1e-3, "cylinder unchanged");

        // Flared horn: the curved wavefronts move the partial ratios.
        let flr_p = WebsterHorn { r1: 1.0, r2: 0.0, r3: 3.0, depth: 6, wavefront: Wavefront::Planar, ..WebsterHorn::default() };
        let flr_s = WebsterHorn { wavefront: Wavefront::Spherical, ..flr_p.clone() };
        let (fp, fs) = (ratio(&flr_p, &mut buf), ratio(&flr_s, &mut buf));
        assert!((fp - fs).abs() > 1e-2, "flared horn: spherical shifts partials ({fp} vs {fs})");
    }

    #[test]
    fn overblow_snaps_and_shapes_by_resonances() {
        let mut buf = ModeBuffer::default();
        // Cylindrical closed-open tube (odd resonances only), played near its
        // fundamental. Overblow should sound a harmonic series with the even
        // harmonics suppressed (they fall between the odd resonances) — hollow.
        let h = WebsterHorn {
            play_mode: PlayMode::Overblow,
            boundary: Boundary::Brass,
            r1: 1.0,
            r2: 0.0,
            r3: 0.0,
            length: 1.0,
            wave_speed: 343.0,
            depth: 8,
            ..WebsterHorn::default()
        };
        h.excite(90.0, 1.0, 48_000.0, &mut buf); // ~ the fundamental resonance
        assert!(buf.n >= 3, "expected a harmonic series");
        // buf.freq[0] = fundamental (h=1), [1] = 2nd (even), [2] = 3rd (odd).
        assert!((buf.freq[1] / buf.freq[0] - 2.0).abs() < 0.05, "2nd harmonic at 2×");
        assert!(buf.amp[1].abs() < buf.amp[0].abs() * 0.25, "even harmonic suppressed");
        assert!(buf.amp[2].abs() > buf.amp[1].abs() * 2.0, "odd harmonic present");
        assert!(buf.sustain, "horn is sustained");
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




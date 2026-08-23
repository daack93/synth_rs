//! A physically-grounded 2-D circular membrane — a drumhead.
//!
//! Every input is a real property of a stretched head: its radius (m), tension
//! per unit length (N/m), areal density (kg/m²), and bending rigidity (N·m).
//! Tension + density set the transverse wave speed `c = √(T/σ)`; the clamped
//! circular boundary quantises the wavenumbers to `k_{ν,j} = α_{ν,j}/R`, the
//! j-th zero of the Bessel function `J_ν`, whose ratios (1 : 1.59 : 2.14 : 2.30…)
//! are the inharmonic voice of a drum. A stiff head adds a bending term, so the
//! dispersion is
//!
//!     ω(k) = k·√((T + D·k²)/σ),
//!
//! the honest membrane-plus-stiffness relation (linear in the bending rigidity
//! `D` — no fudge exponent). Playing a note tunes the head: the fundamental
//! (0,1) mode lands on the key and the geometry sets the inharmonic spread above
//! it. Striking at radius fraction `ρ` weights each mode by `J_ν(α_{ν,j}·ρ)`.
//! Decay is in real seconds.

use serde::{Deserialize, Serialize};

use super::{
    freq_to_midi, midi_name, strike_amplitude, unbounded_slider, Excitation, FtmModel, ModeBuffer,
};

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = std::f64::consts::TAU;
/// First zero of J₀ — the axisymmetric fundamental (0,1) mode.
const ALPHA_01: f64 = 2.404_826;
/// ln(1000): a decay rate of `ln(1000)/T` reaches −60 dB at `t = T` seconds.
const LN_1000: f32 = 6.907_755;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DrumMembrane {
    /// Head radius, metres (a 13" tom ≈ 0.165 m, a 22" kick ≈ 0.28 m).
    pub radius_m: f32,
    /// Head tension per unit length, N/m.
    pub tension_nm: f32,
    /// Head areal density, kg/m² (Mylar drumhead ≈ 0.26).
    pub areal_density_kgm2: f32,
    /// Bending rigidity D, N·m (0 = an ideal membrane; large = a stiff gong/plate).
    pub bending_nm: f32,
    /// Strike position as a fraction of the radius, 0 = centre, 1 = rim.
    pub strike_pos: f32,
    /// Fundamental −60 dB decay time, seconds.
    pub decay_time: f32,
    /// Extra decay on the higher modes, 1/s per (freq-ratio² − 1).
    pub hf_damping: f32,
    /// Number of modes summed.
    pub num_modes: usize,
    /// Struck (a hit that rings and decays) or bowed (driven — sustains, e.g. a
    /// bowed cymbal / singing bowl).
    #[serde(default)]
    pub excitation: Excitation,
    /// Velocity below this is silent (a gate).
    pub play_magnitude: f32,
    /// Velocity mapped to full amplitude.
    pub max_magnitude: f32,
}

impl Default for DrumMembrane {
    fn default() -> Self {
        // A ~13" tom head: radius 0.165 m, 1500 N/m, Mylar → open pitch ≈ 176 Hz.
        Self {
            radius_m: 0.165,
            tension_nm: 1500.0,
            areal_density_kgm2: 0.26,
            bending_nm: 0.0,
            strike_pos: 0.5,
            decay_time: 0.4,
            hf_damping: 3.0,
            num_modes: 32,
            excitation: Excitation::Struck,
            play_magnitude: 0.0,
            max_magnitude: 2500.0,
        }
    }
}

impl DrumMembrane {
    /// Transverse wave speed c = √(T/σ), m/s.
    pub fn wave_speed(&self) -> f32 {
        ((self.tension_nm as f64).max(1e-3) / (self.areal_density_kgm2 as f64).max(1e-5)).sqrt()
            as f32
    }

    /// The (0,1) fundamental frequency from the real geometry, Hz.
    pub fn open_pitch_hz(&self) -> f32 {
        let r = (self.radius_m as f64).max(1e-4);
        let k = ALPHA_01 / r;
        let t = (self.tension_nm as f64).max(1e-3);
        let d = (self.bending_nm as f64).max(0.0);
        let sigma = (self.areal_density_kgm2 as f64).max(1e-5);
        let omega = k * ((t + d * k * k) / sigma).sqrt();
        (omega / TWO_PI) as f32
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
        "A physically-grounded circular drumhead: radius, tension, density and bending set the inharmonic Bessel modes."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        out.sustain = self.excitation == Excitation::Bowed;
        let amp_strike = strike_amplitude(vel, self.play_magnitude, self.max_magnitude);
        if amp_strike <= 0.0 {
            return;
        }
        let f_play = freq_hz.max(1.0) as f64;
        let r = (self.radius_m as f64).max(1e-4);
        let t = (self.tension_nm as f64).max(1e-3);
        let sigma = (self.areal_density_kgm2 as f64).max(1e-5);
        let d = (self.bending_nm as f64).max(0.0);
        let rho = self.strike_pos.clamp(0.0, 0.999) as f64;
        let n_req = self.num_modes.clamp(1, super::MAX_MODES);

        // Enumerate the `n_req` lowest Bessel zeros (a triangular grid covers the
        // lowest α_{ν,j}, which grow with both ν and j).
        let span = ((2 * n_req) as f64).sqrt().ceil() as u32 + 6;
        let mut cands: Vec<(f64, u32)> = Vec::with_capacity((span * span) as usize);
        for nu in 0..=span {
            for j in 1..=span {
                cands.push((bessel_zero(nu, j), nu));
            }
        }
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cands.truncate(n_req);

        // Angular frequency of each mode from the real dispersion, and its strike
        // weight. ω = k·√((T + D k²)/σ), k = α/R.
        let omega_of = |alpha: f64| {
            let k = alpha / r;
            k * ((t + d * k * k) / sigma).sqrt()
        };
        let w0 = omega_of(cands[0].0).max(1e-9); // the (0,1) fundamental
        let a0 = LN_1000 / self.decay_time.max(1e-3);

        let mut amp_sum = 0.0f64;
        let mut tmp: Vec<(f32, f32, f32)> = Vec::with_capacity(cands.len()); // freq, weight, decay
        for &(alpha, nu) in &cands {
            let ratio = omega_of(alpha) / w0; // geometry-derived inharmonic ratio
            let freq = (f_play * ratio) as f32; // fundamental lands on the played note
            if freq >= sr * 0.45 {
                continue;
            }
            let k_weight = bessel_jn(nu, alpha * rho);
            let decay = a0 + self.hf_damping * ((ratio * ratio) as f32 - 1.0);
            amp_sum += k_weight.abs();
            tmp.push((freq, k_weight as f32, decay));
        }
        if tmp.is_empty() {
            return;
        }
        let norm = if amp_sum > 1e-9 { amp_strike / amp_sum as f32 } else { amp_strike };
        for (freq, w, decay) in tmp {
            out.push(freq, w * norm, decay);
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Membrane (physical)");
        let open = self.open_pitch_hz();
        ui.label(
            egui::RichText::new(format!(
                "open pitch ≈ {open:.1} Hz ({})   ·   wave speed {:.0} m/s",
                midi_name(freq_to_midi(open)),
                self.wave_speed()
            ))
            .weak()
            .small(),
        );

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
            .add(unbounded_slider(&mut self.radius_m, 0.03..=0.5, "Radius").suffix(" m"))
            .on_hover_text("Head radius. 13\" tom ≈ 0.165 m, 22\" kick ≈ 0.28 m.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.tension_nm, 100.0..=6000.0, "Tension").suffix(" N/m"))
            .on_hover_text("Head tension: higher = tighter = higher pitch.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.areal_density_kgm2, 0.05..=2.0, "Areal density").suffix(" kg/m²"))
            .on_hover_text("Head mass per area (Mylar ≈ 0.26). Heavier = lower and darker.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.bending_nm, 0.0..=50.0, "Bending rigidity").suffix(" N·m").logarithmic(true))
            .on_hover_text("0 = an ideal membrane; large = a stiff, gong-like/plate head (more inharmonic).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.decay_time, 0.02..=4.0, "Decay time").suffix(" s"))
            .on_hover_text("Fundamental −60 dB ring time, seconds.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.hf_damping, 0.0..=40.0, "HF damping"))
            .on_hover_text("How much faster the high modes die (1/s per freq-ratio²).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.num_modes, 1..=super::MAX_MODES, "Modes"))
            .changed();

        ui.add_space(6.0);
        ui.collapsing("Velocity", |ui| {
            changed |= ui
                .add(unbounded_slider(&mut self.play_magnitude, 0.0..=2500.0, "Play threshold"))
                .changed();
            changed |= ui
                .add(unbounded_slider(&mut self.max_magnitude, 1.0..=5000.0, "Full-velocity level"))
                .changed();
        });
        egui::ComboBox::from_label("Excitation")
            .selected_text(match self.excitation {
                Excitation::Struck => "Struck",
                Excitation::Bowed => "Bowed (sustained)",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.excitation, Excitation::Struck, "Struck")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.excitation, Excitation::Bowed, "Bowed (sustained)")
                    .on_hover_text("Driven — sustains while played (bowed cymbal / singing bowl).")
                    .changed();
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
fn bessel_jn(n: u32, x: f64) -> f64 {
    if x <= 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    let n = n as i64;
    let tox = 2.0 / x;
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
            sum += bj;
        }
        jsum = !jsum;
        if j == n {
            ans = bjp;
        }
        j -= 1;
    }
    if n == 0 {
        ans = bj;
    }
    let norm = 2.0 * sum - bj;
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
        assert!(approx(bessel_jn(0, 1.0), 0.765_197_7, 1e-4));
        assert!(approx(bessel_jn(1, 1.0), 0.440_050_6, 1e-4));
        assert!(bessel_jn(0, 2.404_83).abs() < 1e-3);
    }

    #[test]
    fn bessel_zeros_are_close() {
        assert!(approx(bessel_zero(0, 1), 2.4048, 2e-3));
        assert!(approx(bessel_zero(1, 1), 3.8317, 2e-3));
        assert!(approx(bessel_zero(0, 2), 5.5201, 2e-3));
    }

    #[test]
    fn open_pitch_matches_real_geometry() {
        // 0.165 m head, 1500 N/m, Mylar (0.26 kg/m²) → ≈ 176 Hz.
        let d = DrumMembrane::default();
        let f = d.open_pitch_hz();
        assert!((f - 176.0).abs() < 20.0, "membrane open pitch {f} Hz off");
    }

    #[test]
    fn produces_inharmonic_modes_and_tracks_key() {
        let mut buf = ModeBuffer::default();
        let d = DrumMembrane { num_modes: 12, ..DrumMembrane::default() };
        d.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 3, "drum should have several modes");
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite() && *f > 0.0));
        assert!(buf.decay[..buf.n].iter().all(|dd| dd.is_finite() && *dd >= 0.0));
        assert!((buf.freq[0] - 220.0).abs() < 2.0, "fundamental tracks the key: {}", buf.freq[0]);
        // Second partial is the inharmonic Bessel ratio ~1.59, not 2×.
        let ratio = buf.freq[1] / buf.freq[0];
        assert!(ratio > 1.4 && ratio < 1.8, "second mode ratio ~1.59, got {ratio}");
    }

    #[test]
    fn a_stiff_head_is_more_inharmonic() {
        // Adding bending rigidity stretches the upper modes sharp (gong-like).
        let mut a = ModeBuffer::default();
        let mut b = ModeBuffer::default();
        let membrane = DrumMembrane { bending_nm: 0.0, num_modes: 8, ..DrumMembrane::default() };
        let gong = DrumMembrane { bending_nm: 20.0, num_modes: 8, ..DrumMembrane::default() };
        membrane.excite(220.0, 1.0, 96_000.0, &mut a);
        gong.excite(220.0, 1.0, 96_000.0, &mut b);
        assert!(b.freq[b.n - 1] / b.freq[0] > a.freq[a.n - 1] / a.freq[0], "bending adds inharmonicity");
    }

    #[test]
    fn centre_strike_excites_only_axisymmetric_modes() {
        let mut buf = ModeBuffer::default();
        let d = DrumMembrane { strike_pos: 0.0, num_modes: 10, ..DrumMembrane::default() };
        d.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.amp[1].abs() < 1e-4, "off-axis mode should vanish at a centre strike");
    }

    #[test]
    fn stays_finite_with_degenerate_params() {
        let mut buf = ModeBuffer::default();
        let d = DrumMembrane {
            radius_m: 0.0,
            tension_nm: 0.0,
            areal_density_kgm2: 0.0,
            bending_nm: 0.0,
            decay_time: 0.0,
            num_modes: 2000,
            ..DrumMembrane::default()
        };
        d.excite(110.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite()));
        assert!(buf.decay[..buf.n].iter().all(|dd| dd.is_finite()));
    }
}

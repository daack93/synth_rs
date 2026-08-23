//! A physically-grounded stiff string.
//!
//! Every input is a real, tape-measurable property of a string: its speaking
//! length (m), tension (N), diameter/gauge (mm), and material (density kg/m³,
//! Young's modulus GPa). Those set the wave speed `c = √(T/μ)` (with linear
//! density `μ = ρ·π(d/2)²`) and the bending inharmonicity
//! `B = π²·E·I / (T·L²)`, `I = π(d/2)⁴/4`.
//!
//! Playing a note **frets** the string: the sounding length is `L = c/(2f)`, so
//! higher notes are physically shorter and more inharmonic (B ∝ f²). The
//! fundamental is tuned exactly onto the note — as a real fretted/tuned string
//! is — and B stretches the upper partials off the harmonic series.
//!
//! Pluck position is the one remaining shape control: a centred pluck gives a
//! triangle spectrum (odd modes), a near-end pluck a saw-like one. Decay is in
//! real seconds.

use serde::{Deserialize, Serialize};

use super::{midi_name, freq_to_midi, strike_amplitude, unbounded_slider, Excitation, FtmModel, ModeBuffer};

const PI: f32 = std::f32::consts::PI;
const PI64: f64 = std::f64::consts::PI;
/// ln(1000): a decay rate of `ln(1000)/T` reaches −60 dB (÷1000) at `t = T` s.
const LN_1000: f32 = 6.907_755;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PureString {
    /// Open (unfretted) speaking length, metres. With tension + gauge it sets
    /// the open pitch `(1/2L)·√(T/μ)`; the played note frets up from there.
    pub length_m: f32,
    /// String tension, newtons.
    pub tension_n: f32,
    /// String diameter (gauge), millimetres.
    pub diameter_mm: f32,
    /// Material density, kg/m³ (steel ≈ 7850, nylon ≈ 1150, bronze ≈ 8740).
    pub density_kgm3: f32,
    /// Young's modulus, GPa (steel ≈ 200, nylon ≈ 4, bronze ≈ 105).
    pub youngs_gpa: f32,
    /// Pluck position along the string, fraction 0..1. 0.5 = centre (triangle,
    /// odd modes); near an end = saw-like (all modes).
    pub pluck_pos: f32,
    /// Fundamental −60 dB decay time, seconds.
    pub decay_time: f32,
    /// Extra decay per mode-index² (1/s) — how much faster the highs die
    /// (brightness of the tail). May be negative for a swelling tail.
    pub hf_damping: f32,
    /// Number of partials summed.
    pub num_modes: usize,
    /// Plucked/struck (rings and decays) or bowed (driven — sustains while played).
    #[serde(default)]
    pub excitation: Excitation,
    /// Velocity below this is silent (a gate); 0 keeps soft keypresses audible.
    pub play_magnitude: f32,
    /// Velocity mapped to full amplitude.
    pub max_magnitude: f32,
}

impl Default for PureString {
    fn default() -> Self {
        // A plain steel string ~ a light electric-guitar gauge: 0.65 m, 70 N,
        // 0.5 mm steel → open pitch ≈ 164 Hz (E3), B ≈ 2e-4 (realistic).
        Self {
            length_m: 0.65,
            tension_n: 70.0,
            diameter_mm: 0.5,
            density_kgm3: 7850.0,
            youngs_gpa: 200.0,
            pluck_pos: 0.14,
            decay_time: 1.6,
            hf_damping: 0.5,
            num_modes: 40,
            excitation: Excitation::Struck,
            play_magnitude: 0.0,
            max_magnitude: 2500.0,
        }
    }
}

impl PureString {
    /// Linear (mass-per-length) density μ = ρ·π(d/2)², kg/m.
    fn linear_density(&self) -> f64 {
        let r = 0.5 * (self.diameter_mm as f64 * 1e-3).max(1e-6);
        (self.density_kgm3 as f64 * PI64 * r * r).max(1e-12)
    }

    /// Transverse wave speed c = √(T/μ), m/s.
    pub fn wave_speed(&self) -> f32 {
        ((self.tension_n as f64).max(1e-6) / self.linear_density()).sqrt() as f32
    }

    /// Open-string pitch (1/2L)·√(T/μ), Hz — the string's natural (unfretted)
    /// pitch, from which the played note frets up.
    pub fn open_pitch_hz(&self) -> f32 {
        self.wave_speed() / (2.0 * self.length_m.max(1e-4))
    }
}

impl FtmModel for PureString {
    fn id(&self) -> &'static str {
        "pure_string"
    }

    fn display_name(&self) -> &'static str {
        "Pure String"
    }

    fn description(&self) -> &'static str {
        "A physically-grounded stiff string: length, tension, gauge and material set the pitch and inharmonicity."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        out.sustain = self.excitation == Excitation::Bowed; // bowed = driven/sustained
        let amp_strike = strike_amplitude(vel, self.play_magnitude, self.max_magnitude);
        if amp_strike <= 0.0 {
            return; // below play threshold — silent, like a gentle shake
        }
        let f = freq_hz.max(1.0) as f64;

        // Real string geometry → wave speed and the fret length for this note.
        let r = 0.5 * (self.diameter_mm as f64 * 1e-3).max(1e-6); // radius, m
        let mu = self.linear_density(); // kg/m
        let t = (self.tension_n as f64).max(1e-6); // N
        let c = (t / mu).sqrt(); // m/s
        let l = (c / (2.0 * f)).max(1e-4); // fretted length, m
        // Bending inharmonicity B = π²·E·I / (T·L²), I = π r⁴/4.
        let e = (self.youngs_gpa as f64 * 1e9).max(0.0); // Pa
        let inertia = PI64 * r.powi(4) / 4.0; // m⁴
        let b = (PI64 * PI64 * e * inertia / (t * l * l)).max(0.0); // dimensionless

        let n_req = self.num_modes.clamp(1, super::MAX_MODES);
        let a0 = LN_1000 / self.decay_time.max(1e-3);
        // Pluck weight: Fourier coefficient of a triangular initial displacement,
        // K[m] = 2 sin(mπp) / (m²π² p(1-p)). p = 0.5 kills even modes.
        let p = self.pluck_pos.clamp(1e-3, 1.0 - 1e-3);
        let pq = (p * (1.0 - p)) as f64;
        // Normalise so the m = 1 partial lands exactly on the played note.
        let denom = (1.0 + b).max(1e-9);

        let mut freqs = [0.0f32; super::MAX_MODES];
        let mut ks = [0.0f32; super::MAX_MODES];
        let mut decays = [0.0f32; super::MAX_MODES];
        let mut amp_sum = 0.0f32;
        let mut count = 0usize;
        for i in 0..n_req {
            let m = (i + 1) as f64;
            let m2 = m * m;
            // Stiff-string partial: f_m = m·f·√((1+B m²)/(1+B)).
            let ratio = m * ((1.0 + b * m2) / denom).sqrt();
            let freq = (f * ratio) as f32;
            if freq >= sr * 0.45 {
                break; // near Nyquist: drop the rest (ascending order)
            }
            let mf = (i + 1) as f32;
            let k = (2.0 / (mf * mf * PI * PI * pq as f32)) * (mf * PI * p).sin();
            freqs[count] = freq;
            ks[count] = k;
            decays[count] = a0 + self.hf_damping * mf * mf;
            amp_sum += k.abs();
            count += 1;
        }
        if count == 0 {
            return;
        }
        let norm = if amp_sum > 1e-6 { amp_strike / amp_sum } else { amp_strike };
        for i in 0..count {
            out.push(freqs[i], ks[i] * norm, decays[i]);
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.strong("String (physical)");
        // Live read-out: what real string this is, and its open pitch.
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
            .add(unbounded_slider(&mut self.length_m, 0.1..=2.0, "Length").suffix(" m"))
            .on_hover_text("Open speaking length. Sets the open pitch; the note frets up from there.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.tension_n, 1.0..=1000.0, "Tension").suffix(" N"))
            .on_hover_text("String tension. Higher = brighter/less inharmonic at a given pitch (pianos run ~700 N).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.diameter_mm, 0.05..=3.0, "Gauge").suffix(" mm"))
            .on_hover_text("String diameter. Thicker = more inharmonic (the m⁴ bending term).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.density_kgm3, 500.0..=20000.0, "Density").suffix(" kg/m³"))
            .on_hover_text("Material density: steel ≈ 7850, nylon ≈ 1150, bronze ≈ 8740, gut ≈ 1300.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.youngs_gpa, 0.5..=250.0, "Young's modulus").suffix(" GPa"))
            .on_hover_text("Stiffness of the material: steel ≈ 200, bronze ≈ 105, nylon ≈ 4, gut ≈ 6.")
            .changed();

        ui.add_space(6.0);
        changed |= ui
            .add(
                unbounded_slider(&mut self.pluck_pos, 0.0..=1.0, "Pluck position").custom_formatter(
                    |v, _| {
                        if (v - 0.5).abs() < 0.02 {
                            "center (triangle)".into()
                        } else if v < 0.08 || v > 0.92 {
                            "near end (saw)".into()
                        } else {
                            format!("{v:.2}")
                        }
                    },
                ),
            )
            .on_hover_text("Where the string is plucked. Center = odd harmonics (triangle); near an end = fuller, saw-like.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.decay_time, 0.05..=12.0, "Decay time").suffix(" s"))
            .on_hover_text("Fundamental −60 dB ring time, in seconds.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.hf_damping, -2.0..=8.0, "HF damping"))
            .on_hover_text("How much faster the high partials die (1/s per mode²). Negative = they swell in.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.num_modes, 1..=super::MAX_MODES, "Modes"))
            .changed();

        ui.add_space(6.0);
        ui.collapsing("Velocity", |ui| {
            changed |= ui
                .add(unbounded_slider(&mut self.play_magnitude, 0.0..=2500.0, "Play threshold"))
                .on_hover_text("Velocity below this is silent.")
                .changed();
            changed |= ui
                .add(unbounded_slider(&mut self.max_magnitude, 1.0..=5000.0, "Full-velocity level"))
                .changed();
        });

        egui::ComboBox::from_label("Excitation")
            .selected_text(match self.excitation {
                Excitation::Struck => "Plucked / struck",
                Excitation::Bowed => "Bowed (sustained)",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.excitation, Excitation::Struck, "Plucked / struck")
                    .on_hover_text("A one-shot pluck — rings and decays.")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.excitation, Excitation::Bowed, "Bowed (sustained)")
                    .on_hover_text("Continuously driven — holds while played, fades on release.")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pluck_position_shapes_the_spectrum() {
        let mut buf = ModeBuffer::default();
        // Center pluck (0.5): even modes vanish, so mode 2 (index 1) is ~silent.
        let center = PureString { pluck_pos: 0.5, num_modes: 8, ..PureString::default() };
        center.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 2);
        assert!(buf.amp[1].abs() < 1e-3, "even mode should vanish at center pluck");

        // Off-center pluck: even modes come back.
        let edge = PureString { pluck_pos: 0.12, num_modes: 8, ..PureString::default() };
        edge.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.amp[1].abs() > 1e-3, "even mode present when plucked off-center");
    }

    #[test]
    fn open_pitch_matches_real_geometry() {
        // A 0.65 m, 0.5 mm steel string at 70 N sounds ≈ 164 Hz (≈ E3).
        let s = PureString::default();
        let f = s.open_pitch_hz();
        assert!((f - 164.0).abs() < 12.0, "open pitch {f} Hz not near the physical 164 Hz");
    }

    #[test]
    fn higher_notes_are_more_inharmonic() {
        // B ∝ f², so the 2nd partial stretches sharper up high than down low.
        let mut buf = ModeBuffer::default();
        // Thick, low-tension string so stiffness is audible.
        let s = PureString { diameter_mm: 1.2, tension_n: 40.0, pluck_pos: 0.12, num_modes: 8, ..PureString::default() };
        let ratio = |f: f32, buf: &mut ModeBuffer| { s.excite(f, 1.0, 96_000.0, buf); buf.freq[1] / buf.freq[0] };
        let low = ratio(130.81, &mut buf); // C3
        let high = ratio(1046.5, &mut buf); // C6
        assert!(high > 2.0, "stiff string should stretch partial 2 sharp ({high})");
        assert!(high > low + 1e-4, "high notes more inharmonic ({low} -> {high})");
    }

    #[test]
    fn bowed_string_is_sustained() {
        let mut buf = ModeBuffer::default();
        let mut s = PureString::default();
        s.excitation = Excitation::Struck;
        s.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(!buf.sustain, "plucked string is not sustained");
        s.excitation = Excitation::Bowed;
        s.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.sustain, "bowed string is driven/sustained");
        assert!(buf.n > 1);
    }

    #[test]
    fn stays_finite_with_degenerate_params() {
        let mut buf = ModeBuffer::default();
        let m = PureString {
            length_m: 0.0,
            tension_n: 0.0,
            diameter_mm: 0.0,
            decay_time: 0.0,
            hf_damping: -5.0,
            num_modes: 10_000,
            ..PureString::default()
        };
        m.excite(110.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite()));
        assert!(buf.decay[..buf.n].iter().all(|d| d.is_finite()));
    }

    #[test]
    fn play_magnitude_gates_soft_strikes() {
        let mut buf = ModeBuffer::default();
        let m = PureString { play_magnitude: 2000.0, max_magnitude: 2500.0, ..PureString::default() };
        m.excite(220.0, 0.5, 48_000.0, &mut buf); // soft => below threshold
        assert_eq!(buf.n, 0, "a soft strike should be silent");
        m.excite(220.0, 1.0, 48_000.0, &mut buf); // hard => sounds
        assert!(buf.n > 0);
    }
}

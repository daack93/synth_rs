//! Musically-oriented plucked string.
//!
//! Same modal idea as the firmware model, but re-parametrized in terms that are
//! easy to dial for music rather than physics: an inharmonicity knob (piano-like
//! partial stretching), a decay time in seconds, a high-frequency damping knob,
//! and a continuous pluck position that sweeps triangle → saw. The key always
//! sets the pitch.

use serde::{Deserialize, Serialize};

use super::{unbounded_slider, FtmModel, ModeBuffer};

const PI: f32 = std::f32::consts::PI;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MusicalString {
    /// Pluck position 0..1 (0.5 = center = triangle; near an end = saw).
    pub pluck_pos: f32,
    /// Inharmonicity B: partials at m*f0*sqrt((1+B m^2)/(1+B)).
    pub inharmonicity: f32,
    /// Base -60 dB decay time in seconds.
    pub decay_time: f32,
    /// Extra decay per m^2 (brightness of the decay tail).
    pub hf_damping: f32,
    /// Number of modes summed.
    pub num_modes: usize,
}

impl Default for MusicalString {
    fn default() -> Self {
        Self {
            pluck_pos: 0.5,
            inharmonicity: 0.0008,
            decay_time: 2.5,
            hf_damping: 0.35,
            num_modes: 32,
        }
    }
}

/// ln(1000): the factor giving a -60 dB fall over `decay_time`.
const LN_1000: f32 = 6.907_755;

impl FtmModel for MusicalString {
    fn id(&self) -> &'static str {
        "musical_string"
    }

    fn display_name(&self) -> &'static str {
        "Musical String"
    }

    fn description(&self) -> &'static str {
        "Plucked string with music-friendly controls (inharmonicity, decay time, pluck sweep)."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        let b = self.inharmonicity;
        let pp = self.pluck_pos;
        // Guard only the singular pluck point p(1-p) == 0.
        let pq = {
            let x = pp * (1.0 - pp);
            if x.abs() < 1e-4 {
                1e-4
            } else {
                x
            }
        };
        let a0 = LN_1000 / self.decay_time.max(1e-3); // base per-second decay
        let a2 = self.hf_damping; // extra per m^2 (may be negative => swell)
        let denom = (1.0 + b).max(1e-4); // so ratio[1] == 1 for B >= 0

        let n_req = self.num_modes.clamp(1, super::MAX_MODES);
        let mut amp_sum = 0.0f32;
        for m in 1..=n_req {
            let mf = m as f32;
            let ratio = mf * (((1.0 + b * mf * mf) / denom).max(0.0)).sqrt();
            let freq = freq_hz * ratio;
            if freq >= sr * 0.45 || freq <= 0.0 {
                break; // near Nyquist or collapsed to DC
            }
            // Pluck weight: Fourier coefficient of a triangular displacement
            // plucked at fraction pp of the length.
            let k = (2.0 / (mf * mf * PI * PI * pq)) * (mf * PI * pp).sin();
            let decay = a0 + a2 * mf * mf;
            out.push(freq, k, decay);
            amp_sum += k.abs();
        }
        // Normalize loudness across pluck position / mode count, apply velocity.
        let norm = if amp_sum > 1e-6 { vel / amp_sum } else { vel };
        out.scale_amps(norm);
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
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
            .on_hover_text("Center = odd harmonics (triangle); near an end = fuller, saw-like.")
            .changed();
        changed |= ui
            .add(unbounded_slider(
                &mut self.inharmonicity,
                -0.02..=0.05,
                "Stiffness (inharmonicity)",
            ))
            .on_hover_text("Positive stretches partials sharp (piano-like); negative compresses them.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.decay_time, 0.01..=60.0, "Decay time (s)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.hf_damping, -1.0..=10.0, "HF damping"))
            .on_hover_text("Higher = fast high-partial decay; negative = high partials swell in.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.num_modes, 1..=super::MAX_MODES, "Modes"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excite_is_finite_even_at_extremes() {
        let mut buf = ModeBuffer::default();
        // Negative inharmonicity + swelling HF damping + absurd mode count.
        let m = MusicalString {
            pluck_pos: 0.0,
            inharmonicity: -0.5,
            decay_time: 0.001,
            hf_damping: -5.0,
            num_modes: 10_000,
        };
        m.excite(440.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite()));
        assert!(buf.amp[..buf.n].iter().all(|a| a.is_finite()));
    }

    #[test]
    fn fundamental_tracks_the_key() {
        let mut buf = ModeBuffer::default();
        MusicalString::default().excite(440.0, 1.0, 48_000.0, &mut buf);
        assert!((buf.freq[0] - 440.0).abs() < 1.0, "mode 1 should sit on f0");
    }
}

//! FTM string based on the original `set_triangle/saw_string_params` equations
//! and the exact `#define`s from the 2014 `main.h`. The firmware exposed only
//! two pluck geometries (center = triangle, end = saw); here the pluck position
//! is a continuous control, which those two are just special cases of.

use serde::{Deserialize, Serialize};

use super::{strike_amplitude, unbounded_slider, FtmModel, ModeBuffer, PitchMode, TICK_RATE};

const PI: f32 = std::f32::consts::PI;
const TWO_PI: f32 = std::f32::consts::TAU;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PureString {
    pub stiffness: f32,         // STRING_STIFFNESS (S)
    pub prop_speed: f32,        // STRING_PROP_SPEED (c)
    pub damping: f32,           // STRING_DAMPING (d1)
    pub freq_dep_damping: f32,  // STRING_FREQ_DEPENDENT_DAMPING (d3)
    pub string_length: f32,     // STRING_LENGTH (l)
    pub depth: usize,           // DEPTH
    /// Pluck position along the string, 0..1. 0.5 = center (triangle, odd modes);
    /// near an end = saw-like (all modes). The firmware's two modes are the ends.
    pub pluck_pos: f32,
    pub damp_period: f32,       // DAMP_PERIOD
    pub time_scale: f32,        // TIME_SCALE
    pub play_magnitude: f32,    // PLAY_MAGNITUDE
    pub max_magnitude: f32,     // MAX_MAGNITUDE
    /// If true the key sets pitch; if false, c/2l does (original air-guitar).
    pub key_tracks_pitch: bool,
    /// How the key sets pitch: transpose a fixed string, or shorten the string
    /// for higher notes (note-dependent inharmonicity + decay).
    #[serde(default)]
    pub pitch_mode: PitchMode,
}

impl Default for PureString {
    fn default() -> Self {
        // Straight from main.h (pluck at center = the old MODE_TRIANGLE_STRING).
        Self {
            stiffness: 1.0,
            prop_speed: 500.0,
            damping: 1.0,
            freq_dep_damping: -1.0,
            string_length: 10.0,
            depth: 10,
            pluck_pos: 0.5,
            damp_period: 100.0,
            time_scale: 10_000.0,
            play_magnitude: 0.0, // orig 2000; 0 keeps soft keypresses audible
            max_magnitude: 2500.0,
            key_tracks_pitch: true,
            pitch_mode: PitchMode::Transpose,
        }
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
        "The original FTM string equations and main.h #defines, with a continuous pluck position."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        let amp_strike = strike_amplitude(vel, self.play_magnitude, self.max_magnitude);
        if amp_strike <= 0.0 {
            return; // below play threshold — silent, like a gentle shake
        }

        // Reference length; in Physical mode the string shortens for higher
        // notes so it matches Transpose at C4 and gets more inharmonic above it.
        let l_ref = self.string_length;
        let l_ref_safe = if l_ref.abs() < 1e-3 { 1e-3 } else { l_ref };
        let (l_freq, decay_scale) =
            if self.key_tracks_pitch && self.pitch_mode == PitchMode::Physical {
                let ratio = (super::REF_PITCH_HZ / freq_hz.max(1.0)).clamp(0.02, 50.0);
                // Full length scaling for pitch/inharmonicity; a gentler power for
                // the decay so highs speed up without vanishing (the raw 1/l² is
                // too aggressive).
                (l_ref_safe * ratio, (freq_hz.max(1.0) / super::REF_PITCH_HZ).powf(0.6))
            } else {
                (l_ref_safe, 1.0)
            };

        // Frequency (inharmonicity) terms use the note's length.
        let c = self.prop_speed / l_freq;
        let s = self.stiffness / l_freq;
        let wm_a = (PI * s).powi(4); // W^2 = wm_a*m^4 + wm_b*m^2 - O^2
        let wm_b = (c * PI).powi(2);
        // Decay terms use the reference length; Physical mode scales the whole
        // decay by `decay_scale` instead.
        let d3 = self.freq_dep_damping / (l_ref_safe * l_ref_safe);
        let om_m = PI * PI * d3 / 2.0; // sigma[m] = om_m*m^2 + om_c
        let om_c = -self.damping / 2.0;

        let damp_per = self.damp_period.max(1e-3);
        let n_req = self.depth.clamp(1, super::MAX_MODES);

        // Pluck weight: Fourier coefficient of a triangular initial displacement
        // plucked at fraction p of the length, K[m] = 2 sin(mπp) / (m²π² p(1-p)).
        // At p = 0.5 the even modes vanish (the firmware's triangle); as p → 0
        // it fills in as ~1/m (the firmware's saw). Guarded away from the poles.
        let p = self.pluck_pos.clamp(1e-3, 1.0 - 1e-3);
        let pq = p * (1.0 - p);

        // Precompute W, sigma, K per mode (also gives W[0] for key normalization).
        let mut w = [0.0f32; super::MAX_MODES];
        let mut sig = [0.0f32; super::MAX_MODES];
        let mut kk = [0.0f32; super::MAX_MODES];
        let mut amp_sum = 0.0f32;
        let mut count = 0usize;
        for i in 0..n_req {
            let m = (i + 1) as f32;
            let m2 = m * m;
            let m4 = m2 * m2;
            let sigma = om_m * m2 + om_c;
            let o_lin = (sigma / damp_per).exp(); // firmware O[i], ~1
            let w2 = (wm_a * m4 + wm_b * m2 - o_lin * o_lin).max(0.0);
            let k = (2.0 / (m2 * PI * PI * pq)) * (m * PI * p).sin();
            w[count] = w2.sqrt();
            sig[count] = sigma;
            kk[count] = k;
            amp_sum += k.abs();
            count += 1;
        }
        if count == 0 {
            return;
        }

        let w0 = if w[0] > 1e-6 { w[0] } else { 1.0 };
        let ts = self.time_scale.max(1.0);
        let norm = if amp_sum > 1e-6 { amp_strike / amp_sum } else { amp_strike };

        for i in 0..count {
            let freq = if self.key_tracks_pitch {
                freq_hz * (w[i] / w0)
            } else {
                // Absolute firmware mapping W*TICK_RATE/(TIME_SCALE*2pi).
                w[i] * TICK_RATE / (ts * TWO_PI)
            };
            if freq >= sr * 0.45 {
                break; // near Nyquist: drop the rest (ascending order)
            }
            // The firmware multiplied D by O = exp(sigma/DAMP_PERIOD) every
            // DAMP_PERIOD board-ticks; over a second that is a per-second rate of
            // sigma*TICK_RATE/DAMP_PERIOD^2. decay = -that (sigma <= 0 => decay >= 0).
            let decay = -sig[i] * TICK_RATE / (damp_per * damp_per) * decay_scale;
            out.push(freq, kk[i] * norm, decay);
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.strong("String Parameters");
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
            .add(unbounded_slider(&mut self.stiffness, 0.0..=50.0, "STRING_STIFFNESS (S)"))
            .on_hover_text("The m^4 term: stretches upper partials sharp (inharmonicity).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.prop_speed, 1.0..=2000.0, "STRING_PROP_SPEED (c)"))
            .on_hover_text("Wave speed. With length sets the physical pitch (~c/2l).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.damping, -50.0..=200.0, "STRING_DAMPING (d1)"))
            .on_hover_text("Uniform decay of every mode. (Firmware required >= 0.)")
            .changed();
        changed |= ui
            .add(unbounded_slider(
                &mut self.freq_dep_damping,
                -100.0..=20.0,
                "STRING_FREQ_DEP_DAMPING (d3)",
            ))
            .on_hover_text("Extra decay on high modes (negative in the original).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.string_length, 0.1..=100.0, "STRING_LENGTH (l)"))
            .on_hover_text("Affects pitch and the pluck-shape weights K[m].")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.depth, 1..=super::MAX_MODES, "DEPTH (modes)"))
            .changed();

        ui.add_space(6.0);
        ui.strong("Timing / velocity");
        changed |= ui
            .add(unbounded_slider(&mut self.damp_period, 1.0..=1000.0, "DAMP_PERIOD"))
            .on_hover_text("How often damping is applied (board ticks). Larger = longer sustain.")
            .changed();
        changed |= ui
            .add(
                unbounded_slider(&mut self.time_scale, 100.0..=100_000.0, "TIME_SCALE")
                    .logarithmic(true),
            )
            .on_hover_text("Divides modal frequency in physical-pitch mode (ignored when the key tracks pitch).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.play_magnitude, 0.0..=2500.0, "PLAY_MAGNITUDE"))
            .on_hover_text("Velocity threshold; below it a strike is silent. Firmware: 2000.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.max_magnitude, 1.0..=5000.0, "MAX_MAGNITUDE"))
            .on_hover_text("Velocity mapped to full amplitude. Firmware: 2500.")
            .changed();

        changed |= ui
            .checkbox(&mut self.key_tracks_pitch, "Key tracks pitch")
            .on_hover_text("Off: c/2l sets the pitch and the key transposes — the original air-guitar behavior.")
            .changed();
        ui.add_enabled_ui(self.key_tracks_pitch, |ui| {
            egui::ComboBox::from_label("Pitch mode")
                .selected_text(match self.pitch_mode {
                    PitchMode::Transpose => "Transpose",
                    PitchMode::Physical => "Physical length",
                })
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut self.pitch_mode, PitchMode::Transpose, "Transpose")
                        .on_hover_text("One string stretched to each note — uniform timbre.")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.pitch_mode, PitchMode::Physical, "Physical length")
                        .on_hover_text("Shorten the string for higher notes: more inharmonic and faster-decaying up top.")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pluck_position_shapes_the_spectrum() {
        let mut buf = ModeBuffer::default();
        // Center pluck (0.5): even modes vanish, so mode 2 (index 1) is ~silent.
        let center = PureString { pluck_pos: 0.5, depth: 8, ..PureString::default() };
        center.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 2);
        assert!(buf.amp[1].abs() < 1e-3, "even mode should vanish at center pluck");

        // Off-center pluck: even modes come back.
        let edge = PureString { pluck_pos: 0.12, depth: 8, ..PureString::default() };
        edge.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.amp[1].abs() > 1e-3, "even mode present when plucked off-center");
    }

    #[test]
    fn physical_mode_stretches_high_notes() {
        let mut buf = ModeBuffer::default();
        let ratio = |m: &PureString, f: f32, buf: &mut ModeBuffer| {
            m.excite(f, 1.0, 48_000.0, buf);
            buf.freq[1] / buf.freq[0]
        };
        // Real stiffness so inharmonicity is visible.
        let phys = PureString { pitch_mode: PitchMode::Physical, stiffness: 8.0, pluck_pos: 0.12, depth: 8, ..PureString::default() };
        let r_low = ratio(&phys, 261.63, &mut buf); // C4
        let r_high = ratio(&phys, 1046.5, &mut buf); // C6
        assert!(r_high > r_low + 1e-3, "physical: high notes more inharmonic ({r_low} -> {r_high})");

        // Transpose mode: the partial ratio is the same at every pitch.
        let trans = PureString { pitch_mode: PitchMode::Transpose, stiffness: 8.0, pluck_pos: 0.12, depth: 8, ..PureString::default() };
        let t_low = ratio(&trans, 261.63, &mut buf);
        let t_high = ratio(&trans, 1046.5, &mut buf);
        assert!((t_low - t_high).abs() < 1e-4, "transpose: ratio is note-independent");
        // Physical matches Transpose at the C4 reference.
        assert!((r_low - t_low).abs() < 1e-3, "physical == transpose at C4 ({r_low} vs {t_low})");
    }

    #[test]
    fn stays_finite_with_degenerate_params() {
        let mut buf = ModeBuffer::default();
        let m = PureString {
            string_length: 0.0,
            damping: -3.0,
            freq_dep_damping: 2.0,
            depth: 10_000,
            damp_period: 0.0,
            ..PureString::default()
        };
        m.excite(110.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite()));
        assert!(buf.decay[..buf.n].iter().all(|d| d.is_finite()));
    }

    #[test]
    fn play_magnitude_gates_soft_strikes() {
        let mut buf = ModeBuffer::default();
        // Threshold at 2000/2500 = 0.8 of full velocity.
        let m = PureString { play_magnitude: 2000.0, max_magnitude: 2500.0, ..PureString::default() };
        m.excite(220.0, 0.5, 48_000.0, &mut buf); // soft => below threshold
        assert_eq!(buf.n, 0, "a soft strike should be silent");
        m.excite(220.0, 1.0, 48_000.0, &mut buf); // hard => sounds
        assert!(buf.n > 0);
    }
}

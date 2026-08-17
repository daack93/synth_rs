//! Firmware-faithful FTM string — the original `set_triangle_string_params` /
//! `set_saw_string_params` equations, driven by the exact `#define`s from the
//! 2014 `main.h`. Triangle vs. saw is the pluck geometry (center vs. end), which
//! only changes the mode weights K[m].

use serde::{Deserialize, Serialize};

use super::{strike_amplitude, unbounded_slider, FtmModel, ModeBuffer, TICK_RATE};

const PI: f32 = std::f32::consts::PI;
const TWO_PI: f32 = std::f32::consts::TAU;

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pluck {
    /// `MODE_TRIANGLE_STRING`: plucked at the center — odd modes only.
    Triangle,
    /// `MODE_SAW_STRING`: plucked near the end — all modes.
    Saw,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PureString {
    pub stiffness: f32,         // STRING_STIFFNESS (S)
    pub prop_speed: f32,        // STRING_PROP_SPEED (c)
    pub damping: f32,           // STRING_DAMPING (d1)
    pub freq_dep_damping: f32,  // STRING_FREQ_DEPENDENT_DAMPING (d3)
    pub string_length: f32,     // STRING_LENGTH (l)
    pub depth: usize,           // DEPTH
    pub pluck: Pluck,           // which MODE_*_STRING
    pub damp_period: f32,       // DAMP_PERIOD
    pub time_scale: f32,        // TIME_SCALE
    pub play_magnitude: f32,    // PLAY_MAGNITUDE
    pub max_magnitude: f32,     // MAX_MAGNITUDE
    /// If true the key sets pitch; if false, c/2l does (original air-guitar).
    pub key_tracks_pitch: bool,
}

impl Default for PureString {
    fn default() -> Self {
        // Straight from main.h.
        Self {
            stiffness: 1.0,
            prop_speed: 500.0,
            damping: 1.0,
            freq_dep_damping: -1.0,
            string_length: 10.0,
            depth: 10,
            pluck: Pluck::Triangle,
            damp_period: 100.0,
            time_scale: 10_000.0,
            play_magnitude: 0.0, // orig 2000; 0 keeps soft keypresses audible
            max_magnitude: 2500.0,
            key_tracks_pitch: true,
        }
    }
}

impl FtmModel for PureString {
    fn id(&self) -> &'static str {
        "pure_string"
    }

    fn display_name(&self) -> &'static str {
        "Pure String (firmware)"
    }

    fn description(&self) -> &'static str {
        "The original set_triangle/saw_string_params equations, driven by the exact main.h #defines."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        let amp_strike = strike_amplitude(vel, self.play_magnitude, self.max_magnitude);
        if amp_strike <= 0.0 {
            return; // below play threshold — silent, like a gentle shake
        }

        // Length-normalize exactly as the firmware did.
        let l = self.string_length;
        let l_safe = if l.abs() < 1e-3 { 1e-3 } else { l };
        let d3 = self.freq_dep_damping / (l_safe * l_safe);
        let c = self.prop_speed / l_safe;
        let s = self.stiffness / l_safe;

        let om_m = PI * PI * d3 / 2.0; // sigma[m] = om_m*m^2 + om_c
        let om_c = -self.damping / 2.0;
        let wm_a = (PI * s).powi(4); // W^2 = wm_a*m^4 + wm_b*m^2 - O^2
        let wm_b = (c * PI).powi(2);

        let damp_per = self.damp_period.max(1e-3);
        let saw = self.pluck == Pluck::Saw;
        let n_req = self.depth.clamp(1, super::MAX_MODES);

        // Precompute W, sigma, K per mode (also gives W[0] for key normalization).
        let mut w = [0.0f32; super::MAX_MODES];
        let mut sig = [0.0f32; super::MAX_MODES];
        let mut kk = [0.0f32; super::MAX_MODES];
        let mut amp_sum = 0.0f32;
        let mut sign = 1.0f32;
        let mut count = 0usize;
        for i in 0..n_req {
            // Triangle wave: f_m == 0 for even m, so only odd m. Saw: all m.
            let m = if saw { (i + 1) as f32 } else { (2 * i + 1) as f32 };
            let m2 = m * m;
            let m4 = m2 * m2;
            let sigma = om_m * m2 + om_c;
            let o_lin = (sigma / damp_per).exp(); // firmware O[i], ~1
            let w2 = (wm_a * m4 + wm_b * m2 - o_lin * o_lin).max(0.0);
            let k = if saw {
                -(m * PI / (l_safe * 2.0)).sin() / (m * PI)
            } else {
                let v = 8.0 * (m * PI / (l_safe * 2.0)).sin() / (m2 * PI * PI) * sign;
                sign = -sign;
                v
            };
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
            let decay = -sig[i] * TICK_RATE / (damp_per * damp_per);
            out.push(freq, kk[i] * norm, decay);
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.strong("String Parameters");
        egui::ComboBox::from_label("Pluck (MODE)")
            .selected_text(match self.pluck {
                Pluck::Triangle => "MODE_TRIANGLE_STRING",
                Pluck::Saw => "MODE_SAW_STRING",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.pluck, Pluck::Triangle, "MODE_TRIANGLE_STRING")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.pluck, Pluck::Saw, "MODE_SAW_STRING")
                    .changed();
            });

        changed |= ui
            .add(unbounded_slider(&mut self.stiffness, 0.0..=50.0, "STRING_STIFFNESS (S)"))
            .on_hover_text("The m^4 term: stretches upper partials sharp (inharmonicity).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.prop_speed, 1.0..=2000.0, "STRING_PROP_SPEED (c)"))
            .on_hover_text("Wave speed. With length sets the physical pitch (~c/2l).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.damping, -5.0..=20.0, "STRING_DAMPING (d1)"))
            .on_hover_text("Uniform decay of every mode. (Firmware required >= 0.)")
            .changed();
        changed |= ui
            .add(unbounded_slider(
                &mut self.freq_dep_damping,
                -10.0..=2.0,
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
    fn triangle_and_saw_both_produce_modes() {
        let mut buf = ModeBuffer::default();
        PureString::default().excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 1, "triangle string should have modes");

        let saw = PureString { pluck: Pluck::Saw, ..PureString::default() };
        saw.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 1, "saw string should have modes");
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

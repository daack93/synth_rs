//! Snare drum — a membrane head plus the snare wires.
//!
//! The head is a small, heavily-damped drum membrane (reused from
//! [`DrumMembrane`], so it shares the Bessel-mode maths): a short "thwack". The
//! *snare* character — the buzz/rattle of the wires against the bottom head — is
//! a band-passed **noise** burst (via the noise component) that rings a bit
//! longer than the head. The mix of the two is the snare.

use serde::{Deserialize, Serialize};

use super::drum_membrane::DrumMembrane;
use super::{FtmModel, ModeBuffer};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Snare {
    /// Head "tension" (drum wave speed) — brighter/higher with more.
    pub tension: f32,
    /// Head damping — how fast the drum tone dies (snares are short).
    pub damping: f32,
    /// Strike position across the head (0 = centre, 1 = rim).
    pub strike_pos: f32,
    /// Head modes summed.
    pub depth: usize,
    /// Snare-wire amount (the noise level).
    pub snares: f32,
    /// How long the wires rattle (per-second decay of the noise).
    pub snare_decay: f32,
    /// Brightness of the wire buzz (raises the noise band).
    pub tone: f32,
    /// If true the key sets the head pitch.
    pub key_tracks_pitch: bool,
}

impl Default for Snare {
    fn default() -> Self {
        Self {
            tension: 700.0,
            damping: 12.0,
            strike_pos: 0.4,
            depth: 24,
            snares: 0.6,
            snare_decay: 15.0, // ~0.45 s tail
            tone: 0.6,
            key_tracks_pitch: true,
        }
    }
}

impl FtmModel for Snare {
    fn id(&self) -> &'static str {
        "snare"
    }

    fn display_name(&self) -> &'static str {
        "Snare"
    }

    fn description(&self) -> &'static str {
        "A drum head plus snare-wire noise — a membrane thwack with a buzzing rattle."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        // Build the head from a small, damped membrane (clears `out`).
        let head = DrumMembrane {
            radius_m: 0.165, // 14" snare head
            tension_nm: (self.tension * 2.5).max(100.0), // "tension" knob → N/m
            areal_density_kgm2: 0.26,
            bending_nm: 0.02,
            decay_time: (2.0 / self.damping.max(0.1)).clamp(0.05, 2.0),
            hf_damping: 5.0,
            num_modes: self.depth,
            strike_pos: self.strike_pos,
            ..DrumMembrane::default()
        };
        head.excite(freq_hz, vel, sr, out);

        // Snare wires: a band-passed noise burst that outlasts the head.
        if self.snares > 1e-4 {
            out.noise_level = self.snares * vel;
            out.noise_decay = self.snare_decay.max(0.5);
            out.noise_hp = 800.0 + self.tone.clamp(0.0, 1.0) * 3200.0;
            out.noise_lp = 16_000.0;
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        use super::unbounded_slider;
        let mut changed = false;
        ui.strong("Head");
        changed |= ui
            .add(unbounded_slider(&mut self.tension, 100.0..=2000.0, "Tension"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.damping, 0.0..=60.0, "Head damping"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.strike_pos, 0.0..=1.0, "Strike position"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.depth, 1..=64, "Modes"))
            .changed();
        ui.add_space(6.0);
        ui.strong("Wires");
        changed |= ui
            .add(unbounded_slider(&mut self.snares, 0.0..=1.5, "Snares (buzz)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.snare_decay, 2.0..=60.0, "Rattle decay"))
            .on_hover_text("Higher = shorter rattle.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.tone, 0.0..=1.0, "Wire tone (brightness)"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snare_has_head_and_wire_noise() {
        let mut buf = ModeBuffer::default();
        Snare::default().excite(200.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 1, "head modes present");
        assert!(buf.noise_level > 0.0, "wire noise present");
        assert!(!buf.sustain, "snare is struck, not sustained");
    }

    #[test]
    fn wires_off_leaves_just_the_head() {
        let mut buf = ModeBuffer::default();
        Snare { snares: 0.0, ..Snare::default() }.excite(200.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 1);
        assert_eq!(buf.noise_level, 0.0);
    }
}

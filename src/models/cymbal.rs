//! Cymbal / plate — a stiff metal plate.
//!
//! A plate is bending-stiffness dominated (the biharmonic ∇⁴ term) rather than
//! tension dominated (∇²) like a drum head, so its modes disperse as ω ∝ k²:
//! dense, highly inharmonic, metallic. We get exactly that by reusing the
//! [`DrumMembrane`] with the tension turned down and the **stiffness** turned up
//! so the k⁴ term dominates. Many closely-spaced modes plus a bright **noise
//! wash** give the shimmer.
//!
//! (Honest limit: the *evolving* metallic wash of a real crash comes from
//! nonlinear mode coupling, which a linear modal engine can't do — this is a
//! dense metallic hit with a hiss, not a living shimmer.)

use serde::{Deserialize, Serialize};

use super::drum_membrane::DrumMembrane;
use super::{FtmModel, ModeBuffer};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Cymbal {
    /// Plate size (radius): bigger = lower and denser.
    pub size: f32,
    /// Bending stiffness — the plate dispersion / inharmonic spread.
    pub stiffness: f32,
    /// Decay — low for a long crash, higher for a short splash.
    pub damping: f32,
    /// High-frequency sustain (negative = highs ring on → shimmery).
    pub brightness: f32,
    /// Strike position (0 = centre/bell, 1 = edge).
    pub strike_pos: f32,
    /// Mode density (a cymbal wants a lot).
    pub modes: usize,
    /// Bright noise wash (the metallic hiss).
    pub shimmer: f32,
    /// If true the key sets the pitch.
    pub key_tracks_pitch: bool,
}

impl Default for Cymbal {
    fn default() -> Self {
        Self {
            size: 16.0,
            stiffness: 8.0,
            damping: 1.2,
            brightness: -0.4,
            strike_pos: 0.75,
            modes: 140,
            shimmer: 0.35,
            key_tracks_pitch: true,
        }
    }
}

impl FtmModel for Cymbal {
    fn id(&self) -> &'static str {
        "cymbal"
    }

    fn display_name(&self) -> &'static str {
        "Cymbal / Plate"
    }

    fn description(&self) -> &'static str {
        "A stiff metal plate: dense inharmonic modes plus a bright noise wash."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        // Low tension + high stiffness → the k⁴ term dominates → plate dispersion
        // (ω ∝ k²), dense and inharmonic. Reuses the membrane's Bessel modes.
        let plate = DrumMembrane {
            prop_speed: 8.0,
            stiffness: self.stiffness,
            damping: self.damping,
            freq_dep_damping: self.brightness,
            radius: self.size,
            depth: self.modes,
            strike_pos: self.strike_pos,
            key_tracks_pitch: self.key_tracks_pitch,
            ..DrumMembrane::default()
        };
        plate.excite(freq_hz, vel, sr, out);

        // Bright noise wash, its length tied to the plate decay.
        if self.shimmer > 1e-4 {
            out.noise_level = self.shimmer * vel;
            out.noise_decay = (self.damping * 1.5).max(0.5);
            out.noise_hp = 4000.0;
            out.noise_lp = 18_000.0;
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        use super::unbounded_slider;
        let mut changed = false;
        ui.strong("Plate");
        changed |= ui.add(unbounded_slider(&mut self.size, 1.0..=60.0, "Size")).changed();
        changed |= ui
            .add(unbounded_slider(&mut self.stiffness, 0.5..=30.0, "Stiffness (inharmonicity)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.damping, 0.1..=20.0, "Damping (decay)"))
            .on_hover_text("Low = long crash, high = short splash.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.brightness, -5.0..=2.0, "HF sustain"))
            .on_hover_text("Negative = highs ring on (shimmery).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.strike_pos, 0.0..=1.0, "Strike (bell → edge)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.modes, 1..=super::MAX_MODES, "Mode density"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.shimmer, 0.0..=1.5, "Shimmer (noise)"))
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
    fn cymbal_is_dense_inharmonic_with_wash() {
        let mut buf = ModeBuffer::default();
        Cymbal::default().excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 20, "dense mode set, got {}", buf.n);
        assert!(buf.noise_level > 0.0, "shimmer noise present");
        // Stiffness-dominated ⇒ the 2nd partial is spread well past a 2:1 (plate
        // dispersion), not the drum's ~1.6.
        let ratio = buf.freq[1] / buf.freq[0];
        assert!(ratio > 2.0, "plate dispersion spreads the modes, got {ratio}");
    }
}

//! Metal bell / cowbell — a small struck idiophone: a handful of strong
//! **inharmonic** partials plus a short noise "clank" from the stick.
//!
//! Unlike the drum (Bessel modes) or string (harmonic), a bell/cowbell is
//! defined by just a few stretched partial ratios. The classic 808 cowbell is
//! two square waves at ~1 : 1.48; here the ratios are generated from a base gap
//! + an inharmonicity stretch, so it spans cowbells, small bells and clanks.

use serde::{Deserialize, Serialize};

use super::{unbounded_slider, FtmModel, ModeBuffer};

/// ln(1000): -60 dB fall over `decay_time`.
const LN_1000: f32 = 6.907_755;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetalBell {
    /// Number of partials.
    pub partials: usize,
    /// Base spacing between partials (0.48 ≈ the 808 cowbell's 1 : 1.48).
    pub spread: f32,
    /// Extra stretch that grows with partial index — more = clangier/inharmonic.
    pub inharmonicity: f32,
    /// Amplitude rolloff of the upper partials (higher = darker).
    pub brightness: f32,
    /// -60 dB decay time in seconds (base; upper partials decay faster).
    pub decay_time: f32,
    /// Amount of stick-click noise on the attack.
    pub strike_noise: f32,
    /// If true the key sets the fundamental partial.
    pub key_tracks_pitch: bool,
}

impl Default for MetalBell {
    fn default() -> Self {
        Self {
            partials: 3,
            spread: 0.48,
            inharmonicity: 0.05,
            brightness: 0.6,
            decay_time: 0.5,
            strike_noise: 0.25,
            key_tracks_pitch: true,
        }
    }
}

impl FtmModel for MetalBell {
    fn id(&self) -> &'static str {
        "metal_bell"
    }

    fn display_name(&self) -> &'static str {
        "Metal Bell / Cowbell"
    }

    fn description(&self) -> &'static str {
        "A struck idiophone: a few inharmonic partials plus a stick-click of noise."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        let f0 = if self.key_tracks_pitch { freq_hz } else { 220.0 };
        let n = self.partials.clamp(1, super::MAX_MODES);
        let gap = self.spread.max(0.0);
        let a0 = LN_1000 / self.decay_time.max(0.02);
        let mut amp_sum = 0.0f32;
        for i in 0..n {
            let fi = i as f32;
            let ratio = 1.0 + fi * gap + self.inharmonicity * fi * fi;
            let freq = f0 * ratio;
            if freq >= sr * 0.45 {
                break;
            }
            let amp = (1.0 / (fi + 1.0)).powf(self.brightness.max(0.0));
            let decay = a0 * (1.0 + fi * 0.4); // upper partials ring shorter
            out.push(freq, amp, decay);
            amp_sum += amp;
        }
        let norm = if amp_sum > 1e-6 { vel / amp_sum } else { vel };
        out.scale_amps(norm);

        // Stick-click: a short, bright noise transient.
        if self.strike_noise > 1e-4 {
            out.noise_level = self.strike_noise * vel;
            out.noise_decay = 120.0; // ~ 57 ms tail
            out.noise_hp = 2500.0;
            out.noise_lp = 14_000.0;
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Bell");
        changed |= ui
            .add(unbounded_slider(&mut self.partials, 1..=12, "Partials"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.spread, 0.1..=2.0, "Spread"))
            .on_hover_text("Spacing of the partials (0.48 ≈ the 808 cowbell).")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.inharmonicity, 0.0..=1.0, "Inharmonicity"))
            .on_hover_text("Extra stretch on the upper partials — clangier.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.brightness, 0.0..=3.0, "Brightness rolloff"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.decay_time, 0.05..=5.0, "Decay time (s)"))
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.strike_noise, 0.0..=1.0, "Stick click"))
            .on_hover_text("Noise transient on the attack.")
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
    fn cowbell_has_inharmonic_partials_and_noise() {
        let mut buf = ModeBuffer::default();
        let m = MetalBell::default();
        m.excite(540.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n >= 2, "at least two partials");
        // Second partial ~1.48× the first (not a 2:1 harmonic).
        let ratio = buf.freq[1] / buf.freq[0];
        assert!(ratio > 1.3 && ratio < 1.7, "cowbell 1:1.5-ish ratio, got {ratio}");
        assert!(buf.noise_level > 0.0, "stick click present");
    }

    #[test]
    fn no_click_when_disabled() {
        let mut buf = ModeBuffer::default();
        let m = MetalBell { strike_noise: 0.0, ..MetalBell::default() };
        m.excite(540.0, 1.0, 48_000.0, &mut buf);
        assert_eq!(buf.noise_level, 0.0);
    }
}

//! Basic band-limited oscillator: triangle or sawtooth.
//!
//! This is the framework's reference plugin — the simplest thing that fills a
//! [`ModeBuffer`]. Each classic waveform is just its Fourier series (a bank of
//! harmonics), so it drops straight into the modal engine and is band-limited
//! (no aliasing) for free. It doubles as the desktop stand-in for the firmware's
//! `MODE_TRIANGLE_WAVE`.

use super::{unbounded_slider, FtmModel, ModeBuffer};

const PI: f32 = std::f32::consts::PI;
/// ln(1000): the factor giving a -60 dB fall over `decay_time`.
const LN_1000: f32 = 6.907_755;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Waveform {
    Triangle,
    Saw,
}

#[derive(Clone)]
pub struct BasicWave {
    pub waveform: Waveform,
    /// Number of harmonics summed.
    pub harmonics: usize,
    /// -60 dB decay time in seconds (0 for a sustained tone → very long).
    pub decay_time: f32,
}

impl Default for BasicWave {
    fn default() -> Self {
        Self {
            waveform: Waveform::Triangle,
            harmonics: 24,
            decay_time: 2.0,
        }
    }
}

impl FtmModel for BasicWave {
    fn display_name(&self) -> &'static str {
        "Basic Wave"
    }

    fn description(&self) -> &'static str {
        "Band-limited triangle / sawtooth — the framework's reference oscillator."
    }

    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer) {
        out.clear();
        if freq_hz <= 0.0 {
            return;
        }
        let decay = LN_1000 / self.decay_time.max(1e-3); // uniform across harmonics
        let n = self.harmonics.clamp(1, super::MAX_MODES);

        match self.waveform {
            Waveform::Triangle => {
                // x = (8/pi^2) Σ_{k>=0} (-1)^k sin(2π(2k+1)ft)/(2k+1)^2
                let scale = 8.0 / (PI * PI);
                for k in 0..n {
                    let m = (2 * k + 1) as f32;
                    let fm = freq_hz * m;
                    if fm >= sr * 0.45 {
                        break;
                    }
                    let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                    out.push(fm, scale * sign / (m * m) * vel, decay);
                }
            }
            Waveform::Saw => {
                // x = (2/pi) Σ_{m>=1} (-1)^{m+1} sin(2π m f t)/m
                let scale = 2.0 / PI;
                for i in 0..n {
                    let m = (i + 1) as f32;
                    let fm = freq_hz * m;
                    if fm >= sr * 0.45 {
                        break;
                    }
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    out.push(fm, scale * sign / m * vel, decay);
                }
            }
        }
    }

    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("Waveform")
            .selected_text(match self.waveform {
                Waveform::Triangle => "Triangle",
                Waveform::Saw => "Saw",
            })
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut self.waveform, Waveform::Triangle, "Triangle")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.waveform, Waveform::Saw, "Saw")
                    .changed();
            });
        changed |= ui
            .add(unbounded_slider(&mut self.harmonics, 1..=64, "Harmonics"))
            .on_hover_text("More harmonics = sharper waveform.")
            .changed();
        changed |= ui
            .add(unbounded_slider(&mut self.decay_time, 0.05..=30.0, "Decay time (s)"))
            .changed();
        changed
    }

    fn box_clone(&self) -> Box<dyn FtmModel> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_a_sane_bank() {
        let mut buf = ModeBuffer::default();
        let m = BasicWave::default();
        m.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 1, "triangle should have several harmonics");
        assert!(buf.freq[..buf.n].iter().all(|f| f.is_finite() && *f > 0.0));
        assert!(buf.freq[..buf.n].iter().all(|f| *f < 24_000.0), "no partials past Nyquist");

        // Saw includes even harmonics, so at the same count it reaches lower m's
        // fundamental too; just confirm it also produces a valid bank.
        let saw = BasicWave { waveform: Waveform::Saw, ..BasicWave::default() };
        saw.excite(220.0, 1.0, 48_000.0, &mut buf);
        assert!(buf.n > 1);
    }
}

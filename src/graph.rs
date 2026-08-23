//! The per-sample voice graph — the coupled, real-time synthesis path.
//!
//! Where the mode-bank engine ([`crate::instrument`]) precomputes each note as a
//! set of free-decaying sinusoids that never interact, this runs a small graph
//! of DSP blocks **one sample at a time**, so components can drive each other —
//! including in feedback loops. Every node is a [`Node`]: it `tick`s once per
//! sample, reads its input signal(s), and emits an output sample. Edges just
//! carry that one scalar signal.
//!
//! The bridge to the existing physics: a resonator still gets its modes
//! (`frequency, amplitude, decay`) from a model's [`ModeBuffer`] exactly as
//! today — but instead of playing each mode as a free oscillator with a fixed
//! decay envelope, it runs each as a **driven two-pole resonator**. Struck with
//! a unit impulse, a driven resonator rings out the *same* decaying sinusoid; fed
//! a continuous signal, it responds continuously; wired into a loop, it couples.
//! So the modal physics is unchanged — only how it's rendered, which is what
//! unlocks coupling.

use crate::models::ModeBuffer;

const TAU: f32 = std::f32::consts::TAU;
const PI: f32 = std::f32::consts::PI;

/// A per-voice, per-sample DSP block. `tick` advances one sample: it reads its
/// input signals (empty for a source like an exciter) and returns its output.
pub trait Node: Send {
    fn tick(&mut self, inputs: &[f32]) -> f32;
}

/// One resonant mode as a two-pole resonator. Its impulse response is
/// `amp · rⁿ · sin(θ(n+1))`, i.e. a sinusoid at the mode frequency decaying at
/// the mode's rate — the same thing the old free-oscillator bank produced —
/// but because it filters its input each sample it can be driven and coupled.
struct Mode {
    a1: f32,
    a2: f32,
    b0: f32,
    y1: f32,
    y2: f32,
}

impl Mode {
    fn new(freq: f32, decay: f32, amp: f32, sr: f32) -> Self {
        let theta = TAU * (freq / sr); // resonant angle
        let r = (-decay / sr).exp().clamp(0.0, 0.999_999); // per-sample pole radius
        Mode {
            a1: 2.0 * r * theta.cos(),
            a2: -(r * r),
            // Chosen so a unit impulse yields peak output ≈ `amp`.
            b0: amp * theta.sin(),
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.a1 * self.y1 + self.a2 * self.y2 + self.b0 * x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// A parallel bank of driven two-pole resonators — a resonator node built from a
/// model's mode bank. Each sample it sums every mode's response to the input.
pub struct ModalResonator {
    modes: Vec<Mode>,
}

impl ModalResonator {
    /// Build the resonator from a model's computed mode bank (freq/amp/decay).
    pub fn from_bank(bank: &ModeBuffer, sr: f32) -> Self {
        let modes = (0..bank.n)
            .map(|i| Mode::new(bank.freq[i], bank.decay[i].max(0.0), bank.amp[i], sr))
            .collect();
        ModalResonator { modes }
    }
}

impl Node for ModalResonator {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        // Sum all incoming signals — several exciters, or an upstream resonator.
        let x: f32 = inputs.iter().sum();
        let mut s = 0.0;
        for m in &mut self.modes {
            s += m.tick(x);
        }
        s
    }
}

/// Snare wires resting against a head: band-passed noise (the rattle) whose
/// level tracks how hard its input (the head's motion) is moving. Wired *both*
/// ways in the graph — driven by the bottom membrane and fed back into it — it
/// buzzes when the head moves and keeps re-exciting it, the coupled-snare sound.
pub struct SnareWires {
    rng: u32,
    follow: f32,
    follow_decay: f32,
    hp_a: f32,
    lp_a: f32,
    hp_s: f32,
    lp_s: f32,
    level: f32,
}

impl SnareWires {
    pub fn new(level: f32, hp: f32, lp: f32, sr: f32) -> Self {
        let a = |hz: f32| 1.0 - (-2.0 * PI * hz.max(1.0) / sr).exp();
        SnareWires {
            rng: 0x2545_f491,
            follow: 0.0,
            follow_decay: (-1.0 / (0.02 * sr)).exp(), // ~20 ms rattle release
            hp_a: a(hp),
            lp_a: a(lp),
            hp_s: 0.0,
            lp_s: 0.0,
            level,
        }
    }
}

impl Node for SnareWires {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let x: f32 = inputs.iter().sum();
        // Track how much the head is moving; the wires only rattle while it does.
        self.follow = x.abs().max(self.follow * self.follow_decay);
        // White noise → band-pass (one-pole HP then LP).
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        self.hp_s += self.hp_a * (white - self.hp_s);
        let hp = white - self.hp_s;
        self.lp_s += self.lp_a * (hp - self.lp_s);
        // Saturate: a real rattle clips, and this bounds the feedback loop when
        // the wires are wired back into the head (keeps coupling stable).
        (self.lp_s * self.follow * self.level).tanh()
    }
}

/// A small graph of [`Node`]s wired into a per-voice system, itself a [`Node`].
///
/// Every edge carries one signal, scaled by its **gain** (coupling strength),
/// with a **one-sample delay** (each node reads the previous frame's outputs):
/// a few samples of latency through a chain (inaudible), and — crucially — it
/// makes feedback loops (coupling) stable without special cases. `inputs[i]`
/// lists the `(source node, gain)` edges feeding node `i`; `output` is the node
/// whose sample is the voice's output.
pub struct Graph {
    nodes: Vec<Box<dyn Node>>,
    inputs: Vec<Vec<(usize, f32)>>,
    output: usize,
    last: Vec<f32>,
    cur: Vec<f32>,
    in_buf: Vec<f32>,
}

impl Graph {
    pub fn new(nodes: Vec<Box<dyn Node>>, inputs: Vec<Vec<(usize, f32)>>, output: usize) -> Self {
        let n = nodes.len();
        let fan_in = inputs.iter().map(|e| e.len()).max().unwrap_or(0);
        Graph {
            nodes,
            inputs,
            output,
            last: vec![0.0; n],
            cur: vec![0.0; n],
            in_buf: Vec::with_capacity(fan_in),
        }
    }
}

impl Node for Graph {
    fn tick(&mut self, _external: &[f32]) -> f32 {
        for i in 0..self.nodes.len() {
            self.in_buf.clear();
            for &(j, gain) in &self.inputs[i] {
                self.in_buf.push(self.last[j] * gain);
            }
            self.cur[i] = self.nodes[i].tick(&self.in_buf);
        }
        std::mem::swap(&mut self.last, &mut self.cur);
        self.last[self.output]
    }
}

/// A one-shot strike: emits `amp` on the first sample, then silence. A struck
/// resonator rings it out. (Velocity is carried here as `amp`.)
pub struct ImpulseExciter {
    amp: f32,
    fired: bool,
}

impl ImpulseExciter {
    pub fn new(amp: f32) -> Self {
        ImpulseExciter { amp, fired: false }
    }
}

impl Node for ImpulseExciter {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        if self.fired {
            0.0
        } else {
            self.fired = true;
            self.amp
        }
    }
}

/// A continuous drive — band-passed noise ("breath") that ramps in at note-on
/// and keeps sounding while the note is held. Driving a resonator with it makes
/// the resonator *sustain* (each mode holds a driven steady state) rather than
/// ring and decay — a blown/bowed tone. Note-off is handled by the voice's
/// release gate, which fades and frees it. Its own output ignores inputs.
pub struct DriveExciter {
    rng: u32,
    hp_a: f32,
    lp_a: f32,
    hp_s: f32,
    lp_s: f32,
    level: f32,
    atk: f32,
    atk_rate: f32,
}

impl DriveExciter {
    pub fn new(level: f32, hp: f32, lp: f32, sr: f32) -> Self {
        let a = |hz: f32| 1.0 - (-2.0 * PI * hz.max(1.0) / sr).exp();
        DriveExciter {
            rng: 0x9e37_79b9,
            hp_a: a(hp),
            lp_a: a(lp),
            hp_s: 0.0,
            lp_s: 0.0,
            level,
            atk: 0.0,
            atk_rate: 1.0 - (-1.0 / (0.01 * sr)).exp(), // ~10 ms breath onset
        }
    }
}

impl Node for DriveExciter {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        self.atk += (1.0 - self.atk) * self.atk_rate;
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        self.hp_s += self.hp_a * (white - self.hp_s);
        let hp = white - self.hp_s;
        self.lp_s += self.lp_a * (hp - self.lp_s);
        self.lp_s * self.level * self.atk
    }
}

/// A passthrough mixer: outputs the (already edge-scaled) sum of its inputs.
/// Used as a graph's output node so several components (e.g. a dry primary and a
/// wet body) can be blended by their edge gains.
pub struct Sum;

impl Node for Sum {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        inputs.iter().sum()
    }
}

/// A fixed-formant modal resonator (a body / oral cavity), built from a small
/// bank of resonances independent of the played note. Pure resonance — mix it
/// with the dry path via edge gains in the graph.
pub struct FormantResonator {
    inner: ModalResonator,
}

impl FormantResonator {
    pub fn new(formants: &ModeBuffer, sr: f32) -> Self {
        FormantResonator { inner: ModalResonator::from_bank(formants, sr) }
    }
}

impl Node for FormantResonator {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        self.inner.tick(inputs)
    }
}

/// The simplest complete voice graph: a one-shot strike driving a modal
/// resonator. This is what the struck graph instruments play per note —
/// `ImpulseExciter → ModalResonator`.
pub struct StruckVoice {
    exciter: ImpulseExciter,
    resonator: ModalResonator,
}

impl StruckVoice {
    /// Build from a model's mode bank (velocity is already baked into the bank's
    /// amplitudes, so the strike impulse is unit).
    pub fn new(bank: &ModeBuffer, sr: f32) -> Self {
        StruckVoice {
            exciter: ImpulseExciter::new(1.0),
            resonator: ModalResonator::from_bank(bank, sr),
        }
    }
}

impl Node for StruckVoice {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        let e = self.exciter.tick(&[]);
        self.resonator.tick(&[e])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Goertzel magnitude of `x` at frequency `f`.
    fn mag_at(x: &[f32], f: f32, sr: f32) -> f32 {
        let w = -TAU * f / sr;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (n, &v) in x.iter().enumerate() {
            re += v as f64 * (w as f64 * n as f64).cos();
            im += v as f64 * (w as f64 * n as f64).sin();
        }
        ((re * re + im * im).sqrt() / x.len() as f64) as f32
    }

    #[test]
    fn one_mode_rings_at_its_frequency_and_decays() {
        let sr = 48_000.0;
        let (f, decay, amp) = (440.0f32, 3.0f32, 1.0f32);
        let mut bank = ModeBuffer::default();
        bank.push(f, amp, decay);

        let mut exc = ImpulseExciter::new(1.0);
        let mut reso = ModalResonator::from_bank(&bank, sr);

        let n = sr as usize; // 1 second
        let mut y = vec![0.0f32; n];
        for s in y.iter_mut() {
            let e = exc.tick(&[]);
            *s = reso.tick(&[e]);
        }

        // Energy concentrated at 440 Hz, not at an unrelated frequency.
        let on = mag_at(&y, 440.0, sr);
        let off = mag_at(&y, 700.0, sr);
        assert!(on > 20.0 * off, "rings at 440 Hz (on={on:.4}, off={off:.4})");

        // Envelope decays as exp(-decay*t): compare early vs late RMS windows.
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        let early = rms(&y[0..4800]); // ~0..0.1 s
        let late = rms(&y[38400..43200]); // ~0.8..0.9 s
        // Over ~0.85 s at decay 3.0, amplitude drops by ~exp(-2.55) ≈ 0.078.
        let ratio = late / early;
        assert!(ratio < 0.2 && ratio > 0.01, "decays roughly as expected (ratio={ratio:.4})");
    }

    #[test]
    fn plate_bank_reproduces_the_decaying_sinusoid_sum() {
        // The per-sample resonator bank should render the same modal content as
        // the "ideal" sum of decaying sinusoids the old engine approximates.
        use crate::models::pure_plate::PurePlate;
        use crate::models::FtmModel;

        let sr = 48_000.0;
        let mut bank = ModeBuffer::default();
        PurePlate::default().excite(220.0, 1.0, sr, &mut bank);
        assert!(bank.n > 8);

        // Reference: sum_i amp_i * exp(-decay_i * t) * sin(2π f_i t).
        let n = 24_000usize; // 0.5 s
        let mut reference = vec![0.0f32; n];
        for (k, s) in reference.iter_mut().enumerate() {
            let t = k as f32 / sr;
            let mut acc = 0.0f32;
            for i in 0..bank.n {
                acc += bank.amp[i] * (-bank.decay[i] * t).exp() * (TAU * bank.freq[i] * t).sin();
            }
            *s = acc;
        }

        // Per-sample graph render.
        let mut exc = ImpulseExciter::new(1.0);
        let mut reso = ModalResonator::from_bank(&bank, sr);
        let mut got = vec![0.0f32; n];
        for s in got.iter_mut() {
            let e = exc.tick(&[]);
            *s = reso.tick(&[e]);
        }

        // Compare the two at each mode frequency: the per-mode magnitudes should
        // track (same freqs, amplitudes, decays → same spectral envelope).
        let mut worst = 0.0f32;
        for i in 0..bank.n {
            let f = bank.freq[i];
            if f < 40.0 || f > 18_000.0 {
                continue;
            }
            let a = mag_at(&reference, f, sr);
            let b = mag_at(&got, f, sr);
            if a > 1e-4 {
                let rel = (a - b).abs() / a;
                worst = worst.max(rel);
            }
        }
        assert!(worst < 0.25, "per-mode magnitudes match within 25% (worst={worst:.3})");
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    use crate::models::pure_string::PureString;
    use crate::models::FtmModel;

    fn render(node: &mut dyn Node, n: usize) -> Vec<f32> {
        (0..n).map(|_| node.tick(&[])).collect()
    }
    fn mag_at(x: &[f32], f: f32, sr: f32) -> f32 {
        let w = -std::f32::consts::TAU * f / sr;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (k, &v) in x.iter().enumerate() {
            re += v as f64 * (w as f64 * k as f64).cos();
            im += v as f64 * (w as f64 * k as f64).sin();
        }
        ((re * re + im * im).sqrt() / x.len() as f64) as f32
    }

    #[test]
    fn body_graph_colours_and_differs_from_bare_string() {
        let sr = 48_000.0;
        let mut bank = ModeBuffer::default();
        // A low note so the string's own modes sit above the body formants.
        PureString::default().excite(110.0, 1.0, sr, &mut bank);

        // Bare string (Impulse → String).
        let mut bare = StruckVoice::new(&bank, sr);
        let bare_y = render(&mut bare, 24_000);

        // Strike → String → Body(formants) → Mix(dry string + wet body).
        let mut body = ModeBuffer::default();
        body.push(100.0, 0.6, 8.0);
        body.push(210.0, 0.4, 11.0);
        body.push(390.0, 0.3, 15.0);
        let nodes: Vec<Box<dyn Node>> = vec![
            Box::new(ImpulseExciter::new(1.0)),
            Box::new(ModalResonator::from_bank(&bank, sr)),
            Box::new(FormantResonator::new(&body, sr)),
            Box::new(Sum),
        ];
        // edges: strike→string, string→body, string→mix (dry), body→mix (wet 0.5)
        let edges = vec![
            vec![],
            vec![(0, 1.0)],
            vec![(1, 1.0)],
            vec![(1, 1.0), (2, 0.5)],
        ];
        let mut g = Graph::new(nodes, edges, 3);
        let bodied_y = render(&mut g, 24_000);

        // Both make sound.
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        assert!(rms(&bare_y) > 1e-4 && rms(&bodied_y) > 1e-4, "both non-silent");

        // The body boosts its formant region: 210 Hz is louder relative to the
        // string's fundamental in the bodied version than in the bare one.
        let ratio = |y: &[f32]| mag_at(y, 210.0, sr) / mag_at(y, 110.0, sr).max(1e-9);
        assert!(
            ratio(&bodied_y) > ratio(&bare_y) * 1.2,
            "body adds resonance at its formant (bodied {:.3} vs bare {:.3})",
            ratio(&bodied_y),
            ratio(&bare_y)
        );
    }
}

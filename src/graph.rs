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

/// A control message broadcast to every node in a voice's graph (not audio —
/// these arrive at control rate, on a bend move or a key-up).
#[derive(Clone, Copy)]
pub enum Control {
    /// Pitch-bend ratio (1.0 = no bend): resonators retune, exciters ignore it.
    Bend(f32),
    /// Key gate: `false` on note-off — driven exciters stop so the resonator
    /// rings out. Struck/one-shot exciters ignore it.
    Gate(bool),
}

/// A per-voice, per-sample DSP block. `tick` advances one sample: it reads its
/// input signals (empty for a source like an exciter) and returns its output.
pub trait Node: Send {
    fn tick(&mut self, inputs: &[f32]) -> f32;
    /// Receive a control message. Default: ignore.
    fn control(&mut self, _c: Control) {}
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
    theta0: f32, // base resonant angle (for retuning under pitch bend)
    r: f32,      // pole radius
}

impl Mode {
    fn new(freq: f32, decay: f32, amp: f32, sr: f32) -> Self {
        let theta = TAU * (freq / sr); // resonant angle
        let r = (-decay / sr).exp().clamp(0.0, 0.999_999); // per-sample pole radius
        Mode {
            a1: 2.0 * r * theta.cos(),
            a2: -(r * r),
            // Chosen so a unit impulse yields peak output ≈ `amp` (struck use).
            b0: amp * theta.sin(),
            y1: 0.0,
            y2: 0.0,
            theta0: theta,
            r,
        }
    }

    /// A **constant-peak-gain** band-pass: its magnitude at resonance is exactly
    /// `peak_gain`, independent of Q. Used when the resonator is a *filter* driven
    /// by another signal (a body/oral tract colouring a string, an air column
    /// blown by breath) — so it *colours* the input rather than amplifying it by
    /// ~1/(1−r), which is what makes a naive high-Q resonator explode when driven.
    fn new_filter(freq: f32, decay: f32, peak_gain: f32, sr: f32) -> Self {
        let theta = TAU * (freq / sr);
        let r = (-decay / sr).exp().clamp(0.0, 0.999_999);
        // For D(z) = 1 − a1 z⁻¹ − a2 z⁻², a1 = 2r cosθ, a2 = −r², the response is
        // H = b0/D. Evaluated at the resonance z = e^{jθ}:
        //   Re D = (1−r)[(1+r) − 2r cos²θ],   Im D = r(1−r) sin2θ.
        // Setting b0 = peak_gain·|D(θ)| makes |H(θ)| = peak_gain exactly.
        let c = theta.cos();
        let d_re = (1.0 - r) * ((1.0 + r) - 2.0 * r * c * c);
        let d_im = r * (1.0 - r) * (2.0 * theta).sin();
        let d_mag = (d_re * d_re + d_im * d_im).sqrt();
        Mode {
            a1: 2.0 * r * c,
            a2: -(r * r),
            b0: peak_gain * d_mag,
            y1: 0.0,
            y2: 0.0,
            theta0: theta,
            r,
        }
    }

    /// Retune the pole to `bend`× the base frequency (pitch wheel).
    #[inline]
    fn retune(&mut self, bend: f32) {
        let theta = (self.theta0 * bend).clamp(0.0, PI * 0.9);
        self.a1 = 2.0 * self.r * theta.cos();
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
    /// Struck normalization: a unit impulse rings out at the modes' amplitudes.
    pub fn from_bank(bank: &ModeBuffer, sr: f32) -> Self {
        let modes = (0..bank.n)
            .map(|i| Mode::new(bank.freq[i], bank.decay[i].max(0.0), bank.amp[i], sr))
            .collect();
        ModalResonator { modes }
    }

    /// Build the resonator as a bank of **constant-peak-gain filters** — each
    /// mode's amplitude becomes its passband gain. For a resonator used as a
    /// filter (a body/tract), so it colours a driving signal instead of
    /// amplifying it by its Q.
    pub fn from_bank_filter(bank: &ModeBuffer, sr: f32) -> Self {
        let modes = (0..bank.n)
            .map(|i| Mode::new_filter(bank.freq[i], bank.decay[i].max(0.0), bank.amp[i], sr))
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
    fn control(&mut self, c: Control) {
        if let Control::Bend(b) = c {
            for m in &mut self.modes {
                m.retune(b);
            }
        }
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
    fn control(&mut self, c: Control) {
        for n in &mut self.nodes {
            n.control(c);
        }
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
    env: f32,
    env_target: f32,
    atk_rate: f32,
    rel_rate: f32,
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
            env: 0.0,
            env_target: 1.0, // driving while the note is held
            atk_rate: 1.0 - (-1.0 / (0.01 * sr)).exp(), // ~10 ms onset
            rel_rate: 1.0 - (-1.0 / (0.03 * sr)).exp(), // ~30 ms stop on key-up
        }
    }
}

impl Node for DriveExciter {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk_rate } else { self.rel_rate };
        self.env += (self.env_target - self.env) * rate;
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        self.hp_s += self.hp_a * (white - self.hp_s);
        let hp = white - self.hp_s;
        self.lp_s += self.lp_a * (hp - self.lp_s);
        self.lp_s * self.level * self.env
    }
    fn control(&mut self, c: Control) {
        // Key-up stops the breath, so the resonator rings out on its own.
        if let Control::Gate(on) = c {
            self.env_target = if on { 1.0 } else { 0.0 };
        }
    }
}

/// A self-oscillating reed/lip — a nonlinear exciter that reads the resonator's
/// pressure back (a feedback edge) and produces flow through a reed valve. Wired
/// `Reed → Resonator` and `Resonator → Reed`, the nonlinearity + the resonator's
/// feedback form a limit cycle: it oscillates on its own (a reed/brass tone)
/// rather than just ringing a struck/breathed resonance. `tanh` bounds the loop.
pub struct ReedExciter {
    pressure: f32,
    stiffness: f32,
    rng: u32,
    env: f32,
    env_target: f32,
    atk_rate: f32,
    rel_rate: f32,
}

impl ReedExciter {
    pub fn new(pressure: f32, stiffness: f32, sr: f32) -> Self {
        ReedExciter {
            pressure,
            stiffness,
            rng: 0x1234_5678,
            env: 0.0,
            env_target: 1.0,
            atk_rate: 1.0 - (-1.0 / (0.02 * sr)).exp(), // ~20 ms onset
            rel_rate: 1.0 - (-1.0 / (0.03 * sr)).exp(),
        }
    }
}

impl Node for ReedExciter {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk_rate } else { self.rel_rate };
        self.env += (self.env_target - self.env) * rate;
        let bore: f32 = inputs.iter().sum(); // resonator pressure fed back
        let p = self.pressure * self.env; // gated mouth pressure
        let delta = p - bore; // pressure across the reed
        // Reed opening closes as the pressure difference rises (nonlinear valve).
        let opening = (1.0 - self.stiffness * delta).clamp(0.0, 1.0);
        // Turbulent breath noise through the aperture — a continuous broadband
        // stimulus (∝ the open flow) that kicks and keeps the bore singing,
        // rather than waiting for the feedback to build. It is also the breathy
        // hiss of a real reed.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let flow = delta * opening;
        let turbulence = white * (opening * p).abs() * 0.25;
        (flow + turbulence).tanh() // flow into the bore, bounded
    }
    fn control(&mut self, c: Control) {
        if let Control::Gate(on) = c {
            self.env_target = if on { 1.0 } else { 0.0 };
        }
    }
}

/// A piano / dulcimer **hammer**: a felt-covered mass making nonlinear Hertzian
/// contact with the string. The hammer flies in at the key's velocity; while the
/// felt is compressed it pushes back with `F = K·δ^p` (δ = compression, `p` the
/// felt nonlinearity ≈ 2.5), which decelerates it until it rebounds and
/// separates. It reads the string surface over a feedback edge, so a harder
/// strike compresses the felt deeper, leaves *sooner*, and sounds brighter — the
/// hallmark of real piano dynamics. A one-shot: silent once the hammer departs.
pub struct HammerExciter {
    pos: f32,       // hammer position (internal, well-scaled units)
    vel: f32,       // hammer velocity (from key velocity)
    stiffness: f32, // felt stiffness K
    exponent: f32,  // felt nonlinearity p
    mass: f32,      // hammer mass
    inv_sr: f32,
    done: bool,
}

impl HammerExciter {
    // The string's fed-back signal is read at this scale (a real string barely
    // moves compared to the hammer's travel) and the felt force is emitted at
    // this scale — so a graph can wire the hammer with ordinary ~1.0 edges.
    const READ: f32 = 0.02;
    const OUT: f32 = 0.003;

    /// `hardness` 0..1 maps to felt stiffness K = 10^(5.5 + 1.5·hardness), i.e.
    /// ~3×10⁵ (soft/dark, long contact) to ~10⁷ (hard/bright, ~1 ms contact).
    /// `felt` is the compression exponent p (piano felt ≈ 2.2–3.5). Mass is fixed
    /// so the contact lands in the real ~1–9 ms range across the hardness sweep.
    pub fn new(velocity: f32, hardness: f32, felt: f32, sr: f32) -> Self {
        let k = 10f32.powf(5.5 + 1.5 * hardness.clamp(0.0, 1.0));
        HammerExciter {
            pos: 0.0,
            // Key velocity → approach speed; ×8 puts the contact and the felt
            // compression in a well-conditioned numeric range.
            vel: velocity.clamp(0.02, 1.0) * 8.0,
            stiffness: k,
            exponent: felt.clamp(1.0, 4.0),
            mass: 2.0e-4,
            inv_sr: 1.0 / sr,
            done: false,
        }
    }
}

impl Node for HammerExciter {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        if self.done {
            return 0.0;
        }
        let x: f32 = inputs.iter().sum::<f32>() * Self::READ; // string surface, fed back
        let compression = self.pos - x;
        let force = if compression > 0.0 {
            self.stiffness * compression.powf(self.exponent)
        } else {
            0.0
        };
        // Semi-implicit Euler: the felt reaction decelerates the hammer.
        self.vel -= force / self.mass * self.inv_sr;
        self.pos += self.vel * self.inv_sr;
        // The hammer has left once it is clear of the string and moving away.
        if compression <= 0.0 && self.vel <= 0.0 {
            self.done = true;
        }
        force * Self::OUT
    }
}

/// A **bow**: the classic stick-slip friction drive. The hair moves across the
/// string at a steady `speed` under a `force`; the friction it applies depends
/// nonlinearly on the slip velocity (string velocity − bow velocity). Near
/// sticking, friction is high and the string travels with the bow; past a
/// threshold it breaks away and slips back — the Helmholtz motion of a bowed
/// string. It reads the string velocity over a feedback edge; `tanh` bounds the
/// friction so the loop is stable. Continuous — gated off on note-off.
pub struct BowExciter {
    speed: f32,      // bow velocity
    force: f32,      // bow pressure
    slip: f32,       // Stribeck slip-velocity scale
    last: f32,       // previous fed-back sample (to estimate string velocity)
    rng: u32,        // friction-noise generator
    env: f32,
    env_target: f32,
    atk_rate: f32,
    rel_rate: f32,
}

impl BowExciter {
    pub fn new(speed: f32, force: f32, sr: f32) -> Self {
        BowExciter {
            speed,
            force,
            slip: 0.12,
            last: 0.0,
            rng: 0x71fe_1a3b,
            env: 0.0,
            env_target: 1.0,
            atk_rate: 1.0 - (-1.0 / (0.05 * sr)).exp(), // ~50 ms bow onset
            rel_rate: 1.0 - (-1.0 / (0.08 * sr)).exp(), // ~80 ms release
        }
    }
}

impl Node for BowExciter {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk_rate } else { self.rel_rate };
        self.env += (self.env_target - self.env) * rate;
        let x: f32 = inputs.iter().sum();
        // String velocity at the bow ≈ the derivative of the fed-back
        // displacement, scaled so it lands on the same (bow-speed) scale — the
        // fix for the old model, where it was ~30× too small so the relative
        // velocity was dominated by the bow's DC push and stick-slip never
        // engaged (you just heard the string's modes slowly accumulating).
        const VSCALE: f32 = 25.0;
        let v_string = (x - self.last) * VSCALE;
        self.last = x;
        let v_bow = self.speed * self.env;
        let v_rel = v_string - v_bow;
        // Stribeck friction: high near sticking (v_rel → 0), falling off sharply
        // as the string slips faster. The negative slope past the peak is the
        // "negative resistance" that sustains the Helmholtz stick-slip motion.
        let mu = 0.15 + 0.85 * (-(v_rel / self.slip).powi(2)).exp();
        let friction = -self.force * self.env * v_rel.signum() * mu;
        // Continuous friction/scratch noise (∝ bow pressure): a broadband
        // stimulus that keeps the string excited from the first sample instead
        // of waiting for the stick-slip loop to build — and the natural bow hiss.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let scratch = white * self.force * self.env * 0.15;
        (friction + scratch).tanh()
    }
    fn control(&mut self, c: Control) {
        if let Control::Gate(on) = c {
            self.env_target = if on { 1.0 } else { 0.0 };
        }
    }
}

/// A **voice**: a vocal-fold (glottal) source for singing or growling into an
/// instrument (a didgeridoo, a sax growl) or for playing a vocal-tract resonator
/// directly. It generates a train of glottal-flow pulses (a Rosenberg-style
/// model) at the played pitch — a smooth open phase, a sharper closing, then a
/// closed rest — whose harmonic-rich buzz the following resonator (a body/tract
/// or horn) filters, source-filter style. Continuous; gated on note-off.
pub struct VoiceExciter {
    phase: f32,   // 0..1 within the glottal cycle
    f0: f32,      // phonation frequency (Hz)
    incr: f32,    // phase increment per sample
    open_q: f32,  // open quotient — fraction of the cycle the folds are open
    level: f32,
    inv_sr: f32,
    env: f32,
    env_target: f32,
    atk_rate: f32,
    rel_rate: f32,
}

impl VoiceExciter {
    pub fn new(f0: f32, open_q: f32, level: f32, sr: f32) -> Self {
        let f0 = f0.max(1.0);
        VoiceExciter {
            phase: 0.0,
            f0,
            incr: f0 / sr,
            open_q: open_q.clamp(0.1, 0.95),
            level,
            inv_sr: 1.0 / sr,
            env: 0.0,
            env_target: 1.0,
            atk_rate: 1.0 - (-1.0 / (0.03 * sr)).exp(), // ~30 ms onset
            rel_rate: 1.0 - (-1.0 / (0.04 * sr)).exp(),
        }
    }
}

impl Node for VoiceExciter {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk_rate } else { self.rel_rate };
        self.env += (self.env_target - self.env) * rate;
        self.phase += self.incr;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        let oq = self.open_q;
        // Rosenberg glottal flow: smooth rise over the first 60% of the open
        // phase, sharper fall over the last 40%, then closed (zero).
        let g = if self.phase < oq {
            let t = self.phase / oq;
            if t < 0.6 {
                0.5 * (1.0 - (PI * (t / 0.6)).cos()) // rise 0 → 1
            } else {
                (PI * 0.5 * ((t - 0.6) / 0.4)).cos() // fall 1 → 0 (sharp close)
            }
        } else {
            0.0
        };
        // Centre out the DC of the pulse train so it drives the resonator cleanly.
        (g - oq * 0.5) * self.level * self.env
    }
    fn control(&mut self, c: Control) {
        match c {
            Control::Gate(on) => self.env_target = if on { 1.0 } else { 0.0 },
            Control::Bend(r) => self.incr = (self.f0 * r.max(0.01)) * self.inv_sr,
        }
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
        // A body colours its input, so its modes are constant-peak-gain filters
        // (peak gain = the formant amplitude), not energy-adding resonators.
        FormantResonator { inner: ModalResonator::from_bank_filter(formants, sr) }
    }
}

impl Node for FormantResonator {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        self.inner.tick(inputs)
    }
    fn control(&mut self, c: Control) {
        self.inner.control(c);
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
    fn control(&mut self, c: Control) {
        self.resonator.control(c); // retune on bend
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
            vec![(1, 1.0), (2, 1.0)], // wet body at full edge
        ];
        let mut g = Graph::new(nodes, edges, 3);
        let bodied_y = render(&mut g, 24_000);

        // Both make sound.
        let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
        assert!(rms(&bare_y) > 1e-4 && rms(&bodied_y) > 1e-4, "both non-silent");

        // The body boosts its formant region: 210 Hz is louder relative to the
        // string's fundamental in the bodied version than in the bare one.
        let ratio = |y: &[f32]| mag_at(y, 210.0, sr) / mag_at(y, 110.0, sr).max(1e-9);
        // The body is a constant-peak-gain filter, so it colours gently: the
        // 210 Hz formant region gains relative to the string's fundamental.
        assert!(
            ratio(&bodied_y) > ratio(&bare_y) * 1.05,
            "body adds resonance at its formant (bodied {:.3} vs bare {:.3})",
            ratio(&bodied_y),
            ratio(&bare_y)
        );
    }
    #[test]
    fn bend_retunes_a_mode() {
        let sr = 48_000.0;
        let mut bank = ModeBuffer::default();
        bank.push(440.0, 1.0, 2.0);
        let zcr = |y: &[f32]| y.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        let mut base = StruckVoice::new(&bank, sr);
        let a: Vec<f32> = (0..24_000).map(|_| base.tick(&[])).collect();
        let mut up = StruckVoice::new(&bank, sr);
        up.control(Control::Bend(2.0)); // an octave up
        let b: Vec<f32> = (0..24_000).map(|_| up.tick(&[])).collect();
        assert!(zcr(&b) > zcr(&a) * 3 / 2, "bend up raises pitch ({} vs {})", zcr(&b), zcr(&a));
    }

    #[test]
    fn drive_stops_on_gate_off() {
        let sr = 48_000.0;
        let mut d = DriveExciter::new(1.0, 300.0, 3_000.0, sr);
        for _ in 0..2_400 { d.tick(&[]); } // ramp up
        let on: f32 = (0..2_400).map(|_| d.tick(&[]).abs()).sum::<f32>() / 2_400.0;
        d.control(Control::Gate(false));
        for _ in 0..4_800 { d.tick(&[]); } // ~30 ms release
        let off: f32 = (0..2_400).map(|_| d.tick(&[]).abs()).sum::<f32>() / 2_400.0;
        assert!(off < on * 0.1, "breath stops after key-up (on={on:.4} off={off:.4})");
    }
}

#[cfg(test)]
mod new_exciter_tests {
    use super::*;
    use crate::models::{pure_string::PureString, FtmModel, ModeBuffer};

    fn string_res(sr: f32) -> ModalResonator {
        let mut b = ModeBuffer::default();
        PureString::default().excite(220.0, 1.0, sr, &mut b);
        ModalResonator::from_bank(&b, sr)
    }

    // Hammer(0) → String(1) → Mix(2), with String(1) → Hammer(0) feedback.
    fn hammer_rig(vel: f32, output: usize, sr: f32) -> Graph {
        let nodes: Vec<Box<dyn Node>> = vec![
            Box::new(HammerExciter::new(vel, 0.6, 2.5, sr)),
            Box::new(string_res(sr)),
            Box::new(Sum),
        ];
        let inputs = vec![vec![(1usize, 1.0f32)], vec![(0usize, 1.0f32)], vec![(1usize, 1.0f32)]];
        Graph::new(nodes, inputs, output)
    }

    #[test]
    fn hammer_has_a_finite_velocity_dependent_contact_window() {
        let sr = 48_000.0;
        // Sound out (output = Mix): stable, bounded, and audible.
        let mut g = hammer_rig(1.0, 2, sr);
        let mut peak = 0.0f32;
        let mut energy = 0.0f32;
        for _ in 0..48_000 {
            let y = g.tick(&[]);
            assert!(y.is_finite() && y.abs() < 10.0, "hammer loop stays bounded");
            peak = peak.max(y.abs());
            energy += y * y;
        }
        assert!(energy > 1e-4, "the hammer excites the string");

        // Contact window (output = the hammer force): finite, and a harder hit
        // stays in contact no longer than a soft one (felt gets stiffer).
        let contact = |vel: f32| -> usize {
            let mut g = hammer_rig(vel, 0, sr);
            (0..8000).filter(|_| g.tick(&[]).abs() > 1e-7).count()
        };
        let soft = contact(0.3);
        let hard = contact(1.0);
        let ms = |n: usize| n as f32 / sr * 1000.0;
        assert!((0.3..=12.0).contains(&ms(soft)), "soft contact {} ms in piano range", ms(soft));
        assert!(hard <= soft + 48, "harder strike is not a longer contact (soft={soft} hard={hard})");
    }

    #[test]
    fn bow_sustains_and_stays_bounded() {
        let sr = 48_000.0;
        let nodes: Vec<Box<dyn Node>> = vec![
            Box::new(BowExciter::new(0.6, 1.0, sr)),
            Box::new(string_res(sr)),
            Box::new(Sum),
        ];
        let inputs = vec![vec![(1usize, 1.0f32)], vec![(0usize, 1.0f32)], vec![(1usize, 1.0f32)]];
        let mut g = Graph::new(nodes, inputs, 2);
        let y: Vec<f32> = (0..48_000).map(|_| g.tick(&[])).collect();
        assert!(y.iter().all(|v| v.is_finite() && v.abs() < 10.0), "bow loop stays bounded");
        let rms = |a: &[f32]| (a.iter().map(|s| s * s).sum::<f32>() / a.len() as f32).sqrt();
        assert!(rms(&y[24_000..]) > 1e-3, "the bow sustains a tone while held");
    }

    #[test]
    fn voice_is_pitched_and_gates_off() {
        let sr = 48_000.0;
        let mut v = VoiceExciter::new(110.0, 0.6, 0.4, sr);
        let y: Vec<f32> = (0..48_000).map(|_| v.tick(&[])).collect();
        assert!(y.iter().all(|s| s.is_finite()));
        // Zero-crossings over a second ≈ 2·f0 for a pitched buzz (period-accurate).
        let zc = y[9600..].windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        let approx_f0 = zc as f32 / 2.0 / ((48_000 - 9600) as f32 / sr);
        assert!((approx_f0 - 110.0).abs() < 12.0, "voice pitched near 110 Hz (got {approx_f0})");
        // Note-off: the glottis stops and the source falls silent.
        v.control(Control::Gate(false));
        let tail: Vec<f32> = (0..12_000).map(|_| v.tick(&[])).collect();
        let end = &tail[8_000..];
        assert!(end.iter().all(|s| s.abs() < 1e-2), "voice goes silent after note-off");
    }
}


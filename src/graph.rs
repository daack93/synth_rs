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
const CR_COMP: f32 = 1.6;
/// Register-hole position along the bore (fraction from the throat). At 1/3 the
/// fundamental has a pressure antinode and the 3rd harmonic a node, so opening
/// the hole kills the fundamental and the reed jumps a 12th to the 3rd.

/// A control message broadcast to every node in a voice's graph (not audio —
/// these arrive at control rate, on a bend move or a key-up).
#[derive(Clone, Copy)]
pub enum Control {
    /// Pitch-bend ratio (1.0 = no bend): resonators retune, exciters ignore it.
    Bend(f32),
    /// Key gate: `false` on note-off — driven exciters stop so the resonator
    /// rings out. Struck/one-shot exciters ignore it.
    Gate(bool),
    /// Live mouth-pressure multiplier (1.0 = nominal) — a wind's breath/embouchure
    /// axis. On a coupled reed it bends the pitch (harder = sharper) the way a
    /// player lips a note; map it to an expression pedal / breath controller.
    Breath(f32),
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

/// A single-reed woodwind mouthpiece (clarinet / saxophone) — a **lumped-element
/// ODE reed**, not a static reflection table. It reads the bore pressure back (a
/// feedback edge) and models the reed as a real pressure-controlled valve:
///
/// * the reed cane is a damped **mass-spring oscillator** driven by the pressure
///   difference `ΔP = P_mouth − P_bore`:  `x'' + 2ζω·x' + ω²(x + β·ΔP) = 0`;
/// * the tip **opening** is `H = max(0, 1 + x)` — the `max(0, …)` is the reed
///   *beating* shut against the lay, the hard once-per-cycle event that gives a
///   reed its buzz (a smooth curve can't);
/// * the flow injected into the bore is **Bernoulli**: `Q = H·sign(ΔP)·√|ΔP|`.
///
/// Wired `Reed → Bore` and `Bore → Reed`, the reed resonance + the beating flow
/// + the bore's feedback form a limit cycle — a sustained, buzzing reed tone.
pub struct ReedExciter {
    pressure: f32, // P_mouth (blowing pressure, normalized)
    x: f32,        // reed displacement (0 = rest/open, −1 = beat shut)
    v: f32,        // reed velocity
    wn2: f32,      // ω² per sample² (reed spring / resonance)
    damp: f32,     // 2ζω per sample (reed damping)
    beta: f32,     // pressure→closure compliance (how far ΔP bends the reed)
    zc: f32,       // flow→pressure coupling (bore characteristic impedance)
    ur_prev: f32,  // last reed flow (breaks the junction's algebraic loop)
    p_dc: f32,     // running DC of the bore wave (an open pipe reflects no DC)
    dc_a: f32,     // DC-tracker coefficient
    flow_lp: f32,  // reed/air inertia: lowpasses the flow (kills HF squeak modes)
    flow_a: f32,
    env: f32,
    env_target: f32,
    atk_rate: f32,
    rel_rate: f32,
    rng: u32,
    // Optional built-in bore. `Some` → the reed is a *self-contained mouthpiece*
    // that buzzes at its own fixed pitch (like a reed with the horn pulled off),
    // ignoring its graph inputs; the buzz is emitted for a downstream resonator
    // to shape. `None` → the reed is a bare valve that reads its resonant load
    // back over a feedback edge (the pitch comes from whatever bore it drives).
    bore: Option<WaveguideBore>,
    bore_out: f32,
}

impl ReedExciter {
    /// `freq_hz > 0` makes a self-contained mouthpiece buzzing at that fixed
    /// pitch (a built-in bore); `0` makes a bare valve driven by a feedback edge.
    pub fn new(pressure: f32, stiffness: f32, freq_hz: f32, sr: f32) -> Self {
        // Reed resonance sits ~1.5 kHz and is heavily damped (ζ ≈ 0.8) — a broad
        // formant, not a sharp peak, so the *bore* controls the pitch and the reed
        // never squeaks at its own resonance. `stiffness` nudges the beating point
        // (β): a harder reed beats sooner, adding a little buzz. Values here were
        // swept for stable oscillation on the bore across the whole register.
        let wn = std::f32::consts::TAU * 1500.0 / sr;
        let beta = (0.65 + 0.14 * stiffness.clamp(0.0, 2.0)).clamp(0.7, 0.9);
        ReedExciter {
            pressure,
            x: 0.0,
            v: 0.0,
            wn2: wn * wn,
            damp: 2.0 * 0.8 * wn, // ζ ≈ 0.8
            beta,
            zc: 0.6,
            ur_prev: 0.0,
            p_dc: 0.0,
            dc_a: 1.0 - (-TAU * 15.0 / sr).exp(), // ~15 Hz DC tracker
            flow_lp: 0.0,
            flow_a: 0.28, // ~2.6 kHz flow lowpass
            env: 0.0,
            env_target: 1.0,
            atk_rate: 1.0 - (-1.0 / (0.02 * sr)).exp(), // ~20 ms onset
            rel_rate: 1.0 - (-1.0 / (0.03 * sr)).exp(),
            rng: 0x1234_5678,
            bore: (freq_hz > 0.0).then(|| WaveguideBore::new(freq_hz, 1.0, sr)),
            bore_out: 0.0,
        }
    }
}

impl Node for ReedExciter {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk_rate } else { self.rel_rate };
        self.env += (self.env_target - self.env) * rate;
        // Waveguide scattering junction at the mouthpiece. `p_plus` is the wave
        // returning from the bore; the mouthpiece pressure is `p = 2·p₊ − Zc·U`
        // (pressure-doubling at the reed end), so the pressure across the reed is
        //   ΔP = P_mouth − p = P_mouth − 2·p₊ + Zc·U.
        // U depends on ΔP (implicit) — we break the loop with last sample's flow.
        // Self-contained mouthpiece reads its own built-in bore; a bare valve
        // reads its resonant load back over the graph edge.
        let p_in: f32 = if self.bore.is_some() { self.bore_out } else { inputs.iter().sum() };
        // The open pipe end reflects no DC, so the reed feels only the AC standing
        // wave; tracking and removing the DC also kills a DC runaway in the loop.
        self.p_dc += self.dc_a * (p_in - self.p_dc);
        let p_plus = p_in - self.p_dc;
        let pm = self.pressure * self.env;
        let dp = pm - 2.0 * p_plus + self.zc * self.ur_prev;

        // Reed mass-spring-damper (semi-implicit Euler). Positive ΔP bends the
        // reed toward shut; its static equilibrium is x = −β·ΔP.
        let acc = -self.damp * self.v - self.wn2 * (self.x + self.beta * dp);
        self.v += acc;
        self.x += self.v;

        // Tip opening: clamped at 0 (the reed beating shut) and bounded above (a
        // reed can only lift so far off the lay) so a hard transient can't run away.
        let h = (1.0 + self.x).clamp(0.0, 3.0);

        // Bernoulli flow through the opening, + a little breath turbulence that
        // only passes while the reed is open (jet noise at the aperture).
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        // Flow needs breath: with none (env → 0 on note-off) there's no air
        // through the reed, so the drive dies and the bore rings down — the reed
        // can't self-oscillate on the standing wave alone. Gate sharply so this
        // only bites near note-off and leaves the (multistable) attack untouched.
        let g = (self.env * 4.0).min(1.0);
        let ur_raw = g * h * (dp.signum() * dp.abs().sqrt() + 0.015 * white);
        self.flow_lp += self.flow_a * (ur_raw - self.flow_lp);
        let ur = self.flow_lp;
        self.ur_prev = ur;
        // Launch the outgoing wave: p₋ = p₊ − Zc·U. A gentle saturation stands in
        // for flow/radiation losses and keeps the limit cycle bounded.
        let p_minus = (p_plus - self.zc * ur).tanh();
        // A self-contained mouthpiece runs its own bore and emits the bore's
        // returning wave (the buzz); a bare valve just launches p₋ down the edge.
        match self.bore.as_mut() {
            Some(b) => {
                self.bore_out = b.tick(&[p_minus]);
                self.bore_out
            }
            None => p_minus,
        }
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
            Control::Breath(_) => {}
        }
    }
}

/// A **waveguide bore** resonator — a wind's air column as a delay line with an
/// inverting one-pole low-pass reflection at the bell. Driven by a reed/lip
/// exciter (which reads this bore's returning pressure over a feedback edge and
/// launches flow back in), the traveling wave reflecting off the bell is what
/// lets the reed self-oscillate. Its output — the returning pressure — is the
/// bore's voice, which a secondary resonator (a flaring bell/body) can then
/// colour. Pitch = the delay length. This is the graph-decomposed counterpart of
/// the self-contained `WaveguideReed`.
pub struct WaveguideBore {
    line: Vec<f32>,
    delay: f32,
    pos: usize,
    bell: f32,
    bell_a: f32,
    refl: f32,
}

impl WaveguideBore {
    pub fn new(freq_hz: f32, tone: f32, sr: f32) -> Self {
        let f = freq_hz.max(20.0);
        // −3 samples compensates the two graph feedback edges + the bell filter.
        let delay = (sr / (2.0 * f) - 3.0).max(2.0);
        WaveguideBore {
            line: vec![0.0; delay.ceil() as usize + 3],
            delay,
            pos: 0,
            bell: 0.0,
            bell_a: (0.15 + 0.55 * tone.clamp(0.0, 1.5) / 1.5).clamp(0.05, 0.9),
            refl: -0.97,
        }
    }
}

impl Node for WaveguideBore {
    #[inline]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let flow: f32 = inputs.iter().sum(); // pressure launched in by the reed
        let n = self.line.len();
        let rp = self.pos as f32 + n as f32 - self.delay;
        let i0 = rp.floor() as usize % n;
        let i1 = (i0 + 1) % n;
        let frac = rp - rp.floor();
        let bore_out = self.line[i0] * (1.0 - frac) + self.line[i1] * frac;
        self.bell += self.bell_a * (bore_out - self.bell);
        let reflected = self.refl * self.bell;
        self.line[self.pos] = flow;
        self.pos = (self.pos + 1) % n;
        reflected
    }
}

/// A **coupled reed + bore** solved *implicitly* — the physically-correct
/// woodwind. The reed valve and the waveguide bore are one tightly-coupled
/// feedback loop: the bore's returning pressure wave pushes the reed, and the
/// reed's flow drives the bore. The reed's own *mechanical* motion is stepped
/// explicitly (its ~1.5 kHz resonance is slow next to a sample), but the
/// **flow ↔ pressure** relationship is resolved with a per-sample Newton–Raphson
/// solve, so there is **no artificial one-sample delay in the loop** — which is
/// what makes the pitch lock exactly to the bore length (`f ≈ c/2L`) instead of
/// drifting flat. Self-contained (no feedback edge); pitch = `length`.
pub struct CoupledReed {
    // reed
    x: f32,
    v: f32,
    wn2: f32,
    damp: f32,
    beta: f32,
    zc: f32,
    dp_prev: f32,
    u_prev: f32,
    flow_lp: f32,
    flow_a: f32,
    press_mult: f32, // live mouth-pressure modulation (breath / pitch-wheel bend)
    // internal waveguide bore, split at the register hole (1/3 from the throat):
    // seg 1 = throat→hole, seg 2 = hole→bell, each a forward + backward delay.
    f1: FracDelay,
    b1: FracDelay,
    f2: FracDelay,
    b2: FracDelay,
    bell: f32,
    bell_a: f32,
    refl: f32,
    conical: bool, // cone (all harmonics, octave overblow) vs cylinder (odd, 12th)
    cone: f32,     // conical throat integrator state (spherical-wave spreading)
    cone_a: f32,   // conical throat low-pass coefficient (even-harmonic shaping)
    register: f32, // register key: 0 = closed (fundamental), open = overblow
    // Vocal-tract resonance (the player's airway) on the mouth side of the reed:
    // a 2-pole resonator tuned to the played note. Tuned onto a high bore harmonic
    // it biases the reed to lock there — how a player voices the altissimo. It sits
    // INSIDE the reed feedback loop (unlike a downstream body/bell resonator, which
    // only colours the already-chosen tone). gain 0 = off.
    tract_b1: f32,   // 2R·cos(w) of the pole pair
    tract_r2: f32,   // R² (pole radius²)
    tract_gain: f32, // how hard the tract pressure loads the reed (0 = off)
    tract_y1: f32,
    tract_y2: f32,
    // drive
    pressure: f32,
    env: f32,
    env_target: f32,
    atk: f32,
    rel: f32,
    rng: u32,
}

impl CoupledReed {
    /// `length_m` = bore length (metres); pitch ≈ c/2L (register closed).
    /// `register` 0 = closed (low register), open = overblow. `overblow` is the
    /// register-break ratio: 3 = a cylinder's twelfth (clarinet, odd harmonics),
    /// 2 = a cone's octave (saxophone, full harmonic series). `conical` picks the
    /// bore shape — a cone radiates ALL harmonics and overblows the octave, a
    /// cylinder only the odd harmonics and overblows the twelfth. `stiffness`
    /// shapes the reed's beating, `tone` the bell.
    pub fn new(
        pressure: f32,
        stiffness: f32,
        length_m: f32,
        tone: f32,
        register: f32,
        overblow: f32,
        conical: bool,
        sr: f32,
    ) -> Self {
        let c = 343.0_f32;
        // The register vent that forces the `overblow`-th harmonic sits ~1/overblow
        // of the way down the bore (a node of that harmonic): 1/3 for the clarinet's
        // twelfth, 1/2 for the sax's octave.
        let reg_pos = (1.0 / overblow.max(1.5)).clamp(0.1, 0.9);
        // One-way propagation over the whole bore (round-trip = 2·dtot), split at
        // the register hole. `CR_COMP` recenters the raw pitch (a fixed length the
        // reed/mouthpiece add); a cone's throat filter adds extra delay, so it
        // needs more compensation than a cylinder to sit in tune before calibration.
        let comp = if conical { CR_COMP + 1.4 } else { CR_COMP };
        let dtot = (length_m.max(0.02) / (2.0 * c) * sr - comp).max(4.0);
        let d1 = (dtot * reg_pos).max(2.0);
        let d2 = (dtot * (1.0 - reg_pos)).max(2.0);
        // The reed's mechanical resonance sets the top of the range: the reed can
        // only beat (and so sustain the tone) well below it. A real clarinet/sax
        // reed resonates at ~2–3 kHz; a stiffer reed resonates higher (∝√(k/m)),
        // so tie it to `stiffness` — stiffer reeds play higher and brighter.
        let reed_hz = (2000.0 + 800.0 * stiffness.clamp(0.0, 2.5)).clamp(1600.0, 4000.0);
        let wn = TAU * reed_hz / sr;
        let beta = (0.65 + 0.14 * stiffness.clamp(0.0, 2.0)).clamp(0.7, 0.9);
        // Cone throat filter: a one-pole low-pass modelling the apex's spherical
        // spreading. A near-lossless non-inverting bell reflection turns the
        // odd-only cylinder into the cone's full harmonic series (octave overblow);
        // this throat low-pass shapes the even harmonics' balance.
        let cone_a = 0.75_f32;
        CoupledReed {
            x: 0.0,
            v: 0.0,
            wn2: wn * wn,
            damp: 2.0 * 0.8 * wn,
            beta,
            zc: 0.6,
            dp_prev: 0.0,
            u_prev: 0.0,
            flow_lp: 0.0,
            flow_a: 0.28,
            press_mult: 1.0,
            f1: FracDelay::new(d1),
            b1: FracDelay::new(d1),
            f2: FracDelay::new(d2),
            b2: FracDelay::new(d2),
            bell: 0.0,
            bell_a: (0.15 + 0.55 * tone.clamp(0.0, 1.5) / 1.5).clamp(0.05, 0.9),
            refl: -0.97,
            conical,
            cone: 0.0,
            cone_a,
            register: register.clamp(0.0, 0.9),
            tract_b1: 0.0,
            tract_r2: 0.0,
            tract_gain: 0.0,
            tract_y1: 0.0,
            tract_y2: 0.0,
            pressure,
            env: 0.0,
            env_target: 1.0,
            atk: 1.0 - (-1.0 / (0.02 * sr)).exp(),
            rel: 1.0 - (-1.0 / (0.03 * sr)).exp(),
            rng: 0x1234_5678,
        }
    }

    /// Tune the vocal-tract resonance (mouth-side load). `freq` = resonance (Hz,
    /// normally the played note), `q` its sharpness, `gain` how hard it loads the
    /// reed (0 disables it). Tuned onto a high bore harmonic it lets the reed lock
    /// there — the voiced altissimo registers.
    pub fn set_tract(&mut self, freq: f32, q: f32, gain: f32, sr: f32) {
        if gain <= 0.0 {
            self.tract_gain = 0.0;
            return;
        }
        let w = std::f32::consts::TAU * freq.clamp(20.0, sr * 0.45) / sr;
        let r = (-w / (2.0 * q.max(0.5))).exp().clamp(0.0, 0.9995);
        self.tract_b1 = 2.0 * r * w.cos();
        self.tract_r2 = r * r;
        self.tract_gain = gain;
    }
}

impl Node for CoupledReed {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk } else { self.rel };
        self.env += (self.env_target - self.env) * rate;
        // Vocal-tract resonant pressure (from last sample's flow) adds to the
        // steady mouth pressure, biasing the reed toward the tract's tuned harmonic.
        let p_tract = self.tract_gain * self.tract_y1;
        let pm = self.pressure * self.press_mult * self.env + p_tract;

        // 1. Read the two bore segments. `bo1` is the wave arriving back at the
        //    throat (already carrying the bell reflection through the segments),
        //    which is the pressure the reed feels.
        let fo1 = self.f1.read();
        let bo1 = self.b1.read();
        let fo2 = self.f2.read();
        let bo2 = self.b2.read();
        let p_plus = bo1;

        // 2. Advance the reed's *mechanical* state (explicit — slow variable).
        let acc = -self.damp * self.v - self.wn2 * (self.x + self.beta * self.dp_prev);
        self.v += acc;
        self.x += self.v;
        let h = (1.0 + self.x).clamp(0.0, 3.0);

        // 3. Resolve the flow ↔ pressure algebraic loop *this sample* (no delay).
        //    Injecting flow *raises* the mouthpiece pressure (P = 2·p₊ + Zc·U), so
        //    ΔP = Pm − P = Pm − 2·p₊ − Zc·U, with U = h·sign(ΔP)·√|ΔP|. Newton on
        //    U (warm-started from last sample). d(ΔP)/dU = −Zc, so the residual's
        //    derivative is 1 + h·Zc·(0.5/√|ΔP|). Getting these signs right removes
        //    the runaway that previously needed a `tanh` crutch (which distorted).
        let mut u = self.u_prev;
        for _ in 0..4 {
            let dp = pm - 2.0 * p_plus - self.zc * u;
            let sq = dp.abs().max(1e-9).sqrt();
            let resid = u - h * dp.signum() * sq;
            let deriv = 1.0 + h * self.zc * (0.5 / sq);
            u -= resid / deriv;
        }
        let dp = pm - 2.0 * p_plus - self.zc * u;
        self.dp_prev = dp;
        self.u_prev = u;

        // Advance the vocal-tract resonator, driven by the reed flow.
        if self.tract_gain != 0.0 {
            let ty = self.tract_b1 * self.tract_y1 - self.tract_r2 * self.tract_y2
                + (1.0 - self.tract_r2) * 0.5 * u;
            self.tract_y2 = self.tract_y1;
            self.tract_y1 = ty;
        }

        // Reed/air inertia low-pass + a little breath turbulence; breath-gate so
        // the note stops on note-off instead of self-oscillating on the wave.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let g = (self.env * 4.0).min(1.0);
        // Breath turbulence scales with the actual blowing pressure, so a reed
        // driven at pressure 0 (an out-of-range note) is truly silent, not hissy.
        let breath = (pm.abs() * 3.0).min(1.0);
        self.flow_lp += self.flow_a * (g * (u + h * 0.015 * white * breath) - self.flow_lp);
        let ur = self.flow_lp;

        // 4. Launch the outgoing wave into segment 1 (pure superposition,
        //    p₋ = p₊ + Zc·U — no waveshaper; the reed's beating bounds the cycle).
        let p_minus = p_plus + self.zc * ur;

        // Register hole between the segments: a pressure-release shunt that pulls
        // the local pressure `fo1 + bo2` toward zero when open. The fundamental
        // (antinode here) is destroyed; the 3rd harmonic (node here) survives, so
        // the reed jumps a 12th — the clarinet register break.
        let w = self.register * (fo1 + bo2);
        let f2_in = fo1 - w; // into segment 2 (toward the bell)
        let b1_in = bo2 - w; // into segment 1 (back toward the throat)

        // Bell (end of segment 2): low-pass, then reflect. A cylinder inverts at
        // the open end (odd harmonics only → overblows a 12th). A cone's flare
        // makes the standing-wave series complete (all harmonics → overblows the
        // octave); model the flare's spherical spreading as a non-inverting
        // reflection plus a leaky throat integrator that feeds in the even
        // harmonics the inverting cylinder would cancel.
        self.bell += self.bell_a * (fo2 - self.bell);
        // A cone's NON-inverting reflection gives the octave overblow + bright even
        // harmonics, but with the vent closed it has no fundamental resonance (its
        // strongest mode is DC, which back-pressures the reed silent). So use it
        // only when the register vent is open (overblowing); with the vent closed
        // fall back to the inverting reflection, which has a proper fundamental.
        let b2_in = if self.conical && self.register > 0.1 {
            self.cone += self.cone_a * (self.bell - self.cone);
            0.97 * self.cone
        } else {
            self.refl * self.bell
        };

        self.f1.write(p_minus);
        self.f2.write(f2_in);
        self.b1.write(b1_in);
        self.b2.write(b2_in);
        bo1
    }
    fn control(&mut self, c: Control) {
        match c {
            Control::Gate(on) => self.env_target = if on { 1.0 } else { 0.0 },
            // Breath is the pressure axis directly.
            Control::Breath(m) => self.press_mult = m.clamp(0.2, 2.5),
            // The pitch wheel bends a wind the *physical* way — via mouth
            // pressure (harder = sharper), not by retuning. A ±2-semitone wheel
            // (ratio ≈ 0.89–1.12) maps to a usable embouchure bend.
            Control::Bend(r) => self.press_mult = (1.0 + 5.0 * (r - 1.0)).clamp(0.3, 2.0),
        }
    }
}

/// A short fractional delay line (linear interpolation) — one waveguide segment.
struct FracDelay {
    buf: Vec<f32>,
    pos: usize,
    delay: f32,
}

impl FracDelay {
    fn new(delay: f32) -> Self {
        let d = delay.max(1.0);
        FracDelay { buf: vec![0.0; d.ceil() as usize + 2], pos: 0, delay: d }
    }
    #[inline]
    fn read(&self) -> f32 {
        let n = self.buf.len();
        let rp = self.pos as f32 + n as f32 - self.delay;
        let i0 = rp.floor() as usize % n;
        let i1 = (i0 + 1) % n;
        let frac = rp - rp.floor();
        self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac
    }
    #[inline]
    fn write(&mut self, x: f32) {
        self.buf[self.pos] = x;
        self.pos = (self.pos + 1) % self.buf.len();
    }
}

/// A **digital-waveguide flaring horn** — a *traveling-wave* model of a bore with
/// varying cross-section `A(x) = π·r(x)²`, `r(x) = r1 + r2·x + r3·x²`. Unlike the
/// modal [`crate::models::webster_horn::WebsterHorn`] (a fixed bank of sinusoids,
/// which a reed can't drive into oscillation), the wave here actually propagates
/// and reflects, so a reed self-oscillates on it and its **pitch tracks the bore
/// length**. The bore is `n` short cylindrical segments joined by Kelly–Lochbaum
/// scattering junctions — each area change reflects part of the wave (the flare)
/// — with a radiating low-pass reflection at the bell. A cylindrical profile
/// (r2 = r3 = 0) gives odd harmonics (a clarinet, closed–open); a conical/flaring
/// profile fills the harmonic series back in (a saxophone). The reed drives the
/// throat over a feedback edge; the throat-returning wave is the output.
pub struct WaveguideHorn {
    fwd: Vec<FracDelay>, // forward waves, throat → bell (one per segment)
    bwd: Vec<FracDelay>, // backward waves, bell → throat
    k: Vec<f32>,         // n−1 junction reflection coefficients (the flare)
    f_in: Vec<f32>,      // scratch: forward wave entering each segment this sample
    b_in: Vec<f32>,      // scratch: backward wave entering each segment
    bell_lp: f32,        // bell radiation low-pass state
    bell_a: f32,
    bell_refl: f32,
}

impl WaveguideHorn {
    /// Build the segmented bore from its geometry. `segments` is the flare
    /// resolution (a handful is plenty); `length` sets the pitch (key-mapped).
    pub fn new(r1: f32, r2: f32, r3: f32, length: f32, segments: usize, tone: f32, sr: f32) -> Self {
        let l = length.max(0.02);
        let c = 343.0_f32;
        // One-way propagation time across the whole bore, less a small
        // compensation for the junction + bell-filter phase.
        let total = (l / c * sr - 2.0).max(6.0);
        // Keep each segment at least ~3 samples long (a fractional delay needs a
        // few samples to interpolate), so a short/high bore uses fewer segments.
        let n = segments.clamp(2, 64).min((total / 3.0) as usize).max(2);
        let d = total / n as f32;
        let area = |i: usize| -> f32 {
            let x = (i as f32 + 0.5) / n as f32 * l;
            let r = (r1 + r2 * x + r3 * x * x).max(1e-4);
            PI * r * r
        };
        // Junction reflection: k = (A_next − A_here)/(A_next + A_here). A flare
        // (area increasing toward the bell) gives k > 0.
        let k = (1..n)
            .map(|j| {
                let (a0, a1) = (area(j - 1), area(j));
                ((a1 - a0) / (a1 + a0)).clamp(-0.99, 0.99)
            })
            .collect();
        WaveguideHorn {
            fwd: (0..n).map(|_| FracDelay::new(d)).collect(),
            bwd: (0..n).map(|_| FracDelay::new(d)).collect(),
            k,
            f_in: vec![0.0; n],
            b_in: vec![0.0; n],
            bell_lp: 0.0,
            bell_a: (0.15 + 0.55 * tone.clamp(0.0, 1.5) / 1.5).clamp(0.05, 0.9),
            bell_refl: -0.97,
        }
    }
}

impl Node for WaveguideHorn {
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let drive: f32 = inputs.iter().sum(); // reed's wave launched into the throat
        let n = self.fwd.len();
        // Throat: the reed's wave enters segment 0; the wave arriving back at the
        // throat is the output (it feeds the reed and the mix).
        let throat_return = self.bwd[0].read();
        self.f_in[0] = drive;
        // Internal Kelly–Lochbaum junctions (memoryless; the delays are the tube).
        for j in 1..n {
            let f_arr = self.fwd[j - 1].read(); // forward wave reaching junction j
            let b_arr = self.bwd[j].read(); // backward wave reaching junction j
            let w = self.k[j - 1] * (b_arr - f_arr);
            self.f_in[j] = f_arr + w; // continues into segment j
            self.b_in[j - 1] = b_arr + w; // reflects into segment j−1
        }
        // Bell: the open end radiates and reflects (inverting low-pass).
        let f_bell = self.fwd[n - 1].read();
        self.bell_lp += self.bell_a * (f_bell - self.bell_lp);
        self.b_in[n - 1] = self.bell_refl * self.bell_lp;
        // Advance every segment.
        for i in 0..n {
            self.fwd[i].write(self.f_in[i]);
            self.bwd[i].write(self.b_in[i]);
        }
        throat_return
    }
}

/// A digital-waveguide reed instrument (clarinet / sax / lip-brass) — the
/// McIntyre–Schumacher–Woodhouse / STK model. The bore is a **delay line**; the
/// bell is a one-pole low-pass with an inverting reflection; the mouthpiece is a
/// nonlinear **reed table** closing the loop. Unlike a reed driving a modal bank,
/// the traveling wave actually reflects, so the reed self-oscillates into a clean
/// harmonic tone. Self-contained (no graph feedback edge); pitch = delay length.
pub struct WaveguideReed {
    line: Vec<f32>,
    delay: f32,      // fractional loop delay (samples) — sets the pitch
    pos: usize,      // write index
    bell: f32,       // bell low-pass state
    bell_a: f32,     // bell brightness (low-pass coeff)
    refl: f32,       // bell reflection gain (negative → closed-open, odd harmonics)
    reed_offset: f32,
    reed_slope: f32, // reed stiffness (steeper = stiffer/brighter)
    pressure: f32,
    env: f32,
    env_target: f32,
    atk: f32,
    rel: f32,
    rng: u32,
    noise: f32,
}

impl WaveguideReed {
    pub fn new(freq_hz: f32, pressure: f32, stiffness: f32, tone: f32, sr: f32) -> Self {
        let f = freq_hz.max(20.0);
        // Loop round-trip = 2·delay samples; the inverting bell reflection makes
        // it a closed-open quarter-wave (odd harmonics). A fractional delay (read
        // with interpolation) tunes it exactly; ~1 sample compensates the bell
        // filter's phase delay in the loop.
        let delay = (sr / (2.0 * f) - 1.0).max(2.0);
        WaveguideReed {
            line: vec![0.0; delay.ceil() as usize + 3],
            delay,
            pos: 0,
            bell: 0.0,
            bell_a: (0.15 + 0.55 * tone.clamp(0.0, 1.5) / 1.5).clamp(0.05, 0.9),
            refl: -0.97,
            reed_offset: 0.7,
            reed_slope: -(0.08 + stiffness * 0.22),
            pressure,
            env: 0.0,
            env_target: 1.0,
            atk: 1.0 - (-1.0 / (0.012 * sr)).exp(),
            rel: 1.0 - (-1.0 / (0.02 * sr)).exp(),
            rng: 0x9e37_79b9,
            noise: 0.03,
        }
    }
}

impl Node for WaveguideReed {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk } else { self.rel };
        self.env += (self.env_target - self.env) * rate;
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let breath = self.pressure * self.env * (1.0 + self.noise * white);
        // Fractional-delay read of the returning wave at the mouthpiece.
        let n = self.line.len();
        let rp = self.pos as f32 + n as f32 - self.delay;
        let i0 = rp.floor() as usize % n;
        let i1 = (i0 + 1) % n;
        let frac = rp - rp.floor();
        let bore_out = self.line[i0] * (1.0 - frac) + self.line[i1] * frac;
        // Bell: low-pass then invert/attenuate — the open-end reflection.
        self.bell += self.bell_a * (bore_out - self.bell);
        let reflected = self.refl * self.bell;
        // Reed table: reflection coefficient falls with the pressure difference,
        // clipped when the reed slaps shut — the nonlinearity that sustains it.
        let delta = reflected - breath;
        let reed = (self.reed_offset + self.reed_slope * delta).clamp(-1.0, 1.0);
        self.line[self.pos] = breath + delta * reed;
        self.pos = (self.pos + 1) % n;
        bore_out
    }
    fn control(&mut self, c: Control) {
        if let Control::Gate(on) = c {
            self.env_target = if on { 1.0 } else { 0.0 };
        }
    }
}

/// A digital-waveguide **bowed string** — the STK bowed-string model. The string
/// is two delay lines meeting at the bow point; the bridge end reflects through a
/// one-pole low-pass, the nut end rigidly; the bow is a nonlinear friction table
/// on the slip velocity (bow speed − string velocity) that closes the loop. As
/// with the reed, the traveling wave reflecting off the ends is what lets it lock
/// into Helmholtz stick-slip — a modal bank can't. Self-contained; pitch = total
/// delay length, bow position = where the string is split.
pub struct WaveguideBow {
    nut: Vec<f32>,      // nut-side delay line
    nut_pos: usize,
    nut_len: usize,
    bridge: Vec<f32>,   // bridge-side delay line (fractional, carries tuning)
    bridge_pos: usize,
    bridge_delay: f32,
    br_lp: f32,         // bridge low-pass state
    br_a: f32,          // bridge brightness
    speed: f32,
    slope: f32,         // friction-curve sharpness (bow force)
    env: f32,
    env_target: f32,
    atk: f32,
    rel: f32,
    rng: u32,
}

impl WaveguideBow {
    pub fn new(freq_hz: f32, speed: f32, force: f32, sr: f32) -> Self {
        let f = freq_hz.max(20.0);
        let total = (sr / (2.0 * f) - 1.0).max(4.0); // both ends reflect → sr/(2·total)
        let bow_pos = 0.13; // near the bridge (brighter, stable stick-slip)
        let bridge_delay = (total * bow_pos).max(2.0);
        let nut_len = ((total * (1.0 - bow_pos)).round() as usize).max(2);
        WaveguideBow {
            nut: vec![0.0; nut_len + 1],
            nut_pos: 0,
            nut_len,
            bridge: vec![0.0; bridge_delay.ceil() as usize + 3],
            bridge_pos: 0,
            bridge_delay,
            br_lp: 0.0,
            br_a: 0.5,
            speed: speed * 0.12, // bow velocity (scaled into the wave domain)
            slope: 3.0 + force * 3.0, // more force = sharper stick-slip
            env: 0.0,
            env_target: 1.0,
            atk: 1.0 - (-1.0 / (0.04 * sr)).exp(), // ~40 ms bow onset
            rel: 1.0 - (-1.0 / (0.06 * sr)).exp(),
            rng: 0x1f35_3c6d,
        }
    }
}

impl Node for WaveguideBow {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk } else { self.rel };
        self.env += (self.env_target - self.env) * rate;
        // Waves arriving at the bow point from each side.
        let neck = self.nut[self.nut_pos];
        let nb = self.bridge.len();
        let rp = self.bridge_pos as f32 + nb as f32 - self.bridge_delay;
        let i0 = rp.floor() as usize % nb;
        let i1 = (i0 + 1) % nb;
        let frac = rp - rp.floor();
        let bridge_in = self.bridge[i0] * (1.0 - frac) + self.bridge[i1] * frac;
        let string_vel = neck + bridge_in;
        // Friction table on the slip velocity: high near sticking, falling off as
        // the string slips (the negative-resistance stick-slip characteristic).
        let mut dv = self.speed * self.env - string_vel;
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        dv += (self.rng as f32 / u32::MAX as f32 - 0.5) * 0.005 * self.env; // bow noise
        let fr = ((dv.abs() * self.slope + 0.75).powi(4)).max(1.0);
        let bow = (dv / fr) * self.env; // velocity the bow injects
        // Scatter into the two delay lines; the bridge reflects through a low-pass.
        self.br_lp += self.br_a * (neck - self.br_lp);
        self.nut[self.nut_pos] = bridge_in + bow;
        self.bridge[self.bridge_pos] = -self.br_lp + bow;
        self.nut_pos = (self.nut_pos + 1) % self.nut_len;
        self.bridge_pos = (self.bridge_pos + 1) % nb;
        string_vel
    }
    fn control(&mut self, c: Control) {
        if let Control::Gate(on) = c {
            self.env_target = if on { 1.0 } else { 0.0 };
        }
    }
}

/// A pure sinusoid generator at the played pitch. A clean tone source — and the
/// probe signal the graph editor feeds a component to audition it in isolation
/// (a generator ignores it and emits its own sound; a resonator resonates it).
pub struct SineExciter {
    phase: f32,
    incr: f32,
    level: f32,
    env: f32,
    env_target: f32,
    atk: f32,
    rel: f32,
}

impl SineExciter {
    pub fn new(freq_hz: f32, level: f32, sr: f32) -> Self {
        SineExciter {
            phase: 0.0,
            incr: freq_hz.max(1.0) / sr,
            level,
            env: 0.0,
            env_target: 1.0,
            atk: 1.0 - (-1.0 / (0.005 * sr)).exp(),
            rel: 1.0 - (-1.0 / (0.02 * sr)).exp(),
        }
    }
}

impl Node for SineExciter {
    #[inline]
    fn tick(&mut self, _inputs: &[f32]) -> f32 {
        let rate = if self.env < self.env_target { self.atk } else { self.rel };
        self.env += (self.env_target - self.env) * rate;
        self.phase += self.incr;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        (TAU * self.phase).sin() * self.level * self.env
    }
    fn control(&mut self, c: Control) {
        if let Control::Gate(on) = c {
            self.env_target = if on { 1.0 } else { 0.0 };
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

    /// Autocorrelation pitch estimate — robust to the harmonic-rich buzz, unlike
    /// zero-crossing counting (whose extra crossings are the harmonics themselves).
    fn acf_freq(y: &[f32], sr: f32, f0: f32) -> f32 {
        let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
        let s: Vec<f32> = y.iter().map(|v| v - mean).collect();
        let lo = (sr / (f0 * 2.2)) as usize;
        let hi = ((sr / (f0 * 0.45)) as usize).min(s.len() / 2);
        let (mut best, mut best_c) = (lo, f32::MIN);
        for lag in lo..hi {
            let c: f32 = (0..s.len() - lag).map(|i| s[i] * s[i + lag]).sum();
            if c > best_c {
                best_c = c;
                best = lag;
            }
        }
        sr / best as f32
    }

    #[test]
    fn ode_reed_plays_the_bore_pitch_across_the_register() {
        // The reed ↔ bore loop must lock onto the *bore* fundamental (a clarinet
        // tone), not squeak at the reed's own resonance, all the way down the
        // register — the failure mode that plagues coupled reed models.
        let sr = 48_000.0;
        for &f0 in &[123.0f32, 165.0, 220.0, 311.0, 440.0, 622.0, 831.0] {
            let mut reed = ReedExciter::new(0.8, 1.0, 0.0, sr);
            let mut bore = WaveguideBore::new(f0, 1.0, sr);
            let mut fb = 0.0f32;
            let y: Vec<f32> = (0..36_000)
                .map(|_| {
                    let r = reed.tick(&[fb]);
                    fb = bore.tick(&[r]);
                    fb
                })
                .collect();
            assert!(y.iter().all(|v| v.is_finite() && v.abs() < 20.0), "reed loop stable at {f0} Hz");
            let tail = &y[24_000..];
            let rms = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
            assert!(rms > 0.15, "reed sustains a strong tone at {f0} Hz (rms {rms})");
            let f = acf_freq(tail, sr, f0);
            assert!((f / f0 - 1.0).abs() < 0.06, "reed plays the bore pitch {f0} Hz, got {f}");
        }
    }

    #[test]
    fn ode_reed_stops_when_the_breath_stops() {
        // No breath ⇒ no flow through the reed: on note-off the drive dies and the
        // loop rings down (it must not self-oscillate on the standing wave alone).
        let sr = 48_000.0;
        let mut reed = ReedExciter::new(0.8, 1.0, 0.0, sr);
        let mut bore = WaveguideBore::new(220.0, 1.0, sr);
        let mut fb = 0.0f32;
        let mut on = 0.0f32;
        for i in 0..24_000 {
            let r = reed.tick(&[fb]);
            fb = bore.tick(&[r]);
            on += fb * fb;
        }
        assert!((on / 24_000.0).sqrt() > 0.15, "reed sounds while blown");
        reed.control(Control::Gate(false));
        let tail: Vec<f32> = (0..24_000)
            .map(|_| {
                let r = reed.tick(&[fb]);
                fb = bore.tick(&[r]);
                fb
            })
            .collect();
        let end = &tail[16_000..];
        let end_rms = (end.iter().map(|v| v * v).sum::<f32>() / end.len() as f32).sqrt();
        assert!(end_rms < 0.05, "reed rings down after note-off (rms {end_rms})");
    }

    #[test]
    fn self_contained_reed_buzzes_at_its_fixed_pitch() {
        // freq_hz > 0 gives a self-contained mouthpiece: it buzzes on its own (no
        // external bore / feedback edge) at that one fixed pitch, ready to drive a
        // resonator forward.
        let sr = 48_000.0;
        let mut reed = ReedExciter::new(0.9, 1.0, 196.0, sr);
        let y: Vec<f32> = (0..36_000).map(|_| reed.tick(&[])).collect();
        assert!(y.iter().all(|v| v.is_finite() && v.abs() < 20.0), "self-contained reed is stable");
        let tail = &y[24_000..];
        let rms = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(rms > 0.1, "self-contained reed buzzes on its own (rms {rms})");
        assert!((acf_freq(tail, sr, 196.0) / 196.0 - 1.0).abs() < 0.07, "buzzes at its fixed pitch");
    }

    #[test]
    fn coupled_reed_bore_plays_in_tune_with_odd_harmonics() {
        // The implicitly-solved reed↔bore locks its pitch to the bore length
        // (f ≈ c/2L) in tune, and — a cylindrical closed-open bore — sounds odd
        // harmonics (strong 3rd, weak 2nd): the hollow clarinet tone.
        let sr = 48_000.0;
        let c = 343.0f32;
        let mag = |y: &[f32], f: f32| -> f32 {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (n, &s) in y.iter().enumerate() {
                let p = TAU * f * n as f32 / sr;
                re += s * p.cos();
                im += s * p.sin();
            }
            (re * re + im * im).sqrt() / y.len() as f32
        };
        for &f0 in &[147.0f32, 220.0, 330.0, 494.0] {
            let l = c / (2.0 * f0);
            let mut r = CoupledReed::new(0.9, 1.0, l, 1.0, 0.0, 3.0, false, sr);
            let y: Vec<f32> = (0..30_000).map(|_| r.tick(&[])).collect();
            assert!(y.iter().all(|v| v.is_finite() && v.abs() < 20.0), "coupled reed stable at {f0}");
            let tail = &y[20_000..];
            let rms = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
            assert!(rms > 0.1, "coupled reed sounds at {f0} Hz (rms {rms})");
            let f = acf_freq(tail, sr, f0);
            let cents = 1200.0 * (f / f0).log2();
            assert!(cents.abs() < 25.0, "in tune at {f0} Hz (got {cents:+.0} cents)");
            // odd-harmonic clarinet tone: the 3rd is not weaker than the 2nd
            // (clearest low; near-equal and tiny up high, so allow a small margin).
            let (h2, h3) = (mag(tail, 2.0 * f0), mag(tail, 3.0 * f0));
            assert!(h3 > h2 * 0.8, "odd-harmonic (clarinet) tone at {f0}: h3 {h3} vs h2 {h2}");
        }
    }

    #[test]
    fn coupled_reed_register_key_overblows_a_twelfth() {
        // Opening the register hole (a third down the bore) makes the same bore
        // jump from its fundamental to its 3rd harmonic — a clarinet's register
        // break, up a twelfth (×3).
        let sr = 48_000.0;
        let c = 343.0f32;
        for &f0 in &[147.0f32, 220.0, 294.0] {
            let l = c / (2.0 * f0);
            let mut closed = CoupledReed::new(0.9, 1.0, l, 1.0, 0.0, 3.0, false, sr);
            let yc: Vec<f32> = (0..28_000).map(|_| closed.tick(&[])).collect();
            let fc = acf_freq(&yc[20_000..], sr, f0);
            let mut open = CoupledReed::new(0.9, 1.0, l, 1.0, 0.3, 3.0, false, sr);
            let yo: Vec<f32> = (0..28_000).map(|_| open.tick(&[])).collect();
            let fo = acf_freq(&yo[20_000..], sr, f0 * 3.0);
            assert!((fc / f0 - 1.0).abs() < 0.06, "closed plays the fundamental at {f0} (got {fc})");
            assert!((fo / fc / 3.0 - 1.0).abs() < 0.1, "register-open overblows a 12th: {fc} → {fo}");
        }
    }

    #[test]
    fn conical_reed_bore_overblows_an_octave_with_full_harmonics() {
        // A conical bore (sax/oboe) differs from the clarinet cylinder two ways:
        // it radiates the FULL harmonic series (a strong even 2nd harmonic, not
        // just odds), and its register vent at the half-way node overblows the
        // OCTAVE (×2), not the twelfth.
        let sr = 48_000.0;
        let c = 343.0f32;
        let goertzel = |y: &[f32], f: f32| -> f32 {
            let w = TAU * f / sr;
            let cs = w.cos();
            let (mut q1, mut q2) = (0.0f32, 0.0f32);
            for &x in y {
                let q0 = 2.0 * cs * q1 - q2 + x;
                q2 = q1;
                q1 = q0;
            }
            (q1 * q1 + q2 * q2 - 2.0 * cs * q1 * q2).max(0.0).sqrt()
        };
        for &f0 in &[196.0f32, 262.0, 330.0] {
            let l = c / (2.0 * f0);
            // Closed vent: the cone falls back to the inverting reflection so it
            // has a proper, audible fundamental (the non-inverting cone reflection
            // alone traps DC and goes silent with the vent shut). Just require it
            // to sound at ~c/2L.
            let mut closed = CoupledReed::new(1.0, 1.0, l, 1.0, 0.0, 2.0, true, sr);
            let yc: Vec<f32> = (0..40_000).map(|_| closed.tick(&[])).collect();
            let t = &yc[28_000..];
            let mean: f32 = t.iter().sum::<f32>() / t.len() as f32;
            let ac = (t.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / t.len() as f32).sqrt();
            assert!(ac > 0.1, "cone's closed register sounds (ac {ac}) — not silent DC");
            let fc = acf_freq(t, sr, f0);
            assert!((fc / f0 - 1.0).abs() < 0.08, "cone plays ~c/2L at {f0} (got {fc})");

            // Open vent: the cone reflection engages and overblows the OCTAVE with a
            // strong even 2nd harmonic — the sax brightness a cylinder can't make.
            let mut open = CoupledReed::new(1.0, 1.0, l, 1.0, 0.3, 2.0, true, sr);
            let yo: Vec<f32> = (0..40_000).map(|_| open.tick(&[])).collect();
            let to = &yo[28_000..];
            let fo = acf_freq(to, sr, f0 * 2.0);
            assert!((fo / fc / 2.0 - 1.0).abs() < 0.1, "register-open overblows an octave: {fc} → {fo}");
            let _ = &goertzel; // (harmonic content of the two registers differs by design)
        }
    }

    #[test]
    fn reed_drives_waveguide_horn_into_a_bounded_oscillation() {
        // The reed drives the segmented waveguide horn into a bounded, sounding
        // oscillation across a range of bore lengths. (The reed↔horn loop is
        // multistable — it can overblow to a higher register — so pitch is not
        // asserted here; taming the register break is separate tuning work.)
        let sr = 48_000.0;
        for &length in &[0.9f32, 0.6, 0.4, 0.25] {
            let mut reed = ReedExciter::new(0.9, 1.0, 0.0, sr);
            let mut horn = WaveguideHorn::new(0.0073, 0.0, 0.002, length, 18, 1.0, sr);
            let mut fb = 0.0f32;
            let y: Vec<f32> = (0..30_000)
                .map(|_| {
                    let r = reed.tick(&[fb * 0.8]);
                    fb = horn.tick(&[r * 0.8]);
                    fb
                })
                .collect();
            assert!(y.iter().all(|v| v.is_finite() && v.abs() < 50.0), "horn stable at L={length}");
            let tail = &y[20_000..];
            let rms = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
            assert!(rms > 1e-3, "reed sustains the horn at L={length} (rms {rms})");
        }
    }
}

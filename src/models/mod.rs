//! Pluggable synthesis models.
//!
//! A **model** is a plugin that produces a sound. Most models here use modal
//! synthesis — a vibrating object (a string, a membrane, a solid) represented as
//! a finite sum of exponentially-decaying sinusoids, its *modes* — but a model
//! is free to generate its output however it likes. That representation is the
//! seam the plugin system is built on: **a model's only job is to fill a
//! [`ModeBuffer`] with partials** for a played note. The engine
//! ([`crate::instrument`]) owns everything generic — polyphony, envelopes,
//! voice-stealing, live parameter rebuilds — and plays whatever bank the active
//! model produced.
//!
//! Adding a model (a 2-D drum head, a 3-D solid, a different excitation) is just
//! a new file implementing [`FtmModel`]: fill in the frequencies, amplitudes and
//! decay rates of its partials and register it in [`registry`].

pub mod basic_wave;
pub mod cymbal;
pub mod drum_membrane;
pub mod metal_bell;
pub mod musical_string;
pub mod pure_plate;
pub mod pure_string;
pub mod snare;
pub mod webster_horn;
pub mod graph_voice;

/// Maximum partials a single voice can hold (hard array bound).
pub const MAX_MODES: usize = 512;

/// The modeling tick rate (~10 kHz) that the per-mode decay is expressed in.
/// `DAMP_PERIOD` and `TIME_SCALE` are counted in these ticks, so this constant
/// converts them onto real seconds.
pub const TICK_RATE: f32 = 10_000.0;

/// Reference pitch (C4). Physical-pitch mode scales the geometry so it matches
/// transpose mode at this note and diverges from there.
#[allow(dead_code)] // reserved for physical-pitch models
pub const REF_PITCH_HZ: f32 = 261.625_57;

/// How pitch is realized for a played note.
#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PitchMode {
    /// Transpose a fixed modal template to the note — uniform timbre across the
    /// keyboard.
    Transpose,
    /// Modulate the geometry (string length / drum size) with pitch, so higher
    /// notes are physically more inharmonic and decay faster.
    Physical,
}

impl Default for PitchMode {
    fn default() -> Self {
        // Physical (size tracks pitch) is the standard for the geometric models.
        PitchMode::Physical
    }
}

/// How a resonator is excited.
#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Excitation {
    /// A one-shot pluck/strike: the modes ring and decay.
    Struck,
    /// Continuously driven (bowed): the modes are sustained while the note is
    /// played and only fade on release, and loss shapes the steady-state tone.
    Bowed,
}

impl Default for Excitation {
    fn default() -> Self {
        Excitation::Struck
    }
}

/// A bank of modes: parallel arrays of frequency (Hz), linear amplitude, and
/// per-second decay rate. `env(t) = amp * exp(-decay * t)`; a negative `decay`
/// is a swell (the engine bounds it so it can't run away).
pub struct ModeBuffer {
    pub n: usize,
    pub freq: [f32; MAX_MODES],
    pub amp: [f32; MAX_MODES],
    pub decay: [f32; MAX_MODES],
    /// If true the modes are **driven/sustained** (a blown wind instrument): they
    /// hold at constant amplitude while the note is played and only fade on
    /// release, and `decay` shapes the steady-state spectrum instead of ending
    /// the note. Struck/plucked models (string, drum) leave this false.
    pub sustain: bool,

    // --- Optional filtered-noise component (snare wires, cymbal wash, stick
    // click, breath). `noise_level == 0` means none. ---
    /// Overall noise amplitude (0 = no noise).
    pub noise_level: f32,
    /// Per-second decay of the noise burst (ignored when `sustain` holds it).
    pub noise_decay: f32,
    /// Band edges in Hz: the noise is high-passed at `noise_hp` and low-passed
    /// at `noise_lp` (a band-pass).
    pub noise_hp: f32,
    pub noise_lp: f32,
}

impl Default for ModeBuffer {
    fn default() -> Self {
        ModeBuffer {
            n: 0,
            freq: [0.0; MAX_MODES],
            amp: [0.0; MAX_MODES],
            decay: [0.0; MAX_MODES],
            sustain: false,
            noise_level: 0.0,
            noise_decay: 0.0,
            noise_hp: 20.0,
            noise_lp: 20_000.0,
        }
    }
}

impl ModeBuffer {
    pub fn clear(&mut self) {
        self.n = 0;
        self.sustain = false;
        self.noise_level = 0.0;
        self.noise_decay = 0.0;
        self.noise_hp = 20.0;
        self.noise_lp = 20_000.0;
    }

    /// Append one mode. Silently ignores modes past `MAX_MODES`.
    #[inline]
    pub fn push(&mut self, freq: f32, amp: f32, decay: f32) {
        if self.n < MAX_MODES {
            self.freq[self.n] = freq;
            self.amp[self.n] = amp;
            self.decay[self.n] = decay;
            self.n += 1;
        }
    }

    /// Scale every amplitude (used to normalize then apply velocity).
    pub fn scale_amps(&mut self, factor: f32) {
        for a in self.amp[..self.n].iter_mut() {
            *a *= factor;
        }
    }
}

/// A synthesis model plugin. Implementors live in their own file and are
/// registered in [`registry`].
pub trait FtmModel: Send {
    /// Stable identifier used to persist/restore presets (e.g. "musical_string").
    /// Must be unique across models and never change once presets exist.
    fn id(&self) -> &'static str;

    /// Short name shown in the model picker.
    fn display_name(&self) -> &'static str;

    /// One-line description shown under the picker.
    fn description(&self) -> &'static str;

    /// Fill `out` with the modes of a note struck at `freq_hz` (the played
    /// key's pitch) with velocity `vel` in `[0, 1]`. `sr` is the sample rate,
    /// used to drop modes at/above Nyquist. Runs on the audio thread at
    /// note-on and on every live parameter change, so keep it allocation-free.
    fn excite(&self, freq_hz: f32, vel: f32, sr: f32, out: &mut ModeBuffer);

    /// Draw this model's parameter editor. Returns `true` if anything changed.
    fn params_ui(&mut self, ui: &mut egui::Ui) -> bool;

    /// Clone into a box so the UI thread can hand a snapshot to the audio thread.
    fn box_clone(&self) -> Box<dyn FtmModel>;

    /// Serialize this model's parameters (for presets).
    fn to_json(&self) -> serde_json::Value;

    /// If this model renders through the **per-sample voice graph**, build the
    /// graph for one note. Returning `None` (the default) means "render via the
    /// classic free-oscillator mode bank" — so every existing model is
    /// unchanged. Graph instruments override this to wire up driven resonators
    /// (and, for multi-component instruments, several components + edges).
    fn build_graph(
        &self,
        _freq_hz: f32,
        _vel: f32,
        _sr: f32,
    ) -> Option<Box<dyn crate::graph::Node>> {
        None
    }
}

/// Rebuild a model from its `id` and serialized parameters (the inverse of
/// [`FtmModel::to_json`]). Returns `None` for an unknown id or params that don't
/// deserialize. Every model in [`registry`] must be handled here.
pub fn model_from_id(id: &str, params: &serde_json::Value) -> Option<Box<dyn FtmModel>> {
    fn boxed<M>(v: &serde_json::Value) -> Option<Box<dyn FtmModel>>
    where
        M: FtmModel + serde::de::DeserializeOwned + 'static,
    {
        serde_json::from_value::<M>(v.clone())
            .ok()
            .map(|m| Box::new(m) as Box<dyn FtmModel>)
    }
    match id {
        "basic_wave" => boxed::<basic_wave::BasicWave>(params),
        "musical_string" => boxed::<musical_string::MusicalString>(params),
        "pure_string" => boxed::<pure_string::PureString>(params),
        "drum_membrane" => boxed::<drum_membrane::DrumMembrane>(params),
        "webster_horn" => boxed::<webster_horn::WebsterHorn>(params),
        "metal_bell" => boxed::<metal_bell::MetalBell>(params),
        "snare" => boxed::<snare::Snare>(params),
        "cymbal" => boxed::<cymbal::Cymbal>(params),
        "pure_plate" => boxed::<pure_plate::PurePlate>(params),
        "graph_string" => boxed::<graph_voice::GraphString>(params),
        "graph_plate" => boxed::<graph_voice::GraphPlate>(params),
        "graph_drum" => boxed::<graph_voice::GraphDrum>(params),
        "graph_musical_string" => boxed::<graph_voice::GraphMusicalString>(params),
        "bodied_string" => boxed::<graph_voice::GraphBodiedString>(params),
        _ => None,
    }
}

impl Clone for Box<dyn FtmModel> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// All available models, in picker order. Add new plugins here.
pub fn registry() -> Vec<Box<dyn FtmModel>> {
    vec![
        Box::new(musical_string::MusicalString::default()),
        Box::new(pure_string::PureString::default()),
        Box::new(drum_membrane::DrumMembrane::default()),
        Box::new(webster_horn::WebsterHorn::default()),
        Box::new(metal_bell::MetalBell::default()),
        Box::new(snare::Snare::default()),
        Box::new(cymbal::Cymbal::default()),
        Box::new(pure_plate::PurePlate::default()),
        Box::new(graph_voice::GraphString::default()),
        Box::new(graph_voice::GraphPlate::default()),
        Box::new(graph_voice::GraphDrum::default()),
        Box::new(graph_voice::GraphMusicalString::default()),
        Box::new(graph_voice::GraphBodiedString::default()),
        Box::new(basic_wave::BasicWave::default()),
    ]
}

/// The model selected on startup (the first in the registry).
pub fn default_model() -> Box<dyn FtmModel> {
    registry()
        .into_iter()
        .next()
        .expect("model registry is empty")
}

/// Maps note velocity to a strike level, shared by the struck/plucked models:
/// velocity sets the strike magnitude, and
/// `(mag - play_magnitude)/(max_magnitude - play_magnitude)` sets the level.
#[inline]
pub fn strike_amplitude(vel: f32, play_magnitude: f32, max_magnitude: f32) -> f32 {
    let mag = vel * max_magnitude;
    let span = max_magnitude - play_magnitude;
    let a = if span.abs() < 1e-6 {
        vel
    } else {
        (mag - play_magnitude) / span
    };
    a.clamp(0.0, 1.0)
}

/// An egui slider with range clamping disabled, so any value can be typed in.
/// Every model uses this so the "no limits" behavior is uniform.
pub fn unbounded_slider<'a, N: egui::emath::Numeric>(
    value: &'a mut N,
    range: std::ops::RangeInclusive<N>,
    text: &str,
) -> egui::Slider<'a> {
    egui::Slider::new(value, range)
        .clamping(egui::SliderClamping::Never)
        .text(text.to_owned())
}

// --- Note / pitch helpers, shared by pitch inputs (kit ranges, horn anchor) ---

const NOTE_NAMES: [&str; 12] =
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/// MIDI note number → concert-pitch name, e.g. 60 → "C4".
pub fn midi_name(n: i32) -> String {
    let n = n.clamp(0, 127);
    format!("{}{}", NOTE_NAMES[(n % 12) as usize], n / 12 - 1)
}

/// Parse a note name ("E2", "Bb3", "F#4") or a raw MIDI number → MIDI note.
pub fn parse_note(s: &str) -> Option<i32> {
    let s = s.trim();
    if let Ok(n) = s.parse::<i32>() {
        return Some(n.clamp(0, 127));
    }
    let bytes = s.as_bytes();
    let base = match bytes.first()?.to_ascii_uppercase() {
        b'C' => 0, b'D' => 2, b'E' => 4, b'F' => 5, b'G' => 7, b'A' => 9, b'B' => 11,
        _ => return None,
    };
    let rest = &s[1..];
    let (semi, oct_str) = match rest.as_bytes().first() {
        Some(b'#') | Some(b's') | Some(b'S') => (base + 1, &rest[1..]),
        Some(b'b') | Some(b'B') => (base - 1, &rest[1..]),
        _ => (base, rest),
    };
    let oct: i32 = oct_str.trim().parse().ok()?;
    Some(((oct + 1) * 12 + semi).clamp(0, 127))
}

/// MIDI note → frequency (Hz), A4 = 440.
pub fn midi_freq(n: i32) -> f32 {
    440.0 * 2f32.powf((n as f32 - 69.0) / 12.0)
}

/// Frequency (Hz) → nearest MIDI note.
pub fn freq_to_midi(hz: f32) -> i32 {
    (69.0 + 12.0 * (hz.max(1.0) / 440.0).log2()).round() as i32
}

/// A pitch input showing a concert-pitch name, editable by typing ("E2") **or**
/// by clicking 🎹 and playing a note (MIDI / keyboard / on-screen piano) —
/// "MIDI-learn". `id_salt` must be unique per field. Returns `true` if changed.
///
/// Reads the most-recent note from the app's `NoteMonitor`, stashed in egui
/// context data under `"note_monitor"` as `(generation, note)`.
pub fn note_field(ui: &mut egui::Ui, id_salt: &str, note: &mut i32) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui
            .add(
                egui::DragValue::new(note)
                    .range(0..=127)
                    .speed(0.15)
                    .custom_formatter(|n, _| midi_name(n as i32))
                    .custom_parser(|s| parse_note(s).map(|m| m as f64)),
            )
            .changed();
        let armed_id = egui::Id::new(("note_learn", id_salt));
        let seen_id = egui::Id::new(("note_learn_gen", id_salt));
        let mut armed = ui.data(|d| d.get_temp::<bool>(armed_id).unwrap_or(false));
        let monitor_id = egui::Id::new("note_monitor");
        if ui
            .add(egui::Button::new("🎹").selected(armed))
            .on_hover_text("MIDI-learn: click, then play a note (MIDI / keyboard) to set this")
            .clicked()
        {
            armed = !armed;
            let (gen, _) = ui.ctx().data(|d| d.get_temp::<(u32, u8)>(monitor_id)).unwrap_or((0, 0));
            ui.data_mut(|d| {
                d.insert_temp(armed_id, armed);
                d.insert_temp(seen_id, gen); // baseline: only capture the NEXT note
            });
        }
        if armed {
            let (gen, n) = ui.ctx().data(|d| d.get_temp::<(u32, u8)>(monitor_id)).unwrap_or((0, 0));
            let seen = ui.data(|d| d.get_temp::<u32>(seen_id)).unwrap_or(gen);
            if seen != gen {
                *note = n as i32;
                changed = true;
                ui.data_mut(|d| d.insert_temp(armed_id, false));
            }
            ui.ctx().request_repaint(); // keep polling while armed
        }
    });
    changed
}

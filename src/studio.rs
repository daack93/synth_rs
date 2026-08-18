//! The studio: a multi-timbral host + looper.
//!
//! It owns one *live* instrument (what you play) plus one instrument per loop
//! track, mixes them, and runs the looper transport. Each track remembers the
//! instrument it was recorded with, so you can lay a bass line, switch presets,
//! and overdub a guitar on top — they play back with their own sounds.
//!
//! Timing is sample-accurate: the studio advances a sample clock and fires each
//! track's recorded note events at the exact frame they land on.
//!
//! ## Transport (spacebar "pedal")
//! * **Tap** — mode-specific primary action.
//! * **Stop** — stop playback, keep the loops (spacebar hold ~0.5 s).
//! * **Reset** — clear everything (spacebar hold ~1.5 s).
//!
//! ### Modes
//! * **Pedal** — tap: Idle→Record→Play→Stop→Play… Extra tracks via [`Command::ArmOverdub`].
//! * **Overdub** — tap: Record base, then each tap finalizes the current take and
//!   starts a new track, layering hands-free.

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use crate::instrument::{make_sine_table, EngineParams, Instrument};
use crate::kit::{Kit, Playable};
use crate::models::{default_model, model_from_id, FtmModel};
use crate::project::{AutoPoint, LoopData, LoopEvent, LoopTrack, TempoGrid, ZoneData};

const TWO_PI_F64: f64 = std::f64::consts::TAU;
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Equal-power stereo pan gains for `pan` in `[-1, 1]` (centre = -3 dB each side,
/// constant perceived loudness across the sweep).
#[inline]
fn pan_gains(pan: f32) -> (f32, f32) {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * (std::f32::consts::FRAC_PI_4);
    (angle.cos(), angle.sin())
}

/// Which spacebar-tap behavior is active.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LooperMode {
    Pedal,
    Overdub,
}

impl LooperMode {
    fn as_u8(self) -> u8 {
        match self {
            LooperMode::Pedal => 0,
            LooperMode::Overdub => 1,
        }
    }
}

/// Transport state, published for the UI.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TransportState {
    Idle,
    Recording,
    Playing,
    Stopped,
}

impl TransportState {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => TransportState::Recording,
            2 => TransportState::Playing,
            3 => TransportState::Stopped,
            _ => TransportState::Idle,
        }
    }
    fn as_u8(self) -> u8 {
        match self {
            TransportState::Idle => 0,
            TransportState::Recording => 1,
            TransportState::Playing => 2,
            TransportState::Stopped => 3,
        }
    }
}

/// Messages from the UI / MIDI threads to the audio thread.
pub enum Command {
    NoteOn { note: u8, vel: f32 },
    NoteOff { note: u8 },
    SetModel(Box<dyn FtmModel>),
    SetEngine(EngineParams),
    /// Replace the whole live slot — used to enter/exit kit mode or rebuild a
    /// kit's zones. `SetModel`/`SetEngine` still handle single-instrument edits.
    SetLive(LiveConfig),
    AllNotesOff,
    SetLooperMode(LooperMode),
    /// Primary transport tap (spacebar).
    Tap,
    /// Stop playback, keep loops.
    Stop,
    /// Clear all loops.
    Reset,
    /// Pedal mode: record one extra pass into a new track.
    ArmOverdub,
    ToggleMute(usize),
    DeleteTrack(usize),
    /// Erase a track's recorded parameter automation.
    ClearTrackAutomation(usize),
    /// Chop / crop / rearrange a track's timeline.
    RegionEdit { track: usize, op: RegionOp },
    /// Set a track's mixer level (linear) and pan (-1..=1).
    SetTrackMix { track: usize, volume: f32, pan: f32 },
    /// Toggle a track's solo.
    ToggleSolo(usize),
    /// Master output level (linear).
    SetMasterVolume(f32),
    /// Swap a track's instrument model live (rebuilds its sounding voices).
    SetTrackModel(usize, Box<dyn FtmModel>),
    /// Update a track's engine params live.
    SetTrackEngine(usize, EngineParams),
    /// Replace the current loop + tracks with a saved loop, and play it.
    LoadLoop(LoopData),
    /// Install a song arrangement (resolved loops + repeat counts); doesn't play.
    SetSong(Vec<SongSection>),
    /// Start playing the installed song from the top.
    PlaySong,
    /// Update tempo / grid / metronome settings.
    SetTempo(TempoGrid),
}

/// A non-destructive edit to a track's timeline, over a time range in seconds.
/// Positions and the loop length are preserved; only the selected notes /
/// automation move. `dest` is the target start time for copies/moves.
pub enum RegionOp {
    /// Delete notes starting in `[a, b)` (and automation in range).
    Delete { a: f32, b: f32 },
    /// Keep only what starts in `[a, b)`; delete the rest (crop).
    Keep { a: f32, b: f32 },
    /// Copy the `[a, b)` content to start at `dest`.
    Duplicate { a: f32, b: f32, dest: f32 },
    /// Copy `[a, b)` to `dest`, then delete the original range.
    Move { a: f32, b: f32, dest: f32 },
    /// Slide the whole track by `delta` seconds (wraps within the loop).
    Shift { delta: f32 },
}

/// How the live slot should be configured: a single instrument or a kit.
pub enum LiveConfig {
    Single { model_id: String, params: serde_json::Value, engine: EngineParams },
    Kit { zones: Vec<ZoneData> },
}

/// A resolved song section: a loop and how many times to play it. The UI builds
/// these from `project::Section` + the project's loops.
pub struct SongSection {
    pub loop_data: LoopData,
    pub repeats: u32,
}

#[derive(Clone, Copy)]
enum EvMsg {
    On { note: u8, vel: f32 },
    Off { note: u8 },
}

#[derive(Clone, Copy)]
struct Event {
    pos: u64,
    msg: EvMsg,
}

/// One captured parameter move, timed in samples.
#[derive(Clone)]
struct AutoEv {
    pos: u64,
    target: String,
    value: f32,
}

struct Track {
    inst: Playable,
    events: Vec<Event>,
    cursor: usize,
    muted: bool,
    /// Mixer level (linear) and stereo pan (-1..=1), plus solo.
    volume: f32,
    pan: f32,
    solo: bool,
    name: String,
    label: String,

    // --- Parameter automation (single-instrument tracks only) ---
    /// Recorded parameter moves, sorted by `pos`.
    auto: Vec<AutoEv>,
    auto_cursor: usize,
    /// The track's model id, used to rebuild it when automation fires.
    model_id: String,
    /// Model params + engine at the track's creation — the state each loop
    /// resets to before automation replays.
    base_json: serde_json::Value,
    base_engine: EngineParams,
    /// The evolving state as automation is applied through the loop.
    cur_json: serde_json::Value,
    cur_engine: EngineParams,
}

// --- Shared view (audio thread writes, UI reads) ---

/// A recorded note as a normalized span for drawing (positions in `[0,1)` of the loop).
#[derive(Clone)]
pub struct NoteSpan {
    pub start: f32,
    pub end: f32,
    pub note: u8,
}

#[derive(Clone)]
pub struct TrackView {
    pub name: String,
    pub instrument: String,
    pub muted: bool,
    pub notes: Vec<NoteSpan>,
    /// The track instrument's plugin id + serialized params + engine, so the UI
    /// can open it in the editor. For a kit `model_id` is `"kit"` and `zones`
    /// holds its mapping; otherwise `zones` is empty.
    pub model_id: String,
    pub params: serde_json::Value,
    pub engine: EngineParams,
    pub zones: Vec<ZoneData>,
    /// Number of recorded automation points on this track.
    pub automation: usize,
    /// Mixer state.
    pub volume: f32,
    pub pan: f32,
    pub solo: bool,
}

/// Lock-free scalars + an occasionally-rebuilt track list for the UI.
pub struct SharedView {
    state: AtomicU8,
    mode: AtomicU8,
    pos: AtomicU64,
    loop_len: AtomicU64,
    tracks: Mutex<Vec<TrackView>>,
    /// A full serializable snapshot of the current loop (for saving to a project).
    snapshot: Mutex<LoopData>,
    /// Current song section index while a song plays, else -1.
    song_section: AtomicI64,
}

impl SharedView {
    fn new() -> Self {
        SharedView {
            state: AtomicU8::new(TransportState::Idle.as_u8()),
            mode: AtomicU8::new(0),
            pos: AtomicU64::new(0),
            loop_len: AtomicU64::new(0),
            tracks: Mutex::new(Vec::new()),
            snapshot: Mutex::new(LoopData::default()),
            song_section: AtomicI64::new(-1),
        }
    }

    /// The section index currently playing in the song, or `None`.
    pub fn song_section(&self) -> Option<usize> {
        let v = self.song_section.load(Ordering::Relaxed);
        if v < 0 {
            None
        } else {
            Some(v as usize)
        }
    }

    /// The current loop as serializable data (for "add to project").
    pub fn snapshot(&self) -> LoopData {
        self.snapshot.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn state(&self) -> TransportState {
        TransportState::from_u8(self.state.load(Ordering::Relaxed))
    }
    /// Playhead position within the loop, in `[0,1)`. 0 if no loop yet.
    pub fn play_fraction(&self) -> f32 {
        let len = self.loop_len.load(Ordering::Relaxed);
        if len == 0 {
            0.0
        } else {
            (self.pos.load(Ordering::Relaxed) % len) as f32 / len as f32
        }
    }
    pub fn loop_seconds(&self, sr: f32) -> f32 {
        self.loop_len.load(Ordering::Relaxed) as f32 / sr
    }
    pub fn tracks(&self) -> Vec<TrackView> {
        self.tracks.lock().map(|t| t.clone()).unwrap_or_default()
    }
}

pub struct Studio {
    sr: f32,
    sine: std::sync::Arc<[f32]>,
    live: Playable,
    tracks: Vec<Track>,
    mode: LooperMode,

    playing: bool,
    loop_len: Option<u64>,
    pos: u64,
    /// Track index currently capturing live input, if any.
    recording: Option<usize>,
    /// Armed to capture into a *new* track; the track is created lazily on the
    /// first recorded note, so empty takes never materialize.
    armed: bool,
    /// True while recording the very first pass (loop length still being defined).
    defining: bool,
    /// For a pedal-mode one-pass overdub: finalize after this many recorded frames.
    auto_finalize_at: Option<u64>,
    /// Frames elapsed since arming (measures the one-pass window).
    rec_frames: u64,

    // Song arrangement playback
    song: Vec<SongSection>,
    song_active: bool,
    song_pos: usize,
    song_rep: u32,

    // Tempo / grid / metronome
    tempo: TempoGrid,
    /// Count-in samples remaining before a pending recording starts.
    pre_roll: u64,
    /// True while `pre_roll` is counting toward a first (fixed-bars) recording.
    pending_record: bool,
    /// Metronome beat clock: samples into the current beat, and beat index.
    metro_phase: u64,
    beat_index: u32,
    // Metronome click voice.
    click_env: f32,
    click_phase: f32,
    click_freq: f32,
    click_decay: f32,

    /// Master output level (linear).
    master: f32,

    view: Arc<SharedView>,
    /// Structure snapshot waiting to be published to the UI (flushed each render).
    pending_structure: Option<Vec<TrackView>>,
    /// Serializable loop snapshot waiting to be published (for saving).
    pending_snapshot: Option<LoopData>,
}

impl Studio {
    pub fn new(sample_rate: f32) -> Self {
        let sine = make_sine_table();
        Studio {
            sr: sample_rate,
            sine: sine.clone(),
            live: Playable::Single(Instrument::new(sample_rate, sine)),
            tracks: Vec::new(),
            mode: LooperMode::Pedal,
            playing: false,
            loop_len: None,
            pos: 0,
            recording: None,
            armed: false,
            defining: false,
            auto_finalize_at: None,
            rec_frames: 0,
            song: Vec::new(),
            song_active: false,
            song_pos: 0,
            song_rep: 0,
            tempo: TempoGrid::default(),
            pre_roll: 0,
            pending_record: false,
            metro_phase: 0,
            beat_index: 0,
            click_env: 0.0,
            click_phase: 0.0,
            click_freq: 1000.0,
            click_decay: (-1.0 / (0.03 * sample_rate)).exp(),
            master: 1.0,
            view: Arc::new(SharedView::new()),
            pending_structure: None,
            pending_snapshot: None,
        }
    }

    pub fn view(&self) -> Arc<SharedView> {
        self.view.clone()
    }

    // ---- command handling ----

    pub fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::NoteOn { note, vel } => {
                self.live.note_on(note, vel);
                self.record_event(EvMsg::On { note, vel });
            }
            Command::NoteOff { note } => {
                self.live.note_off(note);
                self.record_event(EvMsg::Off { note });
            }
            Command::SetModel(m) => {
                self.capture_model_auto(m.as_ref());
                self.live.set_model(m);
            }
            Command::SetEngine(e) => {
                self.capture_engine_auto(&e);
                self.live.set_engine(e);
            }
            Command::SetLive(cfg) => {
                self.live.all_notes_off();
                self.live = self.build_live(cfg);
            }
            Command::AllNotesOff => {
                self.live.all_notes_off();
                for t in &mut self.tracks {
                    t.inst.all_notes_off();
                }
            }
            Command::SetLooperMode(m) => {
                self.mode = m;
                self.view.mode.store(m.as_u8(), Ordering::Relaxed);
            }
            Command::Tap => self.tap(),
            Command::Stop => self.stop(),
            Command::Reset => self.reset(),
            Command::ArmOverdub => self.arm_overdub_one_pass(),
            Command::ToggleMute(i) => {
                if let Some(t) = self.tracks.get_mut(i) {
                    t.muted = !t.muted;
                    if t.muted {
                        t.inst.all_notes_off();
                    }
                    self.mark_structure_dirty();
                }
            }
            Command::SetTrackModel(i, m) => {
                let mut relabel = false;
                if let Some(t) = self.tracks.get_mut(i) {
                    if t.label != m.display_name() {
                        relabel = true;
                        t.label = m.display_name().to_string();
                    }
                    // Re-tune the automation base so live edits stick and any
                    // recorded deltas ride on top of the new base.
                    t.model_id = m.id().to_string();
                    t.base_json = m.to_json();
                    t.cur_json = t.base_json.clone();
                    t.inst.set_model(m);
                }
                if relabel {
                    self.mark_structure_dirty(); // instrument name changed
                }
            }
            Command::SetTrackEngine(i, e) => {
                if let Some(t) = self.tracks.get_mut(i) {
                    t.base_engine = e.clone();
                    t.cur_engine = e.clone();
                    t.inst.set_engine(e);
                }
            }
            Command::LoadLoop(data) => self.load_loop(data),
            Command::SetSong(sections) => self.song = sections,
            Command::PlaySong => self.play_song(),
            Command::SetTempo(t) => self.tempo = t,
            Command::RegionEdit { track, op } => self.region_edit(track, op),
            Command::SetTrackMix { track, volume, pan } => {
                if let Some(t) = self.tracks.get_mut(track) {
                    t.volume = volume;
                    t.pan = pan.clamp(-1.0, 1.0);
                    self.mark_structure_dirty();
                }
            }
            Command::ToggleSolo(i) => {
                if let Some(t) = self.tracks.get_mut(i) {
                    t.solo = !t.solo;
                    self.mark_structure_dirty();
                }
            }
            Command::SetMasterVolume(v) => self.master = v.max(0.0),
            Command::ClearTrackAutomation(i) => {
                if let Some(t) = self.tracks.get_mut(i) {
                    t.auto.clear();
                    t.auto_cursor = 0;
                    t.cur_json = t.base_json.clone();
                    t.cur_engine = t.base_engine.clone();
                    self.mark_structure_dirty();
                }
            }
            Command::DeleteTrack(i) => {
                if i < self.tracks.len() {
                    self.tracks.remove(i);
                    // Fix up the recording index if needed.
                    self.recording = match self.recording {
                        Some(r) if r == i => None,
                        Some(r) if r > i => Some(r - 1),
                        other => other,
                    };
                    if self.tracks.is_empty() {
                        self.loop_len = None;
                        self.playing = false;
                        self.pos = 0;
                    }
                    self.renumber_tracks();
                    self.mark_structure_dirty();
                }
            }
        }
    }

    // ---- transport state machine ----

    /// Append a recorded event, lazily creating the armed track on first note.
    fn record_event(&mut self, msg: EvMsg) {
        if self.armed && self.recording.is_none() {
            self.create_track();
        }
        if let Some(r) = self.recording {
            // Quantize note-ons to the grid; leave note-offs at their real time.
            let pos = match msg {
                EvMsg::On { .. } => self.quantize_pos(self.pos),
                EvMsg::Off { .. } => self.pos,
            };
            self.tracks[r].events.push(Event { pos, msg });
        }
    }

    /// While recording a single-instrument track, capture the model params that
    /// changed as automation — stored as a **delta from the track's base value**,
    /// so re-tuning the base later shifts the whole automated move with it.
    fn capture_model_auto(&mut self, new_model: &dyn FtmModel) {
        let Some(r) = self.recording else { return };
        let old = self.live.parts().1; // detect which field the user moved
        let new = new_model.to_json();
        let pos = self.pos;
        for (id, val) in numeric_fields(&new) {
            let changed = numeric_at(&old, &id).map(|o| o != val).unwrap_or(true);
            if changed {
                let base = numeric_at(&self.tracks[r].base_json, &id).unwrap_or(val);
                self.tracks[r].auto.push(AutoEv { pos, target: id, value: (val - base) as f32 });
            }
        }
    }

    /// While recording, capture changed engine params as a delta from the base.
    fn capture_engine_auto(&mut self, new_engine: &EngineParams) {
        let Some(r) = self.recording else { return };
        let (_, _, live_old, _) = self.live.parts();
        let pos = self.pos;
        let base = self.tracks[r].base_engine.clone();
        let t = &mut self.tracks[r];
        let mut push = |name: &str, ov: f32, nv: f32, base_v: f32| {
            if ov != nv {
                t.auto.push(AutoEv { pos, target: format!("eng:{name}"), value: nv - base_v });
            }
        };
        push("gain", live_old.gain, new_engine.gain, base.gain);
        push("attack", live_old.attack_ms, new_engine.attack_ms, base.attack_ms);
        push("release", live_old.release_ms, new_engine.release_ms, base.release_ms);
        push("retrigger", live_old.retrigger_ms, new_engine.retrigger_ms, base.retrigger_ms);
    }

    // ---- tempo / grid helpers ----

    /// Samples per beat at the current tempo.
    fn spb(&self) -> f64 {
        self.sr as f64 * 60.0 / self.tempo.bpm.max(1.0) as f64
    }
    /// Samples in one bar.
    fn bar_samples(&self) -> u64 {
        (self.spb() * self.tempo.beats_per_bar.max(1) as f64).round() as u64
    }
    /// Samples in the fixed loop length (bars × bar), or 0 if free.
    fn fixed_loop_samples(&self) -> u64 {
        (self.bar_samples() * self.tempo.bars as u64).max(if self.tempo.bars > 0 { 1 } else { 0 })
    }

    /// Snap a sample position to the quantize grid (nearest step), wrapping at
    /// the loop length. Returns `pos` unchanged when quantize is off.
    fn quantize_pos(&self, pos: u64) -> u64 {
        let steps = self.tempo.quantize;
        if steps == 0 {
            return pos;
        }
        let grid = self.spb() / steps as f64;
        if grid < 1.0 {
            return pos;
        }
        let q = ((pos as f64 / grid).round() * grid).round() as u64;
        match self.loop_len {
            Some(len) if q >= len => 0,
            _ => q,
        }
    }

    /// Begin the first take from idle (free, or fixed-length if bars is set).
    fn begin_first_take(&mut self) {
        self.arm(true);
        self.playing = true;
        self.pos = 0;
        self.metro_phase = 0;
        self.beat_index = 0;
        if self.tempo.bars > 0 {
            self.loop_len = Some(self.fixed_loop_samples());
        }
    }

    /// Close a fixed-length first take at the bar boundary (called from the
    /// loop-wrap handler when `defining` with a known length).
    fn close_defining(&mut self) {
        self.disarm_and_finalize();
        self.defining = false;
        if self.tracks.is_empty() {
            // Nothing recorded — cancel the loop.
            self.loop_len = None;
            self.playing = false;
        } else if self.mode == LooperMode::Overdub {
            self.arm(false);
        }
        self.mark_structure_dirty();
    }

    // ---- metronome ----

    fn trigger_click(&mut self, accent: bool) {
        self.click_env = 1.0;
        self.click_phase = 0.0;
        self.click_freq = if accent { 1568.0 } else { 1047.0 };
    }

    /// Advance the beat clock one sample and fire a click at each beat.
    fn metro_tick(&mut self) {
        if !self.tempo.metronome {
            return;
        }
        if self.metro_phase == 0 {
            let accent = self.beat_index % self.tempo.beats_per_bar.max(1) == 0;
            self.trigger_click(accent);
        }
        self.metro_phase += 1;
        if self.metro_phase >= self.bar_samples() / self.tempo.beats_per_bar.max(1) as u64 {
            self.metro_phase = 0;
            self.beat_index += 1;
        }
    }

    /// One sample of the metronome click (decaying sine), 0 when silent.
    fn click_sample(&mut self) -> f32 {
        if self.click_env < 1e-4 {
            return 0.0;
        }
        let s = (self.click_phase as f64 * TWO_PI_F64).sin() as f32 * self.click_env * 0.25;
        self.click_phase += self.click_freq / self.sr;
        if self.click_phase >= 1.0 {
            self.click_phase -= 1.0;
        }
        self.click_env *= self.click_decay;
        s
    }

    fn tap(&mut self) {
        // Ignore taps during a count-in or a fixed-length first take (it
        // auto-closes at the bar boundary).
        if self.pre_roll > 0 || (self.defining && self.loop_len.is_some()) {
            return;
        }
        let recording = self.recording.is_some();
        match (self.loop_len, self.defining) {
            // Idle: arm the first (loop-defining) take.
            (None, false) if !self.armed && !recording => {
                if self.tempo.bars > 0 && self.tempo.count_in {
                    // Count in one bar of clicks, then begin recording.
                    self.pre_roll = self.bar_samples();
                    self.pending_record = true;
                    self.playing = false;
                    self.metro_phase = 0;
                    self.beat_index = 0;
                } else {
                    self.begin_first_take();
                }
            }
            // Closing the first pass: fix loop length, start looping.
            (None, true) => {
                if recording {
                    // A take was actually recorded — close the loop around it.
                    self.loop_len = Some(self.pos.max(1));
                    self.disarm_and_finalize();
                    self.defining = false;
                    self.pos = 0;
                    self.reset_cursors();
                    self.playing = true;
                    if self.mode == LooperMode::Overdub {
                        self.arm(false); // arm the next take from the top
                    }
                } else {
                    // Nothing recorded — cancel back to idle.
                    self.armed = false;
                    self.defining = false;
                    self.playing = false;
                }
            }
            // Loop already exists.
            (Some(_), _) => match self.mode {
                LooperMode::Pedal => {
                    if recording || self.armed {
                        self.disarm_and_finalize(); // finish this overdub, keep playing
                    } else if self.playing {
                        self.arm(false); // record over into a new track
                    } else {
                        self.playing = true; // resume from a stop
                    }
                }
                LooperMode::Overdub => {
                    self.disarm_and_finalize(); // close current take (if any)
                    self.arm(false); // arm the next
                    self.playing = true;
                }
            },
            _ => {}
        }
        self.mark_structure_dirty();
    }

    fn stop(&mut self) {
        self.disarm_and_finalize();
        self.playing = false;
        self.defining = false;
        self.song_active = false;
        self.pre_roll = 0;
        self.pending_record = false;
        self.live.all_notes_off();
        for t in &mut self.tracks {
            t.inst.all_notes_off();
        }
        self.mark_structure_dirty();
    }

    fn reset(&mut self) {
        self.tracks.clear();
        self.recording = None;
        self.armed = false;
        self.defining = false;
        self.auto_finalize_at = None;
        self.loop_len = None;
        self.pos = 0;
        self.playing = false;
        self.song_active = false;
        self.pre_roll = 0;
        self.pending_record = false;
        self.live.all_notes_off();
        self.mark_structure_dirty();
    }

    /// Load a single saved loop (leaving any song stopped) and play it.
    fn load_loop(&mut self, data: LoopData) {
        self.song_active = false;
        self.install_loop(data);
        self.mark_structure_dirty();
    }

    /// Start playing the installed song from the top.
    fn play_song(&mut self) {
        if self.song.is_empty() {
            return;
        }
        self.song_active = true;
        self.song_pos = 0;
        self.song_rep = 0;
        let data = self.song[0].loop_data.clone();
        self.install_loop(data);
        self.mark_structure_dirty();
    }

    /// Build the live [`Playable`] from a UI config (single instrument or kit).
    fn build_live(&self, cfg: LiveConfig) -> Playable {
        match cfg {
            LiveConfig::Single { model_id, params, engine } => {
                let model = model_from_id(&model_id, &params).unwrap_or_else(default_model);
                Playable::Single(Instrument::with_config(self.sr, self.sine.clone(), model, engine))
            }
            LiveConfig::Kit { zones } => {
                Playable::Kit(Kit::from_data(self.sr, self.sine.clone(), &zones))
            }
        }
    }

    /// Build a track's [`Playable`] from its serialized form: a kit if it has
    /// zones, otherwise a single instrument.
    fn track_playable(&self, lt: &LoopTrack) -> Playable {
        if lt.zones.is_empty() {
            let model = model_from_id(&lt.model_id, &lt.params).unwrap_or_else(default_model);
            Playable::Single(Instrument::with_config(
                self.sr,
                self.sine.clone(),
                model,
                lt.engine.clone(),
            ))
        } else {
            Playable::Kit(Kit::from_data(self.sr, self.sine.clone(), &lt.zones))
        }
    }

    /// Rebuild the tracks/instruments from a saved loop and start it playing.
    /// Positions in `data` are in seconds; converted to samples at this rate.
    /// Does not touch song-transport state (used by both single-loop load and
    /// song section advances).
    fn install_loop(&mut self, data: LoopData) {
        self.tracks.clear();
        self.recording = None;
        self.armed = false;
        self.defining = false;
        self.auto_finalize_at = None;

        if data.is_empty() {
            self.loop_len = None;
            self.pos = 0;
            self.playing = false;
            return;
        }

        let len = ((data.length * self.sr).round() as u64).max(1);
        self.loop_len = Some(len);
        for lt in data.tracks {
            let inst = self.track_playable(&lt);
            let label = inst.label();
            let mut events: Vec<Event> = lt
                .events
                .iter()
                .map(|e| {
                    let pos = ((e.t * self.sr).round() as u64).min(len - 1);
                    let msg = if e.on {
                        EvMsg::On { note: e.note, vel: e.vel }
                    } else {
                        EvMsg::Off { note: e.note }
                    };
                    Event { pos, msg }
                })
                .collect();
            events.sort_by_key(|e| e.pos);
            let mut auto: Vec<AutoEv> = lt
                .automation
                .iter()
                .map(|a| AutoEv {
                    pos: ((a.t * self.sr).round() as u64).min(len - 1),
                    target: a.target.clone(),
                    value: a.value,
                })
                .collect();
            auto.sort_by_key(|a| a.pos);
            self.tracks.push(Track {
                inst,
                events,
                cursor: 0,
                muted: lt.muted,
                volume: lt.volume,
                pan: lt.pan,
                solo: false,
                name: lt.name,
                label,
                auto,
                auto_cursor: 0,
                model_id: lt.model_id.clone(),
                base_json: lt.params.clone(),
                base_engine: lt.engine.clone(),
                cur_json: lt.params,
                cur_engine: lt.engine,
            });
        }
        self.pos = 0;
        self.reset_cursors();
        self.playing = true;
        // Callers (load_loop / play_song / section advance) publish the structure.
    }

    /// Build a serializable snapshot of the current loop (positions in seconds).
    fn snapshot_loop(&self) -> LoopData {
        let sr = self.sr;
        let length = self.loop_len.unwrap_or(0) as f32 / sr;
        let tracks = self
            .tracks
            .iter()
            .map(|t| {
                let (model_id, params, engine, zones) = t.inst.parts();
                LoopTrack {
                name: t.name.clone(),
                model_id,
                params,
                engine,
                muted: t.muted,
                volume: t.volume,
                pan: t.pan,
                zones,
                automation: t
                    .auto
                    .iter()
                    .map(|a| AutoPoint {
                        t: a.pos as f32 / sr,
                        target: a.target.clone(),
                        value: a.value,
                    })
                    .collect(),
                events: t
                    .events
                    .iter()
                    .map(|e| {
                        let (on, note, vel) = match e.msg {
                            EvMsg::On { note, vel } => (true, note, vel),
                            EvMsg::Off { note } => (false, note, 0.0),
                        };
                        LoopEvent { t: e.pos as f32 / sr, on, note, vel }
                    })
                    .collect(),
                }
            })
            .collect();
        LoopData { length, tracks }
    }

    /// Pedal-mode "+ Rec track": record exactly one loop pass into a new track.
    fn arm_overdub_one_pass(&mut self) {
        let Some(len) = self.loop_len else { return };
        if self.recording.is_some() || self.armed {
            return;
        }
        self.arm(false);
        self.auto_finalize_at = Some(len);
        self.playing = true;
        self.mark_structure_dirty();
    }

    /// Arm capture into a new (not-yet-created) track.
    fn arm(&mut self, defining: bool) {
        self.armed = true;
        self.defining = defining;
        self.recording = None;
        self.rec_frames = 0;
        self.auto_finalize_at = None;
    }

    /// Materialize the armed track, bound to a snapshot of the live instrument.
    fn create_track(&mut self) {
        let idx = self.tracks.len();
        let inst = self.live.snapshot();
        let label = inst.label();
        let (model_id, base_json, base_engine, _zones) = self.live.parts();
        self.tracks.push(Track {
            inst,
            events: Vec::new(),
            cursor: 0,
            muted: false,
            volume: 1.0,
            pan: 0.0,
            solo: false,
            name: format!("Track {}", idx + 1),
            label,
            auto: Vec::new(),
            auto_cursor: 0,
            model_id,
            base_json: base_json.clone(),
            base_engine: base_engine.clone(),
            cur_json: base_json,
            cur_engine: base_engine,
        });
        self.recording = Some(idx);
        self.armed = false;
    }

    /// Close any in-progress recording and clear the armed state.
    fn disarm_and_finalize(&mut self) {
        self.armed = false;
        self.auto_finalize_at = None;
        if let Some(idx) = self.recording.take() {
            if let Some(t) = self.tracks.get_mut(idx) {
                t.events.sort_by_key(|e| e.pos);
                t.auto.sort_by_key(|a| a.pos);
                let pos = self.pos;
                t.cursor = t.events.partition_point(|e| e.pos < pos);
                t.auto_cursor = t.auto.partition_point(|a| a.pos < pos);
                if t.events.is_empty() {
                    self.tracks.remove(idx);
                    self.renumber_tracks();
                }
            }
        }
    }

    fn reset_cursors(&mut self) {
        for t in &mut self.tracks {
            t.cursor = 0;
            // Rewind automation to the track's base state so the loop repeats.
            if !t.auto.is_empty() {
                t.auto_cursor = 0;
                t.cur_json = t.base_json.clone();
                t.cur_engine = t.base_engine.clone();
                if let Some(model) = model_from_id(&t.model_id, &t.base_json) {
                    t.inst.set_model(model);
                }
                t.inst.set_engine(t.base_engine.clone());
            }
        }
    }

    fn renumber_tracks(&mut self) {
        for (i, t) in self.tracks.iter_mut().enumerate() {
            t.name = format!("Track {}", i + 1);
        }
    }

    /// Chop / crop / rearrange a track's timeline. Positions and loop length are
    /// preserved; only the selected notes + automation move.
    fn region_edit(&mut self, i: usize, op: RegionOp) {
        let sr = self.sr;
        let Some(len) = self.loop_len else { return };
        let pos = self.pos;
        let p = |secs: f32| ((secs * sr).round().max(0.0) as u64).min(len);
        {
            let Some(t) = self.tracks.get_mut(i) else { return };
            match op {
                RegionOp::Delete { a, b } => {
                    let (a, b) = (p(a), p(b));
                    let spans = events_to_spans(&t.events, len);
                    let kept: Vec<Span> =
                        spans.into_iter().filter(|s| !(s.start >= a && s.start < b)).collect();
                    t.events = spans_to_events(&kept);
                    t.auto.retain(|x| !(x.pos >= a && x.pos < b));
                }
                RegionOp::Keep { a, b } => {
                    let (a, b) = (p(a), p(b));
                    let spans = events_to_spans(&t.events, len);
                    let kept: Vec<Span> =
                        spans.into_iter().filter(|s| s.start >= a && s.start < b).collect();
                    t.events = spans_to_events(&kept);
                    t.auto.retain(|x| x.pos >= a && x.pos < b);
                }
                RegionOp::Duplicate { a, b, dest } => {
                    let (a, b, dest) = (p(a), p(b), p(dest));
                    let spans = events_to_spans(&t.events, len);
                    t.events = spans_to_events(&dup_spans(&spans, a, b, dest, len));
                    t.auto = dup_auto(&t.auto, a, b, dest, len);
                }
                RegionOp::Move { a, b, dest } => {
                    let (a, b, dest) = (p(a), p(b), p(dest));
                    let spans = events_to_spans(&t.events, len);
                    let dup = dup_spans(&spans, a, b, dest, len);
                    let kept: Vec<Span> =
                        dup.into_iter().filter(|s| !(s.start >= a && s.start < b)).collect();
                    t.events = spans_to_events(&kept);
                    let mut au = dup_auto(&t.auto, a, b, dest, len);
                    au.retain(|x| !(x.pos >= a && x.pos < b));
                    t.auto = au;
                }
                RegionOp::Shift { delta } => {
                    let d = (delta * sr).round() as i64;
                    let spans = events_to_spans(&t.events, len);
                    let shifted: Vec<Span> = spans
                        .iter()
                        .map(|s| {
                            let dur = s.end.saturating_sub(s.start);
                            let ns = wrap_pos(s.start as i64 + d, len);
                            Span { start: ns, end: (ns + dur).min(len), note: s.note, vel: s.vel }
                        })
                        .collect();
                    t.events = spans_to_events(&shifted);
                    for x in &mut t.auto {
                        x.pos = wrap_pos(x.pos as i64 + d, len);
                    }
                }
            }
            t.events.sort_by_key(|e| e.pos);
            t.auto.sort_by_key(|a| a.pos);
            t.cursor = t.events.partition_point(|e| e.pos < pos);
            t.auto_cursor = t.auto.partition_point(|a| a.pos < pos);
        }
        self.mark_structure_dirty();
    }

    // ---- offline export ----

    /// Render the currently-installed loop/song for `frames` samples (transport
    /// running), then let voices ring for `tail` more with the transport halted
    /// (no new events, no hard cutoff), into one mono buffer. For WAV export.
    pub fn render_offline(&mut self, frames: usize, tail: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; frames];
        self.render(&mut buf, 1);
        // Halt the transport but keep sounding voices so they decay naturally.
        self.playing = false;
        self.song_active = false;
        if tail > 0 {
            let mut ring = vec![0.0f32; tail];
            self.render(&mut ring, 1);
            buf.extend_from_slice(&ring);
        }
        buf
    }

    // ---- audio rendering ----

    /// Render `out` (interleaved by `channels`).
    pub fn render(&mut self, out: &mut [f32], channels: usize) {
        self.flush_structure();
        for frame in out.chunks_mut(channels) {
            if self.pre_roll > 0 {
                // Count-in: clicks only, no playback/recording.
                self.metro_tick();
                self.pre_roll -= 1;
                if self.pre_roll == 0 && self.pending_record {
                    self.pending_record = false;
                    self.begin_first_take();
                }
            } else if self.playing {
                self.fire_events();
                self.advance_clock();
                self.metro_tick();
            }

            // Mix into a stereo bus: live is centred; each track is panned and
            // levelled. Every track is still rendered (to advance envelopes) even
            // when silenced by mute/solo, so unmuting doesn't pop.
            let any_solo = self.tracks.iter().any(|t| t.solo);
            let live_s = self.live.render_frame();
            let (mut l, mut r) = (live_s * FRAC_1_SQRT_2, live_s * FRAC_1_SQRT_2);
            for t in &mut self.tracks {
                let f = t.inst.render_frame();
                let audible = !t.muted && (!any_solo || t.solo);
                if audible {
                    let (lg, rg) = pan_gains(t.pan);
                    l += f * t.volume * lg;
                    r += f * t.volume * rg;
                }
            }
            let click = self.click_sample() * FRAC_1_SQRT_2;
            l = (l + click) * self.master;
            r = (r + click) * self.master;
            let (l, r) = (l.clamp(-1.0, 1.0), r.clamp(-1.0, 1.0));
            for (ch, out) in frame.iter_mut().enumerate() {
                *out = if channels < 2 {
                    (l + r) * 0.5
                } else if ch % 2 == 0 {
                    l
                } else {
                    r
                };
            }
        }
        self.publish_scalars();
    }

    fn fire_events(&mut self) {
        let pos = self.pos;
        let recording = self.recording;
        for (i, t) in self.tracks.iter_mut().enumerate() {
            if Some(i) == recording {
                continue; // don't play the take we're currently recording
            }
            // Skip anything already behind us, then fire everything on this frame.
            while t.cursor < t.events.len() && t.events[t.cursor].pos < pos {
                t.cursor += 1;
            }
            while t.cursor < t.events.len() && t.events[t.cursor].pos == pos {
                match t.events[t.cursor].msg {
                    EvMsg::On { note, vel } => t.inst.note_on(note, vel),
                    EvMsg::Off { note } => t.inst.note_off(note),
                }
                t.cursor += 1;
            }

            // Fire any parameter automation landing on this frame.
            if !t.auto.is_empty() {
                while t.auto_cursor < t.auto.len() && t.auto[t.auto_cursor].pos < pos {
                    t.auto_cursor += 1;
                }
                let (mut model_dirty, mut engine_dirty) = (false, false);
                while t.auto_cursor < t.auto.len() && t.auto[t.auto_cursor].pos == pos {
                    let (target, value) = {
                        let ev = &t.auto[t.auto_cursor];
                        (ev.target.clone(), ev.value)
                    };
                    let (m, e) = apply_auto(
                        &mut t.cur_json,
                        &mut t.cur_engine,
                        &t.base_json,
                        &t.base_engine,
                        &target,
                        value,
                    );
                    model_dirty |= m;
                    engine_dirty |= e;
                    t.auto_cursor += 1;
                }
                if model_dirty {
                    if let Some(model) = model_from_id(&t.model_id, &t.cur_json) {
                        t.inst.set_model(model);
                    }
                }
                if engine_dirty {
                    t.inst.set_engine(t.cur_engine.clone());
                }
            }
        }
    }

    fn advance_clock(&mut self) {
        self.pos += 1;
        if self.armed || self.recording.is_some() {
            self.rec_frames += 1;
            if let Some(limit) = self.auto_finalize_at {
                if self.rec_frames >= limit {
                    self.disarm_and_finalize(); // one-pass overdub done
                    self.mark_structure_dirty();
                }
            }
        }
        if let Some(len) = self.loop_len {
            if self.pos >= len {
                self.pos = 0;
                self.reset_cursors();
                if self.defining {
                    // A fixed-length first take just completed one bar-count pass.
                    self.close_defining();
                } else if self.song_active {
                    self.advance_song();
                }
            }
        }
    }

    /// At a loop boundary during song playback, count the repeat and move to the
    /// next section (or end the song) when this section's repeats are done.
    fn advance_song(&mut self) {
        self.song_rep += 1;
        let reps = self
            .song
            .get(self.song_pos)
            .map(|s| s.repeats.max(1))
            .unwrap_or(1);
        if self.song_rep < reps {
            return; // keep repeating this section
        }
        self.song_pos += 1;
        self.song_rep = 0;
        if self.song_pos >= self.song.len() {
            // Song finished.
            self.song_active = false;
            self.playing = false;
            for t in &mut self.tracks {
                t.inst.all_notes_off();
            }
            self.mark_structure_dirty();
            return;
        }
        let data = self.song[self.song_pos].loop_data.clone();
        self.install_loop(data); // swap in the next section's tracks
        self.mark_structure_dirty();
    }

    // ---- publishing to the UI ----

    fn current_state(&self) -> TransportState {
        if self.recording.is_some() || self.armed || self.pre_roll > 0 {
            TransportState::Recording
        } else if self.playing {
            TransportState::Playing
        } else if self.tracks.is_empty() && self.loop_len.is_none() {
            TransportState::Idle
        } else {
            TransportState::Stopped
        }
    }

    fn publish_scalars(&self) {
        self.view
            .state
            .store(self.current_state().as_u8(), Ordering::Relaxed);
        self.view.pos.store(self.pos, Ordering::Relaxed);
        self.view
            .loop_len
            .store(self.loop_len.unwrap_or(0), Ordering::Relaxed);
        self.view.song_section.store(
            if self.song_active {
                self.song_pos as i64
            } else {
                -1
            },
            Ordering::Relaxed,
        );
    }

    fn mark_structure_dirty(&mut self) {
        let len = self.loop_len.unwrap_or(0).max(1) as f32;
        let views = self
            .tracks
            .iter()
            .map(|t| {
                let (model_id, params, engine, zones) = t.inst.parts();
                TrackView {
                    name: t.name.clone(),
                    instrument: t.label.clone(),
                    muted: t.muted,
                    notes: note_spans(&t.events, len),
                    model_id,
                    params,
                    engine,
                    zones,
                    automation: t.auto.len(),
                    volume: t.volume,
                    pan: t.pan,
                    solo: t.solo,
                }
            })
            .collect();
        let snap = self.snapshot_loop();
        self.pending_structure = Some(views);
        self.pending_snapshot = Some(snap);
        self.publish_scalars();
    }

    fn flush_structure(&mut self) {
        if self.pending_structure.is_some() {
            if let Ok(mut guard) = self.view.tracks.try_lock() {
                *guard = self.pending_structure.take().unwrap();
            }
        }
        if self.pending_snapshot.is_some() {
            if let Ok(mut guard) = self.view.snapshot.try_lock() {
                *guard = self.pending_snapshot.take().unwrap();
            }
        }
    }
}

/// Pair note-on/off events into normalized spans for drawing.
/// A paired note: on at `start`, off at `end`.
#[derive(Clone)]
struct Span {
    start: u64,
    end: u64,
    note: u8,
    vel: f32,
}

/// Pair note-on/off events into spans. A note still held at the loop end is
/// closed at `loop_len`.
fn events_to_spans(events: &[Event], loop_len: u64) -> Vec<Span> {
    let mut sorted: Vec<&Event> = events.iter().collect();
    sorted.sort_by_key(|e| e.pos);
    let mut open: Vec<(u8, u64, f32)> = Vec::new();
    let mut spans = Vec::new();
    for e in sorted {
        match e.msg {
            EvMsg::On { note, vel } => open.push((note, e.pos, vel)),
            EvMsg::Off { note } => {
                if let Some(idx) = open.iter().rposition(|(n, _, _)| *n == note) {
                    let (n, start, vel) = open.remove(idx);
                    spans.push(Span { start, end: e.pos.max(start), note: n, vel });
                }
            }
        }
    }
    for (note, start, vel) in open {
        spans.push(Span { start, end: loop_len, note, vel });
    }
    spans
}

/// Rebuild sorted on/off events from spans.
fn spans_to_events(spans: &[Span]) -> Vec<Event> {
    let mut ev = Vec::with_capacity(spans.len() * 2);
    for s in spans {
        ev.push(Event { pos: s.start, msg: EvMsg::On { note: s.note, vel: s.vel } });
        ev.push(Event { pos: s.end, msg: EvMsg::Off { note: s.note } });
    }
    ev.sort_by_key(|e| e.pos);
    ev
}

/// Copy the spans starting in `[a, b)` to begin at `dest`, appended to the set.
/// Copies whose start would land outside the loop are dropped.
fn dup_spans(spans: &[Span], a: u64, b: u64, dest: u64, loop_len: u64) -> Vec<Span> {
    let shift = dest as i64 - a as i64;
    let mut out = spans.to_vec();
    for s in spans {
        if s.start >= a && s.start < b {
            let ns = s.start as i64 + shift;
            let ne = s.end as i64 + shift;
            if ns >= 0 && (ns as u64) < loop_len {
                out.push(Span {
                    start: ns as u64,
                    end: (ne.max(ns + 1) as u64).min(loop_len),
                    note: s.note,
                    vel: s.vel,
                });
            }
        }
    }
    out
}

/// Copy automation points in `[a, b)` to start at `dest`, appended to the set.
fn dup_auto(auto: &[AutoEv], a: u64, b: u64, dest: u64, loop_len: u64) -> Vec<AutoEv> {
    let shift = dest as i64 - a as i64;
    let mut out = auto.to_vec();
    for x in auto {
        if x.pos >= a && x.pos < b {
            let np = x.pos as i64 + shift;
            if np >= 0 && (np as u64) < loop_len {
                out.push(AutoEv { pos: np as u64, target: x.target.clone(), value: x.value });
            }
        }
    }
    out
}

/// Wrap a (possibly negative) sample position into `[0, len)`.
fn wrap_pos(p: i64, len: u64) -> u64 {
    let l = len as i64;
    (((p % l) + l) % l) as u64
}

/// The numeric (`f64`-valued) top-level fields of a params object — the
/// automatable parameters. Non-numbers (enums, bools) are skipped.
fn numeric_fields(v: &serde_json::Value) -> Vec<(String, f64)> {
    match v.as_object() {
        Some(map) => map
            .iter()
            .filter_map(|(k, val)| val.as_f64().map(|f| (k.clone(), f)))
            .collect(),
        None => Vec::new(),
    }
}

/// The numeric value of one field, if present and numeric.
fn numeric_at(v: &serde_json::Value, id: &str) -> Option<f64> {
    v.get(id).and_then(|x| x.as_f64())
}

/// Apply one automation move — a **delta from the base** — to the evolving
/// state (`effective = base + delta`), returning whether the model or the engine
/// needs rebuilding. Because the delta rides on the current base, editing the
/// track's base value shifts the automated move with it. Integer-valued fields
/// (mode counts) keep their integer JSON type so the model still deserializes.
fn apply_auto(
    cur_json: &mut serde_json::Value,
    cur_engine: &mut EngineParams,
    base_json: &serde_json::Value,
    base_engine: &EngineParams,
    target: &str,
    delta: f32,
) -> (bool, bool) {
    if let Some(name) = target.strip_prefix("eng:") {
        let base = match name {
            "gain" => base_engine.gain,
            "attack" => base_engine.attack_ms,
            "release" => base_engine.release_ms,
            "retrigger" => base_engine.retrigger_ms,
            _ => return (false, false),
        };
        let v = base + delta;
        match name {
            "gain" => cur_engine.gain = v,
            "attack" => cur_engine.attack_ms = v,
            "release" => cur_engine.release_ms = v,
            "retrigger" => cur_engine.retrigger_ms = v,
            _ => {}
        }
        return (false, true);
    }
    if let Some(base_slot) = base_json.get(target) {
        let (Some(base), is_int) = (base_slot.as_f64(), base_slot.is_i64() || base_slot.is_u64())
        else {
            return (false, false);
        };
        let v = base + delta as f64;
        if let Some(slot) = cur_json.get_mut(target) {
            *slot = if is_int {
                serde_json::json!(v.round() as i64)
            } else {
                serde_json::json!(v)
            };
            return (true, false);
        }
    }
    (false, false)
}

fn note_spans(events: &[Event], loop_len: f32) -> Vec<NoteSpan> {
    let mut sorted: Vec<Event> = events.to_vec();
    sorted.sort_by_key(|e| e.pos);
    let mut open: Vec<(u8, u64)> = Vec::new();
    let mut spans = Vec::new();
    for e in &sorted {
        match e.msg {
            EvMsg::On { note, .. } => open.push((note, e.pos)),
            EvMsg::Off { note } => {
                if let Some(idx) = open.iter().rposition(|(n, _)| *n == note) {
                    let (_, start) = open.remove(idx);
                    spans.push(NoteSpan {
                        start: start as f32 / loop_len,
                        end: e.pos as f32 / loop_len,
                        note,
                    });
                }
            }
        }
    }
    // Notes still held at loop end run to the boundary.
    for (note, start) in open {
        spans.push(NoteSpan {
            start: start as f32 / loop_len,
            end: 1.0,
            note,
        });
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(studio: &mut Studio, frames: usize) {
        let mut buf = vec![0.0f32; frames];
        studio.render(&mut buf, 1);
        assert!(buf.iter().all(|s| s.is_finite() && s.abs() <= 1.0001));
    }

    #[test]
    fn pedal_record_then_play_fires_track() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::SetLooperMode(LooperMode::Pedal));
        // Tap: start recording.
        s.handle(Command::Tap);
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 2400); // record ~50 ms
        s.handle(Command::NoteOff { note: 60 });
        drain(&mut s, 2400);
        // Tap: close loop (~100 ms) and play from the top.
        s.handle(Command::Tap);
        assert!(s.loop_len.is_some());
        assert_eq!(s.tracks.len(), 1, "one recorded track");
        assert!(s.recording.is_none(), "pedal mode stops recording after close");

        // Play through the loop: the track's note should re-fire.
        s.live.all_notes_off();
        drain(&mut s, 200); // reach the note onset near pos 0
        assert!(
            s.tracks[0].inst.active_voices() > 0,
            "recorded note should play back"
        );
    }

    #[test]
    fn pedal_tap_records_over_while_playing() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::SetLooperMode(LooperMode::Pedal));
        s.handle(Command::Tap); // record base
        s.handle(Command::NoteOn { note: 48, vel: 1.0 });
        drain(&mut s, 4800);
        s.handle(Command::NoteOff { note: 48 });
        s.handle(Command::Tap); // close loop -> playing (not recording)
        assert_eq!(s.tracks.len(), 1);
        assert!(s.playing && !s.armed && s.recording.is_none());

        // Tap while playing = start recording over into a NEW track.
        s.handle(Command::Tap);
        assert!(s.armed || s.recording.is_some(), "tap should start recording over");
        assert!(s.playing, "the loop keeps playing while overdubbing");
        s.handle(Command::NoteOn { note: 55, vel: 1.0 });
        drain(&mut s, 2400);
        s.handle(Command::NoteOff { note: 55 });

        // Tap again = finish the overdub, keep playing.
        s.handle(Command::Tap);
        assert_eq!(s.tracks.len(), 2, "overdub added a second track");
        assert!(s.playing && !s.armed && s.recording.is_none());
    }

    #[test]
    fn overdub_layers_multiple_tracks() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::SetLooperMode(LooperMode::Overdub));
        s.handle(Command::Tap); // record base
        s.handle(Command::NoteOn { note: 48, vel: 1.0 });
        drain(&mut s, 4800);
        s.handle(Command::NoteOff { note: 48 });
        s.handle(Command::Tap); // close loop + arm track 2
        assert_eq!(s.tracks.len(), 1, "only the base take exists so far");
        assert!(s.armed, "overdub arms the next take (created on first note)");
        s.handle(Command::NoteOn { note: 64, vel: 1.0 });
        drain(&mut s, 4800);
        s.handle(Command::NoteOff { note: 64 });
        s.handle(Command::Tap); // finalize track 2, arm track 3
        assert_eq!(s.tracks.len(), 2, "two layered tracks");
    }

    #[test]
    fn empty_take_is_discarded() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::Tap); // record, but play nothing
        drain(&mut s, 4800);
        s.handle(Command::Tap); // close
        assert_eq!(s.tracks.len(), 0, "a silent take is dropped");
    }

    #[test]
    fn set_track_model_swaps_instrument_live() {
        use crate::models::drum_membrane::DrumMembrane;
        let mut s = Studio::new(48_000.0);
        s.handle(Command::Tap);
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 2400);
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Tap); // close loop
        assert_eq!(s.tracks.len(), 1);
        let before = s.tracks[0].inst.model_id();
        assert_ne!(before, "drum_membrane");

        // Swap the track's instrument to a drum while it loops.
        s.handle(Command::SetTrackModel(0, Box::new(DrumMembrane::default())));
        assert_eq!(s.tracks[0].inst.model_id(), "drum_membrane");
        assert_eq!(s.tracks[0].label, "Drum (2D membrane)");
        drain(&mut s, 4800); // must keep rendering finite / in range
    }

    #[test]
    fn kit_records_and_roundtrips() {
        use crate::project::ZoneData;
        let zone = |name: &str, lo, hi, id: &str| ZoneData {
            name: name.into(),
            lo,
            hi,
            fixed_note: None,
            transpose: 0,
            model_id: id.into(),
            params: serde_json::json!({}),
            engine: EngineParams::default(),
        };
        let mut s = Studio::new(48_000.0);
        s.handle(Command::SetLive(LiveConfig::Kit {
            zones: vec![zone("lo", 0, 59, "drum_membrane"), zone("hi", 60, 127, "pure_string")],
        }));
        s.handle(Command::Tap); // begin first take
        s.handle(Command::NoteOn { note: 40, vel: 1.0 }); // low half → drum zone
        drain(&mut s, 2400);
        s.handle(Command::NoteOff { note: 40 });
        s.handle(Command::Tap); // close loop
        assert_eq!(s.tracks.len(), 1);

        let snap = s.snapshot_loop();
        assert_eq!(snap.tracks[0].model_id, "kit", "kit track serializes as a kit");
        assert_eq!(snap.tracks[0].zones.len(), 2, "both zones kept");

        // Reload at a different rate and confirm it plays back through the kit.
        let mut s2 = Studio::new(44_100.0);
        s2.handle(Command::LoadLoop(snap));
        drain(&mut s2, 200);
        assert!(s2.tracks[0].inst.active_voices() > 0, "reloaded kit plays");
    }

    #[test]
    fn automation_is_captured_and_serialized() {
        use crate::models::drum_membrane::DrumMembrane;
        let mut s = Studio::new(48_000.0);
        s.handle(Command::SetModel(Box::new(DrumMembrane::default())));
        s.handle(Command::Tap); // begin take
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 1000);
        // Tweak a param mid-take → captured as automation on the recording track.
        let mut d = DrumMembrane::default();
        d.damping += 10.0;
        s.handle(Command::SetModel(Box::new(d)));
        s.handle(Command::SetEngine(EngineParams { gain: 1.5, ..EngineParams::default() }));
        drain(&mut s, 1000);
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Tap); // close

        assert_eq!(s.tracks.len(), 1);
        // Captured as deltas from the track's base: damping moved +10, gain to 1.5
        // (base 0.6 → +0.9).
        let damp = s.tracks[0].auto.iter().find(|a| a.target == "damping");
        assert!(damp.is_some_and(|a| (a.value - 10.0).abs() < 1e-3), "damping delta ~+10");
        let gain = s.tracks[0].auto.iter().find(|a| a.target == "eng:gain");
        assert!(gain.is_some_and(|a| (a.value - 0.9).abs() < 1e-3), "gain delta ~+0.9");

        let snap = s.snapshot_loop();
        assert!(snap.tracks[0].automation.iter().any(|a| a.target == "damping"));
    }

    #[test]
    fn automation_replays_and_changes_the_model() {
        use crate::models::drum_membrane::DrumMembrane;
        use crate::models::FtmModel;
        use crate::project::{AutoPoint, LoopData, LoopEvent, LoopTrack};
        let data = LoopData {
            length: 0.05,
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "drum_membrane".into(),
                params: DrumMembrane::default().to_json(),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                zones: Vec::new(),
                // Delta of +100 from the base damping (default 8) → effective 108.
                automation: vec![AutoPoint { t: 0.005, target: "damping".into(), value: 100.0 }],
                events: vec![LoopEvent { t: 0.0, on: true, note: 60, vel: 1.0 }],
            }],
        };
        let base = DrumMembrane::default().to_json().get("damping").and_then(|v| v.as_f64()).unwrap();
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(data));
        // Before the automation point (t=0.005 → 240 samples) it sits at the base.
        drain(&mut s, 100);
        let before = s.tracks[0].inst.parts().1.get("damping").and_then(|v| v.as_f64());
        assert_eq!(before, Some(base), "still at base before the point");
        // After the point, effective = base + delta.
        drain(&mut s, 300);
        let after = s.tracks[0].inst.parts().1.get("damping").and_then(|v| v.as_f64());
        assert_eq!(after, Some(base + 100.0), "automation delta rides on the base");
    }

    #[test]
    fn editing_the_base_shifts_the_automated_value() {
        use crate::models::drum_membrane::DrumMembrane;
        use crate::models::FtmModel;
        use crate::project::{AutoPoint, LoopData, LoopEvent, LoopTrack};
        let data = LoopData {
            length: 0.05,
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "drum_membrane".into(),
                params: DrumMembrane::default().to_json(),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                zones: Vec::new(),
                automation: vec![AutoPoint { t: 0.005, target: "damping".into(), value: 100.0 }],
                events: vec![LoopEvent { t: 0.0, on: true, note: 60, vel: 1.0 }],
            }],
        };
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(data));
        // Re-tune the base live: damping 8 → 20.
        let mut d = DrumMembrane::default();
        d.damping = 20.0;
        s.handle(Command::SetTrackModel(0, Box::new(d)));
        // Past the point, effective = new base (20) + delta (100) = 120.
        drain(&mut s, 400);
        let after = s.tracks[0].inst.parts().1.get("damping").and_then(|v| v.as_f64());
        assert_eq!(after, Some(120.0), "the recorded delta rides on the edited base");
    }

    fn two_note_loop() -> crate::project::LoopData {
        use crate::project::{LoopData, LoopEvent, LoopTrack};
        LoopData {
            length: 1.0,
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "musical_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                zones: Vec::new(),
                automation: Vec::new(),
                events: vec![
                    LoopEvent { t: 0.0, on: true, note: 60, vel: 1.0 },
                    LoopEvent { t: 0.1, on: false, note: 60, vel: 0.0 },
                    LoopEvent { t: 0.5, on: true, note: 62, vel: 1.0 },
                    LoopEvent { t: 0.6, on: false, note: 62, vel: 0.0 },
                ],
            }],
        }
    }

    fn ons_of(s: &Studio) -> Vec<u8> {
        s.snapshot_loop().tracks[0].events.iter().filter(|e| e.on).map(|e| e.note).collect()
    }

    #[test]
    fn region_delete_removes_notes_in_range() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(two_note_loop()));
        s.handle(Command::RegionEdit { track: 0, op: RegionOp::Delete { a: 0.4, b: 0.7 } });
        assert_eq!(ons_of(&s), vec![60], "note 62 (starts at 0.5) deleted");
    }

    #[test]
    fn region_keep_crops_to_range() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(two_note_loop()));
        s.handle(Command::RegionEdit { track: 0, op: RegionOp::Keep { a: 0.4, b: 0.7 } });
        assert_eq!(ons_of(&s), vec![62], "cropped to the [0.4,0.7) window");
    }

    #[test]
    fn region_duplicate_copies_range() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(two_note_loop()));
        s.handle(Command::RegionEdit { track: 0, op: RegionOp::Duplicate { a: 0.0, b: 0.2, dest: 0.5 } });
        let n60 = s.snapshot_loop().tracks[0].events.iter().filter(|e| e.on && e.note == 60).count();
        assert_eq!(n60, 2, "note 60 now appears twice");
    }

    #[test]
    fn region_move_relocates_range() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(two_note_loop()));
        s.handle(Command::RegionEdit { track: 0, op: RegionOp::Move { a: 0.0, b: 0.2, dest: 0.5 } });
        let snap = s.snapshot_loop();
        let n60: Vec<i32> = snap.tracks[0]
            .events
            .iter()
            .filter(|e| e.on && e.note == 60)
            .map(|e| (e.t * 10.0).round() as i32)
            .collect();
        assert_eq!(n60, vec![5], "note 60 moved from 0.0 to 0.5, not duplicated");
    }

    #[test]
    fn region_shift_wraps_within_loop() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(two_note_loop()));
        s.handle(Command::RegionEdit { track: 0, op: RegionOp::Shift { delta: 0.5 } });
        let snap = s.snapshot_loop();
        let mut got: Vec<(u8, i32)> = snap.tracks[0]
            .events
            .iter()
            .filter(|e| e.on)
            .map(|e| (e.note, (e.t * 10.0).round() as i32))
            .collect();
        got.sort();
        assert!(got.contains(&(60, 5)), "note 60 → 0.5s: {got:?}");
        assert!(got.contains(&(62, 0)), "note 62 → wrapped to 0.0s: {got:?}");
    }

    fn held_note_loop(pan: f32, volume: f32) -> crate::project::LoopData {
        use crate::project::{LoopData, LoopEvent, LoopTrack};
        LoopData {
            length: 0.1,
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "musical_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
                muted: false,
                volume,
                pan,
                zones: Vec::new(),
                automation: Vec::new(),
                events: vec![LoopEvent { t: 0.0, on: true, note: 60, vel: 1.0 }],
            }],
        }
    }

    fn stereo_energy(s: &mut Studio, frames: usize) -> (f32, f32) {
        let mut buf = vec![0.0f32; frames * 2];
        s.render(&mut buf, 2);
        let (mut le, mut re) = (0.0f32, 0.0f32);
        for f in buf.chunks(2) {
            le += f[0].abs();
            re += f[1].abs();
        }
        (le, re)
    }

    #[test]
    fn pan_splits_the_stereo_field() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(-1.0, 1.0)));
        let (le, re) = stereo_energy(&mut s, 512);
        assert!(le > re * 5.0 + 1.0, "hard-left: L≫R (L={le}, R={re})");
    }

    #[test]
    fn master_volume_scales_output() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        let (l1, _) = stereo_energy(&mut s, 512);
        assert!(l1 > 0.0, "makes sound at unity");

        let mut s2 = Studio::new(48_000.0);
        s2.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        s2.handle(Command::SetMasterVolume(0.0));
        let (l0, r0) = stereo_energy(&mut s2, 512);
        assert!(l0 + r0 < 1e-3, "silenced at master 0");
    }

    #[test]
    fn loop_snapshot_and_load_roundtrip() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::Tap);
        s.handle(Command::NoteOn { note: 60, vel: 0.9 });
        drain(&mut s, 2400);
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Tap); // close loop
        assert_eq!(s.tracks.len(), 1);

        let snap = s.snapshot_loop();
        assert!(snap.length > 0.0);
        assert_eq!(snap.tracks.len(), 1);
        assert_eq!(snap.tracks[0].events.len(), 2, "note-on + note-off saved");

        // Load into a fresh studio at a *different* sample rate (seconds-based).
        let mut s2 = Studio::new(44_100.0);
        s2.handle(Command::LoadLoop(snap.clone()));
        assert_eq!(s2.tracks.len(), 1);
        assert!(s2.playing);
        let expected_len = (snap.length * 44_100.0).round() as u64;
        assert_eq!(s2.loop_len, Some(expected_len));

        // Playing from the top fires the recorded note.
        drain(&mut s2, 200);
        assert!(s2.tracks[0].inst.active_voices() > 0, "loaded loop should play");
    }

    #[test]
    fn song_plays_through_sections() {
        use crate::project::{LoopData, LoopEvent, LoopTrack};
        let mk = |note: u8| LoopData {
            length: 0.05, // 2400 samples @ 48k
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "musical_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                zones: Vec::new(),
                automation: Vec::new(),
                events: vec![
                    LoopEvent { t: 0.0, on: true, note, vel: 1.0 },
                    LoopEvent { t: 0.02, on: false, note, vel: 0.0 },
                ],
            }],
        };
        let mut s = Studio::new(48_000.0);
        s.handle(Command::SetSong(vec![
            SongSection { loop_data: mk(60), repeats: 2 },
            SongSection { loop_data: mk(67), repeats: 1 },
        ]));
        s.handle(Command::PlaySong);
        assert!(s.song_active);
        assert_eq!(s.song_pos, 0);

        // Two repeats of section 0 (2 * 2400 samples) → advance to section 1.
        drain(&mut s, 2400 * 2 + 50);
        assert_eq!(s.song_pos, 1, "advanced after section 0's repeats");
        assert!(s.song_active);

        // One repeat of section 1 → song ends.
        drain(&mut s, 2400 + 50);
        assert!(!s.song_active, "song ends after the last section");
        assert!(!s.playing);
    }

    #[test]
    fn fixed_bars_recording_auto_closes() {
        use crate::project::TempoGrid;
        let mut s = Studio::new(48_000.0);
        // 120 BPM, 4/4 => beat 24000, bar 96000 samples. One-bar loop.
        s.handle(Command::SetTempo(TempoGrid {
            bars: 1,
            ..TempoGrid::default()
        }));
        s.handle(Command::Tap);
        assert!(s.defining);
        assert_eq!(s.loop_len, Some(96_000), "loop length fixed to one bar");
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 96_000 + 20); // play through the bar → auto-close
        assert!(!s.defining, "fixed take auto-closes at the bar boundary");
        assert_eq!(s.loop_len, Some(96_000));
        assert_eq!(s.tracks.len(), 1);
    }

    #[test]
    fn quantize_snaps_note_ons() {
        use crate::project::TempoGrid;
        let mut s = Studio::new(48_000.0);
        // 120 BPM, 1/16 grid => 6000 samples per step.
        s.handle(Command::SetTempo(TempoGrid {
            quantize: 4,
            ..TempoGrid::default()
        }));
        s.handle(Command::Tap); // free defining
        drain(&mut s, 6100); // just past a grid line
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        let r = s.recording.expect("recording");
        assert_eq!(s.tracks[r].events[0].pos, 6000, "note-on snapped to the 1/16 grid");
    }

    #[test]
    fn reset_clears_everything() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::Tap);
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 2400);
        s.handle(Command::Tap);
        assert!(!s.tracks.is_empty());
        s.handle(Command::Reset);
        assert!(s.tracks.is_empty());
        assert!(s.loop_len.is_none());
        assert!(!s.playing);
    }

    #[test]
    fn mute_and_delete() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::Tap);
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 2400);
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Tap);
        assert_eq!(s.tracks.len(), 1);
        s.handle(Command::ToggleMute(0));
        assert!(s.tracks[0].muted);
        s.handle(Command::DeleteTrack(0));
        assert!(s.tracks.is_empty());
    }
}

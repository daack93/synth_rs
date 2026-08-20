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

use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::instrument::{make_sine_table, EngineParams, Instrument};
use crate::kit::{Kit, Playable};
use crate::models::{default_model, model_from_id, FtmModel};
use crate::project::{AutoPoint, ClipData, LoopData, LoopEvent, LoopTrack, TempoGrid, ZoneData};

const TWO_PI_F64: f64 = std::f64::consts::TAU;
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Linear fade-in/out gain at time `secs` into a loop of length `total`, given
/// fade lengths in seconds (0 = no fade).
#[inline]
fn fade_gain(secs: f32, total: f32, fade_in: f32, fade_out: f32) -> f32 {
    let mut g = 1.0;
    if fade_in > 0.0 && secs < fade_in {
        g *= (secs / fade_in).clamp(0.0, 1.0);
    }
    if fade_out > 0.0 && total > 0.0 && secs > total - fade_out {
        g *= ((total - secs) / fade_out).clamp(0.0, 1.0);
    }
    g
}

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
    /// Whammy / pitch-wheel: global pitch-bend of the live instrument, in
    /// semitones (0 = no bend). Recorded into the take's automation.
    SetBend(f32),
    /// Replace the whole live slot — used to enter/exit kit mode or rebuild a
    /// kit's zones. `SetModel`/`SetEngine` still handle single-instrument edits.
    SetLive(LiveConfig),
    AllNotesOff,
    /// Legacy pedal/overdub selector (kept for tests / a future pedal plugin).
    #[allow(dead_code)]
    SetLooperMode(LooperMode),
    /// Primary transport tap (legacy combined record/play cycle; kept for tests).
    #[allow(dead_code)]
    Tap,
    /// Start / resume playback without recording (Play button).
    Play,
    /// Smart record (Record button): first take, or punch in at the seek cursor.
    /// From a stop it count-ins first (when enabled); pressed again it finishes.
    Record,
    /// Stop playback, keep loops.
    Stop,
    /// Clear all loops.
    Reset,
    /// Pedal mode: record one extra pass into a new track (legacy; kept for tests).
    #[allow(dead_code)]
    ArmOverdub,
    ToggleMute(usize),
    DeleteTrack(usize),
    /// Erase a track's recorded parameter automation.
    ClearTrackAutomation(usize),
    /// Set a track's mixer level (linear) and pan (-1..=1).
    SetTrackMix { track: usize, volume: f32, pan: f32 },
    /// Toggle a track's solo.
    ToggleSolo(usize),
    /// Master output level (linear).
    SetMasterVolume(f32),
    /// Undo / redo the last destructive edit.
    Undo,
    Redo,
    /// Stretch the whole loop in time by `factor` (>1 = longer/slower).
    TimeStretch(f32),
    /// Add a clip placing `track` at `start` for `length` seconds (0 = one loop).
    AddClip { track: usize, start: f32, length: f32 },
    /// Move/resize the clip at `index`: new `start` and `length` (seconds).
    SetClip { index: usize, start: f32, length: f32 },
    /// Trim the clip's front edge: set `start`/`length` and advance the loop
    /// `offset` (seconds) so the content stays anchored (front trim, not move).
    SetClipTrim { index: usize, start: f32, length: f32, offset: f32 },
    /// Copy the clip at `index` to start at `dest` (seconds) as an independent
    /// (forked) clip.
    DuplicateClip { index: usize, dest: f32 },
    /// Remove the clip at `index`.
    RemoveClip { index: usize },
    /// Set the playhead position (seconds).
    Seek(f32),
    /// Set a clip's transpose (semitones) + velocity scale.
    SetClipLayer { index: usize, transpose: i32, vel: f32 },
    // ---- in-clip selection edits (all fork into independent clips) ----
    /// Crop the clip to the timeline range `[a, b)` (seconds).
    CropClip { index: usize, a: f32, b: f32 },
    /// Delete `[a, b)` — trims an edge, or splits into two clips with a gap.
    SplitDeleteClip { index: usize, a: f32, b: f32 },
    /// Reverse the clip's notes within `[a, b)` (seconds).
    ReverseClipRange { index: usize, a: f32, b: f32 },
    /// Make `[a, b)` the clip's loop unit (turns looping on).
    LoopClipRange { index: usize, a: f32, b: f32 },
    /// Toggle a clip's looping; `loop_len` (secs, 0 = whole extent) sets the unit.
    SetClipLoop { index: usize, looping: bool, loop_len: f32 },
    /// Bake a looping clip's playback into a single un-looped raw clip.
    FlattenClip { index: usize },
    /// Set a track's fade-in / fade-out length in seconds.
    SetTrackFades { track: usize, fade_in: f32, fade_out: f32 },
    /// Swap a track's instrument model live (rebuilds its sounding voices).
    SetTrackModel(usize, Box<dyn FtmModel>),
    /// Update a track's engine params live.
    SetTrackEngine(usize, EngineParams),
    /// Replace the current loop + tracks with a saved loop, and play it.
    LoadLoop(LoopData),
    /// Update tempo / grid / metronome settings.
    SetTempo(TempoGrid),
}

/// How the live slot should be configured: a single instrument or a kit.
pub enum LiveConfig {
    Single { model_id: String, params: serde_json::Value, engine: EngineParams },
    Kit { zones: Vec<ZoneData> },
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

/// What a count-in should start once it finishes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pending {
    /// The first (loop-defining) take, recorded from bar 0.
    FirstTake,
    /// A one-pass overdub, punched in at the current playhead (the seek cursor).
    PunchIn,
}

/// One placement of a track on the arrangement timeline. Several clips may
/// A self-contained placement on the arrangement. A clip owns *what plays and
/// when*: its notes, its loop config, and its window on the timeline. Clips
/// never share content — every content edit (crop/delete/reverse/duplicate)
/// forks into an independent clip. The track only provides the *sound*
/// (instrument + mix). `was_active` is runtime only.
///
/// **Content**: the note list is `own_events` if forked, else the track's
/// recording; its natural length is `content_len` (0 ⇒ the track's `period`).
/// **Loop**: when `looping`, the content repeats every `loop_len` samples
/// (0 ⇒ the content length); if `loop_len` exceeds the content the remainder is
/// silence each cycle. When not looping the content plays once, then silence.
/// **Window**: `offset` is the entry point into the content (front-trim /
/// loop phase); `length` is how much timeline the clip occupies.
#[derive(Clone)]
struct Clip {
    track: usize,
    start: u64,
    length: u64,
    /// Entry offset (samples) into the content — front-trim / loop phase.
    offset: u64,
    /// Natural content length (samples); 0 ⇒ use the track `period`.
    content_len: u64,
    /// Loop unit (samples); 0 ⇒ use the resolved content length (seamless).
    loop_len: u64,
    /// Whether the content repeats. Off ⇒ play once, then silence.
    looping: bool,
    was_active: bool,
    /// Per-clip transpose (semitones) and velocity scale.
    transpose: i32,
    vel: f32,
    /// The clip's own notes once forked; `None` ⇒ use the track's recording.
    own_events: Option<Vec<Event>>,
}

impl Clip {
    fn at(track: usize, start: u64, length: u64) -> Clip {
        Clip {
            track,
            start,
            length,
            offset: 0,
            content_len: 0,
            loop_len: 0,
            looping: true,
            was_active: false,
            transpose: 0,
            vel: 1.0,
            own_events: None,
        }
    }
}

/// A track is pure **content** now — its instrument, notes, mix, and its own
/// loop length (`period`). *Where* it plays lives in [`Clip`]s. Firing state is
/// held per-clip in the playlist, not here.
struct Track {
    inst: Playable,
    events: Vec<Event>,
    /// This track's own loop length in samples — it repeats every `period`.
    /// Events are stored relative to it (`[0, period)`).
    period: u64,
    muted: bool,
    /// Mixer level (linear) and stereo pan (-1..=1), plus solo.
    volume: f32,
    pan: f32,
    solo: bool,
    /// Fade-in / fade-out lengths in seconds (0 = none).
    fade_in: f32,
    fade_out: f32,
    name: String,
    label: String,

    // --- Parameter automation (single-instrument tracks only) ---
    /// Recorded parameter moves, sorted by `pos`.
    auto: Vec<AutoEv>,
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
    #[allow(dead_code)] // available for pitch-aware note rendering
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
    pub fade_in: f32,
    pub fade_out: f32,
    /// This track's own loop length in seconds.
    pub period: f32,
}

/// One clip placement, for the arrangement editor (positions in seconds).
#[derive(Clone)]
pub struct ClipView {
    pub track: usize,
    pub start: f32,
    pub length: f32,
    /// Loop-phase offset (seconds) where playback begins — front-trim amount.
    pub offset: f32,
    /// Natural content length (seconds); the loop's played window.
    pub content_len: f32,
    /// Loop unit (seconds) when looping.
    pub loop_len: f32,
    /// Whether the clip repeats.
    pub looping: bool,
    pub transpose: i32,
    pub vel: f32,
    /// True if the clip owns its own (forked) notes.
    pub unique: bool,
}

/// Lock-free scalars + an occasionally-rebuilt track list for the UI.
pub struct SharedView {
    state: AtomicU8,
    mode: AtomicU8,
    pos: AtomicU64,
    loop_len: AtomicU64,
    tracks: Mutex<Vec<TrackView>>,
    /// The clip arrangement, for the arrangement editor.
    arrangement: Mutex<Vec<ClipView>>,
    /// A full serializable snapshot of the current loop (for saving to a project).
    snapshot: Mutex<LoopData>,
    /// Depth of the undo / redo stacks (for enabling the UI buttons).
    undo_depth: AtomicUsize,
    redo_depth: AtomicUsize,
}

impl SharedView {
    fn new() -> Self {
        SharedView {
            state: AtomicU8::new(TransportState::Idle.as_u8()),
            mode: AtomicU8::new(0),
            pos: AtomicU64::new(0),
            loop_len: AtomicU64::new(0),
            tracks: Mutex::new(Vec::new()),
            arrangement: Mutex::new(Vec::new()),
            snapshot: Mutex::new(LoopData::default()),
            undo_depth: AtomicUsize::new(0),
            redo_depth: AtomicUsize::new(0),
        }
    }

    pub fn undo_depth(&self) -> usize {
        self.undo_depth.load(Ordering::Relaxed)
    }
    pub fn redo_depth(&self) -> usize {
        self.redo_depth.load(Ordering::Relaxed)
    }
    pub fn arrangement(&self) -> Vec<ClipView> {
        self.arrangement.lock().map(|a| a.clone()).unwrap_or_default()
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
    /// Clip placements of the tracks on the arrangement timeline.
    arrangement: Vec<Clip>,
    /// The active firing list for the current play mode (rebuilt on change).
    playlist: Vec<Clip>,
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
    /// The period (loop length) for the take currently being armed/recorded, if
    /// known ahead of time. `None` for a free first take (derived on close).
    arm_period: Option<u64>,
    /// Frames elapsed since arming (measures the one-pass window).
    rec_frames: u64,
    /// The global position a take is recorded relative to (0 in Loop mode; the
    /// playhead when punching into the Arrangement). Events store `pos - origin`.
    rec_origin: u64,

    // Tempo / grid / metronome
    tempo: TempoGrid,
    /// Count-in samples remaining before a pending recording starts.
    pre_roll: u64,
    /// What the count-in should start when `pre_roll` reaches 0 (if anything).
    pending: Option<Pending>,
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

    /// Loop snapshots for undo / redo of destructive edits.
    undo_stack: Vec<LoopData>,
    redo_stack: Vec<LoopData>,

    view: Arc<SharedView>,
    /// Structure snapshot waiting to be published to the UI (flushed each render).
    pending_structure: Option<Vec<TrackView>>,
    pending_arrangement: Option<Vec<ClipView>>,
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
            arrangement: Vec::new(),
            playlist: Vec::new(),
            mode: LooperMode::Pedal,
            playing: false,
            loop_len: None,
            pos: 0,
            recording: None,
            armed: false,
            defining: false,
            auto_finalize_at: None,
            arm_period: None,
            rec_frames: 0,
            rec_origin: 0,
            tempo: TempoGrid::default(),
            pre_roll: 0,
            pending: None,
            metro_phase: 0,
            beat_index: 0,
            click_env: 0.0,
            click_phase: 0.0,
            click_freq: 1000.0,
            click_decay: (-1.0 / (0.03 * sample_rate)).exp(),
            master: 1.0,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            view: Arc::new(SharedView::new()),
            pending_structure: None,
            pending_arrangement: None,
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
            Command::SetBend(semitones) => {
                self.live.set_bend(2f32.powf(semitones / 12.0));
                // Record the bend into the take so a dive replays with the loop.
                if let Some(r) = self.recording {
                    let pos = self.pos.saturating_sub(self.rec_origin);
                    self.tracks[r].auto.push(AutoEv {
                        pos,
                        target: "@bend".to_string(),
                        value: semitones,
                    });
                }
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
            Command::Play => self.play(),
            Command::Record => self.record(),
            Command::Stop => self.stop(),
            Command::Reset => self.reset(),
            Command::ArmOverdub => self.request_overdub_one_pass(),
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
            Command::SetTempo(t) => self.tempo = t,
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
            Command::Undo => self.undo(),
            Command::Redo => self.redo(),
            Command::TimeStretch(factor) => self.time_stretch(factor),
            Command::AddClip { track, start, length } => {
                if track < self.tracks.len() {
                    self.push_undo();
                    let sr = self.sr;
                    let period = self.tracks[track].period.max(1);
                    // A clip always has a concrete length (never fills the song).
                    let len = {
                        let l = (length.max(0.0) * sr).round() as u64;
                        if l == 0 { period } else { l }
                    };
                    let want = (start.max(0.0) * sr).round() as u64;
                    let at = self.free_slot(track, usize::MAX, want, len);
                    let mut c = Clip::at(track, at, len);
                    self.fork_content(&mut c); // independent — never a shared ref
                    self.arrangement.push(c);
                    self.after_arrangement_change();
                }
            }
            Command::SetClip { index, start, length } => {
                if index < self.arrangement.len() {
                    self.push_undo();
                    let sr = self.sr;
                    let c = &mut self.arrangement[index];
                    c.start = (start.max(0.0) * sr).round() as u64;
                    c.length = ((length.max(0.0) * sr).round() as u64).max(1);
                    self.after_arrangement_change();
                }
            }
            Command::SetClipTrim { index, start, length, offset } => {
                if index < self.arrangement.len() {
                    self.push_undo();
                    let sr = self.sr;
                    let period = self.tracks.get(self.arrangement[index].track).map(|t| t.period.max(1)).unwrap_or(1);
                    let c = &mut self.arrangement[index];
                    c.start = (start.max(0.0) * sr).round() as u64;
                    c.length = ((length.max(0.0) * sr).round() as u64).max(1);
                    // Reduce the offset mod period (it may arrive negative when the
                    // front edge is dragged left past the loop start).
                    let off = (offset * sr).round() as i64;
                    c.offset = off.rem_euclid(period as i64) as u64;
                    self.after_arrangement_change();
                }
            }
            Command::DuplicateClip { index, dest } => {
                if index < self.arrangement.len() {
                    self.push_undo();
                    let sr = self.sr;
                    let mut c = self.arrangement[index].clone();
                    self.fork_content(&mut c); // an independent copy, never a shared ref
                    let len = c.length.max(1);
                    let want = (dest.max(0.0) * sr).round() as u64;
                    // Drop the copy into the nearest free space so it never lands
                    // on top of the original (or any other clip on this track).
                    c.start = self.free_slot(c.track, usize::MAX, want, len);
                    c.length = len;
                    c.was_active = false;
                    self.arrangement.push(c);
                    self.after_arrangement_change();
                }
            }
            Command::RemoveClip { index } => {
                if index < self.arrangement.len() {
                    self.push_undo();
                    self.arrangement.remove(index);
                    self.after_arrangement_change();
                }
            }
            Command::SetClipLayer { index, transpose, vel } => {
                if index < self.arrangement.len() {
                    self.push_undo();
                    let c = &mut self.arrangement[index];
                    c.transpose = transpose.clamp(-48, 48);
                    c.vel = vel.clamp(0.0, 2.0);
                    self.mark_structure_dirty();
                }
            }
            Command::CropClip { index, a, b } => self.crop_clip(index, a, b),
            Command::SplitDeleteClip { index, a, b } => self.split_delete_clip(index, a, b),
            Command::ReverseClipRange { index, a, b } => self.reverse_clip_range(index, a, b),
            Command::LoopClipRange { index, a, b } => self.loop_clip_range(index, a, b),
            Command::SetClipLoop { index, looping, loop_len } => {
                if index < self.arrangement.len() {
                    self.push_undo();
                    let sr = self.sr;
                    let c = &mut self.arrangement[index];
                    c.looping = looping;
                    if looping {
                        // 0 ⇒ snapshot the current extent as the loop unit.
                        let unit = (loop_len.max(0.0) * sr).round() as u64;
                        c.loop_len = if unit > 0 { unit } else { c.length.max(1) };
                    }
                    self.after_arrangement_change();
                }
            }
            Command::FlattenClip { index } => self.flatten_clip(index),
            Command::Seek(secs) => {
                let p = (secs.max(0.0) * self.sr).round() as u64;
                self.pos = self.loop_len.map(|l| p.min(l.saturating_sub(1))).unwrap_or(0);
                for t in &mut self.tracks {
                    t.inst.all_notes_off();
                }
                self.reset_cursors();
                self.publish_scalars();
            }
            Command::SetTrackFades { track, fade_in, fade_out } => {
                if let Some(t) = self.tracks.get_mut(track) {
                    t.fade_in = fade_in.max(0.0);
                    t.fade_out = fade_out.max(0.0);
                    self.mark_structure_dirty();
                }
            }
            Command::ClearTrackAutomation(i) => {
                if i < self.tracks.len() {
                    self.push_undo();
                }
                if let Some(t) = self.tracks.get_mut(i) {
                    t.auto.clear();
                    t.cur_json = t.base_json.clone();
                    t.cur_engine = t.base_engine.clone();
                    self.mark_structure_dirty();
                }
            }
            Command::DeleteTrack(i) => {
                if i < self.tracks.len() {
                    self.push_undo();
                    self.remove_track(i); // drops the track's clips + reindexes
                    // Fix up the recording index if needed.
                    self.recording = match self.recording {
                        Some(r) if r == i => None,
                        Some(r) if r > i => Some(r - 1),
                        other => other,
                    };
                    self.rebuild_playlist();
                    if self.tracks.is_empty() {
                        self.loop_len = None;
                        self.playing = false;
                        self.pos = 0;
                    } else {
                        self.recompute_song_len();
                    }
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
            // Events are stored relative to the take's origin (the playhead when
            // punching into the arrangement; 0 in Loop mode).
            let pos = match msg {
                EvMsg::On { .. } => self.quantize_pos(self.pos),
                EvMsg::Off { .. } => self.pos,
            }
            .saturating_sub(self.rec_origin);
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
        let pos = self.pos.saturating_sub(self.rec_origin);
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
        let pos = self.pos.saturating_sub(self.rec_origin);
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
        self.rec_origin = 0; // first take defines the song start
        self.metro_phase = 0;
        self.beat_index = 0;
        if self.tempo.bars > 0 {
            let fixed = self.fixed_loop_samples();
            self.loop_len = Some(fixed);
            self.arm_period = Some(fixed);
        } else {
            self.arm_period = None; // free take — period set on close
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

    /// Advance the beat clock one sample and fire a click at each beat boundary.
    fn beat_clock_tick(&mut self) {
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

    /// The metronome during playback — clicks only when the click is enabled.
    fn metro_tick(&mut self) {
        if !self.tempo.metronome {
            return;
        }
        self.beat_clock_tick();
    }

    /// The count-in tick: always clicks, so a count-in is audible even when the
    /// metronome click is turned off for normal playback.
    fn count_in_tick(&mut self) {
        self.beat_clock_tick();
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
                if self.tempo.count_in {
                    self.start_count_in(Pending::FirstTake);
                } else {
                    self.begin_first_take();
                }
            }
            // Closing the first pass: fix loop length, start looping.
            (None, true) => {
                if recording {
                    // A take was actually recorded — its length becomes its period.
                    let len = self.pos.max(1);
                    if let Some(r) = self.recording {
                        self.tracks[r].period = len;
                    }
                    self.disarm_and_finalize();
                    self.recompute_song_len();
                    self.defining = false;
                    self.pos = 0;
                    self.reset_cursors();
                    self.playing = true;
                    if self.mode == LooperMode::Overdub {
                        self.arm_period = Some(self.take_period());
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
                        self.arm_overdub(); // record over into a new track
                    } else {
                        self.playing = true; // resume from a stop
                    }
                }
                LooperMode::Overdub => {
                    self.disarm_and_finalize(); // close current take (if any)
                    self.arm_overdub(); // arm the next
                    self.playing = true;
                }
            },
            _ => {}
        }
        self.mark_structure_dirty();
    }

    /// Play button: start / resume playback, never recording. Ignored during a
    /// count-in; a no-op when there is nothing recorded yet.
    fn play(&mut self) {
        if self.pre_roll > 0 || self.recording.is_some() || self.armed {
            return;
        }
        if self.loop_len.is_some() {
            self.playing = !self.playing; // toggle play / pause
            if self.playing {
                self.reset_cursors();
            }
            self.mark_structure_dirty();
        }
    }

    /// Record button: the record entry point, decoupled from Play.
    /// - Pressed during a count-in → cancel it.
    /// - Pressed while a take is in progress → finish that take.
    /// - No loop yet → the first (loop-defining) take.
    /// - A loop exists and playing → punch in immediately at the playhead.
    /// - A loop exists and stopped → count in (when enabled), then punch in at
    ///   the seek cursor.
    fn record(&mut self) {
        if self.pre_roll > 0 {
            self.pre_roll = 0; // cancel a running count-in
            self.pending = None;
            self.playing = false;
            self.mark_structure_dirty();
            return;
        }
        if self.recording.is_some() || self.armed {
            self.finish_take();
            return;
        }
        match self.loop_len {
            None => {
                if self.tempo.count_in {
                    self.start_count_in(Pending::FirstTake);
                } else {
                    self.begin_first_take();
                }
            }
            Some(_) => {
                if self.playing {
                    self.arm_overdub_one_pass(); // punch in now, no count-in
                } else if self.tempo.count_in {
                    self.start_count_in(Pending::PunchIn);
                } else {
                    self.arm_overdub_one_pass();
                }
            }
        }
        self.mark_structure_dirty();
    }

    /// Close the take in progress. The first (defining) take fixes the loop
    /// length and starts looping; a later overdub just finalizes and keeps
    /// playing.
    fn finish_take(&mut self) {
        if self.defining && self.recording.is_some() {
            let len = self.pos.max(1);
            if let Some(r) = self.recording {
                self.tracks[r].period = len;
            }
            self.disarm_and_finalize();
            self.recompute_song_len();
            self.defining = false;
            self.pos = 0;
            self.reset_cursors();
            self.playing = true;
        } else {
            self.disarm_and_finalize(); // finish the overdub, keep playing
        }
        self.mark_structure_dirty();
    }

    fn stop(&mut self) {
        self.disarm_and_finalize();
        self.playing = false;
        self.defining = false;
        self.pre_roll = 0;
        self.pending = None;
        self.live.all_notes_off();
        for t in &mut self.tracks {
            t.inst.all_notes_off();
        }
        self.mark_structure_dirty();
    }

    fn reset(&mut self) {
        self.tracks.clear();
        self.arrangement.clear();
        self.playlist.clear();
        self.recording = None;
        self.armed = false;
        self.defining = false;
        self.auto_finalize_at = None;
        self.loop_len = None;
        self.pos = 0;
        self.playing = false;
        self.pre_roll = 0;
        self.pending = None;
        self.clear_history();
        self.live.all_notes_off();
        self.mark_structure_dirty();
    }

    /// Load a single saved loop (leaving any song stopped) and play it.
    fn load_loop(&mut self, data: LoopData) {
        self.clear_history();
        self.install_loop(data);
        self.mark_structure_dirty();
    }

    /// Drop undo/redo history (called when the editing context changes).
    fn clear_history(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Stretch the whole loop in time: scale every event + automation position
    /// and the loop length by `factor` (>1 = longer/slower).
    fn time_stretch(&mut self, factor: f32) {
        let Some(len) = self.loop_len else { return };
        if !(factor > 0.0) || (factor - 1.0).abs() < 1e-4 {
            return;
        }
        let _ = len;
        self.push_undo();
        let f = factor as f64;
        // Scale every track's period and its (period-relative) events/automation.
        for t in &mut self.tracks {
            let new_period = ((t.period as f64 * f).round() as u64).max(1);
            let scale = |pos: u64| ((pos as f64 * f).round() as u64).min(new_period - 1);
            for e in &mut t.events {
                e.pos = scale(e.pos);
            }
            for a in &mut t.auto {
                a.pos = scale(a.pos);
            }
            t.period = new_period;
            t.events.sort_by_key(|e| e.pos);
            t.auto.sort_by_key(|a| a.pos);
        }
        // Scale clip placements to match.
        for c in &mut self.arrangement {
            c.start = (c.start as f64 * f).round() as u64;
            c.length = (c.length as f64 * f).round() as u64;
        }
        self.rebuild_playlist();
        self.recompute_song_len();
        self.pos = 0;
        self.reset_cursors();
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
        self.arrangement.clear();
        self.playlist.clear();
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

        let song = ((data.length * self.sr).round() as u64).max(1);
        self.loop_len = Some(song);
        // Migration: old projects stored a single placement per track on the
        // track itself; build one clip per track from it as a fallback.
        let mut migrated: Vec<Clip> = Vec::new();
        for (ti, lt) in data.tracks.into_iter().enumerate() {
            migrated.push(Clip::at(
                ti,
                lt.start.map(|s| (s * self.sr).round() as u64).unwrap_or(0),
                lt.span.map(|s| (s * self.sr).round() as u64).unwrap_or(0),
            ));
            let inst = self.track_playable(&lt);
            let label = inst.label();
            // Each track has its own period (defaults to the whole loop for
            // projects saved before per-track lengths existed).
            let period = lt
                .period
                .map(|s| (s * self.sr).round() as u64)
                .unwrap_or(song)
                .max(1);
            let mut events: Vec<Event> = lt
                .events
                .iter()
                .map(|e| {
                    let pos = ((e.t * self.sr).round() as u64).min(period - 1);
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
                    pos: ((a.t * self.sr).round() as u64).min(period - 1),
                    target: a.target.clone(),
                    value: a.value,
                })
                .collect();
            auto.sort_by_key(|a| a.pos);
            self.tracks.push(Track {
                inst,
                events,
                period,
                muted: lt.muted,
                volume: lt.volume,
                pan: lt.pan,
                solo: false,
                fade_in: lt.fade_in,
                fade_out: lt.fade_out,
                name: lt.name,
                label,
                auto,
                model_id: lt.model_id.clone(),
                base_json: lt.params.clone(),
                base_engine: lt.engine.clone(),
                cur_json: lt.params,
                cur_engine: lt.engine,
            });
        }
        // Use the saved arrangement if present, else the migrated clips.
        self.arrangement = if data.arrangement.is_empty() {
            migrated
        } else {
            let sr = self.sr;
            data.arrangement
                .iter()
                .map(|c| Clip {
                    track: c.track,
                    start: (c.start * sr).round() as u64,
                    length: (c.length * sr).round() as u64,
                    offset: (c.offset * sr).round() as u64,
                    content_len: (c.content_len * sr).round() as u64,
                    loop_len: (c.loop_len * sr).round() as u64,
                    looping: c.looping,
                    was_active: false,
                    transpose: c.transpose,
                    vel: c.vel,
                    own_events: c.own_events.as_ref().map(|evs| {
                        evs.iter()
                            .map(|e| Event {
                                pos: (e.t * sr).round() as u64,
                                msg: if e.on {
                                    EvMsg::On { note: e.note, vel: e.vel }
                                } else {
                                    EvMsg::Off { note: e.note }
                                },
                            })
                            .collect()
                    }),
                })
                .collect()
        };
        // Normalize any legacy fill clips (length 0) to a concrete length equal
        // to the song, preserving their old "loops for the whole song" playback
        // while removing the auto-stretch behaviour going forward.
        for c in &mut self.arrangement {
            if c.length == 0 {
                c.length = song.saturating_sub(c.start).max(1);
            }
        }
        self.rebuild_playlist();
        self.recompute_song_len();
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
                fade_in: t.fade_in,
                fade_out: t.fade_out,
                period: Some(t.period as f32 / sr),
                start: None, // placement lives in the arrangement now
                span: None,
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
        let arrangement = self
            .arrangement
            .iter()
            .map(|c| ClipData {
                track: c.track,
                start: c.start as f32 / sr,
                length: c.length as f32 / sr,
                offset: c.offset as f32 / sr,
                content_len: c.content_len as f32 / sr,
                loop_len: c.loop_len as f32 / sr,
                looping: c.looping,
                transpose: c.transpose,
                vel: c.vel,
                own_events: c.own_events.as_ref().map(|evs| {
                    evs.iter()
                        .map(|e| {
                            let (on, note, vel) = match e.msg {
                                EvMsg::On { note, vel } => (true, note, vel),
                                EvMsg::Off { note } => (false, note, 0.0),
                            };
                            LoopEvent { t: e.pos as f32 / sr, on, note, vel }
                        })
                        .collect()
                }),
            })
            .collect();
        LoopData { length, tracks, arrangement }
    }

    /// Pedal-mode "+ Rec track": record exactly one loop pass into a new track.
    /// Arm an overdub take: choose its period (fixed bars or match the song),
    /// extend the transport so a longer take fits, and arm.
    fn arm_overdub(&mut self) {
        let period = self.take_period().max(1);
        self.arm_period = Some(period);
        // In Arrange mode, punch in at the playhead; in Loop mode, record from 0.
        self.rec_origin = self.pos; // punch in at the playhead
        let cur = self.loop_len.unwrap_or(0);
        self.loop_len = Some(cur.max(self.rec_origin + period));
        self.arm(false);
    }

    fn arm_overdub_one_pass(&mut self) {
        if self.loop_len.is_none() || self.recording.is_some() || self.armed {
            return;
        }
        self.arm_overdub();
        // Auto-close after exactly one period of the take.
        self.auto_finalize_at = self.arm_period;
        self.playing = true;
        self.mark_structure_dirty();
    }

    /// "+ Rec track": punch in a one-pass overdub at the seek cursor. With
    /// count-in enabled this first plays a bar of clicks (the transport parked at
    /// the cursor), then arms and records from there.
    fn request_overdub_one_pass(&mut self) {
        if self.loop_len.is_none() || self.recording.is_some() || self.armed || self.pre_roll > 0 {
            return;
        }
        if self.tempo.count_in {
            self.start_count_in(Pending::PunchIn);
            self.mark_structure_dirty();
        } else {
            self.arm_overdub_one_pass();
        }
    }

    /// Begin a count-in: park the transport and play one bar of clicks, then run
    /// `pending` when it finishes. Clicks sound regardless of the metronome
    /// toggle so the count-in is always audible.
    fn start_count_in(&mut self, pending: Pending) {
        self.pre_roll = self.bar_samples().max(1);
        self.pending = Some(pending);
        self.playing = false;
        self.metro_phase = 0;
        self.beat_index = 0;
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
            period: self.arm_period.unwrap_or(0), // 0 = free take, set on close
            muted: false,
            volume: 1.0,
            pan: 0.0,
            solo: false,
            fade_in: 0.0,
            fade_out: 0.0,
            name: format!("Track {}", idx + 1),
            label,
            auto: Vec::new(),
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
        self.arm_period = None;
        let origin = self.rec_origin;
        let recorded_len = self.pos.saturating_sub(origin);
        if let Some(idx) = self.recording.take() {
            let mut removed = false;
            if let Some(t) = self.tracks.get_mut(idx) {
                t.events.sort_by_key(|e| e.pos);
                t.auto.sort_by_key(|a| a.pos);
                if t.period == 0 {
                    t.period = recorded_len.max(1); // free take with no set period
                }
                if t.events.is_empty() {
                    removed = true;
                }
            }
            if removed {
                self.remove_track(idx);
            } else if !self.arrangement.iter().any(|c| c.track == idx) {
                // Auto-place the new track — at the playhead (arrange punch-in) or 0.
                // Give it a concrete one-loop length so it never fills to the song
                // end (which would grow/overlap as the arrangement changes).
                let period = self.tracks.get(idx).map(|t| t.period.max(1)).unwrap_or(1);
                self.arrangement.push(Clip::at(idx, origin, period));
            }
        }
        self.rec_origin = 0;
        self.rebuild_playlist();
        self.recompute_song_len();
    }

    /// Remove a track and fix up the arrangement (drop its clips, reindex the
    /// clips that referenced later tracks).
    fn remove_track(&mut self, idx: usize) {
        if idx >= self.tracks.len() {
            return;
        }
        self.tracks.remove(idx);
        self.arrangement.retain(|c| c.track != idx);
        for c in &mut self.arrangement {
            if c.track > idx {
                c.track -= 1;
            }
        }
        self.renumber_tracks();
    }

    fn reset_cursors(&mut self) {
        for t in &mut self.tracks {
            reset_track_automation(t);
        }
        for c in &mut self.playlist {
            c.was_active = false;
        }
    }

    /// Refresh runtime state after the arrangement (clips) changed.
    fn after_arrangement_change(&mut self) {
        self.rebuild_playlist();
        self.recompute_song_len();
        self.mark_structure_dirty();
    }

    /// Rebuild the firing list for the current play mode: one clip per track from
    /// 0 in Loop mode, or the arrangement's clips in Arrange mode.
    fn rebuild_playlist(&mut self) {
        // The playlist is always the clip arrangement.
        self.playlist =
            self.arrangement.iter().map(|c| Clip { was_active: false, ..c.clone() }).collect();
    }

    /// The global song length (samples): the furthest clip end in the current
    /// firing list (0 if empty). A fill clip contributes `start + track period`.
    fn song_len(&self) -> u64 {
        self.playlist
            .iter()
            .map(|c| {
                let period = self.tracks.get(c.track).map(|t| t.period.max(1)).unwrap_or(1);
                c.start + if c.length > 0 { c.length } else { period }
            })
            .max()
            .unwrap_or(0)
    }

    /// Find a start (samples) at or after `desired` where a `len`-long clip on
    /// `track` fits without overlapping any existing clip on that track (the
    /// clip at `exclude` is ignored — pass `usize::MAX` for none). Overlaps are
    /// resolved by pushing right to just past each clip in the way.
    fn free_slot(&self, track: usize, exclude: usize, desired: u64, len: u64) -> u64 {
        let mut sibs: Vec<(u64, u64)> = self
            .arrangement
            .iter()
            .enumerate()
            .filter(|(i, c)| c.track == track && *i != exclude)
            .map(|(_, c)| (c.start, c.start + c.length.max(1)))
            .collect();
        sibs.sort_by_key(|s| s.0);
        let mut s = desired;
        let mut changed = true;
        let mut guard = 0;
        while changed && guard <= sibs.len() {
            changed = false;
            for &(a, b) in &sibs {
                if s < b && s + len > a {
                    s = b; // shove past this clip
                    changed = true;
                }
            }
            guard += 1;
        }
        s
    }

    /// Set the transport wrap length to the current song length.
    fn recompute_song_len(&mut self) {
        let len = self.song_len();
        self.loop_len = if len > 0 { Some(len) } else { None };
    }

    /// The period (samples) a newly-armed take should use: a fixed bar-count if
    /// the grid is set, otherwise the current song length (match the longest).
    fn take_period(&self) -> u64 {
        if self.tempo.bars > 0 {
            self.fixed_loop_samples()
        } else {
            self.song_len()
        }
    }

    fn renumber_tracks(&mut self) {
        for (i, t) in self.tracks.iter_mut().enumerate() {
            t.name = format!("Track {}", i + 1);
        }
    }

    /// Snapshot the current loop onto the undo stack before a destructive edit
    /// (and drop the redo history). Bounded so it can't grow without limit.
    fn push_undo(&mut self) {
        const CAP: usize = 64;
        let snap = self.snapshot_loop();
        self.undo_stack.push(snap);
        if self.undo_stack.len() > CAP {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    fn undo(&mut self) {
        if let Some(prev) = self.undo_stack.pop() {
            let cur = self.snapshot_loop();
            self.redo_stack.push(cur);
            self.install_loop(prev);
            self.mark_structure_dirty();
        }
    }

    fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            let cur = self.snapshot_loop();
            self.undo_stack.push(cur);
            self.install_loop(next);
            self.mark_structure_dirty();
        }
    }

    /// Resolve a clip's dimensions (samples): `(period, note_span, content, loop)`.
    /// `note_span` is the modulo base for note positions (the forked span, or the
    /// track period); `content` is the played window; `loop` is the repeat unit.
    fn clip_dims(&self, c: &Clip) -> (u64, u64, u64, u64) {
        let period = self.tracks.get(c.track).map(|t| t.period.max(1)).unwrap_or(1);
        let span = if c.own_events.is_some() { c.content_len.max(1) } else { period };
        let s = if c.content_len > 0 { c.content_len } else { period };
        let l = if c.loop_len > 0 { c.loop_len } else { s };
        (period, span, s, l)
    }

    /// Give a clip its own copy of the notes so it is independent of the track
    /// (no-op if already forked). Operates on a detached clip clone.
    fn fork_content(&self, c: &mut Clip) {
        if c.own_events.is_none() {
            if let Some(t) = self.tracks.get(c.track) {
                c.own_events = Some(t.events.clone());
                if c.content_len == 0 {
                    c.content_len = t.period.max(1);
                }
            }
        }
    }

    /// Clamp a seconds pair to the clip's timeline window, in samples.
    fn clip_range(&self, c: &Clip, a: f32, b: f32) -> (u64, u64) {
        let sr = self.sr;
        let lo = c.start;
        let hi = c.start + c.length;
        let a = ((a.max(0.0) * sr).round() as u64).clamp(lo, hi);
        let b = ((b.max(0.0) * sr).round() as u64).clamp(lo, hi);
        (a.min(b), a.max(b))
    }

    /// The content-note position playing at timeline sample `t` in this clip.
    fn content_pos_at(&self, c: &Clip, t: u64) -> u64 {
        let (_, span, _, l) = self.clip_dims(c);
        let rel = t.saturating_sub(c.start);
        let phase = if c.looping { rel % l.max(1) } else { rel };
        (c.offset + phase) % span.max(1)
    }

    /// Crop the clip to `[a, b)` — one independent clip = that window.
    fn crop_clip(&mut self, index: usize, a: f32, b: f32) {
        if index >= self.arrangement.len() {
            return;
        }
        self.push_undo();
        let c = self.arrangement[index].clone();
        let (a, b) = self.clip_range(&c, a, b);
        if b <= a {
            return;
        }
        let off = self.content_pos_at(&c, a);
        let mut nc = c.clone();
        self.fork_content(&mut nc);
        nc.start = a;
        nc.length = b - a;
        nc.offset = off % nc.content_len.max(1);
        nc.was_active = false;
        self.arrangement[index] = nc;
        self.after_arrangement_change();
    }

    /// Delete `[a, b)`: trims an edge, or splits into two clips with a gap. A
    /// selection covering the whole clip removes it.
    fn split_delete_clip(&mut self, index: usize, a: f32, b: f32) {
        if index >= self.arrangement.len() {
            return;
        }
        self.push_undo();
        let c = self.arrangement[index].clone();
        let (a, b) = self.clip_range(&c, a, b);
        let end = c.start + c.length;
        let mut pieces: Vec<Clip> = Vec::new();
        if a > c.start {
            let mut left = c.clone();
            self.fork_content(&mut left);
            left.length = a - c.start; // start / offset unchanged
            left.was_active = false;
            pieces.push(left);
        }
        if b < end {
            let off = self.content_pos_at(&c, b);
            let mut right = c.clone();
            self.fork_content(&mut right);
            right.start = b;
            right.length = end - b;
            right.offset = off % right.content_len.max(1);
            right.was_active = false;
            pieces.push(right);
        }
        self.arrangement.remove(index);
        self.arrangement.extend(pieces);
        self.after_arrangement_change();
    }

    /// Reverse the clip's notes within `[a, b)` (forks first).
    fn reverse_clip_range(&mut self, index: usize, a: f32, b: f32) {
        if index >= self.arrangement.len() {
            return;
        }
        self.push_undo();
        let c = self.arrangement[index].clone();
        let (a, b) = self.clip_range(&c, a, b);
        let lo = self.content_pos_at(&c, a);
        let hi = self.content_pos_at(&c, b.saturating_sub(1)) + 1;
        let mut nc = c.clone();
        self.fork_content(&mut nc);
        if let Some(ev) = nc.own_events.as_mut() {
            reverse_events_in(ev, lo.min(hi), lo.max(hi));
            ev.sort_by_key(|e| e.pos);
        }
        self.arrangement[index] = nc;
        self.after_arrangement_change();
    }

    /// Make `[a, b)` the clip's loop unit (turns looping on). Non-destructive.
    fn loop_clip_range(&mut self, index: usize, a: f32, b: f32) {
        if index >= self.arrangement.len() {
            return;
        }
        self.push_undo();
        let c = self.arrangement[index].clone();
        let (a, b) = self.clip_range(&c, a, b);
        if b <= a {
            return;
        }
        let off = self.content_pos_at(&c, a);
        let (_, span, _, _) = self.clip_dims(&c);
        let cc = &mut self.arrangement[index];
        cc.offset = off % span.max(1);
        cc.loop_len = b - a;
        cc.looping = true;
        self.after_arrangement_change();
    }

    /// Bake a looping clip's playback across its extent into one raw, un-looped
    /// clip (repeats become concrete notes; looping turns off).
    fn flatten_clip(&mut self, index: usize) {
        if index >= self.arrangement.len() {
            return;
        }
        self.push_undo();
        let c = self.arrangement[index].clone();
        let (_, span, s, l) = self.clip_dims(&c);
        let src: Vec<Event> = match &c.own_events {
            Some(ev) => ev.clone(),
            None => self.tracks.get(c.track).map(|t| t.events.clone()).unwrap_or_default(),
        };
        let length = c.length.max(1);
        let bound = s.min(l);
        let mut flat: Vec<Event> = Vec::new();
        for e in &src {
            // Timeline phase where this content position first plays.
            let phase0 = (e.pos + span - (c.offset % span.max(1))) % span.max(1);
            if phase0 >= bound {
                continue; // in the silent part of the cycle / outside the window
            }
            if c.looping {
                let mut u = phase0;
                while u < length {
                    flat.push(Event { pos: u, msg: e.msg });
                    u += l.max(1);
                }
            } else if phase0 < length {
                flat.push(Event { pos: phase0, msg: e.msg });
            }
        }
        flat.sort_by_key(|e| e.pos);
        let cc = &mut self.arrangement[index];
        cc.own_events = Some(flat);
        cc.content_len = length;
        cc.loop_len = 0;
        cc.looping = false;
        cc.offset = 0;
        self.after_arrangement_change();
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
                // Count-in: clicks only (always audible), no playback/recording.
                self.count_in_tick();
                self.pre_roll -= 1;
                if self.pre_roll == 0 {
                    match self.pending.take() {
                        Some(Pending::FirstTake) => self.begin_first_take(),
                        Some(Pending::PunchIn) => self.arm_overdub_one_pass(),
                        None => {}
                    }
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
            let secs = self.pos as f32 / self.sr;
            let total = self.loop_len.unwrap_or(0) as f32 / self.sr;
            let live_s = self.live.render_frame();
            let (mut l, mut r) = (live_s * FRAC_1_SQRT_2, live_s * FRAC_1_SQRT_2);
            for t in &mut self.tracks {
                let f = t.inst.render_frame();
                let audible = !t.muted && (!any_solo || t.solo);
                if audible {
                    let (lg, rg) = pan_gains(t.pan);
                    let g = t.volume * fade_gain(secs, total, t.fade_in, t.fade_out);
                    l += f * g * lg;
                    r += f * g * rg;
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
        let song = self.loop_len.unwrap_or(0);
        for k in 0..self.playlist.len() {
            let (ti, start, length, offset, content_len, loop_len, looping, transpose, velscale) = {
                let c = &self.playlist[k];
                (c.track, c.start, c.length, c.offset, c.content_len, c.loop_len, c.looping, c.transpose, c.vel)
            };
            if ti >= self.tracks.len() {
                continue;
            }
            let period = self.tracks[ti].period.max(1);
            // Note-space span: forked clips index their own events over
            // `content_len`; linked clips index the track's recording over `period`.
            let span = if self.playlist[k].own_events.is_some() {
                content_len.max(1)
            } else {
                period
            };
            let s = if content_len > 0 { content_len } else { period }; // content window
            let l = if loop_len > 0 { loop_len } else { s }; // loop unit
            let end = start + if length > 0 { length } else { song.saturating_sub(start) };
            let active = pos >= start && pos < end;
            if Some(ti) == recording {
                // Don't play the take being recorded, but track its window state.
                self.playlist[k].was_active = active;
                continue;
            }
            if !active {
                if self.playlist[k].was_active {
                    self.tracks[ti].inst.all_notes_off(); // clean stop at the clip's end
                    self.playlist[k].was_active = false;
                }
                continue;
            }
            let u = pos - start;
            let phase = if looping { u % l } else { u };
            // Silence portion of a cycle (loop unit longer than the content, or a
            // one-shot that has finished): fire nothing; release at the boundary.
            if phase >= s {
                let prev = u.wrapping_sub(1);
                let prev_phase = if looping { prev % l } else { prev };
                if u > 0 && prev_phase < s {
                    self.tracks[ti].inst.all_notes_off(); // clean edge into silence
                }
                self.playlist[k].was_active = true;
                continue;
            }
            // Position within the content (front-trim / loop phase applied).
            let local = (offset + phase) % span;
            // At the top of each cycle rewind automation.
            if phase == 0 {
                reset_track_automation(&mut self.tracks[ti]);
            }
            let to_fire: Vec<(bool, u8, f32)> = {
                let events: &[Event] = match &self.playlist[k].own_events {
                    Some(ev) => ev,
                    None => &self.tracks[ti].events,
                };
                let lo = events.partition_point(|e| e.pos < local);
                events[lo..]
                    .iter()
                    .take_while(|e| e.pos == local)
                    .map(|e| match e.msg {
                        EvMsg::On { note, vel } => {
                            (true, (note as i32 + transpose).clamp(0, 127) as u8, (vel * velscale).clamp(0.0, 4.0))
                        }
                        EvMsg::Off { note } => (false, (note as i32 + transpose).clamp(0, 127) as u8, 0.0),
                    })
                    .collect()
            };
            let inst = &mut self.tracks[ti].inst;
            for (on, note, vel) in to_fire {
                if on {
                    inst.note_on(note, vel);
                } else {
                    inst.note_off(note);
                }
            }
            fire_track_auto(&mut self.tracks[ti], local);
            self.playlist[k].was_active = true;
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
                }
            }
        }
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
        self.view.undo_depth.store(self.undo_stack.len(), Ordering::Relaxed);
        self.view.redo_depth.store(self.redo_stack.len(), Ordering::Relaxed);
    }

    fn mark_structure_dirty(&mut self) {
        let sr = self.sr;
        let views = self
            .tracks
            .iter()
            .map(|t| {
                let (model_id, params, engine, zones) = t.inst.parts();
                let period = t.period.max(1) as f32;
                TrackView {
                    name: t.name.clone(),
                    instrument: t.label.clone(),
                    muted: t.muted,
                    notes: note_spans(&t.events, period),
                    model_id,
                    params,
                    engine,
                    zones,
                    automation: t.auto.len(),
                    volume: t.volume,
                    pan: t.pan,
                    solo: t.solo,
                    fade_in: t.fade_in,
                    fade_out: t.fade_out,
                    period: period / sr,
                }
            })
            .collect();
        let snap = self.snapshot_loop();
        let sr = self.sr;
        let arr: Vec<ClipView> = self
            .arrangement
            .iter()
            .map(|c| {
                let (_, _, s, l) = self.clip_dims(c);
                ClipView {
                    track: c.track,
                    start: c.start as f32 / sr,
                    length: c.length as f32 / sr,
                    offset: c.offset as f32 / sr,
                    content_len: s as f32 / sr,
                    loop_len: l as f32 / sr,
                    looping: c.looping,
                    transpose: c.transpose,
                    vel: c.vel,
                    unique: c.own_events.is_some(),
                }
            })
            .collect();
        self.pending_structure = Some(views);
        self.pending_snapshot = Some(snap);
        self.pending_arrangement = Some(arr);
        self.publish_scalars();
    }

    fn flush_structure(&mut self) {
        if self.pending_structure.is_some() {
            if let Ok(mut guard) = self.view.tracks.try_lock() {
                *guard = self.pending_structure.take().unwrap();
            }
        }
        if self.pending_arrangement.is_some() {
            if let Ok(mut guard) = self.view.arrangement.try_lock() {
                *guard = self.pending_arrangement.take().unwrap();
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
/// Restore a track's automation to its base (model, engine, bend) — called at
/// the top of each of the track's loop periods so automation replays.
fn reset_track_automation(t: &mut Track) {
    if t.auto.is_empty() {
        return;
    }
    t.cur_json = t.base_json.clone();
    t.cur_engine = t.base_engine.clone();
    if let Some(model) = model_from_id(&t.model_id, &t.base_json) {
        t.inst.set_model(model);
    }
    t.inst.set_engine(t.base_engine.clone());
    t.inst.set_bend(1.0); // bend resets each loop; @bend events re-apply
}

/// Fire a track's automation moves landing on local time `local`, rebuilding the
/// model / engine only if something changed.
fn fire_track_auto(t: &mut Track, local: u64) {
    if t.auto.is_empty() {
        return;
    }
    let lo = t.auto.partition_point(|a| a.pos < local);
    let (mut model_dirty, mut engine_dirty) = (false, false);
    let mut j = lo;
    while j < t.auto.len() && t.auto[j].pos == local {
        let (target, value) = (t.auto[j].target.clone(), t.auto[j].value);
        if target == "@bend" {
            t.inst.set_bend(2f32.powf(value / 12.0));
        } else {
            let (m, e) =
                apply_auto(&mut t.cur_json, &mut t.cur_engine, &t.base_json, &t.base_engine, &target, value);
            model_dirty |= m;
            engine_dirty |= e;
        }
        j += 1;
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

/// Reverse the rhythm of note events whose position lies in `[lo, hi)` by
/// mirroring each about the window (kept simple: positions mirror, types stay,
/// which reverses strike timing faithfully for the struck/plucked models).
fn reverse_events_in(events: &mut Vec<Event>, lo: u64, hi: u64) {
    if hi <= lo {
        return;
    }
    for e in events.iter_mut() {
        if e.pos >= lo && e.pos < hi {
            e.pos = lo + (hi - 1 - e.pos);
        }
    }
}

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
        use crate::models::basic_wave::BasicWave;
        let mut s = Studio::new(48_000.0);
        s.handle(Command::SetModel(Box::new(BasicWave::default())));
        s.handle(Command::Tap); // begin take
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 1000);
        // Tweak a param mid-take → captured as automation on the recording track.
        let mut d = BasicWave::default();
        d.decay_time += 10.0;
        s.handle(Command::SetModel(Box::new(d)));
        s.handle(Command::SetEngine(EngineParams { gain: 1.5, ..EngineParams::default() }));
        drain(&mut s, 1000);
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Tap); // close

        assert_eq!(s.tracks.len(), 1);
        // Captured as deltas from the track's base: decay_time moved +10, gain to
        // 1.5 (base 0.6 → +0.9).
        let dt = s.tracks[0].auto.iter().find(|a| a.target == "decay_time");
        assert!(dt.is_some_and(|a| (a.value - 10.0).abs() < 1e-3), "decay_time delta ~+10");
        let gain = s.tracks[0].auto.iter().find(|a| a.target == "eng:gain");
        assert!(gain.is_some_and(|a| (a.value - 0.9).abs() < 1e-3), "gain delta ~+0.9");

        let snap = s.snapshot_loop();
        assert!(snap.tracks[0].automation.iter().any(|a| a.target == "decay_time"));
    }

    #[test]
    fn automation_replays_and_changes_the_model() {
        use crate::models::basic_wave::BasicWave;
        use crate::models::FtmModel;
        use crate::project::{AutoPoint, LoopData, LoopEvent, LoopTrack};
        let data = LoopData {
            length: 0.05,
            arrangement: Vec::new(),
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "basic_wave".into(),
                params: BasicWave::default().to_json(),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                fade_in: 0.0,
                fade_out: 0.0,
                period: None,
                start: None,
                span: None,
                zones: Vec::new(),
                // Delta of +100 from the base decay_time (default 2) → effective 102.
                automation: vec![AutoPoint { t: 0.005, target: "decay_time".into(), value: 100.0 }],
                events: vec![LoopEvent { t: 0.0, on: true, note: 60, vel: 1.0 }],
            }],
        };
        let base = BasicWave::default().to_json().get("decay_time").and_then(|v| v.as_f64()).unwrap();
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(data));
        // Before the automation point (t=0.005 → 240 samples) it sits at the base.
        drain(&mut s, 100);
        let before = s.tracks[0].inst.parts().1.get("decay_time").and_then(|v| v.as_f64());
        assert_eq!(before, Some(base), "still at base before the point");
        // After the point, effective = base + delta.
        drain(&mut s, 300);
        let after = s.tracks[0].inst.parts().1.get("decay_time").and_then(|v| v.as_f64());
        assert_eq!(after, Some(base + 100.0), "automation delta rides on the base");
    }

    #[test]
    fn whammy_bend_records_into_the_take() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::Tap); // start recording
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 500);
        s.handle(Command::SetBend(-2.0)); // dive mid-take
        drain(&mut s, 500);
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Tap); // close
        assert_eq!(s.tracks.len(), 1);
        assert!(
            s.tracks[0].auto.iter().any(|a| a.target == "@bend" && (a.value + 2.0).abs() < 1e-3),
            "whammy dive captured as @bend"
        );
        let snap = s.snapshot_loop();
        assert!(snap.tracks[0].automation.iter().any(|a| a.target == "@bend"));
    }

    #[test]
    fn record_button_is_separate_from_play() {
        use crate::project::TempoGrid;
        let mut s = Studio::new(48_000.0);
        // Record from idle → first take; Record again → finish, now a loop plays.
        s.handle(Command::Record);
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 4_800);
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Record); // finish the take
        assert_eq!(s.tracks.len(), 1);
        assert!(s.loop_len.is_some());
        assert!(s.playing, "finishing the first take starts playback");

        // Count-in enabled, but Record WHILE PLAYING punches in immediately.
        s.handle(Command::SetTempo(TempoGrid {
            bpm: 120.0, beats_per_bar: 4, bars: 0, quantize: 0, metronome: false, count_in: true,
        }));
        s.handle(Command::Record);
        assert_eq!(s.pre_roll, 0, "no count-in while already playing");
        assert!(s.armed || s.recording.is_some(), "punches in immediately");
        s.handle(Command::Record); // finish that overdub

        // Play toggles playback without recording.
        s.handle(Command::Play); // pause
        assert!(!s.playing, "Play toggles to paused");
        s.handle(Command::Play); // resume
        assert!(s.playing);

        // From a STOP, Record does a count-in first.
        s.handle(Command::Stop);
        s.handle(Command::Seek(0.03));
        s.handle(Command::Record);
        assert!(s.pre_roll > 0, "count-in when Record is pressed before playing");
        assert!(s.recording.is_none() && !s.armed, "not recording during the count-in");
    }

    #[test]
    fn count_in_precedes_punch_in_and_clicks_without_metronome() {
        use crate::project::TempoGrid;
        let mut s = Studio::new(48_000.0);
        // Lay down a base loop so an overdub has something to punch into.
        s.handle(Command::Tap);
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 4_800);
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Tap); // close
        assert_eq!(s.tracks.len(), 1);

        // Count-in on, metronome (click) OFF, free length.
        s.handle(Command::SetTempo(TempoGrid {
            bpm: 120.0,
            beats_per_bar: 4,
            bars: 0,
            quantize: 0,
            metronome: false,
            count_in: true,
        }));
        // Seek and request a punch-in (+ Rec track).
        s.handle(Command::Seek(0.05));
        let cursor = s.pos;
        s.handle(Command::ArmOverdub);

        // A count-in is running: not recording yet.
        assert!(s.pre_roll > 0, "count-in started");
        assert!(s.recording.is_none() && !s.armed, "not armed during the count-in");

        // One frame in, a click has fired even though the metronome is off.
        drain(&mut s, 1);
        assert!(s.click_env > 0.0, "count-in is audible without the metronome");

        // One bar at 120 bpm / 4 beats = 2.0s = 96 000 samples; finish it.
        drain(&mut s, 96_000);
        assert!(s.armed || s.recording.is_some(), "recording arms after the count-in");
        assert_eq!(s.rec_origin, cursor, "punches in at the seek cursor");
    }

    #[test]
    fn editing_the_base_shifts_the_automated_value() {
        use crate::models::basic_wave::BasicWave;
        use crate::models::FtmModel;
        use crate::project::{AutoPoint, LoopData, LoopEvent, LoopTrack};
        let data = LoopData {
            length: 0.05,
            arrangement: Vec::new(),
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "basic_wave".into(),
                params: BasicWave::default().to_json(),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                fade_in: 0.0,
                fade_out: 0.0,
                period: None,
                start: None,
                span: None,
                zones: Vec::new(),
                automation: vec![AutoPoint { t: 0.005, target: "decay_time".into(), value: 100.0 }],
                events: vec![LoopEvent { t: 0.0, on: true, note: 60, vel: 1.0 }],
            }],
        };
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(data));
        // Re-tune the base live: decay_time 2 → 20.
        let mut d = BasicWave::default();
        d.decay_time = 20.0;
        s.handle(Command::SetTrackModel(0, Box::new(d)));
        // Past the point, effective = new base (20) + delta (100) = 120.
        drain(&mut s, 400);
        let after = s.tracks[0].inst.parts().1.get("decay_time").and_then(|v| v.as_f64());
        assert_eq!(after, Some(120.0), "the recorded delta rides on the edited base");
    }

    fn two_note_loop() -> crate::project::LoopData {
        use crate::project::{LoopData, LoopEvent, LoopTrack};
        LoopData {
            length: 1.0,
            arrangement: Vec::new(),
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "musical_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                fade_in: 0.0,
                fade_out: 0.0,
                period: None,
                start: None,
                span: None,
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

    #[test]
    fn shorter_track_loops_within_a_longer_song() {
        use crate::project::{LoopData, LoopEvent, LoopTrack};
        let track = |note: u8, period: f32| LoopTrack {
            name: "T".into(),
            model_id: "musical_string".into(),
            params: serde_json::json!({}),
            engine: EngineParams::default(),
            muted: false,
            volume: 1.0,
            pan: 0.0,
            fade_in: 0.0,
            fade_out: 0.0,
            period: Some(period),
            start: None,
            span: None,
            zones: Vec::new(),
            automation: Vec::new(),
            events: vec![
                LoopEvent { t: 0.0, on: true, note, vel: 1.0 },
                LoopEvent { t: 0.02, on: false, note, vel: 0.0 },
            ],
        };
        // Track A repeats every 0.1s; track B (the longest) sets the 0.4s song.
        let data = LoopData { length: 0.4, arrangement: Vec::new(), tracks: vec![track(60, 0.1), track(67, 0.4)] };
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(data));
        assert_eq!(s.loop_len, Some((0.4 * 48_000.0) as u64), "song = longest period");
        assert_eq!(s.tracks[0].period, (0.1 * 48_000.0) as u64, "track A keeps its 0.1s period");

        // Play to ~0.31s: past three of A's re-triggers. A struck a fresh note at
        // 0.3s, so it's audible — under a single-play model it would have decayed
        // to silence long ago.
        drain(&mut s, 14_880);
        assert!(s.tracks[0].inst.active_voices() > 0, "short track re-fired inside the song");
    }

    #[test]
    fn clip_start_offset_delays_playback() {
        use crate::project::{LoopData, LoopEvent, LoopTrack};
        let data = LoopData {
            length: 0.4,
            arrangement: Vec::new(),
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "musical_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
                muted: false,
                volume: 1.0,
                pan: 0.0,
                fade_in: 0.0,
                fade_out: 0.0,
                period: Some(0.2),
                start: Some(0.2), // clip begins at 0.2s
                span: None,
                zones: Vec::new(),
                automation: Vec::new(),
                events: vec![
                    LoopEvent { t: 0.0, on: true, note: 60, vel: 1.0 },
                    LoopEvent { t: 0.02, on: false, note: 60, vel: 0.0 },
                ],
            }],
        };
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(data));
        assert_eq!(s.loop_len, Some((0.4 * 48_000.0) as u64), "song = start + period");
        drain(&mut s, 4_800); // 0.1s — before the clip starts
        assert_eq!(s.tracks[0].inst.active_voices(), 0, "silent before its start");
        drain(&mut s, 5_200); // ~0.208s — the clip has started
        assert!(s.tracks[0].inst.active_voices() > 0, "plays once its start is reached");
    }

    #[test]
    fn clip_ops_edit_the_arrangement() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        assert_eq!(s.arrangement.len(), 1, "one clip auto-placed on load");
        s.handle(Command::AddClip { track: 0, start: 0.4, length: 0.0 });
        assert_eq!(s.arrangement.len(), 2);
        s.handle(Command::SetClip { index: 1, start: 0.6, length: 0.2 });
        assert_eq!(s.arrangement[1].start, (0.6 * 48_000.0) as u64);
        assert_eq!(s.arrangement[1].length, (0.2 * 48_000.0) as u64);
        s.handle(Command::DuplicateClip { index: 0, dest: 0.8 });
        assert_eq!(s.arrangement.len(), 3);
        assert_eq!(s.arrangement[2].start, (0.8 * 48_000.0) as u64);
        s.handle(Command::RemoveClip { index: 0 });
        assert_eq!(s.arrangement.len(), 2);
    }

    #[test]
    fn front_trim_offsets_the_loop_phase_without_moving_content() {
        let mut s = Studio::new(48_000.0);
        // Period 0.1s, a note struck at loop-local 0 (held, retriggers each pass).
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        assert_eq!(s.tracks[0].period, (0.1 * 48_000.0) as u64);

        // Trim the front by half a period: same window start/len, offset 0.05s.
        s.handle(Command::SetClipTrim { index: 0, start: 0.0, length: 0.1, offset: 0.05 });
        assert_eq!(s.arrangement[0].offset, (0.05 * 48_000.0) as u64);

        // Play from 0: the note no longer fires at the clip start — it waits for
        // the loop phase (period − offset = 0.05s) to come around.
        s.handle(Command::Seek(0.0));
        drain(&mut s, 100);
        assert_eq!(s.tracks[0].inst.active_voices(), 0, "front trim delayed the note");
        drain(&mut s, 2_600); // cross 0.05s (2400 samples)
        assert!(s.tracks[0].inst.active_voices() > 0, "note fires once the offset elapses");
    }

    #[test]
    fn duplicate_lands_in_free_space_and_never_resizes_the_original() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        // The auto-placed clip has a concrete length (no "fill to song" clip).
        let orig = s.arrangement[0].clone();
        assert!(orig.length > 0, "clip carries a concrete length");

        // Duplicating onto the original's own spot must shove the copy clear.
        s.handle(Command::DuplicateClip { index: 0, dest: 0.0 });
        assert_eq!(s.arrangement.len(), 2);
        assert_eq!(s.arrangement[0].length, orig.length, "original length untouched");
        let copy = &s.arrangement[1];
        assert!(
            copy.start >= orig.start + orig.length,
            "copy sits after the original with no overlap"
        );
        assert_eq!(copy.length, orig.length, "copy keeps the source length");
    }

    #[test]
    fn added_clip_gets_a_concrete_length_and_avoids_overlap() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        let period = s.tracks[0].period;
        // Ask for a fill (length 0) at bar 0, where the auto clip already sits.
        s.handle(Command::AddClip { track: 0, start: 0.0, length: 0.0 });
        let added = s.arrangement.last().unwrap();
        assert_eq!(added.length, period, "fill request became one concrete loop");
        assert!(added.start >= period, "pushed past the existing clip, no overlap");
    }

    #[test]
    fn arrange_mode_recording_punches_in_at_the_playhead() {
        let mut s = Studio::new(48_000.0);
        // Build a first loop (defines the song; clip at 0).
        s.handle(Command::Tap);
        s.handle(Command::NoteOn { note: 60, vel: 1.0 });
        drain(&mut s, 4_800); // 0.1s
        s.handle(Command::NoteOff { note: 60 });
        s.handle(Command::Tap); // close
        assert_eq!(s.arrangement.len(), 1);
        assert_eq!(s.arrangement[0].start, 0);
        assert_eq!(s.tracks.len(), 1);

        // Seek to 0.05s and punch in a new track (default play mode is Arrange).
        s.handle(Command::Seek(0.05));
        let origin = s.pos;
        assert!(origin > 0);
        s.handle(Command::ArmOverdub);
        s.handle(Command::NoteOn { note: 67, vel: 1.0 });
        drain(&mut s, 200);
        s.handle(Command::Tap); // finalize the overdub

        assert_eq!(s.tracks.len(), 2, "punched-in take made a new track");
        let clip = s.arrangement.iter().find(|c| c.track == 1).expect("clip for the new track");
        assert_eq!(clip.start, origin, "clip placed at the playhead, not bar 0");
        // Its notes are stored relative to the take (first note at local 0).
        assert_eq!(s.tracks[1].events.first().map(|e| e.pos), Some(0));
    }

    #[test]
    fn clip_transpose_stays_linked_but_content_edits_fork() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        assert_eq!(s.arrangement.len(), 1);

        // Transpose + velocity are non-destructive and stay linked to the track.
        s.handle(Command::SetClipLayer { index: 0, transpose: 5, vel: 0.5 });
        assert_eq!(s.arrangement[0].transpose, 5);
        assert!((s.arrangement[0].vel - 0.5).abs() < 1e-6);
        assert!(s.arrangement[0].own_events.is_none(), "transpose stays linked");

        // A content edit (crop) forks the clip into its own independent notes.
        let orig_len = s.arrangement[0].length;
        s.handle(Command::CropClip { index: 0, a: 0.0, b: 0.05 });
        assert!(s.arrangement[0].own_events.is_some(), "crop forks the clip");
        assert!(s.arrangement[0].length < orig_len, "crop shortens the clip");
    }

    #[test]
    fn time_stretch_scales_positions_and_length() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(two_note_loop())); // length 1.0s
        s.handle(Command::TimeStretch(2.0));
        let snap = s.snapshot_loop();
        assert!((snap.length - 2.0).abs() < 0.01, "length doubled: {}", snap.length);
        let t62 = snap.tracks[0]
            .events
            .iter()
            .find(|e| e.on && e.note == 62)
            .map(|e| e.t)
            .unwrap();
        assert!((t62 - 1.0).abs() < 0.01, "note 62 (0.5s) pushed to 1.0s: {t62}");
    }

    #[test]
    fn undo_redo_round_trips_a_clip_edit() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        assert_eq!(s.arrangement.len(), 1);
        // Delete the whole clip via a full-span selection → removed.
        let len = s.arrangement[0].length as f32 / 48_000.0;
        s.handle(Command::SplitDeleteClip { index: 0, a: 0.0, b: len });
        assert_eq!(s.arrangement.len(), 0, "whole-clip delete removes it");
        s.handle(Command::Undo);
        assert_eq!(s.arrangement.len(), 1, "undo restores the clip");
        s.handle(Command::Redo);
        assert_eq!(s.arrangement.len(), 0, "redo re-applies the delete");
    }

    #[test]
    fn split_delete_middle_makes_two_clips_with_a_gap() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        // Give the clip a concrete length of 0.4s so a middle exists.
        s.handle(Command::SetClip { index: 0, start: 0.0, length: 0.4 });
        s.handle(Command::SplitDeleteClip { index: 0, a: 0.1, b: 0.2 });
        assert_eq!(s.arrangement.len(), 2, "split leaves two clips");
        let mut spans: Vec<(u64, u64)> = s.arrangement.iter().map(|c| (c.start, c.start + c.length)).collect();
        spans.sort();
        // Left ends at 0.1s; right starts at 0.2s → a gap in between.
        assert_eq!(spans[0].1, (0.1 * 48_000.0) as u64);
        assert_eq!(spans[1].0, (0.2 * 48_000.0) as u64);
        assert!(s.arrangement.iter().all(|c| c.own_events.is_some()), "both pieces are forked");
    }

    #[test]
    fn loop_range_sets_a_custom_loop_unit() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        s.handle(Command::SetClip { index: 0, start: 0.0, length: 0.4 });
        s.handle(Command::LoopClipRange { index: 0, a: 0.1, b: 0.2 });
        assert!(s.arrangement[0].looping);
        assert_eq!(s.arrangement[0].loop_len, (0.1 * 48_000.0) as u64, "loop unit = selection length");
    }

    #[test]
    fn flatten_bakes_repeats_and_turns_looping_off() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        // Period 0.1s; extend to 0.3s while looping → 3 repeats.
        s.handle(Command::SetClip { index: 0, start: 0.0, length: 0.3 });
        s.handle(Command::FlattenClip { index: 0 });
        let c = &s.arrangement[0];
        assert!(!c.looping, "flatten turns looping off");
        assert!(c.own_events.is_some(), "flatten forks");
        // The single on-note (per 0.1s loop) baked to 3 concrete strikes.
        let ons = c.own_events.as_ref().unwrap().iter().filter(|e| matches!(e.msg, EvMsg::On { .. })).count();
        assert_eq!(ons, 3, "three repeats baked in");
    }

    #[test]
    fn duplicate_forks_into_independent_notes() {
        let mut s = Studio::new(48_000.0);
        s.handle(Command::LoadLoop(held_note_loop(0.0, 1.0)));
        s.handle(Command::DuplicateClip { index: 0, dest: 0.5 });
        assert_eq!(s.arrangement.len(), 2);
        assert!(s.arrangement[1].own_events.is_some(), "the copy owns its notes");
    }

    fn held_note_loop(pan: f32, volume: f32) -> crate::project::LoopData {
        use crate::project::{LoopData, LoopEvent, LoopTrack};
        LoopData {
            length: 0.1,
            arrangement: Vec::new(),
            tracks: vec![LoopTrack {
                name: "T".into(),
                model_id: "musical_string".into(),
                params: serde_json::json!({}),
                engine: EngineParams::default(),
                muted: false,
                volume,
                pan,
                fade_in: 0.0,
                fade_out: 0.0,
                period: None,
                start: None,
                span: None,
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

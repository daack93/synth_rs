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
use crate::models::{default_model, model_from_id, FtmModel};
use crate::project::{LoopData, LoopEvent, LoopTrack, TempoGrid};

const TWO_PI_F64: f64 = std::f64::consts::TAU;

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

struct Track {
    inst: Instrument,
    events: Vec<Event>,
    cursor: usize,
    muted: bool,
    name: String,
    label: String,
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
    /// can open it in the editor.
    pub model_id: String,
    pub params: serde_json::Value,
    pub engine: EngineParams,
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
    live: Instrument,
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
            live: Instrument::new(sample_rate, sine),
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
            Command::SetModel(m) => self.live.set_model(m),
            Command::SetEngine(e) => self.live.set_engine(e),
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
                    t.inst.set_model(m);
                }
                if relabel {
                    self.mark_structure_dirty(); // instrument name changed
                }
            }
            Command::SetTrackEngine(i, e) => {
                if let Some(t) = self.tracks.get_mut(i) {
                    t.inst.set_engine(e);
                }
            }
            Command::LoadLoop(data) => self.load_loop(data),
            Command::SetSong(sections) => self.song = sections,
            Command::PlaySong => self.play_song(),
            Command::SetTempo(t) => self.tempo = t,
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
            let model = model_from_id(&lt.model_id, &lt.params).unwrap_or_else(default_model);
            let inst = Instrument::with_config(self.sr, self.sine.clone(), model, lt.engine);
            let label = inst.model_name().to_string();
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
            self.tracks.push(Track {
                inst,
                events,
                cursor: 0,
                muted: lt.muted,
                name: lt.name,
                label,
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
            .map(|t| LoopTrack {
                name: t.name.clone(),
                model_id: t.inst.model_id().to_string(),
                params: t.inst.model_json(),
                engine: t.inst.engine_params(),
                muted: t.muted,
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
        let label = inst.model_name().to_string();
        self.tracks.push(Track {
            inst,
            events: Vec::new(),
            cursor: 0,
            muted: false,
            name: format!("Track {}", idx + 1),
            label,
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
                let pos = self.pos;
                t.cursor = t.events.partition_point(|e| e.pos < pos);
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
        }
    }

    fn renumber_tracks(&mut self) {
        for (i, t) in self.tracks.iter_mut().enumerate() {
            t.name = format!("Track {}", i + 1);
        }
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

            let mut s = self.live.render_frame();
            for t in &mut self.tracks {
                let f = t.inst.render_frame();
                if !t.muted {
                    s += f;
                }
            }
            s += self.click_sample();
            let sample = s.clamp(-1.0, 1.0);
            for ch in frame.iter_mut() {
                *ch = sample;
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
            .map(|t| TrackView {
                name: t.name.clone(),
                instrument: t.label.clone(),
                muted: t.muted,
                notes: note_spans(&t.events, len),
                model_id: t.inst.model_id().to_string(),
                params: t.inst.model_json(),
                engine: t.inst.engine_params(),
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

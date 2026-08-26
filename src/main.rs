//! Synth.RS Studio — a MIDI synth / production-studio framework. Pluggable
//! synthesizer models (many use physical modeling, but the framework is
//! method-agnostic), played from an on-screen piano, the computer keyboard, or a
//! MIDI controller, recorded and arranged into songs.

mod arrangement_ui;
mod audio;
mod graph;
mod graph_editor;
mod export;
mod instrument;
mod kit;
mod midi;
mod models;
mod presets;
mod project;
mod studio;
mod wav;

use std::collections::HashMap;
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;

use eframe::egui;

use arrangement_ui::DragKind;
use audio::AudioEngine;
use instrument::EngineParams;
use midi::MidiInputHandle;
use models::FtmModel;
use presets::Preset;
use project::{NamedSong, Project, TempoGrid, ZoneData};
use studio::{
    Command, LiveConfig, SharedView, TrackView,
    TransportState,
};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([980.0, 720.0])
            .with_title("Synth.RS Studio"),
        ..Default::default()
    };
    eframe::run_native(
        "Synth.RS Studio",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}

/// Which instrument the right-hand parameter panel is editing.
#[derive(Clone, Copy, PartialEq)]
enum Target {
    /// The live instrument you play from the keyboard.
    Live,
    /// A loop track's instrument.
    Track(usize),
}

/// Editable copy of a track's instrument (the track's real instrument lives on
/// the audio thread; edits are pushed to it via commands).
struct TrackEdit {
    idx: usize,
    model: Box<dyn FtmModel>,
    engine: EngineParams,
}

/// Grid that clip drags snap to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Snap {
    Bar,
    Quarter,
    Eighth,
    Sixteenth,
    Free,
}

impl Snap {
    fn label(self) -> &'static str {
        match self {
            Snap::Bar => "Bar",
            Snap::Quarter => "1/4",
            Snap::Eighth => "1/8",
            Snap::Sixteenth => "1/16",
            Snap::Free => "Free",
        }
    }
    /// Snap length in seconds (None = free / no snap), given bar and beat lengths.
    fn secs(self, bar: f32, beat: f32) -> Option<f32> {
        match self {
            Snap::Bar => Some(bar),
            Snap::Quarter => Some(beat),
            Snap::Eighth => Some(beat * 0.5),
            Snap::Sixteenth => Some(beat * 0.25),
            Snap::Free => None,
        }
    }
}

struct App {
    tx: Sender<Command>,
    _audio: Option<AudioEngine>,
    audio_err: Option<String>,
    /// Which instrument the parameter panel edits.
    edit_target: Target,
    /// Working copy of the track instrument being edited (when `edit_target` is a track).
    track_edit: Option<TrackEdit>,
    /// Transport / track state published by the studio (audio thread).
    view: Option<Arc<SharedView>>,
    sample_rate: f32,

    /// Available synthesis models (plugins); each holds its own parameters.
    models: Vec<Box<dyn FtmModel>>,
    /// Index of the active model.
    selected: usize,
    /// Engine-wide parameters (gain, envelope, retrigger).
    engine: EngineParams,
    /// Whether the live keyboard is a multi-zone kit (split/mapped across several
    /// instruments) rather than one instrument.
    live_is_kit: bool,
    /// Zones for the live kit (each an instrument mapped to a key range).
    kit_zones: Vec<ZoneData>,

    // Preset library
    /// Presets found in the folder (refreshed on save/load/delete).
    preset_list: Vec<Preset>,
    /// Name field for saving the current sound.
    preset_name: String,
    /// Last preset action result, shown in the UI.
    preset_status: String,

    // Project
    /// The in-memory project (the current loop + tempo).
    project: Project,
    /// Project name input.
    project_name: String,
    /// Saved project names on disk.
    project_list: Vec<String>,
    /// Last project action result.
    project_status: String,
    /// True while the "Clear project?" confirmation modal is open.
    confirm_clear: bool,

    /// Master output level (linear).
    master_volume: f32,
    /// Whether the song repeats at the end (default off = play once).
    repeat: bool,
    /// Snap grid for clip drags (default quarter note).
    snap: Snap,

    // Whammy (pitch-bend lever), semitones + configurable range.
    whammy: f32,
    whammy_down: f32,
    whammy_up: f32,

    // Clip editing
    /// In-clip time selection: (clip index, start secs, end secs) — dragged across
    /// a clip body; drives Crop / Delete / Loop / Reverse.
    clip_sel: Option<(usize, f32, f32)>,
    /// The track whose controls the contextual panel shows.
    sel_track: Option<usize>,
    /// Selected clip indices in the arrangement editor.
    sel_clips: Vec<usize>,
    /// Active clip drag: (clip index, kind, a, b) — a/b meaning depends on kind.
    arr_drag: Option<(usize, DragKind, f32, f32)>,
    /// Offset (secs) from the dragged clip's start to where it was grabbed.
    arr_grab: f32,
    /// True while scrubbing the wrapped seek strip inside the arrangement.
    arr_seeking: bool,

    // Export
    export_name: String,
    export_sr: u32,
    export_bit: wav::BitDepth,
    export_hi_res: bool,
    export_status: String,

    // Keyboard state
    base_midi: i32,
    /// On-screen (mouse) currently-held note.
    mouse_note: Option<u8>,
    /// Computer-keyboard held notes: physical key -> midi note.
    held_keys: HashMap<egui::Key, u8>,

    // MIDI
    midi_ports: Vec<String>,
    midi_sel: Option<usize>,
    _midi: Option<MidiInputHandle>,
    /// Novation Launchkey DAW-mode control surface (kept alive; drop = exit DAW).
    _launchkey: Option<midi::LaunchkeyHandle>,
    encoders: midi::EncoderMonitor,
    /// Last transport state pushed to the Launchkey LEDs (to detect changes).
    last_transport: TransportState,
    /// Shared "last note played" for MIDI-learn on pitch fields.
    note_monitor: midi::NoteMonitor,
    midi_status: String,

    /// Visual graph-editor state (node positions, selection, audition).
    ge: graph_editor::GeState,
}

impl App {
    fn new() -> Self {
        let (tx, rx) = channel::<Command>();
        let (audio, audio_err) = match AudioEngine::start(rx) {
            Ok(a) => (Some(a), None),
            Err(e) => (None, Some(e)),
        };
        let view = audio.as_ref().map(|a| a.view.clone());
        let sample_rate = audio.as_ref().map(|a| a.sample_rate).unwrap_or(48_000.0);

        let models = models::registry();
        let selected = 0;
        let engine = EngineParams::default();
        // Prime the audio thread with the initial model + engine + tempo.
        let _ = tx.send(Command::SetModel(models[selected].box_clone()));
        let _ = tx.send(Command::SetEngine(engine.clone()));
        let _ = tx.send(Command::SetTempo(TempoGrid::default()));

        let midi_ports = midi::list_ports();
        // First run seeds the folder with the factory instrument kit.
        let preset_list = presets::load_library();

        App {
            tx,
            _audio: audio,
            audio_err,
            edit_target: Target::Live,
            track_edit: None,
            view,
            sample_rate,
            models,
            selected,
            engine,
            live_is_kit: false,
            kit_zones: Vec::new(),
            preset_list,
            preset_name: String::new(),
            preset_status: String::new(),
            project: Project {
                name: "Untitled".to_string(),
                loops: Vec::new(),
                tempo: TempoGrid::default(),
            },
            project_name: "Untitled".to_string(),
            project_list: project::list(),
            project_status: String::new(),
            confirm_clear: false,
            master_volume: 1.0,
            repeat: false,
            snap: Snap::Quarter,
            whammy: 0.0,
            whammy_down: 12.0,
            whammy_up: 2.0,
            clip_sel: None,
            sel_track: None,
            sel_clips: Vec::new(),
            arr_drag: None,
            arr_grab: 0.0,
            arr_seeking: false,
            export_name: "take".to_string(),
            export_sr: 48_000,
            export_bit: wav::BitDepth::Int16,
            export_hi_res: false,
            export_status: String::new(),
            base_midi: 60, // C4
            mouse_note: None,
            held_keys: HashMap::new(),
            midi_ports,
            midi_sel: None,
            _midi: None,
            _launchkey: None,
            encoders: midi::EncoderMonitor::default(),
            last_transport: TransportState::Recording,
            note_monitor: midi::NoteMonitor::default(),
            midi_status: "not connected".to_string(),
            ge: graph_editor::GeState::default(),
        }
    }

    /// Send the active model's current parameters to the audio thread.
    fn push_model(&self) {
        let _ = self.tx.send(Command::SetModel(self.models[self.selected].box_clone()));
    }

    /// Save the currently-edited instrument (live or a track) as a preset.
    fn save_preset(&mut self) {
        let name = self.preset_name.trim().to_string();
        if name.is_empty() {
            self.preset_status = "Enter a name first.".into();
            return;
        }
        let preset = match self.edit_target {
            // The live keyboard is a kit → save the whole kit (all its zones).
            Target::Live if self.live_is_kit => Preset::capture_kit(&name, self.kit_zones.clone()),
            Target::Track(i) => match &self.track_edit {
                Some(te) if te.idx == i => Preset::capture(&name, te.model.as_ref(), &te.engine),
                _ => Preset::capture(&name, self.models[self.selected].as_ref(), &self.engine),
            },
            Target::Live => Preset::capture(&name, self.models[self.selected].as_ref(), &self.engine),
        };
        match presets::save(&preset) {
            Ok(path) => {
                self.preset_status = format!("Saved “{name}” → {}", path.display());
                self.preset_list = presets::list();
            }
            Err(e) => self.preset_status = format!("Save failed: {e}"),
        }
    }

    /// Load a preset onto the current target (the live instrument or a track).
    fn apply_preset(&mut self, preset: &Preset) {
        // A kit preset always loads onto the live keyboard. Kit tracks
        // aren't editable in place yet — re-record from the live kit to place one.
        if preset.is_kit() {
            self.live_is_kit = true;
            self.kit_zones = preset.zones.clone();
            self.edit_target = Target::Live;
            self.send_live_kit();
            self.preset_name = preset.name.clone();
            self.preset_status =
                format!("Loaded kit “{}” ({} zones) onto the keyboard.", preset.name, preset.zones.len());
            return;
        }

        let Some(model) = preset.build_model() else {
            self.preset_status =
                format!("Can't load “{}”: unknown model “{}”.", preset.name, preset.model_id);
            return;
        };

        if let Target::Track(i) = self.edit_target {
            // Swap this track's instrument live.
            let _ = self.tx.send(Command::SetTrackModel(i, model.box_clone()));
            let _ = self.tx.send(Command::SetTrackEngine(i, preset.engine.clone()));
            self.track_edit = Some(TrackEdit {
                idx: i,
                model,
                engine: preset.engine.clone(),
            });
            self.preset_status = format!("Loaded “{}” onto track {}.", preset.name, i + 1);
            return;
        }

        // Live: replace the matching registry slot so the editor shows these params.
        match self.models.iter().position(|m| m.id() == preset.model_id) {
            Some(idx) => {
                self.models[idx] = model;
                self.selected = idx;
            }
            None => {
                self.models.push(model);
                self.selected = self.models.len() - 1;
            }
        }
        self.engine = preset.engine.clone();
        self.preset_name = preset.name.clone();
        self.push_model();
        let _ = self.tx.send(Command::SetEngine(self.engine.clone()));
        self.preset_status = format!("Loaded “{}”.", preset.name);
    }

    fn delete_preset(&mut self, name: &str) {
        match presets::delete(name) {
            Ok(true) => self.preset_status = format!("Deleted “{name}”."),
            Ok(false) => self.preset_status = format!("“{name}” not found."),
            Err(e) => self.preset_status = format!("Delete failed: {e}"),
        }
        self.preset_list = presets::list();
    }

    /// Save the current studio loop (tracks + arrangement) as the project.
    fn save_project(&mut self) {
        let name = self.project_name.trim();
        self.project.name = if name.is_empty() { "Untitled".into() } else { name.to_string() };
        self.project_name = self.project.name.clone();
        // Capture whatever is currently in the studio as the project's content.
        if let Some(v) = &self.view {
            let data = v.snapshot();
            self.project.loops = if data.is_empty() {
                Vec::new()
            } else {
                vec![NamedSong { name: self.project.name.clone(), data }]
            };
        }
        match project::save(&self.project) {
            Ok(path) => {
                self.project_status = format!("Saved project → {}", path.display());
                self.project_list = project::list();
            }
            Err(e) => self.project_status = format!("Save failed: {e}"),
        }
    }

    fn load_project(&mut self, name: &str) {
        match project::load_named(name) {
            Ok(p) => {
                self.project_name = p.name.clone();
                let _ = self.tx.send(Command::SetTempo(p.tempo));
                // Load its loop straight into the studio.
                if let Some(nl) = p.loops.first() {
                    let _ = self.tx.send(Command::LoadSong(nl.data.clone()));
                }
                self.edit_target = Target::Live;
                self.track_edit = None;
                self.project_status = format!("Loaded “{}”.", p.name);
                self.project = p;
            }
            Err(e) => self.project_status = format!("Load failed: {e}"),
        }
    }

    /// Clear everything: wipe the studio (tracks + arrangement) and start a new,
    /// empty project. Behind a confirmation — there is no undo for this.
    fn clear_project(&mut self) {
        let _ = self.tx.send(Command::Reset);
        self.project = Project {
            name: "Untitled".into(),
            loops: Vec::new(),
            tempo: self.project.tempo,
        };
        self.project_name = "Untitled".into();
        self.edit_target = Target::Live;
        self.track_edit = None;
        self.project_status = "Cleared — new project.".into();
    }

    /// The "Clear project?" confirmation modal: a dimmed, click-blocking backdrop
    /// plus a centered dialog. Only clearing goes through here (there's no undo).
    fn clear_confirm_modal(&mut self, ctx: &egui::Context) {
        if !self.confirm_clear {
            return;
        }
        egui::Area::new(egui::Id::new("clear_backdrop"))
            .order(egui::Order::Foreground)
            .fixed_pos(egui::Pos2::ZERO)
            .interactable(true)
            .show(ctx, |ui| {
                let screen = ctx.screen_rect();
                ui.allocate_response(screen.size(), egui::Sense::click());
                ui.painter().rect_filled(screen, 0.0, egui::Color32::from_black_alpha(160));
            });
        egui::Window::new("Clear project?")
            .order(egui::Order::Tooltip)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("This removes all tracks and the arrangement and starts a new project.");
                ui.label(egui::RichText::new("This can't be undone.").weak());
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.confirm_clear = false;
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("Clear everything").color(egui::Color32::WHITE),
                            )
                            .fill(egui::Color32::from_rgb(200, 60, 60)),
                        )
                        .clicked()
                    {
                        self.clear_project();
                        self.confirm_clear = false;
                    }
                });
            });
    }

    fn connect_midi(&mut self, index: usize) {
        // Selecting the Launchkey's DAW port engages the full DAW-mode control
        // surface (transport, encoders, pads, LED + screen feedback); it finds
        // the matching keys port itself. Any other port is a plain MIDI input.
        let name = self.midi_ports.get(index).cloned().unwrap_or_default();
        let n = name.to_lowercase();
        let is_launchkey_daw =
            (n.contains("launchkey") || n.contains("launch key")) && n.contains("daw");
        if is_launchkey_daw {
            self._midi = None;
            match midi::connect_launchkey(self.tx.clone(), self.note_monitor.clone(), self.encoders.clone()) {
                Ok(h) => {
                    self.midi_status = format!("Launchkey DAW mode: {}", h.daw_port);
                    self._launchkey = Some(h);
                    self.midi_sel = Some(index);
                }
                Err(e) => {
                    self.midi_status = format!("error: {e}");
                    self._launchkey = None;
                    self.midi_sel = None;
                }
            }
            return;
        }
        // A plain input — and leave DAW mode if we were in it (drop = standalone).
        self._launchkey = None;
        match midi::connect(index, self.tx.clone(), self.note_monitor.clone()) {
            Ok(h) => {
                self.midi_status = format!("connected: {}", h.port_name);
                self._midi = Some(h);
                self.midi_sel = Some(index);
            }
            Err(e) => {
                self.midi_status = format!("error: {e}");
                self._midi = None;
                self.midi_sel = None;
            }
        }
    }

    fn note_on(&mut self, note: u8, vel: f32) {
        self.note_monitor.record(note); // feed MIDI-learn (keyboard / piano)
        let _ = self.tx.send(Command::NoteOn { note, vel });
    }
    fn note_off(&mut self, note: u8) {
        let _ = self.tx.send(Command::NoteOff { note });
    }

    /// Translate physical-key events into notes.
    fn handle_computer_keyboard(&mut self, ctx: &egui::Context) {
        // While a text field (e.g. the preset name) has focus, keystrokes are
        // for typing — don't play notes or drive the looper. Release anything
        // currently held so notes don't stick when focus is taken.
        if ctx.wants_keyboard_input() {
            let stuck: Vec<u8> = self.held_keys.values().copied().collect();
            self.held_keys.clear();
            for note in stuck {
                self.note_off(note);
            }
            return;
        }

        let events = ctx.input(|i| i.events.clone());
        for ev in events {
            if let egui::Event::Key {
                key,
                pressed,
                repeat,
                modifiers,
                ..
            } = ev
            {
                if repeat {
                    continue;
                }
                // Undo / redo: ⌘Z (Ctrl+Z), ⌘⇧Z (Ctrl+Shift+Z).
                if pressed && modifiers.command && key == egui::Key::Z {
                    let cmd = if modifiers.shift { Command::Redo } else { Command::Undo };
                    let _ = self.tx.send(cmd);
                    continue;
                }
                // Octave shift with Z / X.
                if pressed && key == egui::Key::Z {
                    self.base_midi = (self.base_midi - 12).max(0);
                    continue;
                }
                if pressed && key == egui::Key::X {
                    self.base_midi = (self.base_midi + 12).min(108);
                    continue;
                }
                if let Some(semi) = key_to_semitone(key) {
                    if pressed {
                        if !self.held_keys.contains_key(&key) {
                            let note = (self.base_midi + semi).clamp(0, 127) as u8;
                            self.held_keys.insert(key, note);
                            self.note_on(note, 0.85);
                        }
                    } else if let Some(note) = self.held_keys.remove(&key) {
                        self.note_off(note);
                    }
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keep animating so key/piano input is polled continuously.
        ctx.request_repaint();

        // Publish the last note played for MIDI-learn pitch fields to read.
        let latest = self.note_monitor.latest();
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("note_monitor"), latest));

        // Apply any Launchkey encoder moves to the live engine params, in sync
        // with the on-screen sliders. Encoders 1-4 → gain / attack / release /
        // retrigger; 5-8 are unmapped for now.
        let enc = self.encoders.take_changes();
        let mut eng_changed = false;
        for (i, v) in enc.iter().enumerate() {
            if let Some(v) = v {
                let n = *v as f32 / 127.0;
                match i {
                    0 => self.engine.gain = n * 2.0,          // 0 .. 2.0
                    1 => self.engine.attack_ms = n * 1000.0,  // 0 .. 1000 ms
                    2 => self.engine.release_ms = (n * 2000.0).max(1.0),
                    3 => self.engine.retrigger_ms = n * 500.0,
                    _ => continue,
                }
                eng_changed = true;
            }
        }
        if eng_changed {
            let _ = self.tx.send(Command::SetEngine(self.engine.clone()));
        }

        // Reflect the transport state on the Launchkey's Play/Record LEDs.
        let state = self
            .view
            .as_ref()
            .map(|v| v.state())
            .unwrap_or(TransportState::Idle);
        if state != self.last_transport {
            self.last_transport = state;
            if let Some(lk) = self._launchkey.as_mut() {
                let (play, rec) = match state {
                    TransportState::Playing => (21, 0),   // green play
                    TransportState::Recording => (21, 5), // green play + red rec
                    _ => (0, 0),
                };
                lk.set_button_led(0x73, play);
                lk.set_button_led(0x75, rec);
                lk.set_button_led(0x74, 3); // stop: dim white
            }
        }

        self.handle_computer_keyboard(ctx);

        // Very top: project management (save / load / clear), above everything.
        egui::TopBottomPanel::top("project").show(ctx, |ui| {
            self.project_bar(ui);
        });

        // Right: instrument configuration — presets on top, then the plugin.
        egui::SidePanel::right("instrument")
            .resizable(false)
            .min_width(340.0)
            .show(ctx, |ui| {
                self.presets_bar(ui);
                ui.separator();
                self.params_panel(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                if let Some(err) = &self.audio_err {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 90, 90),
                        format!("Audio unavailable: {err}"),
                    );
                }

                // Top: MIDI inputs — the virtual keyboard + its controls (and,
                // later, attached MIDI controllers).
                self.midi_inputs(ui);

                ui.add_space(10.0);

                // Below: recording + editing — transport & tempo, then the
                // arrangement / track / clip editor.
                self.transport_bar(ui); // includes the tempo/grid row
                ui.add_space(8.0);
                self.tracks_panel(ui);

                ui.add_space(10.0);
                self.export_ui(ui);
            });
        });

        self.clear_confirm_modal(ctx);
        self.graph_editor_window(ctx);
    }
}

impl App {
    /// The project bar: save/load a project, add the current loop, and the loop
    /// list (load a saved loop back into the studio).
    fn project_bar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.strong("🎹 Synth.RS Studio");
            ui.separator();
            ui.label("Project");
            ui.add(
                egui::TextEdit::singleline(&mut self.project_name)
                    .hint_text("project name")
                    .desired_width(150.0),
            );
            if ui.button("💾 Save").clicked() {
                self.save_project();
            }
            let mut load: Option<String> = None;
            egui::ComboBox::from_id_salt("project_load")
                .selected_text("Open…")
                .show_ui(ui, |ui| {
                    for name in &self.project_list {
                        if ui.selectable_label(false, name).clicked() {
                            load = Some(name.clone());
                        }
                    }
                });
            if let Some(name) = load {
                self.load_project(&name);
            }
            if ui.button("⟳").on_hover_text("Rescan project folder").clicked() {
                self.project_list = project::list();
            }
            ui.separator();
            // Destructive: gated behind a confirmation modal (there's no undo).
            if ui
                .add(egui::Button::new(egui::RichText::new("🗑 Clear project").color(egui::Color32::from_rgb(230, 90, 90))))
                .on_hover_text("Remove all tracks and the arrangement and start fresh")
                .clicked()
            {
                self.confirm_clear = true;
            }
        });

        if !self.project_status.is_empty() {
            ui.label(egui::RichText::new(&self.project_status).weak().small());
        }
        ui.add_space(2.0);
    }

    /// Offline WAV export of the current song (the loop + its clip arrangement).
    fn export_ui(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.horizontal(|ui| {
            ui.strong("Export WAV");
            ui.label("Name");
            ui.add(egui::TextEdit::singleline(&mut self.export_name).desired_width(120.0));
            ui.label("Rate");
            egui::ComboBox::from_id_salt("export_sr")
                .selected_text(format!("{} Hz", self.export_sr))
                .show_ui(ui, |ui| {
                    for sr in [44_100u32, 48_000, 96_000, 192_000] {
                        ui.selectable_value(&mut self.export_sr, sr, format!("{sr} Hz"));
                    }
                });
            ui.label("Depth");
            egui::ComboBox::from_id_salt("export_bit")
                .selected_text(self.export_bit.label())
                .show_ui(ui, |ui| {
                    for d in [wav::BitDepth::Int16, wav::BitDepth::Int24, wav::BitDepth::Int32] {
                        ui.selectable_value(&mut self.export_bit, d, d.label());
                    }
                });
            ui.checkbox(&mut self.export_hi_res, "Hi-res")
                .on_hover_text("Max out the horn eigensolve resolution for the render (higher rate already admits more modes).");
            if ui
                .button("⬇ Export song")
                .on_hover_text("Render the whole arrangement once to a WAV")
                .clicked()
            {
                self.export_loop();
            }
        });
        if !self.export_status.is_empty() {
            ui.label(egui::RichText::new(&self.export_status).weak().small());
        }
    }

    fn export_loop(&mut self) {
        let view = match &self.view {
            Some(v) => v.clone(),
            None => return,
        };
        let data = view.snapshot();
        if data.is_empty() {
            self.export_status = "Nothing to export — record a loop first.".into();
            return;
        }
        let path = export::export_path(&self.export_name);
        let sr = self.export_sr as f32;
        // Render the song once, with a fixed tail so trailing decays aren't cut.
        match export::render_loop_to_wav(data, sr, 1, 3.0, self.export_hi_res, self.export_bit, &path) {
            Ok(()) => self.export_status = format!("Wrote {}", path.display()),
            Err(e) => self.export_status = format!("Export failed: {e}"),
        }
    }


    /// The instrument-library bar: name + save, and load/delete of saved presets.
    fn presets_bar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.strong("Presets");
        // Row 1: name + save + delete.
        ui.horizontal(|ui| {
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.preset_name)
                    .hint_text("preset name")
                    .desired_width(150.0),
            );
            let save_on_enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let save_hint = if self.edit_target == Target::Live && self.live_is_kit {
                "Save the live kit (all its zones) as a preset"
            } else {
                "Save the current instrument as a preset"
            };
            if ui.button("💾 Save").on_hover_text(save_hint).clicked() || save_on_enter {
                self.save_preset();
            }
            let can_delete = self
                .preset_list
                .iter()
                .any(|p| p.name.eq_ignore_ascii_case(self.preset_name.trim()));
            if ui
                .add_enabled(can_delete, egui::Button::new("🗑"))
                .on_hover_text("Delete the saved preset with this name")
                .clicked()
            {
                let name = self.preset_name.trim().to_string();
                self.delete_preset(&name);
            }
        });
        // Row 2: load + rescan + factory.
        ui.horizontal(|ui| {
            let mut to_load: Option<usize> = None;
            let load_label = if self.preset_list.is_empty() {
                "— no presets —".to_string()
            } else {
                "Load preset…".to_string()
            };
            egui::ComboBox::from_id_salt("preset_load")
                .selected_text(load_label)
                .show_ui(ui, |ui| {
                    for (i, p) in self.preset_list.iter().enumerate() {
                        let kind = if p.is_kit() {
                            format!("🥁 Kit ({})", p.zones.len())
                        } else {
                            self.models
                                .iter()
                                .find(|m| m.id() == p.model_id)
                                .map(|m| m.display_name())
                                .unwrap_or(p.model_id.as_str())
                                .to_string()
                        };
                        if ui
                            .selectable_label(false, format!("{}  ·  {}", p.name, kind))
                            .clicked()
                        {
                            to_load = Some(i);
                        }
                    }
                });
            if let Some(i) = to_load {
                let preset = self.preset_list[i].clone();
                self.apply_preset(&preset);
            }
            if ui.button("⟳").on_hover_text("Rescan preset folder").clicked() {
                self.preset_list = presets::list();
            }
            if ui
                .button("★ Factory")
                .on_hover_text("Restore the built-in instrument presets (overwrites same-named)")
                .clicked()
            {
                self.preset_list = presets::restore_factory();
                self.preset_status = "Restored factory presets.".into();
            }
        });
        if !self.preset_status.is_empty() {
            ui.label(egui::RichText::new(&self.preset_status).weak().small());
        }
    }

    /// Point the parameter panel at the live instrument or a specific track,
    /// loading that track's current instrument into an editable working copy.
    fn set_target(&mut self, target: Target, tracks: &[TrackView]) {
        match target {
            Target::Live => {
                self.edit_target = Target::Live;
                self.track_edit = None;
            }
            Target::Track(i) => {
                if let Some(tv) = tracks.get(i) {
                    let model =
                        models::model_from_id(&tv.model_id, &tv.params).unwrap_or_else(models::default_model);
                    self.track_edit = Some(TrackEdit {
                        idx: i,
                        model,
                        engine: tv.engine.clone(),
                    });
                    self.edit_target = Target::Track(i);
                }
            }
        }
    }

    fn params_panel(&mut self, ui: &mut egui::Ui) {
        let tracks = self.view.as_ref().map(|v| v.tracks()).unwrap_or_default();
        // If the edited track vanished (deleted / reset), fall back to Live.
        if let Target::Track(i) = self.edit_target {
            if i >= tracks.len() {
                self.edit_target = Target::Live;
                self.track_edit = None;
            }
        }

        ui.add_space(6.0);
        ui.heading("Instrument");

        // --- Target selector: Live or a track ---
        ui.horizontal(|ui| {
            ui.label("Editing:");
            let current = match self.edit_target {
                Target::Live => "Live (keyboard)".to_string(),
                Target::Track(i) => format!("{} · {}", tracks[i].name, tracks[i].instrument),
            };
            let mut choose: Option<Target> = None;
            egui::ComboBox::from_id_salt("edit_target")
                .width(230.0)
                .selected_text(current)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(self.edit_target == Target::Live, "Live (keyboard)")
                        .clicked()
                    {
                        choose = Some(Target::Live);
                    }
                    for (i, t) in tracks.iter().enumerate() {
                        let sel = self.edit_target == Target::Track(i);
                        if ui
                            .selectable_label(sel, format!("{} · {}", t.name, t.instrument))
                            .clicked()
                        {
                            choose = Some(Target::Track(i));
                        }
                    }
                });
            if let Some(t) = choose {
                self.set_target(t, &tracks);
            }
        });
        ui.separator();

        match self.edit_target {
            Target::Live => self.edit_live(ui),
            Target::Track(i) => self.edit_track(ui, i),
        }
    }

    /// A default zone from the current live instrument, spanning the keyboard.
    fn default_zone(&self) -> ZoneData {
        let m = self.models[self.selected].as_ref();
        ZoneData {
            name: m.display_name().to_string(),
            lo: 0,
            hi: 127,
            fixed_note: None,
            transpose: 0,
            model_id: m.id().to_string(),
            params: m.to_json(),
            engine: self.engine.clone(),
        }
    }

    /// Push the current kit to the audio thread.
    fn send_live_kit(&self) {
        let _ = self.tx.send(Command::SetLive(LiveConfig::Kit { zones: self.kit_zones.clone() }));
    }

    /// Switch the live slot back to the single selected instrument.
    fn send_live_single(&self) {
        let m = self.models[self.selected].as_ref();
        let _ = self.tx.send(Command::SetLive(LiveConfig::Single {
            model_id: m.id().to_string(),
            params: m.to_json(),
            engine: self.engine.clone(),
        }));
    }

    /// Edit the live (keyboard) instrument — single instrument or a kit.
    fn edit_live(&mut self, ui: &mut egui::Ui) {
        let mut kit_mode = self.live_is_kit;
        if ui
            .checkbox(&mut kit_mode, "Kit mode (split / map keys to instruments)")
            .on_hover_text("Route key ranges to different instruments — drum kits, splits, layers.")
            .changed()
        {
            self.live_is_kit = kit_mode;
            if kit_mode {
                if self.kit_zones.is_empty() {
                    self.kit_zones.push(self.default_zone());
                }
                self.send_live_kit();
            } else {
                self.send_live_single();
            }
        }
        ui.separator();

        if self.live_is_kit {
            self.edit_kit(ui);
            return;
        }

        let mut new_selection = self.selected;
        egui::ComboBox::from_id_salt("model_pick_live")
            .width(260.0)
            .selected_text(self.models[self.selected].display_name())
            .show_ui(ui, |ui| {
                for (i, m) in self.models.iter().enumerate() {
                    ui.selectable_value(&mut new_selection, i, m.display_name());
                }
            });
        ui.label(
            egui::RichText::new(self.models[self.selected].description())
                .weak()
                .small(),
        );
        if new_selection != self.selected {
            self.selected = new_selection;
            self.push_model();
        }

        ui.separator();
        if self.models[self.selected].id() == "instrument_graph" {
            if ui.button("🕸 Edit graph (visual)").clicked() {
                self.ge.open = true;
            }
            ui.add_space(4.0);
        }
        egui::ScrollArea::vertical()
            .max_height(320.0)
            .show(ui, |ui| {
                if self.models[self.selected].params_ui(ui) {
                    self.push_model();
                }
            });

        ui.separator();
        ui.strong("Output / Voice");
        if engine_sliders(ui, &mut self.engine) {
            let _ = self.tx.send(Command::SetEngine(self.engine.clone()));
        }

        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if ui.button("Reset model").clicked() {
                self.models = models::registry();
                self.push_model();
            }
            if ui.button("All notes off").clicked() {
                let _ = self.tx.send(Command::AllNotesOff);
            }
        });
    }

    /// Edit the live kit: a list of key-range → instrument zones.
    fn edit_kit(&mut self, ui: &mut egui::Ui) {
        let presets = self.preset_list.clone();
        let mut changed = false;
        let mut remove: Option<usize> = None;

        ui.horizontal(|ui| {
            ui.label(format!("{} zone(s)", self.kit_zones.len()));
            if ui.button("+ Zone").clicked() {
                let z = self.default_zone();
                self.kit_zones.push(z);
                changed = true;
            }
            if ui.button("Clear").clicked() {
                self.kit_zones.clear();
                changed = true;
            }
        });
        ui.label(
            egui::RichText::new("Expand a zone to configure its model. Overlapping ranges layer; a pad plays one fixed pitch.")
                .weak()
                .small(),
        );
        ui.separator();

        let reg = models::registry();
        egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
            for (zi, z) in self.kit_zones.iter_mut().enumerate() {
                // Header summarizes the zone; expand to configure the model.
                let model_name = reg
                    .iter()
                    .find(|m| m.id() == z.model_id)
                    .map(|m| m.display_name())
                    .unwrap_or("?");
                let sound = match z.fixed_note {
                    Some(n) => format!("pad {}", note_name(n)),
                    None if z.transpose != 0 => format!("{:+} st", z.transpose),
                    None => "chromatic".to_string(),
                };
                let title = format!(
                    "{}  ·  {}–{}  ·  {} ({})",
                    z.name,
                    note_name(z.lo),
                    note_name(z.hi),
                    model_name,
                    sound
                );
                egui::CollapsingHeader::new(title)
                    .id_salt(("zone_hdr", zi))
                    .show(ui, |ui| {
                        // --- Identity + quick preset load + remove ---
                        ui.horizontal(|ui| {
                            ui.label("Name");
                            ui.add(egui::TextEdit::singleline(&mut z.name).desired_width(110.0));
                            egui::ComboBox::from_id_salt(("zone_preset", zi))
                                .selected_text("load preset…")
                                .width(140.0)
                                .show_ui(ui, |ui| {
                                    for p in presets.iter() {
                                        if ui.selectable_label(false, &p.name).clicked() {
                                            z.model_id = p.model_id.clone();
                                            z.params = p.params.clone();
                                            z.engine = p.engine.clone();
                                            z.name = p.name.clone();
                                            changed = true;
                                        }
                                    }
                                });
                            if ui.button("✕ Remove").clicked() {
                                remove = Some(zi);
                            }
                        });

                        // --- Key range + pad / transpose --- (note-name fields,
                        // typeable or set by playing a note via 🎹 MIDI-learn)
                        ui.horizontal(|ui| {
                            ui.label("Keys");
                            let mut lo = z.lo as i32;
                            if models::note_field(ui, &format!("kit_lo_{zi}"), &mut lo) {
                                z.lo = lo.clamp(0, 127) as u8;
                                changed = true;
                            }
                            ui.label("–");
                            let mut hi = z.hi as i32;
                            if models::note_field(ui, &format!("kit_hi_{zi}"), &mut hi) {
                                z.hi = hi.clamp(0, 127) as u8;
                                changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            let mut pad = z.fixed_note.is_some();
                            if ui
                                .checkbox(&mut pad, "Pad")
                                .on_hover_text("Any key in range plays one fixed pitch (a drum pad).")
                                .changed()
                            {
                                z.fixed_note = if pad { Some(z.lo) } else { None };
                                changed = true;
                            }
                            if let Some(fixed) = z.fixed_note.as_mut() {
                                let mut n = *fixed as i32;
                                if models::note_field(ui, &format!("kit_pad_{zi}"), &mut n) {
                                    *fixed = n.clamp(0, 127) as u8;
                                    changed = true;
                                }
                            } else {
                                ui.label("transpose");
                                changed |= ui
                                    .add(egui::DragValue::new(&mut z.transpose).range(-48..=48).suffix(" st"))
                                    .changed();
                            }
                        });

                        ui.separator();

                        // --- Model type + its full parameter editor ---
                        ui.horizontal(|ui| {
                            ui.strong("Model");
                            egui::ComboBox::from_id_salt(("zone_model", zi))
                                .selected_text(model_name)
                                .width(200.0)
                                .show_ui(ui, |ui| {
                                    for m in reg.iter() {
                                        if ui
                                            .selectable_label(m.id() == z.model_id, m.display_name())
                                            .clicked()
                                            && m.id() != z.model_id
                                        {
                                            z.model_id = m.id().to_string();
                                            z.params = m.to_json();
                                            changed = true;
                                        }
                                    }
                                });
                        });
                        // Rebuild a live model from the zone's JSON, edit it, write back.
                        let mut model = models::model_from_id(&z.model_id, &z.params)
                            .unwrap_or_else(models::default_model);
                        ui.label(egui::RichText::new(model.description()).weak().small());
                        if model.params_ui(ui) {
                            z.params = model.to_json();
                            changed = true;
                        }

                        ui.separator();
                        ui.strong("Output / Voice");
                        if engine_sliders(ui, &mut z.engine) {
                            changed = true;
                        }
                    });
            }
        });

        if let Some(r) = remove {
            if r < self.kit_zones.len() {
                self.kit_zones.remove(r);
                changed = true;
            }
        }

        ui.add_space(8.0);
        if ui.button("All notes off").clicked() {
            let _ = self.tx.send(Command::AllNotesOff);
        }

        if changed {
            self.send_live_kit();
        }
    }

    /// Show a kit track's zones (read-only for now — kit-track editing lands in a
    /// later step; you can still re-record the track from a live kit).
    fn show_kit_track(&self, ui: &mut egui::Ui, zones: &[ZoneData]) {
        ui.label(egui::RichText::new("Kit track").strong());
        ui.label(
            egui::RichText::new("Editing kit tracks in place isn't wired up yet — tweak the live kit and re-record.")
                .weak()
                .small(),
        );
        ui.separator();
        for z in zones {
            let sound = match z.fixed_note {
                Some(n) => format!("pad → {}", note_name(n)),
                None if z.transpose != 0 => format!("{:+} st", z.transpose),
                None => "chromatic".to_string(),
            };
            ui.label(format!(
                "{}  ·  {}–{}  ·  {}  ·  {}",
                z.name,
                note_name(z.lo),
                note_name(z.hi),
                z.model_id,
                sound
            ));
        }
    }

    /// Edit a loop track's instrument live; edits are pushed to the audio thread.
    fn edit_track(&mut self, ui: &mut egui::Ui, i: usize) {
        // A kit track: show its zones read-only instead of the single editor.
        let tracks = self.view.as_ref().map(|v| v.tracks()).unwrap_or_default();
        if let Some(tv) = tracks.get(i) {
            if tv.model_id == "kit" {
                self.show_kit_track(ui, &tv.zones);
                return;
            }
        }

        // Take the working copy out to sidestep borrow conflicts with self.tx.
        let Some(mut te) = self.track_edit.take() else {
            return;
        };
        if te.idx != i {
            self.track_edit = Some(te);
            return;
        }

        let reg = models::registry();
        let mut model_changed = false;

        egui::ComboBox::from_id_salt("model_pick_track")
            .width(260.0)
            .selected_text(te.model.display_name())
            .show_ui(ui, |ui| {
                let mut pick = None;
                for (k, m) in reg.iter().enumerate() {
                    if ui
                        .selectable_label(m.id() == te.model.id(), m.display_name())
                        .clicked()
                    {
                        pick = Some(k);
                    }
                }
                if let Some(k) = pick {
                    if reg[k].id() != te.model.id() {
                        te.model = reg[k].box_clone(); // switch plugin -> fresh defaults
                        model_changed = true;
                    }
                }
            });
        ui.label(egui::RichText::new(te.model.description()).weak().small());

        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(320.0)
            .show(ui, |ui| {
                if te.model.params_ui(ui) {
                    model_changed = true;
                }
            });
        if model_changed {
            let _ = self.tx.send(Command::SetTrackModel(i, te.model.box_clone()));
        }

        ui.separator();
        ui.strong("Output / Voice");
        if engine_sliders(ui, &mut te.engine) {
            let _ = self.tx.send(Command::SetTrackEngine(i, te.engine.clone()));
        }

        ui.separator();
        ui.strong("Fades");
        let (mut fin, mut fout) = tracks.get(i).map(|tv| (tv.fade_in, tv.fade_out)).unwrap_or((0.0, 0.0));
        let mut fade_changed = false;
        ui.horizontal(|ui| {
            ui.label("In");
            fade_changed |= ui.add(egui::DragValue::new(&mut fin).range(0.0..=10.0).speed(0.05).suffix(" s")).changed();
            ui.label("Out");
            fade_changed |= ui.add(egui::DragValue::new(&mut fout).range(0.0..=10.0).speed(0.05).suffix(" s")).changed();
        });
        if fade_changed {
            let _ = self.tx.send(Command::SetTrackFades { track: i, fade_in: fin, fade_out: fout });
        }

        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if ui.button("← Back to Live").clicked() {
                self.edit_target = Target::Live;
            }
            if ui.button("🗑 Delete track").clicked() {
                let _ = self.tx.send(Command::DeleteTrack(i));
                self.edit_target = Target::Live;
            }
        });

        self.track_edit = Some(te);
    }

    /// The MIDI inputs section: the virtual keyboard with its play controls, and
    /// the hardware MIDI-device selector. Future controllers attach here too.
    fn midi_inputs(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("🎛 MIDI Inputs");
                ui.label(
                    egui::RichText::new("virtual keyboard · attach controllers here later")
                        .weak()
                        .small(),
                );
            });
            self.midi_panel(ui); // hardware device selector
            ui.separator();
            ui.horizontal(|ui| {
                // The keyboard on the left …
                self.piano(ui);
                ui.separator();
                // … its output / voice controls on the right.
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Octave");
                        if ui.button("–").clicked() {
                            self.base_midi = (self.base_midi - 12).max(0);
                        }
                        ui.label(format!("C{}", self.base_midi / 12 - 1));
                        if ui.button("+").clicked() {
                            self.base_midi = (self.base_midi + 12).min(108);
                        }
                    });
                    ui.label(
                        egui::RichText::new("keys A W S E D F T G Y H U J K · Z/X shift octave")
                            .weak()
                            .small(),
                    );
                    ui.add_space(4.0);
                    self.whammy_bar(ui);
                });
            });
        });
    }

    fn midi_panel(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("MIDI input:");
            let selected_text = match self.midi_sel {
                Some(i) => self
                    .midi_ports
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| "?".into()),
                None => "— none —".to_string(),
            };
            let mut choose: Option<usize> = None;
            egui::ComboBox::from_id_salt("midi_ports")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for (i, name) in self.midi_ports.iter().enumerate() {
                        if ui
                            .selectable_label(self.midi_sel == Some(i), name)
                            .clicked()
                        {
                            choose = Some(i);
                        }
                    }
                });
            if let Some(i) = choose {
                self.connect_midi(i);
            }
            if ui.button("Rescan").clicked() {
                self.midi_ports = midi::list_ports();
            }
        });
        ui.label(egui::RichText::new(&self.midi_status).weak());

    }

    /// Transport controls: play / record / repeat and the tempo grid.
    fn transport_bar(&mut self, ui: &mut egui::Ui) {
        let (state, loop_secs) = self
            .view
            .as_ref()
            .map(|v| (v.state(), v.song_seconds(self.sample_rate)))
            .unwrap_or((TransportState::Idle, 0.0));

        ui.horizontal(|ui| {
            ui.strong("Transport");
            ui.separator();
            let (label, color) = match state {
                TransportState::Idle => ("● Idle", egui::Color32::GRAY),
                TransportState::Recording => ("⏺ Recording", egui::Color32::from_rgb(230, 80, 80)),
                TransportState::Playing => ("▶ Playing", egui::Color32::from_rgb(90, 200, 110)),
                TransportState::Stopped => ("⏸ Stopped", egui::Color32::from_rgb(220, 190, 90)),
            };
            ui.colored_label(color, label);
            if loop_secs > 0.0 {
                ui.label(format!("loop {loop_secs:.1}s"));
            }

            ui.separator();
            // Repeat toggle (left of Play) — default off = play once, then stop.
            if ui
                .add(egui::Button::new("🔁").selected(self.repeat))
                .on_hover_text("Repeat: loop the song at the end. Off = play once and stop.")
                .clicked()
            {
                self.repeat = !self.repeat;
                let _ = self.tx.send(Command::SetRepeat(self.repeat));
            }
            // Play / pause — never records.
            let playing = matches!(state, TransportState::Playing | TransportState::Recording);
            let play_label = if playing { "⏸ Pause" } else { "▶ Play" };
            if ui
                .button(play_label)
                .on_hover_text("Start / pause playback (no recording).")
                .clicked()
            {
                let _ = self.tx.send(Command::Play);
            }
            // Record — separate from Play.
            let recording = matches!(state, TransportState::Recording);
            let rec_label = if recording { "⏹ Finish take" } else { "⏺ Record" };
            let rec_hint = if self.project.tempo.count_in {
                "Record (separate from Play). Playing → punches in at the playhead now. Stopped → one bar of count-in, then records from the seek cursor. No loop yet → records the first take."
            } else {
                "Record (separate from Play). Playing → punches in at the playhead. Stopped/idle → first take, or punch-in at the seek cursor. Turn on Count-in for a lead-in bar before recording."
            };
            let rec_btn = egui::Button::new(egui::RichText::new(rec_label).color(egui::Color32::from_rgb(230, 90, 90)));
            if ui.add(rec_btn).on_hover_text(rec_hint).clicked() {
                let _ = self.tx.send(Command::Record);
            }
            if ui.button("⏹ Stop").clicked() {
                let _ = self.tx.send(Command::Stop);
            }
        });

        let hint = "▶ Play and ⏺ Record are separate. Record while playing punches in at the playhead; Record from a stop gives a count-in (when enabled) and records from the seek cursor. Each take becomes a track / clip.";
        ui.label(egui::RichText::new(hint).weak().small());

        self.tempo_bar(ui);
    }

    /// Tempo, bars grid, quantize, and metronome controls.
    fn tempo_bar(&mut self, ui: &mut egui::Ui) {
        let t = &mut self.project.tempo;
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Tempo");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut t.bpm)
                        .range(20.0..=300.0)
                        .speed(0.5)
                        .suffix(" BPM"),
                )
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut t.beats_per_bar).range(1..=16).suffix("/bar"))
                .changed();
            ui.separator();

            ui.label("Recording Duration");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut t.bars)
                        .range(0..=64)
                        .custom_formatter(|n, _| {
                            if n < 0.5 {
                                "Free".to_string()
                            } else {
                                format!("{} bar", n as u32)
                            }
                        }),
                )
                .on_hover_text("How long a recording runs: a fixed number of bars auto-stops the take at that length; Free = you stop it with ⏺ Record.")
                .changed();
            ui.separator();

            ui.label("Quantize");
            let qname = |q: u32| match q {
                0 => "Off",
                1 => "1/4",
                2 => "1/8",
                3 => "1/8T",
                4 => "1/16",
                _ => "?",
            };
            egui::ComboBox::from_id_salt("quantize")
                .selected_text(qname(t.quantize))
                .show_ui(ui, |ui| {
                    for q in [0u32, 1, 2, 3, 4] {
                        changed |= ui.selectable_value(&mut t.quantize, q, qname(q)).changed();
                    }
                });
            ui.separator();

            changed |= ui.checkbox(&mut t.metronome, "🔔 Click").changed();
            changed |= ui
                .checkbox(&mut t.count_in, "Count-in")
                .on_hover_text("Play one audible bar of clicks before recording starts — for the first take and for ＋Rec track (which punches in at the seek cursor). Clicks even when 🔔 Click is off.")
                .changed();
        });
        if changed {
            let _ = self.tx.send(Command::SetTempo(self.project.tempo));
        }
    }

    /// The recorded loop tracks, shown below the keyboard.
    fn tracks_panel(&mut self, ui: &mut egui::Ui) {
        let (tracks, play, loop_secs, clips) = match &self.view {
            Some(v) => (
                v.tracks(),
                v.play_fraction(),
                v.song_seconds(self.sample_rate),
                v.arrangement(),
            ),
            None => return,
        };

        ui.horizontal(|ui| {
            ui.strong("Tracks");
            ui.label(egui::RichText::new(format!("({})", tracks.len())).weak());
            ui.separator();
            ui.label("Master");
            let mut m = self.master_volume;
            if ui
                .add(egui::Slider::new(&mut m, 0.0..=1.5).show_value(false))
                .on_hover_text(format!("Master volume ({:.0}%)", m * 100.0))
                .changed()
            {
                self.master_volume = m;
                let _ = self.tx.send(Command::SetMasterVolume(m));
            }
            ui.separator();
            let (can_undo, can_redo) = self
                .view
                .as_ref()
                .map(|v| (v.undo_depth() > 0, v.redo_depth() > 0))
                .unwrap_or((false, false));
            if ui
                .add_enabled(can_undo, egui::Button::new("↶ Undo"))
                .on_hover_text("Undo the last edit (⌘Z)")
                .clicked()
            {
                let _ = self.tx.send(Command::Undo);
            }
            if ui
                .add_enabled(can_redo, egui::Button::new("↷ Redo"))
                .on_hover_text("Redo (⌘⇧Z)")
                .clicked()
            {
                let _ = self.tx.send(Command::Redo);
            }
        });
        if tracks.is_empty() {
            ui.label(
                egui::RichText::new("No tracks yet. Hit ⏺ Record to lay down a loop — it becomes a track shown here as a clip you can arrange.")
                    .weak()
                    .small(),
            );
            return;
        }

        self.arrangement_editor(ui, &tracks, &clips, loop_secs, play);
        ui.separator();

        // Controls for the selected track / clip.
        self.contextual_panel(ui, &tracks, &clips, play, loop_secs);
    }

    fn whammy_bar(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.label(egui::RichText::new("Whammy").small());
            let resp = ui.add(
                egui::Slider::new(&mut self.whammy, -self.whammy_down..=self.whammy_up)
                    .vertical()
                    .show_value(false),
            );
            if resp.changed() {
                let _ = self.tx.send(Command::SetBend(self.whammy));
            }
            // Spring back toward centre when not being held.
            if !resp.dragged() && self.whammy.abs() > 1e-3 {
                self.whammy *= 0.6;
                if self.whammy.abs() < 1e-3 {
                    self.whammy = 0.0;
                }
                let _ = self.tx.send(Command::SetBend(self.whammy));
                ui.ctx().request_repaint();
            }
            ui.label(egui::RichText::new(format!("{:+.1}", self.whammy)).small());
        });
    }

    /// Draw a clickable two-octave piano and handle mouse input.
    fn piano(&mut self, ui: &mut egui::Ui) {
        let n_octaves = 2;
        let n_white = 7 * n_octaves;
        let width = ui.available_width().min(820.0);
        let height = 150.0;
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        let white_w = rect.width() / n_white as f32;
        let black_w = white_w * 0.62;
        let black_h = height * 0.62;

        // Semitone offset for each white key within an octave.
        let white_semis = [0, 2, 4, 5, 7, 9, 11];
        // Black keys sit after white index 0,1,3,4,5 with these semitones.
        let black_after: [(usize, i32); 5] = [(0, 1), (1, 3), (3, 6), (4, 8), (5, 10)];

        // Figure out which note the pointer is over (blacks are on top).
        let pointer_note = |pos: egui::Pos2| -> Option<u8> {
            if !rect.contains(pos) {
                return None;
            }
            // Check black keys first.
            for oct in 0..n_octaves {
                for (wi, semi) in black_after.iter() {
                    let cx = rect.left() + (oct * 7 + wi) as f32 * white_w + white_w;
                    let bx = cx - black_w / 2.0;
                    let brect = egui::Rect::from_min_size(
                        egui::pos2(bx, rect.top()),
                        egui::vec2(black_w, black_h),
                    );
                    if brect.contains(pos) {
                        return Some((self.base_midi + oct as i32 * 12 + semi).clamp(0, 127) as u8);
                    }
                }
            }
            // Then white keys.
            let rel = pos.x - rect.left();
            let wi_global = (rel / white_w).floor() as i32;
            if wi_global < 0 || wi_global >= n_white as i32 {
                return None;
            }
            let oct = wi_global / 7;
            let wi = (wi_global % 7) as usize;
            Some((self.base_midi + oct * 12 + white_semis[wi]).clamp(0, 127) as u8)
        };

        // Handle mouse press / drag / release.
        let primary_down = ui.input(|i| i.pointer.primary_down());
        if primary_down {
            if let Some(pos) = response.interact_pointer_pos() {
                let note = pointer_note(pos);
                if note != self.mouse_note {
                    if let Some(old) = self.mouse_note.take() {
                        self.note_off(old);
                    }
                    if let Some(n) = note {
                        // Velocity from vertical position: lower = harder.
                        let vy = ((pos.y - rect.top()) / height).clamp(0.2, 1.0);
                        self.note_on(n, 0.4 + 0.6 * vy);
                        self.mouse_note = Some(n);
                    }
                }
            }
        } else if let Some(old) = self.mouse_note.take() {
            self.note_off(old);
        }

        // Which notes are currently sounding (for highlight)?
        let is_down = |note: u8| -> bool {
            self.mouse_note == Some(note) || self.held_keys.values().any(|&n| n == note)
        };

        // Draw white keys.
        for wi_global in 0..n_white {
            let oct = wi_global / 7;
            let wi = (wi_global % 7) as usize;
            let note = (self.base_midi + oct as i32 * 12 + white_semis[wi]).clamp(0, 127) as u8;
            let x = rect.left() + wi_global as f32 * white_w;
            let krect = egui::Rect::from_min_size(
                egui::pos2(x, rect.top()),
                egui::vec2(white_w - 1.0, height),
            );
            let fill = if is_down(note) {
                egui::Color32::from_rgb(120, 180, 255)
            } else {
                egui::Color32::from_gray(245)
            };
            painter.rect_filled(krect, 2.0, fill);
            painter.rect_stroke(krect, 2.0, egui::Stroke::new(1.0_f32, egui::Color32::from_gray(120)));
        }

        // Draw black keys on top.
        for oct in 0..n_octaves {
            for (wi, semi) in black_after.iter() {
                let note = (self.base_midi + oct as i32 * 12 + semi).clamp(0, 127) as u8;
                let cx = rect.left() + (oct * 7 + wi) as f32 * white_w + white_w;
                let bx = cx - black_w / 2.0;
                let brect = egui::Rect::from_min_size(
                    egui::pos2(bx, rect.top()),
                    egui::vec2(black_w, black_h),
                );
                let fill = if is_down(note) {
                    egui::Color32::from_rgb(70, 120, 200)
                } else {
                    egui::Color32::from_gray(25)
                };
                painter.rect_filled(brect, 2.0, fill);
            }
        }
    }
}

/// Map a physical keyboard key to a semitone offset within the base octave,
/// using the common one-octave tracker layout.
/// Shared engine-parameter sliders (gain, envelope, retrigger). Returns true if
/// anything changed. Used by both the live and per-track editors.
/// MIDI note number → name like `C4` (C4 = 60).
fn note_name(n: u8) -> String {
    const NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let oct = (n / 12) as i32 - 1;
    format!("{}{}", NAMES[(n % 12) as usize], oct)
}

fn engine_sliders(ui: &mut egui::Ui, e: &mut EngineParams) -> bool {
    let unbounded = egui::SliderClamping::Never;
    let mut c = false;
    c |= ui
        .add(egui::Slider::new(&mut e.gain, 0.0..=4.0).clamping(unbounded).text("Gain"))
        .changed();
    c |= ui
        .add(egui::Slider::new(&mut e.attack_ms, 0.0..=2000.0).clamping(unbounded).text("Attack (ms)"))
        .changed();
    c |= ui
        .add(egui::Slider::new(&mut e.release_ms, 1.0..=5000.0).clamping(unbounded).text("Release (ms)"))
        .changed();
    c |= ui
        .add(egui::Slider::new(&mut e.retrigger_ms, 0.0..=2000.0).clamping(unbounded).text("Retrigger (ms)"))
        .on_hover_text("Minimum time between strikes; 0 = off.")
        .changed();
    c
}

/// Draw a track's recorded notes as bars on a timeline, with a moving playhead
/// and (optionally) a shaded selection band. Senses click-and-drag so the caller
/// can drag out a selection; returns the response.

fn key_to_semitone(key: egui::Key) -> Option<i32> {
    use egui::Key::*;
    Some(match key {
        A => 0,   // C
        W => 1,   // C#
        S => 2,   // D
        E => 3,   // D#
        D => 4,   // E
        F => 5,   // F
        T => 6,   // F#
        G => 7,   // G
        Y => 8,   // G#
        H => 9,   // A
        U => 10,  // A#
        J => 11,  // B
        K => 12,  // C (next octave)
        O => 13,  // C#
        L => 14,  // D
        _ => return None,
    })
}

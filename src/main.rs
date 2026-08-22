//! Synth.RS Studio — a MIDI synth / production-studio framework. Pluggable
//! synthesizer models (many use physical modeling, but the framework is
//! method-agnostic), played from an on-screen piano, the computer keyboard, or a
//! MIDI controller, recorded and arranged into songs.

mod audio;
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

use audio::AudioEngine;
use instrument::EngineParams;
use midi::MidiInputHandle;
use models::FtmModel;
use presets::Preset;
use project::{NamedSong, Project, TempoGrid, ZoneData};
use studio::{
    ClipView, Command, LiveConfig, SharedView, TrackView,
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

/// What a clip drag is doing. `Move` is the top-middle handle; the edges resize;
/// a drag on the body selects a time range for the Crop/Delete/Loop/Reverse ops.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DragKind {
    Move,
    ResizeR,
    ResizeL,
    Select,
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
    /// Shared "last note played" for MIDI-learn on pitch fields.
    note_monitor: midi::NoteMonitor,
    midi_status: String,
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
            note_monitor: midi::NoteMonitor::default(),
            midi_status: "not connected".to_string(),
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

    /// The arrangement editor: a wrapping multi-lane timeline. Time flows left to
    /// right and wraps to stacked blocks (like a score wrapping systems); each
    /// block shows every track's lane for that time window. Clips can be dragged
    /// to move, edge-dragged to resize, duplicated and deleted; click empty to
    /// seek, double-click an empty lane to place a clip.
    fn arrangement_editor(
        &mut self,
        ui: &mut egui::Ui,
        tracks: &[TrackView],
        clips: &[ClipView],
        song_secs: f32,
        play: f32,
    ) {
        if tracks.is_empty() {
            return;
        }
        self.sel_clips.retain(|&i| i < clips.len());
        let n = tracks.len();
        let song = song_secs.max(0.001);
        ui.horizontal(|ui| {
            ui.strong("Arrangement");
            ui.separator();
            ui.label("Snap");
            egui::ComboBox::from_id_salt("snap_grid")
                .selected_text(self.snap.label())
                .show_ui(ui, |ui| {
                    for sn in [Snap::Bar, Snap::Quarter, Snap::Eighth, Snap::Sixteenth, Snap::Free] {
                        ui.selectable_value(&mut self.snap, sn, sn.label());
                    }
                });
            ui.label(egui::RichText::new(
                "· handle moves · edges resize · drag body to select · strip seeks · dbl-click empty to place",
            ).weak().small());
        });

        let lane_h = 22.0;
        let seek_h = 10.0; // thin scrub strip under each wrapped row
        let row_gap = 10.0;
        let label_w = 66.0;
        let px_per_bar = 84.0;
        let bar_secs =
            60.0 / self.project.tempo.bpm.max(1.0) * self.project.tempo.beats_per_bar.max(1) as f32;
        let avail = ui.available_width().max(360.0);
        let tl_w = (avail - label_w - 8.0).max(60.0);
        let bars_per_row = ((tl_w / px_per_bar).floor() as usize).max(1);
        let row_secs = (bars_per_row as f32 * bar_secs).max(0.001);
        // One spare row past the song end so a clip's right edge can be dragged
        // out to grow the arrangement.
        let n_rows = (((song / row_secs).ceil() as usize) + 1).clamp(1, 200);
        // Full timeline extent (incl. the spare row) — resize can reach here.
        let grid_secs = n_rows as f32 * row_secs;
        let lanes_h = n as f32 * lane_h;
        let row_h = lanes_h + seek_h + row_gap;
        let (rect, resp) =
            ui.allocate_exact_size(egui::vec2(avail, n_rows as f32 * row_h), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, egui::Color32::from_gray(22));
        let tl_x = rect.left() + label_w;
        let font = egui::FontId::proportional(10.0);

        let row_top = |r: usize| rect.top() + r as f32 * row_h;
        let x_in_row = |secs: f32, r: usize| {
            tl_x + ((secs - r as f32 * row_secs) / row_secs).clamp(0.0, 1.0) * tl_w
        };
        // Pointer → absolute time (any row), clamped to the full grid extent
        // (which includes a spare row past the song, so edges can grow it).
        let time_at = |p: egui::Pos2| -> f32 {
            let rel = (p.y - rect.top()).max(0.0);
            let r = ((rel / row_h) as usize).min(n_rows - 1);
            (r as f32 * row_secs + ((p.x - tl_x) / tl_w).clamp(0.0, 1.0) * row_secs).clamp(0.0, grid_secs)
        };
        let lane_at = |p: egui::Pos2| -> Option<usize> {
            if p.x < tl_x {
                return None;
            }
            let rel = p.y - rect.top();
            if rel < 0.0 {
                return None;
            }
            let r = (rel / row_h) as usize;
            if r >= n_rows {
                return None;
            }
            let lane = ((rel - r as f32 * row_h) / lane_h) as usize;
            (lane < n).then_some(lane)
        };
        // Pointer in a row's seek strip (the thin band below its lanes)?
        let in_seek_strip = |p: egui::Pos2| -> bool {
            if p.x < tl_x {
                return false;
            }
            let rel = p.y - rect.top();
            if rel < 0.0 {
                return false;
            }
            let r = (rel / row_h) as usize;
            if r >= n_rows {
                return false;
            }
            let off = rel - r as f32 * row_h;
            off >= lanes_h && off < lanes_h + seek_h
        };
        let hit_clip = |p: egui::Pos2| -> Option<(usize, DragKind)> {
            let lane = lane_at(p)?;
            let t = time_at(p);
            let resize_secs = (8.0 / tl_w) * row_secs;
            let handle_secs = (10.0 / tl_w) * row_secs; // half-width of the move handle
            let rel = (p.y - rect.top()).max(0.0);
            let r = (rel / row_h) as usize;
            let y_in_lane = rel - r as f32 * row_h - lane as f32 * lane_h;
            for (ci, c) in clips.iter().enumerate().rev() {
                if c.track != lane {
                    continue;
                }
                let len = if c.length > 0.0 { c.length } else { (song - c.start).max(0.0) };
                if t >= c.start && t <= c.start + len {
                    let center = c.start + len * 0.5;
                    let kind = if t >= c.start + len - resize_secs {
                        DragKind::ResizeR
                    } else if t <= c.start + resize_secs {
                        DragKind::ResizeL
                    } else if y_in_lane <= lane_h * 0.55 && (t - center).abs() <= handle_secs {
                        DragKind::Move // top-middle handle
                    } else {
                        DragKind::Select // body → time selection
                    };
                    return Some((ci, kind));
                }
            }
            None
        };

        // Row backgrounds: alternate-lane shading + a separator above each row.
        for r in 0..n_rows {
            for li in 0..n {
                if li % 2 == 1 {
                    let y0 = row_top(r) + li as f32 * lane_h;
                    painter.rect_filled(
                        egui::Rect::from_min_size(egui::pos2(tl_x, y0), egui::vec2(tl_w, lane_h)),
                        0.0,
                        egui::Color32::from_gray(30),
                    );
                }
            }
            // Bar gridlines within the row.
            let yr = egui::Rangef::new(row_top(r), row_top(r) + n as f32 * lane_h);
            for b in 0..=bars_per_row {
                let secs = r as f32 * row_secs + b as f32 * bar_secs;
                if secs > song + 1e-3 {
                    break;
                }
                painter.vline(x_in_row(secs, r), yr, egui::Stroke::new(1.0_f32, egui::Color32::from_gray(40)));
            }
            // Lane labels (repeated per row) + row/time marker.
            for (li, t) in tracks.iter().enumerate() {
                painter.text(
                    egui::pos2(rect.left() + 3.0, row_top(r) + li as f32 * lane_h + lane_h * 0.5),
                    egui::Align2::LEFT_CENTER,
                    &t.name,
                    font.clone(),
                    egui::Color32::from_gray(150),
                );
            }
            // Seek strip for this row: a thin scrub band under the lanes, covering
            // exactly this row's slice of the song. Only the part before song end.
            let strip_y0 = row_top(r) + lanes_h;
            let strip_end = ((r as f32 + 1.0) * row_secs).min(song);
            if strip_end > r as f32 * row_secs {
                let strip_x1 = x_in_row(strip_end, r);
                let strip = egui::Rect::from_min_max(
                    egui::pos2(tl_x, strip_y0),
                    egui::pos2(strip_x1.max(tl_x + 1.0), strip_y0 + seek_h),
                );
                painter.rect_filled(strip, 2.0, egui::Color32::from_gray(38));
                for b in 0..=bars_per_row {
                    let secs = r as f32 * row_secs + b as f32 * bar_secs;
                    if secs > strip_end + 1e-3 {
                        break;
                    }
                    painter.vline(
                        x_in_row(secs, r),
                        egui::Rangef::new(strip_y0, strip_y0 + seek_h),
                        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(52)),
                    );
                }
                // Playhead handle, drawn in the row that holds it.
                let pl = play * song;
                if pl >= r as f32 * row_secs && pl <= strip_end + 1e-3 {
                    painter.vline(
                        x_in_row(pl, r),
                        egui::Rangef::new(strip_y0 - 1.0, strip_y0 + seek_h + 1.0),
                        egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(240, 240, 120)),
                    );
                }
            }
        }

        // Clips (drawn as one segment per row they span), with their notes.
        for (ci, c) in clips.iter().enumerate() {
            if c.track >= n {
                continue;
            }
            let t = &tracks[c.track];
            let default_len = if c.length > 0.0 { c.length } else { (song - c.start).max(0.0) };
            let (start, len) = match self.arr_drag {
                // Move / resize previews carry (start, len); a Select drag does
                // NOT resize the clip — its floats are the selection, not a size.
                Some((di, k, ps, pl)) if di == ci && k != DragKind::Select => (ps, pl),
                Some((di, DragKind::Move, ps, _)) if self.sel_clips.contains(&ci) && self.sel_clips.contains(&di) => {
                    let delta = ps - clips.get(di).map(|d| d.start).unwrap_or(0.0);
                    ((c.start + delta).max(0.0), default_len)
                }
                _ => (c.start, default_len),
            };
            let end = start + len;
            // Effective loop offset for drawing: a front-trim (left-edge) drag
            // advances the offset by how far the edge moved, so content stays put.
            let eff_offset = match self.arr_drag {
                Some((di, DragKind::ResizeL, ps, _)) if di == ci => c.offset + (ps - c.start),
                _ => c.offset,
            };
            let selected = self.sel_clips.contains(&ci);
            let fill = if t.muted {
                egui::Color32::from_gray(70)
            } else if selected {
                egui::Color32::from_rgb(90, 140, 100)
            } else {
                egui::Color32::from_rgb(60, 90, 130)
            };
            let r0 = (start / row_secs) as usize;
            let r1 = (((end - 1e-4).max(start)) / row_secs) as usize;
            for r in r0..=r1.min(n_rows - 1) {
                let seg_s = start.max(r as f32 * row_secs);
                let seg_e = end.min((r as f32 + 1.0) * row_secs);
                if seg_e <= seg_s {
                    continue;
                }
                let y0 = row_top(r) + c.track as f32 * lane_h;
                let x0 = x_in_row(seg_s, r);
                let x1 = x_in_row(seg_e, r).max(x0 + 3.0);
                let rrect = egui::Rect::from_min_max(egui::pos2(x0, y0 + 2.0), egui::pos2(x1, y0 + lane_h - 1.0));
                painter.rect_filled(rrect, 2.0, fill);
                if selected {
                    painter.rect_stroke(rrect, 2.0, egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(210, 235, 210)));
                    // Grab handles on the true edges (left edge in the first row,
                    // right edge in the last row) so resizing is discoverable.
                    let handle = egui::Color32::from_rgb(225, 245, 225);
                    if r == r0 {
                        painter.rect_filled(
                            egui::Rect::from_min_max(egui::pos2(x0, y0 + 2.0), egui::pos2(x0 + 6.0, y0 + lane_h - 1.0)),
                            1.0,
                            handle,
                        );
                    }
                    if r == r1.min(n_rows - 1) {
                        painter.rect_filled(
                            egui::Rect::from_min_max(egui::pos2(x1 - 6.0, y0 + 2.0), egui::pos2(x1, y0 + lane_h - 1.0)),
                            1.0,
                            handle,
                        );
                    }
                }
            }
            // Notes, following the clip's loop config: its own content notes
            // (`c.notes`, fractions of the content span `s`), the loop unit `l`
            // (silence beyond the window each cycle), and the front-trim offset.
            let s = c.content_len.max(1e-6);
            let l = if c.looping { c.loop_len.max(1e-6) } else { len.max(1e-6) };
            let base_off = eff_offset.rem_euclid(s);
            let bound = s.min(l);
            for nsp in &c.notes {
                let note_pos = nsp.start * s;
                let dur = ((nsp.end - nsp.start) * s).max(1e-4);
                let phase0 = (note_pos - base_off).rem_euclid(s);
                if phase0 >= bound {
                    continue;
                }
                let mut u = start + phase0;
                let mut guard = 0;
                while u < end && guard < 512 {
                    let r = ((u / row_secs) as usize).min(n_rows - 1);
                    let ne = (u + dur).min(end).min((r as f32 + 1.0) * row_secs);
                    let y = row_top(r) + c.track as f32 * lane_h + lane_h * 0.5;
                    painter.line_segment(
                        [egui::pos2(x_in_row(u, r), y), egui::pos2(x_in_row(ne, r).max(x_in_row(u, r) + 1.0), y)],
                        egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(190, 215, 255)),
                    );
                    if !c.looping {
                        break;
                    }
                    u += l;
                    guard += 1;
                }
            }
            // Move handle: a small grip at the clip's top-middle (in its centre
            // row). Grab here to move; the body selects a range.
            {
                let center = (start + end) * 0.5;
                let r = ((center / row_secs) as usize).min(n_rows - 1);
                let hx = x_in_row(center, r);
                let hy = row_top(r) + c.track as f32 * lane_h;
                let col = if selected { egui::Color32::from_rgb(235, 245, 235) } else { egui::Color32::from_gray(200) };
                painter.rect_filled(
                    egui::Rect::from_min_max(egui::pos2(hx - 9.0, hy + 2.0), egui::pos2(hx + 9.0, hy + 6.0)),
                    1.5,
                    col,
                );
            }
            // In-clip selection overlay (Crop/Delete/Loop/Reverse target).
            if let Some((si, sa, sb)) = self.clip_sel {
                if si == ci {
                    let (sa, sb) = (sa.max(start), sb.min(end));
                    let sr0 = (sa / row_secs) as usize;
                    let sr1 = (((sb - 1e-4).max(sa)) / row_secs) as usize;
                    for r in sr0..=sr1.min(n_rows - 1) {
                        let seg_s = sa.max(r as f32 * row_secs);
                        let seg_e = sb.min((r as f32 + 1.0) * row_secs);
                        if seg_e <= seg_s {
                            continue;
                        }
                        let y0 = row_top(r) + c.track as f32 * lane_h;
                        let rr = egui::Rect::from_min_max(
                            egui::pos2(x_in_row(seg_s, r), y0 + 2.0),
                            egui::pos2(x_in_row(seg_e, r).max(x_in_row(seg_s, r) + 2.0), y0 + lane_h - 1.0),
                        );
                        painter.rect_filled(rr, 0.0, egui::Color32::from_rgba_unmultiplied(240, 230, 140, 70));
                        painter.rect_stroke(rr, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(240, 230, 140)));
                    }
                }
            }
        }

        // Playhead (in its row).
        {
            let pl = play * song;
            let r = ((pl / row_secs) as usize).min(n_rows - 1);
            let yr = egui::Rangef::new(row_top(r), row_top(r) + n as f32 * lane_h);
            painter.vline(x_in_row(pl, r), yr, egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(240, 240, 120)));
        }

        // ---- interaction ----
        // Snap to the configured grid (quarter note by default; Free = no snap).
        let beat_secs = (bar_secs / self.project.tempo.beats_per_bar.max(1) as f32).max(1e-4);
        let snap_len = self.snap.secs(bar_secs, beat_secs);
        let snap = |s: f32| match snap_len {
            Some(d) if d > 0.0 => (s / d).round() * d,
            _ => s,
        };
        // Smallest a clip may be trimmed to — the snap unit (or ~a 32nd when free).
        let min_len = snap_len.unwrap_or(beat_secs * 0.125).max(1e-3);
        let shift = ui.input(|i| i.modifiers.shift);
        if resp.drag_started() {
            // Use the press origin (where the mouse went down), not the current
            // pointer: egui only starts a drag after a few px of movement, and
            // that shift would otherwise pull an edge-grab into the clip body and
            // read as a move.
            let press = ui
                .input(|i| i.pointer.press_origin())
                .or_else(|| resp.interact_pointer_pos());
            if let Some(p) = press {
                if let Some((ci, kind)) = hit_clip(p) {
                    let c = &clips[ci];
                    let len = if c.length > 0.0 { c.length } else { (song - c.start).max(0.0) };
                    if kind == DragKind::Select {
                        // Begin a time selection within this clip.
                        let anchor = time_at(p).clamp(c.start, c.start + len);
                        self.arr_drag = Some((ci, kind, anchor, anchor));
                        self.clip_sel = Some((ci, anchor, anchor));
                        self.sel_clips = vec![ci];
                    } else {
                        self.arr_drag = Some((ci, kind, c.start, len));
                        self.arr_grab = (time_at(p) - c.start).clamp(0.0, len);
                        self.clip_sel = None; // moving/resizing, not selecting
                        if shift {
                            if let Some(k) = self.sel_clips.iter().position(|&x| x == ci) {
                                self.sel_clips.remove(k);
                            } else {
                                self.sel_clips.push(ci);
                            }
                        } else if !self.sel_clips.contains(&ci) {
                            self.sel_clips = vec![ci];
                        }
                    }
                    self.sel_track = Some(c.track);
                } else if in_seek_strip(p) {
                    self.arr_drag = None;
                    self.arr_seeking = true;
                    let _ = self.tx.send(Command::Seek(time_at(p)));
                } else {
                    self.arr_drag = None;
                }
            }
        }
        if resp.dragged() && self.arr_seeking {
            if let Some(p) = resp.interact_pointer_pos() {
                let _ = self.tx.send(Command::Seek(time_at(p)));
            }
        }
        if resp.dragged() {
            if let (Some((ci, kind, _, _)), Some(p)) = (self.arr_drag, resp.interact_pointer_pos()) {
                let c = &clips[ci];
                let orig_len = if c.length > 0.0 { c.length } else { (song - c.start).max(0.0) };
                // A sibling's occupied span on the same track (concrete length).
                let sib_end = |o: &ClipView| o.start + if o.length > 0.0 { o.length } else { tracks[o.track].period.max(1e-4) };
                match kind {
                    DragKind::ResizeR => {
                        // Right edge: keep start, grow/shrink length. Can't cross
                        // into the next clip on this track.
                        let mut limit = f32::INFINITY;
                        for (i, o) in clips.iter().enumerate() {
                            if i != ci && o.track == c.track && o.start >= c.start {
                                limit = limit.min(o.start);
                            }
                        }
                        let max_len = (limit - c.start).max(min_len);
                        let len = snap((time_at(p) - c.start).max(min_len)).min(max_len);
                        self.arr_drag = Some((ci, kind, c.start, len));
                    }
                    DragKind::ResizeL => {
                        // Left edge: keep the end fixed, move start. Can't cross
                        // into the previous clip on this track.
                        let end = c.start + orig_len;
                        let mut lo = 0.0_f32;
                        for (i, o) in clips.iter().enumerate() {
                            if i != ci && o.track == c.track && sib_end(o) <= end {
                                lo = lo.max(sib_end(o));
                            }
                        }
                        let start = snap(time_at(p)).clamp(lo, end - min_len);
                        self.arr_drag = Some((ci, kind, start, end - start));
                    }
                    DragKind::Move => {
                        // Slide the clip, but not through its neighbours on the
                        // same track (non-selected clips block it).
                        let sel = if self.sel_clips.contains(&ci) { self.sel_clips.clone() } else { vec![ci] };
                        let (mut lo, mut hi) = (0.0_f32, f32::INFINITY);
                        for (i, o) in clips.iter().enumerate() {
                            if o.track != c.track || sel.contains(&i) {
                                continue;
                            }
                            if sib_end(o) <= c.start {
                                lo = lo.max(sib_end(o));
                            } else if o.start >= c.start + orig_len {
                                hi = hi.min(o.start - orig_len);
                            }
                        }
                        // Keep the grabbed point under the cursor: offset the
                        // start by where the clip was grabbed, then snap.
                        let start = snap(time_at(p) - self.arr_grab).clamp(lo, hi.max(lo));
                        self.arr_drag = Some((ci, kind, start, orig_len));
                    }
                    DragKind::Select => {
                        // Drag out the in-clip time selection (snapped to beats).
                        let anchor = self.arr_drag.map(|d| d.2).unwrap_or_else(|| time_at(p));
                        let end_c = c.start + orig_len;
                        let cur = time_at(p).clamp(c.start, end_c);
                        self.arr_drag = Some((ci, kind, anchor, cur));
                        let a = snap(anchor.min(cur)).clamp(c.start, end_c);
                        let b = snap(anchor.max(cur)).clamp(c.start, end_c);
                        self.clip_sel = Some((ci, a, b));
                    }
                }
            }
        }
        if resp.drag_stopped() {
            self.arr_seeking = false;
            if let Some((ci, kind, start, len)) = self.arr_drag.take() {
                match kind {
                    DragKind::ResizeR => {
                        let _ = self.tx.send(Command::SetClip { index: ci, start, length: len });
                    }
                    DragKind::ResizeL => {
                        // Front trim: advance the loop offset by how far the left
                        // edge moved so the content stays anchored in place.
                        let c = &clips[ci];
                        let offset = c.offset + (start - c.start);
                        let _ = self.tx.send(Command::SetClipTrim { index: ci, start, length: len, offset });
                    }
                    DragKind::Move => {
                        let delta = start - clips[ci].start;
                        let sel = if self.sel_clips.contains(&ci) { self.sel_clips.clone() } else { vec![ci] };
                        for si in sel {
                            if let Some(c) = clips.get(si) {
                                let _ = self.tx.send(Command::SetClip {
                                    index: si,
                                    start: (c.start + delta).max(0.0),
                                    length: c.length,
                                });
                            }
                        }
                    }
                    // Selection is already stored in `clip_sel`; the action row acts on it.
                    DragKind::Select => {}
                }
            }
        } else if resp.double_clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                if hit_clip(p).is_none() {
                    if let Some(lane) = lane_at(p) {
                        // Default to one loop (the track's period) — a concrete block
                        // the user can then drag out to loop further.
                        let period = tracks.get(lane).map(|t| t.period).unwrap_or(0.0);
                        let _ = self.tx.send(Command::AddClip {
                            track: lane,
                            start: snap(time_at(p)).max(0.0),
                            length: period.max(0.0),
                        });
                    }
                }
            }
        } else if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                // A click on the seek strip moves the playhead; otherwise it
                // selects a clip if it lands on one, else the lane's track.
                if in_seek_strip(p) {
                    let _ = self.tx.send(Command::Seek(time_at(p)));
                } else if let Some((ci, _)) = hit_clip(p) {
                    if shift {
                        if let Some(k) = self.sel_clips.iter().position(|&x| x == ci) {
                            self.sel_clips.remove(k);
                        } else {
                            self.sel_clips.push(ci);
                        }
                    } else {
                        self.sel_clips = vec![ci];
                    }
                    // A plain click clears any in-clip range selection.
                    if !self.clip_sel.map(|s| s.0 == ci).unwrap_or(false) {
                        self.clip_sel = None;
                    }
                    self.sel_track = Some(clips[ci].track);
                } else if let Some(lane) = lane_at(p) {
                    self.sel_clips.clear();
                    self.clip_sel = None;
                    self.sel_track = Some(lane);
                }
            }
        }
        // Cursor hints: resize on the edges, grab/move on the handle.
        if let Some(p) = resp.hover_pos() {
            match hit_clip(p) {
                Some((_, DragKind::ResizeL)) | Some((_, DragKind::ResizeR)) => {
                    ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::ResizeHorizontal);
                }
                Some((_, DragKind::Move)) => {
                    ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Grab);
                }
                _ => {}
            }
        }
        // Keep the in-clip selection valid if clips changed.
        if let Some((ci, _, _)) = self.clip_sel {
            if ci >= clips.len() {
                self.clip_sel = None;
            }
        }
        ui.add_space(4.0);
    }

    /// Controls for the currently-selected track and clip, always on screen.
    fn contextual_panel(
        &mut self,
        ui: &mut egui::Ui,
        tracks: &[TrackView],
        clips: &[ClipView],
        play: f32,
        song_secs: f32,
    ) {
        // Keep the selection valid; default to the first track.
        if self.sel_track.map(|i| i >= tracks.len()).unwrap_or(true) {
            self.sel_track = (!tracks.is_empty()).then_some(0);
        }
        let Some(ti) = self.sel_track else { return };
        let t = &tracks[ti];

        // --- Track row ---
        let mut delete = false;
        ui.horizontal(|ui| {
            ui.strong(&t.name);
            ui.label(egui::RichText::new(format!("· {} · {:.2}s loop", t.instrument, t.period)).weak().small());
            ui.separator();
            let mute = if t.muted { "🔇" } else { "🔊" };
            if ui.button(mute).on_hover_text("Mute / unmute").clicked() {
                let _ = self.tx.send(Command::ToggleMute(ti));
            }
            if ui.add(egui::Button::new("S").selected(t.solo)).on_hover_text("Solo").clicked() {
                let _ = self.tx.send(Command::ToggleSolo(ti));
            }
            let targeting = self.edit_target == Target::Track(ti);
            if ui.add(egui::Button::new("✎ Instrument").selected(targeting)).on_hover_text("Edit this track's instrument in the right-hand panel").clicked() {
                self.set_target(Target::Track(ti), tracks);
            }
            if t.automation > 0
                && ui.button(format!("🎚 {}", t.automation)).on_hover_text("Clear recorded automation").clicked()
            {
                let _ = self.tx.send(Command::ClearTrackAutomation(ti));
            }
            if ui.button("🗑 Delete track").clicked() {
                delete = true;
            }
        });
        ui.horizontal(|ui| {
            let mut vol = t.volume;
            let mut pan = t.pan;
            let vc = ui.add(egui::Slider::new(&mut vol, 0.0..=1.5).text("Vol").clamping(egui::SliderClamping::Never)).changed();
            let pc = ui.add(egui::Slider::new(&mut pan, -1.0..=1.0).text("Pan")).changed();
            if vc || pc {
                let _ = self.tx.send(Command::SetTrackMix { track: ti, volume: vol, pan });
            }
            let mut fi = t.fade_in;
            let mut fo = t.fade_out;
            ui.separator();
            ui.label("Fade");
            let fic = ui.add(egui::DragValue::new(&mut fi).range(0.0..=10.0).speed(0.05).prefix("in ").suffix("s")).changed();
            let foc = ui.add(egui::DragValue::new(&mut fo).range(0.0..=10.0).speed(0.05).prefix("out ").suffix("s")).changed();
            if fic || foc {
                let _ = self.tx.send(Command::SetTrackFades { track: ti, fade_in: fi, fade_out: fo });
            }
        });

        // --- Clip controls ---
        if self.sel_clips.len() == 1 {
            if let Some(c) = clips.get(self.sel_clips[0]).cloned() {
                let ci = self.sel_clips[0];
                let beat = (60.0 / self.project.tempo.bpm.max(1.0)).max(1e-4);
                // Duplicate / delete.
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Clip:").small());
                    if ui.button("Duplicate →").on_hover_text("Independent copy at the playhead").clicked() {
                        let _ = self.tx.send(Command::DuplicateClip { index: ci, dest: play * song_secs.max(0.001) });
                    }
                    if ui.button("🗑 Delete clip").clicked() {
                        let _ = self.tx.send(Command::RemoveClip { index: ci });
                        self.sel_clips.clear();
                        self.clip_sel = None;
                    }
                });
                // Loop config: on/off, loop length (beats), repeats, flatten.
                ui.horizontal(|ui| {
                    let mut looping = c.looping;
                    if ui
                        .checkbox(&mut looping, "🔁 Loop")
                        .on_hover_text("Repeat the content across the clip; off = play once, then silence")
                        .changed()
                    {
                        let _ = self.tx.send(Command::SetClipLoop { index: ci, looping, loop_len: c.loop_len });
                    }
                    if c.looping {
                        let mut unit = (c.loop_len / beat).max(0.25);
                        if ui
                            .add(egui::DragValue::new(&mut unit).range(0.25..=256.0).speed(0.25).suffix(" beat"))
                            .on_hover_text("Loop length — the repeating unit")
                            .changed()
                        {
                            let _ = self.tx.send(Command::SetClipLoop { index: ci, looping: true, loop_len: unit * beat });
                        }
                        let mut reps = (c.length / c.loop_len.max(1e-4)).round().max(1.0) as i32;
                        if ui
                            .add(egui::DragValue::new(&mut reps).range(1..=512).prefix("×"))
                            .on_hover_text("Repeats — clip length in loop units")
                            .changed()
                        {
                            let _ = self.tx.send(Command::SetClip { index: ci, start: c.start, length: reps as f32 * c.loop_len.max(1e-4) });
                        }
                        if ui.button("Flatten").on_hover_text("Bake the repeats into one raw clip and turn looping off").clicked() {
                            let _ = self.tx.send(Command::FlattenClip { index: ci });
                        }
                    }
                    ui.label(egui::RichText::new(format!("{:.2}s", c.length)).weak().small());
                    if c.unique {
                        ui.label(egui::RichText::new("· own notes").weak().small());
                    }
                });
                // Actions on the in-clip time selection.
                if let Some((si, sa, sb)) = self.clip_sel {
                    if si == ci && sb > sa {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("selection ⟦{:.2}–{:.2}s⟧:", sa, sb)).small());
                            if ui.button("Loop").on_hover_text("Make the selection the clip's loop unit").clicked() {
                                let _ = self.tx.send(Command::LoopClipRange { index: ci, a: sa, b: sb });
                                self.clip_sel = None;
                            }
                            if ui.button("Crop").on_hover_text("Keep only the selection (independent clip)").clicked() {
                                let _ = self.tx.send(Command::CropClip { index: ci, a: sa, b: sb });
                                self.clip_sel = None;
                            }
                            if ui.button("Delete").on_hover_text("Cut the selection — splits into two clips with a gap").clicked() {
                                let _ = self.tx.send(Command::SplitDeleteClip { index: ci, a: sa, b: sb });
                                self.clip_sel = None;
                                self.sel_clips.clear();
                            }
                            if ui.button("Reverse").on_hover_text("Reverse the notes in the selection").clicked() {
                                let _ = self.tx.send(Command::ReverseClipRange { index: ci, a: sa, b: sb });
                                self.clip_sel = None;
                            }
                        });
                    }
                } else {
                    ui.label(egui::RichText::new("drag across the clip to select a range → Crop / Delete / Loop / Reverse").weak().small());
                }
            }
        } else if self.sel_clips.len() > 1 {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{} clips selected", self.sel_clips.len())).small());
                if ui.button("🗑 Delete clips").clicked() {
                    let mut idxs = self.sel_clips.clone();
                    idxs.sort_unstable();
                    for i in idxs.into_iter().rev() {
                        let _ = self.tx.send(Command::RemoveClip { index: i });
                    }
                    self.sel_clips.clear();
                }
            });
        }

        if delete {
            let _ = self.tx.send(Command::DeleteTrack(ti));
            if self.edit_target == Target::Track(ti) {
                self.edit_target = Target::Live;
            }
            self.sel_track = None;
            self.sel_clips.clear();
            self.clip_sel = None;
        }
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

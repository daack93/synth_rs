//! FTM Synth — a desktop re-implementation of a 2014 embedded "electric air
//! guitar". Physical string parameters shape the timbre; an on-screen piano,
//! the computer keyboard, and a MIDI controller all play it.

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
use project::{NamedLoop, Project, TempoGrid, ZoneData};
use studio::{
    ClipView, Command, LiveConfig, LooperMode, NoteSpan, PlayMode, RegionOp, SharedView, TrackView,
    TransportState,
};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([880.0, 620.0])
            .with_title("FTM Synth — physical-model string"),
        ..Default::default()
    };
    eframe::run_native(
        "FTM Synth",
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
    /// Live "kit mode": the keyboard is split/mapped across several instruments.
    live_is_kit: bool,
    /// Zones for the live kit (each an instrument mapped to a key range).
    kit_zones: Vec<ZoneData>,

    // Looper
    looper_mode: LooperMode,

    // Preset library
    /// Presets found in the folder (refreshed on save/load/delete).
    preset_list: Vec<Preset>,
    /// Name field for saving the current sound.
    preset_name: String,
    /// Last preset action result, shown in the UI.
    preset_status: String,

    // Project (loop library)
    /// The in-memory project (its loops).
    project: Project,
    /// Project name input.
    project_name: String,
    /// Name input for the loop being added.
    loop_name: String,
    /// Saved project names on disk.
    project_list: Vec<String>,
    /// Last project action result.
    project_status: String,

    /// Master output level (linear).
    master_volume: f32,
    /// Transport play mode: false = Loop (all tracks from 0), true = Arrange.
    arrange_mode: bool,

    // Whammy (pitch-bend lever), semitones + configurable range.
    whammy: f32,
    whammy_down: f32,
    whammy_up: f32,

    // Region editing (chop / crop / rearrange)
    /// Current timeline selection: (track index, start secs, end secs).
    sel: Option<(usize, f32, f32)>,
    /// Drag anchor (loop fraction) while dragging out a selection.
    drag_start: Option<f32>,
    /// The track whose controls the contextual panel shows.
    sel_track: Option<usize>,
    /// The track whose expanded editor section is open (via ✎ Edit).
    edit_track_open: Option<usize>,
    /// Selected clip indices in the arrangement editor.
    sel_clips: Vec<usize>,
    /// Active clip drag: (clip index, is_resize, preview_start_s, preview_len_s).
    arr_drag: Option<(usize, bool, f32, f32)>,
    /// Destination time (secs) for duplicate / move.
    region_dest: f32,
    /// Note-edit params for the selection: transpose semitones, velocity factor.
    region_transpose: i32,
    region_vel: f32,

    // Export
    export_name: String,
    export_sr: u32,
    export_repeats: u32,
    export_tail: f32,
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
        let preset_list = presets::seed_if_empty();

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
            looper_mode: LooperMode::Pedal,
            preset_list,
            preset_name: String::new(),
            preset_status: String::new(),
            project: Project {
                name: "Untitled".to_string(),
                loops: Vec::new(),
                tempo: TempoGrid::default(),
            },
            project_name: "Untitled".to_string(),
            loop_name: String::new(),
            project_list: project::list(),
            project_status: String::new(),
            master_volume: 1.0,
            arrange_mode: true,
            whammy: 0.0,
            whammy_down: 12.0,
            whammy_up: 2.0,
            sel: None,
            drag_start: None,
            sel_track: None,
            edit_track_open: None,
            sel_clips: Vec::new(),
            arr_drag: None,
            region_dest: 0.0,
            region_transpose: 0,
            region_vel: 1.0,
            export_name: "take".to_string(),
            export_sr: 48_000,
            export_repeats: 2,
            export_tail: 1.0,
            export_hi_res: false,
            export_status: String::new(),
            base_midi: 60, // C4
            mouse_note: None,
            held_keys: HashMap::new(),
            midi_ports,
            midi_sel: None,
            _midi: None,
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

    /// Capture the studio's current loop into the project as a named loop.
    fn add_current_loop(&mut self) {
        let data = match &self.view {
            Some(v) => v.snapshot(),
            None => return,
        };
        if data.is_empty() {
            self.project_status = "No loop to add — record one first.".into();
            return;
        }
        let name = if self.loop_name.trim().is_empty() {
            format!("Loop {}", self.project.loops.len() + 1)
        } else {
            self.loop_name.trim().to_string()
        };
        let tracks = data.tracks.len();
        self.project.loops.push(NamedLoop { name, data });
        self.loop_name.clear();
        self.project_status = format!("Added loop ({tracks} tracks).");
    }

    fn save_project(&mut self) {
        let name = self.project_name.trim();
        self.project.name = if name.is_empty() { "Untitled".into() } else { name.to_string() };
        self.project_name = self.project.name.clone();
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
                self.project_status =
                    format!("Loaded “{}” ({} loops).", p.name, p.loops.len());
                let _ = self.tx.send(Command::SetTempo(p.tempo));
                self.project = p;
            }
            Err(e) => self.project_status = format!("Load failed: {e}"),
        }
    }

    fn load_loop_into_studio(&mut self, i: usize) {
        if let Some(nl) = self.project.loops.get(i) {
            let _ = self.tx.send(Command::LoadLoop(nl.data.clone()));
            self.edit_target = Target::Live;
            self.track_edit = None;
            self.project_status = format!("Loaded loop “{}”.", nl.name);
        }
    }

    /// Remove a loop and keep the arrangement's indices valid.
    fn remove_loop(&mut self, i: usize) {
        if i >= self.project.loops.len() {
            return;
        }
        self.project.loops.remove(i);
    }

    fn connect_midi(&mut self, index: usize) {
        match midi::connect(index, self.tx.clone()) {
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

        self.handle_computer_keyboard(ctx);

        egui::TopBottomPanel::top("presets").show(ctx, |ui| {
            self.presets_bar(ui);
        });

        egui::TopBottomPanel::bottom("project").show(ctx, |ui| {
            self.project_panel(ui);
        });

        egui::SidePanel::right("params")
            .resizable(false)
            .min_width(300.0)
            .show(ctx, |ui| {
                self.params_panel(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.heading("FTM Synth");
            ui.label(
                "A modal physical-modeling synth (Function Transformation Method). \
                 Pick a synthesis model on the right and tweak its parameters; \
                 the keyboard plays it.",
            );
            ui.add_space(6.0);

            if let Some(err) = &self.audio_err {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 90, 90),
                    format!("Audio unavailable: {err}"),
                );
            }

            ui.horizontal(|ui| {
                ui.label("Octave:");
                if ui.button("–").clicked() {
                    self.base_midi = (self.base_midi - 12).max(0);
                }
                ui.label(format!("C{}", self.base_midi / 12 - 1));
                if ui.button("+").clicked() {
                    self.base_midi = (self.base_midi + 12).min(108);
                }
                ui.separator();
                ui.label("Type on the keyboard (A W S E D F T G Y H U J K), Z/X shift octave.");
            });

            ui.add_space(10.0);
            self.transport_bar(ui);

            ui.add_space(10.0);
            ui.horizontal(|ui| {
                self.whammy_bar(ui);
                self.piano(ui);
            });

            ui.add_space(8.0);
            self.tracks_panel(ui);

            ui.add_space(10.0);
            self.midi_panel(ui);
            });
        });
    }
}

impl App {
    /// The project bar: save/load a project, add the current loop, and the loop
    /// list (load a saved loop back into the studio).
    fn project_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.strong("Project");
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
            if ui.button("New").clicked() {
                self.project = Project {
                    name: "Untitled".into(),
                    loops: Vec::new(),
                    tempo: self.project.tempo,
                };
                self.project_name = "Untitled".into();
                self.project_status = "New project.".into();
            }
            if ui.button("⟳").on_hover_text("Rescan project folder").clicked() {
                self.project_list = project::list();
            }
        });

        ui.horizontal(|ui| {
            ui.label("Add current loop:");
            ui.add(
                egui::TextEdit::singleline(&mut self.loop_name)
                    .hint_text("loop name")
                    .desired_width(150.0),
            );
            if ui
                .button("＋ Add loop")
                .on_hover_text("Capture the loop currently in the tracks below into this project")
                .clicked()
            {
                self.add_current_loop();
            }
        });

        ui.separator();
        if self.project.loops.is_empty() {
            ui.label(
                egui::RichText::new("No loops yet — record a loop, then “Add loop”.")
                    .weak()
                    .small(),
            );
        } else {
            let mut load = None;
            let mut remove = None;
            egui::ScrollArea::vertical()
                .max_height(110.0)
                .show(ui, |ui| {
                    for (i, nl) in self.project.loops.iter().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.button("▶").on_hover_text("Load this loop").clicked() {
                                load = Some(i);
                            }
                            ui.label(format!("{}. {}", i + 1, nl.name));
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} tracks · {} notes · {:.1}s",
                                    nl.data.tracks.len(),
                                    nl.data.note_count(),
                                    nl.data.length
                                ))
                                .weak()
                                .small(),
                            );
                            if ui.button("🗑").on_hover_text("Remove from project").clicked() {
                                remove = Some(i);
                            }
                        });
                    }
                });
            if let Some(i) = load {
                self.load_loop_into_studio(i);
            }
            if let Some(i) = remove {
                self.remove_loop(i);
            }
        }

        self.export_ui(ui);

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
            ui.checkbox(&mut self.export_hi_res, "Hi-res")
                .on_hover_text("Max out the horn eigensolve resolution for the render (higher rate already admits more modes).");
        });
        ui.horizontal(|ui| {
            ui.label("Loop ×");
            ui.add(egui::DragValue::new(&mut self.export_repeats).range(1..=256));
            ui.label("Tail");
            ui.add(egui::DragValue::new(&mut self.export_tail).range(0.0..=10.0).speed(0.1).suffix(" s"));
            if ui
                .button("⬇ Export song")
                .on_hover_text("Render the whole arrangement to a WAV")
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
        match export::render_loop_to_wav(
            data,
            sr,
            self.export_repeats,
            self.export_tail,
            self.export_hi_res,
            &path,
        ) {
            Ok(()) => self.export_status = format!("Wrote {}", path.display()),
            Err(e) => self.export_status = format!("Export failed: {e}"),
        }
    }


    /// The instrument-library bar: name + save, and load/delete of saved presets.
    fn presets_bar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.strong("Presets");
            ui.separator();

            ui.label("Name:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.preset_name)
                    .hint_text("instrument name")
                    .desired_width(160.0),
            );
            let save_on_enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui.button("💾 Save").clicked() || save_on_enter {
                self.save_preset();
            }

            ui.separator();

            // Load: pick from the saved presets.
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
                        let model_name = self
                            .models
                            .iter()
                            .find(|m| m.id() == p.model_id)
                            .map(|m| m.display_name())
                            .unwrap_or(p.model_id.as_str());
                        if ui
                            .selectable_label(false, format!("{}  ·  {}", p.name, model_name))
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

            // Delete the preset matching the current name field.
            let can_delete = self
                .preset_list
                .iter()
                .any(|p| p.name.eq_ignore_ascii_case(self.preset_name.trim()));
            if ui
                .add_enabled(can_delete, egui::Button::new("🗑 Delete"))
                .on_hover_text("Delete the saved preset with this name")
                .clicked()
            {
                let name = self.preset_name.trim().to_string();
                self.delete_preset(&name);
            }

            if ui.button("⟳").on_hover_text("Rescan preset folder").clicked() {
                self.preset_list = presets::list();
            }
            if ui
                .button("★ Factory")
                .on_hover_text("Restore the built-in instrument kit (overwrites same-named presets)")
                .clicked()
            {
                self.preset_list = presets::restore_factory();
                self.preset_status = "Restored factory presets.".into();
            }
        });
        if !self.preset_status.is_empty() {
            ui.label(egui::RichText::new(&self.preset_status).weak().small());
        }
        ui.add_space(2.0);
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

                        // --- Key range + pad / transpose ---
                        ui.horizontal(|ui| {
                            ui.label("Keys");
                            changed |= ui.add(egui::DragValue::new(&mut z.lo).range(0..=127)).changed();
                            ui.label("–");
                            changed |= ui.add(egui::DragValue::new(&mut z.hi).range(0..=127)).changed();
                            ui.label(
                                egui::RichText::new(format!("{}–{}", note_name(z.lo), note_name(z.hi)))
                                    .weak(),
                            );
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
                                changed |= ui.add(egui::DragValue::new(fixed).range(0..=127)).changed();
                                ui.label(egui::RichText::new(note_name(*fixed)).weak());
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

    /// Transport controls: looper mode, state, and pedal buttons.
    fn transport_bar(&mut self, ui: &mut egui::Ui) {
        let (state, loop_secs) = self
            .view
            .as_ref()
            .map(|v| (v.state(), v.loop_seconds(self.sample_rate)))
            .unwrap_or((TransportState::Idle, 0.0));

        ui.horizontal(|ui| {
            ui.strong("Looper");

            let mut mode = self.looper_mode;
            egui::ComboBox::from_id_salt("looper_mode")
                .selected_text(match mode {
                    LooperMode::Pedal => "Pedal cycle",
                    LooperMode::Overdub => "Overdub",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut mode, LooperMode::Pedal, "Pedal cycle");
                    ui.selectable_value(&mut mode, LooperMode::Overdub, "Overdub");
                });
            if mode != self.looper_mode {
                self.looper_mode = mode;
                let _ = self.tx.send(Command::SetLooperMode(mode));
            }

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
            if ui
                .button("⏺ Rec / Play")
                .on_hover_text("Idle → record the first loop → play → record over into a new track → tap again to finish. Each recorded take becomes a track.")
                .clicked()
            {
                let _ = self.tx.send(Command::Tap);
            }
            if ui.button("⏹ Stop").clicked() {
                let _ = self.tx.send(Command::Stop);
            }
            if ui.button("⟲ Reset").on_hover_text("Clear all tracks").clicked() {
                let _ = self.tx.send(Command::Reset);
            }
            if self.looper_mode == LooperMode::Pedal && loop_secs > 0.0 {
                if ui
                    .button("＋ Rec track")
                    .on_hover_text("Record another track. In 🎬 Arrange mode it punches in at the playhead; in 🔁 Loop mode it records from the top.")
                    .clicked()
                {
                    let _ = self.tx.send(Command::ArmOverdub);
                }
            }
        });

        let hint = match self.looper_mode {
            LooperMode::Pedal => "⏺ Rec/Play: first tap records the base loop, next plays it, next records over into a new track. Each take = a new track shown as a clip in the arrangement.",
            LooperMode::Overdub => "⏺ Rec/Play: first tap records the base track, then each tap layers another track. Every take shows up as a clip.",
        };
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

            ui.label("Loop");
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
                .on_hover_text("Fixed loop length in bars (Free = the take sets the length).")
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
                .on_hover_text("Play one bar of clicks before a fixed-bars recording.")
                .changed();
        });
        if changed {
            let _ = self.tx.send(Command::SetTempo(self.project.tempo));
        }
    }

    /// The recorded loop tracks, shown below the keyboard.
    fn tracks_panel(&mut self, ui: &mut egui::Ui) {
        let (tracks, play, loop_secs, clips, arrange_mode) = match &self.view {
            Some(v) => (
                v.tracks(),
                v.play_fraction(),
                v.loop_seconds(self.sample_rate),
                v.arrangement(),
                v.is_arrange_mode(),
            ),
            None => return,
        };

        ui.horizontal(|ui| {
            ui.strong("Tracks");
            ui.label(egui::RichText::new(format!("({})", tracks.len())).weak());
            ui.separator();
            // Play mode: Loop (build beats, all from 0) vs Arrange (play clips).
            let mut arrange = self.arrange_mode;
            if ui.selectable_label(!arrange, "🔁 Loop").on_hover_text("Play all tracks looping from the start").clicked() {
                arrange = false;
            }
            if ui.selectable_label(arrange, "🎬 Arrange").on_hover_text("Play the clip arrangement").clicked() {
                arrange = true;
            }
            if arrange != self.arrange_mode {
                self.arrange_mode = arrange;
                let _ = self.tx.send(Command::SetPlayMode(if arrange {
                    PlayMode::Arrange
                } else {
                    PlayMode::Loop
                }));
            }
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
            if !tracks.is_empty() {
                ui.separator();
                ui.label("Stretch");
                if ui.button("½×").on_hover_text("Halve the loop time (faster)").clicked() {
                    let _ = self.tx.send(Command::TimeStretch(0.5));
                }
                if ui.button("2×").on_hover_text("Double the loop time (slower)").clicked() {
                    let _ = self.tx.send(Command::TimeStretch(2.0));
                }
            }
        });
        if tracks.is_empty() {
            ui.label(
                egui::RichText::new("No tracks yet. In 🔁 Loop mode, hit ⏺ Record (or ＋Rec track) to record a loop — it becomes a track shown here as a clip.")
                    .weak()
                    .small(),
            );
            return;
        }

        self.arrangement_editor(ui, &tracks, &clips, loop_secs, play, arrange_mode);
        self.seek_bar(ui, loop_secs, play);
        ui.separator();

        // Controls for the selected track / clip.
        self.contextual_panel(ui, &tracks, &clips, play, loop_secs);

        // Expanded editor for the track being edited (via ✎ Edit).
        if let Some(i) = self.edit_track_open {
            if i < tracks.len() {
                self.track_editor_section(ui, i, &tracks);
            } else {
                self.edit_track_open = None;
            }
        }
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
        arrange_mode: bool,
    ) {
        if tracks.is_empty() {
            return;
        }
        self.sel_clips.retain(|&i| i < clips.len());
        let n = tracks.len();
        let song = song_secs.max(0.001);
        ui.horizontal(|ui| {
            ui.strong("Arrangement");
            let hint = if arrange_mode {
                "click to move the playback cursor · drag a clip to move · right edge to resize · dbl-click empty lane to place"
            } else {
                "(Loop mode — playback ignores placement; switch to 🎬 Arrange to hear it)"
            };
            ui.label(egui::RichText::new(hint).weak().small());
        });

        let lane_h = 22.0;
        let row_gap = 12.0;
        let label_w = 66.0;
        let px_per_bar = 84.0;
        let bar_secs =
            60.0 / self.project.tempo.bpm.max(1.0) * self.project.tempo.beats_per_bar.max(1) as f32;
        let avail = ui.available_width().max(360.0);
        let tl_w = (avail - label_w - 8.0).max(60.0);
        let bars_per_row = ((tl_w / px_per_bar).floor() as usize).max(1);
        let row_secs = (bars_per_row as f32 * bar_secs).max(0.001);
        let n_rows = ((song / row_secs).ceil() as usize).clamp(1, 200);
        let row_h = n as f32 * lane_h + row_gap;
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
        // Pointer → absolute time (any row); and → (lane, time) when over a lane.
        let time_at = |p: egui::Pos2| -> f32 {
            let rel = (p.y - rect.top()).max(0.0);
            let r = ((rel / row_h) as usize).min(n_rows - 1);
            (r as f32 * row_secs + ((p.x - tl_x) / tl_w).clamp(0.0, 1.0) * row_secs).clamp(0.0, song)
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
        let hit_clip = |p: egui::Pos2| -> Option<(usize, bool)> {
            let lane = lane_at(p)?;
            let t = time_at(p);
            let resize_secs = (6.0 / tl_w) * row_secs;
            for (ci, c) in clips.iter().enumerate().rev() {
                if c.track != lane {
                    continue;
                }
                let len = if c.length > 0.0 { c.length } else { (song - c.start).max(0.0) };
                if t >= c.start && t <= c.start + len {
                    return Some((ci, t >= c.start + len - resize_secs));
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
        }

        // Clips (drawn as one segment per row they span), with their notes.
        for (ci, c) in clips.iter().enumerate() {
            if c.track >= n {
                continue;
            }
            let t = &tracks[c.track];
            let default_len = if c.length > 0.0 { c.length } else { (song - c.start).max(0.0) };
            let (start, len) = match self.arr_drag {
                Some((di, resize, ps, pl)) if di == ci => (ps, if resize { pl } else { default_len }),
                Some((di, false, ps, _)) if self.sel_clips.contains(&ci) && self.sel_clips.contains(&di) => {
                    let delta = ps - clips.get(di).map(|d| d.start).unwrap_or(0.0);
                    ((c.start + delta).max(0.0), default_len)
                }
                _ => (c.start, default_len),
            };
            let end = start + len;
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
                }
            }
            // Notes, repeated across the clip's length.
            let period = t.period.max(1e-6);
            let reps = ((len / period).ceil() as i32).clamp(1, 512);
            for rep in 0..reps {
                let base = start + rep as f32 * period;
                for nsp in &t.notes {
                    let ns = base + nsp.start * period;
                    if ns >= end {
                        continue;
                    }
                    let r = ((ns / row_secs) as usize).min(n_rows - 1);
                    let ne = (base + nsp.end * period).min(end).min((r as f32 + 1.0) * row_secs);
                    let y = row_top(r) + c.track as f32 * lane_h + lane_h * 0.5;
                    painter.line_segment(
                        [egui::pos2(x_in_row(ns, r), y), egui::pos2(x_in_row(ne, r).max(x_in_row(ns, r) + 1.0), y)],
                        egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(190, 215, 255)),
                    );
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
        let snap = |s: f32| if bar_secs > 0.0 { (s / bar_secs).round() * bar_secs } else { s };
        let shift = ui.input(|i| i.modifiers.shift);
        if resp.drag_started() {
            if let Some(p) = resp.interact_pointer_pos() {
                if let Some((ci, resize)) = hit_clip(p) {
                    let c = &clips[ci];
                    let len = if c.length > 0.0 { c.length } else { (song - c.start).max(0.0) };
                    self.arr_drag = Some((ci, resize, c.start, len));
                    if shift {
                        if let Some(k) = self.sel_clips.iter().position(|&x| x == ci) {
                            self.sel_clips.remove(k);
                        } else {
                            self.sel_clips.push(ci);
                        }
                    } else if !self.sel_clips.contains(&ci) {
                        self.sel_clips = vec![ci];
                    }
                    self.sel_track = Some(c.track);
                } else {
                    self.arr_drag = None;
                }
            }
        }
        if resp.dragged() {
            if let (Some((ci, resize, _, _)), Some(p)) = (self.arr_drag, resp.interact_pointer_pos()) {
                if resize {
                    let len = snap((time_at(p) - clips[ci].start).max(bar_secs.max(0.05)));
                    self.arr_drag = Some((ci, true, clips[ci].start, len));
                } else {
                    let len = self.arr_drag.map(|d| d.3).unwrap_or(0.0);
                    self.arr_drag = Some((ci, false, snap(time_at(p)).max(0.0), len));
                }
            }
        }
        if resp.drag_stopped() {
            if let Some((ci, resize, start, len)) = self.arr_drag.take() {
                if resize {
                    let _ = self.tx.send(Command::SetClip { index: ci, start, length: len });
                } else {
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
            }
        } else if resp.double_clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                if hit_clip(p).is_none() {
                    if let Some(lane) = lane_at(p) {
                        let _ = self.tx.send(Command::AddClip {
                            track: lane,
                            start: snap(time_at(p)).max(0.0),
                            length: 0.0,
                        });
                    }
                }
            }
        } else if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                // A click selects: a clip if it lands on one, else the lane's track.
                if let Some((ci, _)) = hit_clip(p) {
                    if shift {
                        if let Some(k) = self.sel_clips.iter().position(|&x| x == ci) {
                            self.sel_clips.remove(k);
                        } else {
                            self.sel_clips.push(ci);
                        }
                    } else {
                        self.sel_clips = vec![ci];
                    }
                    self.sel_track = Some(clips[ci].track);
                } else if let Some(lane) = lane_at(p) {
                    self.sel_clips.clear();
                    self.sel_track = Some(lane);
                }
            }
        }
        let _ = time_at; // (seeking lives in the seek bar below)
        ui.add_space(4.0);
    }

    /// A thin scrubber below the arrangement: click or drag to move the playhead
    /// across the whole song.
    fn seek_bar(&mut self, ui: &mut egui::Ui, song_secs: f32, play: f32) {
        let song = song_secs.max(0.001);
        let width = ui.available_width().max(120.0);
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, 14.0), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, egui::Color32::from_gray(34));
        // Bar ticks.
        let bar_secs =
            60.0 / self.project.tempo.bpm.max(1.0) * self.project.tempo.beats_per_bar.max(1) as f32;
        if bar_secs > 0.0 {
            let mut b = 0.0;
            while b <= song && (b / bar_secs) < 512.0 {
                let x = rect.left() + (b / song) * rect.width();
                painter.vline(x, rect.y_range(), egui::Stroke::new(1.0_f32, egui::Color32::from_gray(48)));
                b += bar_secs;
            }
        }
        // Playhead marker.
        let px = rect.left() + play.clamp(0.0, 1.0) * rect.width();
        painter.vline(px, rect.y_range(), egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(240, 240, 120)));
        if resp.clicked() || resp.dragged() {
            if let Some(p) = resp.interact_pointer_pos() {
                let frac = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                let _ = self.tx.send(Command::Seek(frac * song));
            }
        }
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
            let editing = self.edit_track_open == Some(ti);
            if ui.add(egui::Button::new("✎ Edit").selected(editing)).on_hover_text("Edit this track's notes + instrument").clicked() {
                if editing {
                    self.edit_track_open = None;
                } else {
                    self.edit_track_open = Some(ti);
                    self.set_target(Target::Track(ti), tracks);
                }
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

        // --- Clip row ---
        if self.sel_clips.len() == 1 {
            if let Some(c) = clips.get(self.sel_clips[0]).cloned() {
                let ci = self.sel_clips[0];
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Clip:").small());
                    let mut tr = c.transpose;
                    let mut vl = c.vel;
                    let ca = ui.add(egui::DragValue::new(&mut tr).range(-24..=24).suffix(" st")).on_hover_text("Transpose (linked to the loop)").changed();
                    let cb = ui.add(egui::DragValue::new(&mut vl).range(0.0..=2.0).speed(0.02).prefix("×")).on_hover_text("Velocity").changed();
                    if ca || cb {
                        let _ = self.tx.send(Command::SetClipLayer { index: ci, transpose: tr, vel: vl });
                    }
                    if c.unique {
                        ui.label(egui::RichText::new("🔓 unique").small());
                    } else if ui.button("Make unique").on_hover_text("Fork this clip's notes for independent editing").clicked() {
                        let _ = self.tx.send(Command::MakeClipUnique { index: ci });
                    }
                    ui.separator();
                    if ui.button("Duplicate →").on_hover_text("Copy to the playhead").clicked() {
                        let _ = self.tx.send(Command::DuplicateClip { index: ci, dest: play * song_secs.max(0.001) });
                    }
                    if ui.button("🗑 Delete clip").clicked() {
                        let _ = self.tx.send(Command::RemoveClip { index: ci });
                        self.sel_clips.clear();
                    }
                });
                // Per-clip chop, using a range selected in the track editor (✎).
                if let Some((st, sa, sb)) = self.sel {
                    if st == c.track {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("chop this clip ⟦{sa:.2}–{sb:.2}s⟧:")).small());
                            if ui.button("Crop").clicked() {
                                let _ = self.tx.send(Command::ClipRegionEdit { index: ci, op: RegionOp::Keep { a: sa, b: sb } });
                            }
                            if ui.button("Delete range").clicked() {
                                let _ = self.tx.send(Command::ClipRegionEdit { index: ci, op: RegionOp::Delete { a: sa, b: sb } });
                            }
                            if ui.button("Reverse").clicked() {
                                let _ = self.tx.send(Command::ClipRegionEdit { index: ci, op: RegionOp::Reverse { a: sa, b: sb } });
                            }
                        });
                    }
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
            if self.edit_track_open == Some(ti) {
                self.edit_track_open = None;
            }
            if self.edit_target == Target::Track(ti) {
                self.edit_target = Target::Live;
            }
            self.sel_track = None;
            self.sel_clips.clear();
        }
    }

    /// The expanded editor for one track: its loop's note timeline + chop tools
    /// (its instrument is edited in the right-hand panel).
    fn track_editor_section(&mut self, ui: &mut egui::Ui, i: usize, tracks: &[TrackView]) {
        let t = &tracks[i];
        let period = t.period.max(1e-6);
        ui.separator();
        ui.horizontal(|ui| {
            ui.strong(format!("Editing loop: {}", t.name));
            ui.label(egui::RichText::new("drag across the loop to select a range, then chop below · instrument on the right").weak().small());
        });
        // Note timeline for this track's loop, with drag-to-select.
        let pos_secs = self.view.as_ref().map(|v| v.play_fraction() * v.loop_seconds(self.sample_rate)).unwrap_or(0.0);
        let track_play = (pos_secs % period) / period;
        let sel_frac = self.sel.filter(|s| s.0 == i).map(|(_, a, b)| (a / period, b / period));
        let resp = draw_track_timeline(ui, &t.notes, track_play, t.muted, sel_frac);
        let w = resp.rect.width().max(1.0);
        let frac_at = |x: f32| ((x - resp.rect.left()) / w).clamp(0.0, 1.0);
        if resp.drag_started() {
            if let Some(p) = resp.interact_pointer_pos() {
                self.drag_start = Some(frac_at(p.x));
            }
        }
        if resp.dragged() {
            if let (Some(st), Some(p)) = (self.drag_start, resp.interact_pointer_pos()) {
                let cur = frac_at(p.x);
                self.sel = Some((i, st.min(cur) * period, st.max(cur) * period));
            }
        }
        if resp.drag_stopped() {
            self.drag_start = None;
        }
        // Chop toolbar (targets this track's loop, or the selected clip if unique).
        if self.sel.map(|s| s.0) == Some(i) {
            self.region_ops_row(ui, i, period);
        }
    }

    /// The chop/crop/rearrange toolbar shown under the selected track.
    fn region_ops_row(&mut self, ui: &mut egui::Ui, i: usize, track_secs: f32) {
        let Some((_, a, b)) = self.sel else { return };
        let beat = 60.0 / self.project.tempo.bpm.max(1.0);
        ui.horizontal(|ui| {
            ui.add_space(28.0);
            ui.label(egui::RichText::new(format!("⟦{a:.2}–{b:.2}s⟧")).small());
            if ui.button("Crop").on_hover_text("Keep only the selection").clicked() {
                let _ = self.tx.send(Command::RegionEdit { track: i, op: RegionOp::Keep { a, b } });
            }
            if ui.button("Delete").on_hover_text("Delete the selection").clicked() {
                let _ = self.tx.send(Command::RegionEdit { track: i, op: RegionOp::Delete { a, b } });
            }
            if ui.button("Dup→").on_hover_text("Duplicate right after the selection").clicked() {
                let _ = self
                    .tx
                    .send(Command::RegionEdit { track: i, op: RegionOp::Duplicate { a, b, dest: b } });
            }
            ui.separator();
            ui.label("dest");
            ui.add(
                egui::DragValue::new(&mut self.region_dest)
                    .range(0.0..=track_secs)
                    .speed(0.01)
                    .suffix(" s"),
            );
            let dest = self.region_dest;
            if ui.button("Dup→dest").clicked() {
                let _ = self
                    .tx
                    .send(Command::RegionEdit { track: i, op: RegionOp::Duplicate { a, b, dest } });
            }
            if ui.button("Move→dest").clicked() {
                let _ = self
                    .tx
                    .send(Command::RegionEdit { track: i, op: RegionOp::Move { a, b, dest } });
            }
            ui.separator();
            if ui.button("◀").on_hover_text("Nudge whole track left one beat").clicked() {
                let _ = self
                    .tx
                    .send(Command::RegionEdit { track: i, op: RegionOp::Shift { delta: -beat } });
            }
            if ui.button("▶").on_hover_text("Nudge whole track right one beat").clicked() {
                let _ = self
                    .tx
                    .send(Command::RegionEdit { track: i, op: RegionOp::Shift { delta: beat } });
            }
            if ui.button("✕ sel").clicked() {
                self.sel = None;
            }
        });
        // Second row: note edits on the selection.
        ui.horizontal(|ui| {
            ui.add_space(28.0);
            ui.label(egui::RichText::new("notes:").small());
            ui.add(egui::DragValue::new(&mut self.region_transpose).range(-24..=24).suffix(" st"));
            if ui.button("Transpose").clicked() {
                let semitones = self.region_transpose;
                let _ = self.tx.send(Command::RegionEdit {
                    track: i,
                    op: RegionOp::Transpose { a, b, semitones },
                });
            }
            ui.separator();
            ui.add(egui::DragValue::new(&mut self.region_vel).range(0.0..=2.0).speed(0.02));
            if ui.button("×Vel").on_hover_text("Scale velocity of the selection").clicked() {
                let factor = self.region_vel;
                let _ = self.tx.send(Command::RegionEdit {
                    track: i,
                    op: RegionOp::VelScale { a, b, factor },
                });
            }
            if ui.button("Crescendo").on_hover_text("Ramp velocity 0.3 → 1.0 across the selection").clicked() {
                let _ = self.tx.send(Command::RegionEdit {
                    track: i,
                    op: RegionOp::VelRamp { a, b, from: 0.3, to: 1.0 },
                });
            }
            ui.separator();
            if ui.button("Quantize").clicked() {
                let _ = self.tx.send(Command::RegionEdit { track: i, op: RegionOp::Quantize { a, b } });
            }
            if ui.button("Reverse").clicked() {
                let _ = self.tx.send(Command::RegionEdit { track: i, op: RegionOp::Reverse { a, b } });
            }
        });
    }

    /// A spring-loaded whammy lever: drag to bend pitch, release snaps back.
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
        .add(egui::Slider::new(&mut e.gain, 0.0..=4.0).clamping(unbounded).text("Gain (SPEAKER_GAIN)"))
        .changed();
    c |= ui
        .add(egui::Slider::new(&mut e.attack_ms, 0.0..=2000.0).clamping(unbounded).text("Attack (ms)"))
        .changed();
    c |= ui
        .add(egui::Slider::new(&mut e.release_ms, 1.0..=5000.0).clamping(unbounded).text("Release (ms)"))
        .changed();
    c |= ui
        .add(egui::Slider::new(&mut e.retrigger_ms, 0.0..=2000.0).clamping(unbounded).text("Retrigger (PLAY_PERIOD)"))
        .on_hover_text("Minimum time between strikes. Firmware: 2000 ms; 0 = off.")
        .changed();
    c
}

/// Draw a track's recorded notes as bars on a timeline, with a moving playhead
/// and (optionally) a shaded selection band. Senses click-and-drag so the caller
/// can drag out a selection; returns the response.
fn draw_track_timeline(
    ui: &mut egui::Ui,
    notes: &[NoteSpan],
    play: f32,
    muted: bool,
    sel: Option<(f32, f32)>,
) -> egui::Response {
    let width = (ui.available_width() - 40.0).max(120.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, 26.0), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 3.0, egui::Color32::from_gray(30));

    // Selection band.
    if let Some((a, b)) = sel {
        let x0 = rect.left() + a.clamp(0.0, 1.0) * rect.width();
        let x1 = rect.left() + b.clamp(0.0, 1.0) * rect.width();
        let band = egui::Rect::from_min_max(
            egui::pos2(x0.min(x1), rect.top()),
            egui::pos2(x0.max(x1), rect.bottom()),
        );
        painter.rect_filled(band, 0.0, egui::Color32::from_rgba_unmultiplied(120, 200, 120, 60));
    }

    // Vertical extent maps MIDI notes 36..=84 (C2..C6) onto the row height.
    let (lo, hi) = (36.0f32, 84.0f32);
    let y_for = |note: u8| -> f32 {
        let t = ((note as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
        rect.bottom() - 3.0 - t * (rect.height() - 6.0)
    };
    let bar_color = if muted {
        egui::Color32::from_gray(90)
    } else {
        egui::Color32::from_rgb(120, 180, 255)
    };
    for n in notes {
        let x0 = rect.left() + n.start.clamp(0.0, 1.0) * rect.width();
        let x1 = rect.left() + n.end.clamp(0.0, 1.0) * rect.width();
        let y = y_for(n.note);
        let bar = egui::Rect::from_min_max(egui::pos2(x0, y - 2.0), egui::pos2(x1.max(x0 + 2.0), y + 2.0));
        painter.rect_filled(bar, 1.0, bar_color);
    }

    // Playhead.
    let px = rect.left() + play.clamp(0.0, 1.0) * rect.width();
    painter.line_segment(
        [egui::pos2(px, rect.top()), egui::pos2(px, rect.bottom())],
        egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(240, 240, 120)),
    );
    response
}

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

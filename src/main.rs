//! FTM Synth — a desktop re-implementation of a 2014 embedded "electric air
//! guitar". Physical string parameters shape the timbre; an on-screen piano,
//! the computer keyboard, and a MIDI controller all play it.

mod audio;
mod instrument;
mod midi;
mod models;
mod presets;
mod project;
mod studio;

use std::collections::HashMap;
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;

use audio::AudioEngine;
use instrument::EngineParams;
use midi::MidiInputHandle;
use models::FtmModel;
use presets::Preset;
use project::{NamedLoop, Project};
use studio::{Command, LooperMode, NoteSpan, SharedView, TrackView, TransportState};

/// Spacebar hold thresholds: a quick press taps, a medium hold stops, a long
/// hold resets.
const HOLD_STOP: f32 = 0.4;
const HOLD_RESET: f32 = 1.5;

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

    // Looper
    looper_mode: LooperMode,
    /// When the spacebar went down (for tap / hold-stop / hold-reset).
    space_down_at: Option<Instant>,

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
        // Prime the audio thread with the initial model + engine params.
        let _ = tx.send(Command::SetModel(models[selected].box_clone()));
        let _ = tx.send(Command::SetEngine(engine.clone()));

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
            looper_mode: LooperMode::Pedal,
            space_down_at: None,
            preset_list,
            preset_name: String::new(),
            preset_status: String::new(),
            project: Project {
                name: "Untitled".to_string(),
                loops: Vec::new(),
            },
            project_name: "Untitled".to_string(),
            loop_name: String::new(),
            project_list: project::list(),
            project_status: String::new(),
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
            self.space_down_at = None;
            return;
        }

        let events = ctx.input(|i| i.events.clone());
        for ev in events {
            if let egui::Event::Key {
                key,
                pressed,
                repeat,
                ..
            } = ev
            {
                if repeat {
                    continue;
                }
                // Spacebar = looper transport pedal: quick tap = primary action,
                // hold ~0.4s = Stop, hold ~1.5s = Reset.
                if key == egui::Key::Space {
                    if pressed {
                        if self.space_down_at.is_none() {
                            self.space_down_at = Some(Instant::now());
                        }
                    } else if let Some(t0) = self.space_down_at.take() {
                        let held = t0.elapsed().as_secs_f32();
                        let cmd = if held >= HOLD_RESET {
                            Command::Reset
                        } else if held >= HOLD_STOP {
                            Command::Stop
                        } else {
                            Command::Tap
                        };
                        let _ = self.tx.send(cmd);
                    }
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
            self.piano(ui);

            ui.add_space(8.0);
            self.tracks_panel(ui);

            ui.add_space(10.0);
            self.midi_panel(ui);
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
                self.project.loops.remove(i);
            }
        }

        if !self.project_status.is_empty() {
            ui.label(egui::RichText::new(&self.project_status).weak().small());
        }
        ui.add_space(2.0);
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

    /// Edit the live (keyboard) instrument.
    fn edit_live(&mut self, ui: &mut egui::Ui) {
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

    /// Edit a loop track's instrument live; edits are pushed to the audio thread.
    fn edit_track(&mut self, ui: &mut egui::Ui, i: usize) {
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
            if ui.button("Tap").clicked() {
                let _ = self.tx.send(Command::Tap);
            }
            if ui.button("Stop").clicked() {
                let _ = self.tx.send(Command::Stop);
            }
            if ui.button("Reset").clicked() {
                let _ = self.tx.send(Command::Reset);
            }
            if self.looper_mode == LooperMode::Pedal && loop_secs > 0.0 {
                if ui
                    .button("＋ Rec track")
                    .on_hover_text("Record one more pass into a new track")
                    .clicked()
                {
                    let _ = self.tx.send(Command::ArmOverdub);
                }
            }
        });

        let tap_hint = match self.looper_mode {
            LooperMode::Pedal => "Space: tap = record → play → record over (new take), tap again to finish. Hold = Stop · hold longer = Reset.",
            LooperMode::Overdub => "Space: tap = record base, then each tap layers a new track. Hold = Stop · hold longer = Reset.",
        };
        ui.label(egui::RichText::new(tap_hint).weak().small());
    }

    /// The recorded loop tracks, shown below the keyboard.
    fn tracks_panel(&mut self, ui: &mut egui::Ui) {
        let (tracks, play) = match &self.view {
            Some(v) => (v.tracks(), v.play_fraction()),
            None => return,
        };

        ui.horizontal(|ui| {
            ui.strong("Tracks");
            ui.label(egui::RichText::new(format!("({})", tracks.len())).weak());
        });
        if tracks.is_empty() {
            ui.label(
                egui::RichText::new("No loops yet — tap Space (or Tap) to record one.")
                    .weak()
                    .small(),
            );
            return;
        }

        let mut toggle_mute = None;
        let mut delete = None;
        let mut edit = None;
        for (i, t) in tracks.iter().enumerate() {
            let editing = self.edit_target == Target::Track(i);
            ui.horizontal(|ui| {
                let mute = if t.muted { "🔇" } else { "🔊" };
                if ui.button(mute).on_hover_text("Mute / unmute").clicked() {
                    toggle_mute = Some(i);
                }
                // Edit this track's instrument (highlighted when active).
                let edit_btn = egui::Button::new("✎").selected(editing);
                if ui
                    .add(edit_btn)
                    .on_hover_text("Edit this track's instrument")
                    .clicked()
                {
                    edit = Some(i);
                }
                ui.allocate_ui_with_layout(
                    egui::vec2(130.0, 24.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.label(&t.name);
                        ui.label(egui::RichText::new(&t.instrument).weak().small());
                    },
                );
                draw_track_timeline(ui, &t.notes, play, t.muted);
                if ui.button("🗑").on_hover_text("Delete track").clicked() {
                    delete = Some(i);
                }
            });
        }
        if let Some(i) = toggle_mute {
            let _ = self.tx.send(Command::ToggleMute(i));
        }
        if let Some(i) = edit {
            self.set_target(Target::Track(i), &tracks);
        }
        if let Some(i) = delete {
            let _ = self.tx.send(Command::DeleteTrack(i));
            if self.edit_target == Target::Track(i) {
                self.edit_target = Target::Live;
            }
        }
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

/// Draw a track's recorded notes as bars on a timeline, with a moving playhead.
fn draw_track_timeline(ui: &mut egui::Ui, notes: &[NoteSpan], play: f32, muted: bool) {
    let width = (ui.available_width() - 40.0).max(120.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 26.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 3.0, egui::Color32::from_gray(30));

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

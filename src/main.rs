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
use std::time::Instant;

use eframe::egui;

use audio::AudioEngine;
use instrument::EngineParams;
use midi::MidiInputHandle;
use models::FtmModel;
use presets::Preset;
use project::{NamedLoop, Project, Section, TempoGrid, ZoneData};
use studio::{
    Command, LiveConfig, LooperMode, NoteSpan, RegionOp, SharedView, SongSection, TrackView,
    TransportState,
};

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
    /// Live "kit mode": the keyboard is split/mapped across several instruments.
    live_is_kit: bool,
    /// Zones for the live kit (each an instrument mapped to a key range).
    kit_zones: Vec<ZoneData>,

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
    /// Arranger: which loop to add as the next section.
    arrange_loop_sel: usize,
    /// Arranger: repeat count for the next section.
    arrange_repeats: u32,

    /// Master output level (linear).
    master_volume: f32,

    // Whammy (pitch-bend lever), semitones + configurable range.
    whammy: f32,
    whammy_down: f32,
    whammy_up: f32,

    // Region editing (chop / crop / rearrange)
    /// Current timeline selection: (track index, start secs, end secs).
    sel: Option<(usize, f32, f32)>,
    /// Drag anchor (loop fraction) while dragging out a selection.
    drag_start: Option<f32>,
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
            space_down_at: None,
            preset_list,
            preset_name: String::new(),
            preset_status: String::new(),
            project: Project {
                name: "Untitled".to_string(),
                loops: Vec::new(),
                arrangement: Vec::new(),
                tempo: TempoGrid::default(),
            },
            project_name: "Untitled".to_string(),
            loop_name: String::new(),
            project_list: project::list(),
            project_status: String::new(),
            arrange_loop_sel: 0,
            arrange_repeats: 4,
            master_volume: 1.0,
            whammy: 0.0,
            whammy_down: 12.0,
            whammy_up: 2.0,
            sel: None,
            drag_start: None,
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
        self.project.arrangement.retain(|s| s.loop_index != i);
        for s in &mut self.project.arrangement {
            if s.loop_index > i {
                s.loop_index -= 1;
            }
        }
        if self.arrange_loop_sel >= self.project.loops.len() {
            self.arrange_loop_sel = self.project.loops.len().saturating_sub(1);
        }
    }

    /// Resolve the arrangement into runtime sections and play the song.
    fn play_song(&mut self) {
        let sections: Vec<SongSection> = self
            .project
            .arrangement
            .iter()
            .filter_map(|s| {
                self.project.loops.get(s.loop_index).map(|nl| SongSection {
                    loop_data: nl.data.clone(),
                    repeats: s.repeats.max(1),
                })
            })
            .collect();
        if sections.is_empty() {
            self.project_status = "Add sections to the song first.".into();
            return;
        }
        let _ = self.tx.send(Command::SetSong(sections));
        let _ = self.tx.send(Command::PlaySong);
        self.project_status = "Playing song…".into();
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
            ui.horizontal(|ui| {
                self.whammy_bar(ui);
                self.piano(ui);
            });

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
                    arrangement: Vec::new(),
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

        self.arrangement_ui(ui);
        self.export_ui(ui);

        if !self.project_status.is_empty() {
            ui.label(egui::RichText::new(&self.project_status).weak().small());
        }
        ui.add_space(2.0);
    }

    /// Offline WAV export of the current loop or the whole song.
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
            if ui.button("⬇ Export loop").clicked() {
                self.export_loop();
            }
            let has_song = !self.project.arrangement.is_empty();
            if ui
                .add_enabled(has_song, egui::Button::new("⬇ Export song"))
                .clicked()
            {
                self.export_song();
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

    fn export_song(&mut self) {
        let sections: Vec<SongSection> = self
            .project
            .arrangement
            .iter()
            .filter_map(|s| {
                self.project.loops.get(s.loop_index).map(|nl| SongSection {
                    loop_data: nl.data.clone(),
                    repeats: s.repeats,
                })
            })
            .collect();
        if sections.is_empty() {
            self.export_status = "Add sections to the song first.".into();
            return;
        }
        let path = export::export_path(&format!("{}_song", self.export_name));
        let sr = self.export_sr as f32;
        match export::render_song_to_wav(sections, sr, self.export_tail, self.export_hi_res, &path) {
            Ok(()) => self.export_status = format!("Wrote {}", path.display()),
            Err(e) => self.export_status = format!("Export failed: {e}"),
        }
    }

    /// The song arranger: a linear timeline of (loop × repeats) sections.
    fn arrangement_ui(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        let playing_section = self.view.as_ref().and_then(|v| v.song_section());

        ui.horizontal(|ui| {
            ui.strong("Song");
            if ui.button("▶ Play song").clicked() {
                self.play_song();
            }
            if ui.button("■ Stop").clicked() {
                let _ = self.tx.send(Command::Stop);
            }
            ui.separator();

            if self.project.loops.is_empty() {
                ui.label(egui::RichText::new("add a loop first").weak().small());
            } else {
                ui.label("Add:");
                if self.arrange_loop_sel >= self.project.loops.len() {
                    self.arrange_loop_sel = 0;
                }
                let sel = self.project.loops[self.arrange_loop_sel].name.clone();
                egui::ComboBox::from_id_salt("arrange_loop")
                    .selected_text(sel)
                    .show_ui(ui, |ui| {
                        for (i, nl) in self.project.loops.iter().enumerate() {
                            ui.selectable_value(&mut self.arrange_loop_sel, i, &nl.name);
                        }
                    });
                ui.label("×");
                ui.add(egui::DragValue::new(&mut self.arrange_repeats).range(1..=64));
                if ui.button("＋ Section").clicked() {
                    self.project.arrangement.push(Section {
                        loop_index: self.arrange_loop_sel,
                        repeats: self.arrange_repeats.max(1),
                    });
                }
            }
        });

        if self.project.arrangement.is_empty() {
            ui.label(
                egui::RichText::new("Song is empty — add sections (loop × repeats) to arrange one.")
                    .weak()
                    .small(),
            );
            return;
        }

        let mut remove = None;
        let mut move_left = None;
        let mut move_right = None;
        egui::ScrollArea::horizontal()
            .id_salt("arrangement_scroll")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (i, sec) in self.project.arrangement.iter().enumerate() {
                        let name = self
                            .project
                            .loops
                            .get(sec.loop_index)
                            .map(|nl| nl.name.as_str())
                            .unwrap_or("?");
                        let active = playing_section == Some(i);
                        let mut frame = egui::Frame::group(ui.style());
                        if active {
                            frame = frame.fill(egui::Color32::from_rgb(60, 90, 60));
                        }
                        frame.show(ui, |ui| {
                            let text = format!("{}. {} ×{}", i + 1, name, sec.repeats);
                            if active {
                                ui.colored_label(egui::Color32::from_rgb(140, 230, 140), text);
                            } else {
                                ui.label(text);
                            }
                            if ui.small_button("←").on_hover_text("Move left").clicked() {
                                move_left = Some(i);
                            }
                            if ui.small_button("✕").on_hover_text("Remove section").clicked() {
                                remove = Some(i);
                            }
                            if ui.small_button("→").on_hover_text("Move right").clicked() {
                                move_right = Some(i);
                            }
                        });
                    }
                });
            });
        if let Some(i) = remove {
            self.project.arrangement.remove(i);
        }
        if let Some(i) = move_left {
            if i > 0 {
                self.project.arrangement.swap(i, i - 1);
            }
        }
        if let Some(i) = move_right {
            if i + 1 < self.project.arrangement.len() {
                self.project.arrangement.swap(i, i + 1);
            }
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
        let (tracks, play, loop_secs) = match &self.view {
            Some(v) => (v.tracks(), v.play_fraction(), v.loop_seconds(self.sample_rate)),
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
                egui::RichText::new("No loops yet — tap Space (or Tap) to record one.")
                    .weak()
                    .small(),
            );
            return;
        }

        ui.label(
            egui::RichText::new("Drag across a track's timeline to select a span, then chop/crop/duplicate/move it below. Move sliders while recording to automate.")
                .weak()
                .small(),
        );

        let mut toggle_mute = None;
        let mut toggle_solo = None;
        let mut mix_change: Option<(usize, f32, f32)> = None;
        let mut delete = None;
        let mut edit = None;
        let mut clear_auto = None;
        let mut drag_start = self.drag_start;
        let mut new_sel: Option<Option<(usize, f32, f32)>> = None;
        for (i, t) in tracks.iter().enumerate() {
            let editing = self.edit_target == Target::Track(i);
            let sel_frac = if loop_secs > 0.0 {
                self.sel.filter(|s| s.0 == i).map(|(_, a, b)| (a / loop_secs, b / loop_secs))
            } else {
                None
            };
            ui.horizontal(|ui| {
                let mute = if t.muted { "🔇" } else { "🔊" };
                if ui.button(mute).on_hover_text("Mute / unmute").clicked() {
                    toggle_mute = Some(i);
                }
                let solo_btn = egui::Button::new("S").selected(t.solo);
                if ui.add(solo_btn).on_hover_text("Solo").clicked() {
                    toggle_solo = Some(i);
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
                let resp = draw_track_timeline(ui, &t.notes, play, t.muted, sel_frac);
                if loop_secs > 0.0 {
                    let w = resp.rect.width().max(1.0);
                    let frac_at = |x: f32| ((x - resp.rect.left()) / w).clamp(0.0, 1.0);
                    if resp.drag_started() {
                        if let Some(p) = resp.interact_pointer_pos() {
                            drag_start = Some(frac_at(p.x));
                        }
                    }
                    if resp.dragged() {
                        if let (Some(st), Some(p)) = (drag_start, resp.interact_pointer_pos()) {
                            let cur = frac_at(p.x);
                            let (a, b) = (st.min(cur), st.max(cur));
                            new_sel = Some(Some((i, a * loop_secs, b * loop_secs)));
                        }
                    }
                    if resp.drag_stopped() {
                        drag_start = None;
                    }
                }
                // Compact per-track mixer: pan + volume.
                let mut pan = t.pan;
                let mut vol = t.volume;
                let pan_resp = ui.add_sized(
                    [64.0, 18.0],
                    egui::Slider::new(&mut pan, -1.0..=1.0).show_value(false),
                );
                let vol_resp = ui.add_sized(
                    [64.0, 18.0],
                    egui::Slider::new(&mut vol, 0.0..=1.5).show_value(false),
                );
                if pan_resp.on_hover_text("Pan").changed() || vol_resp.on_hover_text("Volume").changed() {
                    mix_change = Some((i, vol, pan));
                }
                if t.automation > 0
                    && ui
                        .button(format!("🎚 {}", t.automation))
                        .on_hover_text("Recorded parameter automation — click to clear")
                        .clicked()
                {
                    clear_auto = Some(i);
                }
                if ui.button("🗑").on_hover_text("Delete track").clicked() {
                    delete = Some(i);
                }
            });
            // Region-edit toolbar for the selected track.
            if self.sel.map(|s| s.0) == Some(i) && loop_secs > 0.0 {
                self.region_ops_row(ui, i, loop_secs);
            }
        }
        self.drag_start = drag_start;
        if let Some(sel) = new_sel {
            self.sel = sel;
        }
        if let Some(i) = toggle_mute {
            let _ = self.tx.send(Command::ToggleMute(i));
        }
        if let Some(i) = toggle_solo {
            let _ = self.tx.send(Command::ToggleSolo(i));
        }
        if let Some((i, volume, pan)) = mix_change {
            let _ = self.tx.send(Command::SetTrackMix { track: i, volume, pan });
        }
        if let Some(i) = clear_auto {
            let _ = self.tx.send(Command::ClearTrackAutomation(i));
        }
        if let Some(i) = edit {
            self.set_target(Target::Track(i), &tracks);
        }
        if let Some(i) = delete {
            let _ = self.tx.send(Command::DeleteTrack(i));
            if self.edit_target == Target::Track(i) {
                self.edit_target = Target::Live;
            }
            if self.sel.map(|s| s.0) == Some(i) {
                self.sel = None;
            }
        }
    }

    /// The chop/crop/rearrange toolbar shown under the selected track.
    fn region_ops_row(&mut self, ui: &mut egui::Ui, i: usize, loop_secs: f32) {
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
                    .range(0.0..=loop_secs)
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

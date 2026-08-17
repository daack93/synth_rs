//! FTM Synth — a desktop re-implementation of a 2014 embedded "electric air
//! guitar". Physical string parameters shape the timbre; an on-screen piano,
//! the computer keyboard, and a MIDI controller all play it.

mod audio;
mod midi;
mod models;
mod presets;
mod synth;

use std::collections::HashMap;
use std::sync::mpsc::{channel, Sender};

use eframe::egui;

use audio::AudioEngine;
use midi::MidiInputHandle;
use models::FtmModel;
use presets::Preset;
use synth::{Command, EngineParams};

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

struct App {
    tx: Sender<Command>,
    _audio: Option<AudioEngine>,
    audio_err: Option<String>,

    /// Available synthesis models (plugins); each holds its own parameters.
    models: Vec<Box<dyn FtmModel>>,
    /// Index of the active model.
    selected: usize,
    /// Engine-wide parameters (gain, envelope, retrigger).
    engine: EngineParams,

    // Preset library
    /// Presets found in the folder (refreshed on save/load/delete).
    preset_list: Vec<Preset>,
    /// Name field for saving the current sound.
    preset_name: String,
    /// Last preset action result, shown in the UI.
    preset_status: String,

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

        let models = models::registry();
        let selected = 0;
        let engine = EngineParams::default();
        // Prime the audio thread with the initial model + engine params.
        let _ = tx.send(Command::SetModel(models[selected].box_clone()));
        let _ = tx.send(Command::SetEngine(engine.clone()));

        let midi_ports = midi::list_ports();
        let preset_list = presets::list();

        App {
            tx,
            _audio: audio,
            audio_err,
            models,
            selected,
            engine,
            preset_list,
            preset_name: String::new(),
            preset_status: String::new(),
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

    /// Save the current model + engine as a preset under `preset_name`.
    fn save_preset(&mut self) {
        let name = self.preset_name.trim().to_string();
        if name.is_empty() {
            self.preset_status = "Enter a name first.".into();
            return;
        }
        let preset = Preset::capture(&name, self.models[self.selected].as_ref(), &self.engine);
        match presets::save(&preset) {
            Ok(path) => {
                self.preset_status = format!("Saved “{name}” → {}", path.display());
                self.preset_list = presets::list();
            }
            Err(e) => self.preset_status = format!("Save failed: {e}"),
        }
    }

    /// Load a preset: swap in its model + params + engine and start playing it.
    fn apply_preset(&mut self, preset: &Preset) {
        let Some(model) = preset.build_model() else {
            self.preset_status =
                format!("Can't load “{}”: unknown model “{}”.", preset.name, preset.model_id);
            return;
        };
        // Replace the matching registry slot so the editor shows these params.
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
            self.piano(ui);

            ui.add_space(10.0);
            self.midi_panel(ui);
        });
    }
}

impl App {
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
        });
        if !self.preset_status.is_empty() {
            ui.label(egui::RichText::new(&self.preset_status).weak().small());
        }
        ui.add_space(2.0);
    }

    fn params_panel(&mut self, ui: &mut egui::Ui) {
        let unbounded = egui::SliderClamping::Never;

        ui.add_space(6.0);
        ui.heading("Synthesis model");
        ui.add_space(2.0);

        // --- Model (plugin) picker ---
        let mut new_selection = self.selected;
        egui::ComboBox::from_id_salt("model_pick")
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
            self.push_model(); // swap live; the engine re-excites held notes
        }

        ui.separator();

        // --- The active model's own parameter editor ---
        egui::ScrollArea::vertical()
            .max_height(360.0)
            .show(ui, |ui| {
                if self.models[self.selected].params_ui(ui) {
                    self.push_model();
                }
            });

        ui.separator();

        // --- Engine-wide (shared) parameters ---
        ui.strong("Output / Voice");
        let e = &mut self.engine;
        let mut eng_changed = false;
        eng_changed |= ui
            .add(egui::Slider::new(&mut e.gain, 0.0..=4.0).clamping(unbounded).text("Master gain (SPEAKER_GAIN)"))
            .changed();
        eng_changed |= ui
            .add(egui::Slider::new(&mut e.attack_ms, 0.0..=2000.0).clamping(unbounded).text("Attack (ms)"))
            .changed();
        eng_changed |= ui
            .add(egui::Slider::new(&mut e.release_ms, 1.0..=5000.0).clamping(unbounded).text("Release (ms)"))
            .changed();
        eng_changed |= ui
            .add(egui::Slider::new(&mut e.retrigger_ms, 0.0..=2000.0).clamping(unbounded).text("Retrigger lockout (PLAY_PERIOD)"))
            .on_hover_text("Minimum time between strikes. Firmware: 2000 ms; 0 = off (playable).")
            .changed();
        if eng_changed {
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

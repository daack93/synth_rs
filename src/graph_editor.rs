//! A visual node-graph editor for an `InstrumentGraph`.
//!
//! Components are draggable boxes; you wire an output port to an input port to
//! connect them. Selecting a component (or edge) shows its parameters, and —
//! crucially for diagnosis — **arms it for audition**: while it is selected,
//! playing a key renders *just that component, in isolation*, driven by a pure
//! sinusoid at the played pitch. A signal generator ignores the probe and emits
//! its own sound; a resonator resonates the pure tone — so you hear how each
//! part responds to a clean input, with its note-mapping applied.
//!
//! Audition costs nothing on the audio thread: the model is reshipped on every
//! edit anyway, so we just send a three-node stand-in graph — `Sine → component
//! → Mix` (see [`isolate`]) — in place of the real one while a node is selected.

use eframe::egui;

use crate::models::instrument_graph::{Comp, Edge, InstrumentGraph, KeyBinding};
use crate::models::cymbal::Cymbal;
use crate::models::drum_membrane::DrumMembrane;
use crate::models::metal_bell::MetalBell;
use crate::models::musical_string::MusicalString;
use crate::models::pure_plate::PurePlate;
use crate::models::pure_string::PureString;
use crate::models::webster_horn::WebsterHorn;
use crate::models::FtmModel;

const NODE_W: f32 = 152.0;
const NODE_H: f32 = 52.0;
const PORT_R: f32 = 6.0;

/// What is currently selected in the editor.
#[derive(Clone, Copy, PartialEq)]
pub enum GeSel {
    Node(usize),
    Edge(usize),
}

/// Editor UI state (positions + selection + in-progress drags). Lives on `App`.
pub struct GeState {
    pub open: bool,
    /// Node positions (canvas-relative), one per component. Rebuilt if stale.
    pub layout: Vec<[f32; 2]>,
    pub sel: Option<GeSel>,
    /// Node currently being dragged (moved).
    drag: Option<usize>,
    /// Output port a wire is being dragged from.
    wire_from: Option<usize>,
    /// Which component output the audio thread is currently auditioning.
    audition: Option<usize>,
}

impl Default for GeState {
    fn default() -> Self {
        GeState {
            open: false,
            layout: Vec::new(),
            sel: None,
            drag: None,
            wire_from: None,
            audition: None,
        }
    }
}

fn node_color(c: &Comp) -> egui::Color32 {
    if c.is_exciter() {
        egui::Color32::from_rgb(196, 120, 60) // exciters: warm
    } else if matches!(c, Comp::Mix) {
        egui::Color32::from_rgb(96, 104, 112) // output: grey
    } else if matches!(c, Comp::Wires { .. }) {
        egui::Color32::from_rgb(150, 110, 180) // coupling: violet
    } else {
        egui::Color32::from_rgb(60, 140, 150) // resonators: teal
    }
}

impl crate::App {
    /// Draw the graph-editor window when open, and drive per-component audition.
    /// Only edits the **live** model, and only when it is an `InstrumentGraph`.
    pub fn graph_editor_window(&mut self, ctx: &egui::Context) {
        if !self.ge.open {
            return;
        }
        // Take a working copy of the live graph (releases the model borrow).
        let mut ig = match self.models[self.selected].as_instrument_graph_mut() {
            Some(g) => g.clone(),
            None => {
                self.ge.open = false;
                return;
            }
        };
        // Keep layout in sync with the component count.
        if self.ge.layout.len() != ig.components.len() {
            self.ge.layout = ig.auto_layout();
            if let Some(GeSel::Node(i)) = self.ge.sel {
                if i >= ig.components.len() {
                    self.ge.sel = None;
                }
            }
        }

        let mut open = self.ge.open;
        let mut changed = false;
        egui::Window::new("Graph editor")
            .open(&mut open)
            .default_size([760.0, 520.0])
            .resizable(true)
            .show(ctx, |ui| {
                changed = draw_editor(ui, &mut ig, &mut self.ge);
            });
        self.ge.open = open;

        // Write edits back to the live model.
        if changed {
            self.models[self.selected] = Box::new(ig.clone());
        }

        // Audition: while a node (or an edge's source) is selected, the audio
        // model is a clone with `output` repointed there; otherwise the real one.
        let desired = if self.ge.open {
            match self.ge.sel {
                Some(GeSel::Node(i)) => Some(i),
                Some(GeSel::Edge(e)) => ig.edges.get(e).map(|ed| ed.from),
                None => None,
            }
        } else {
            None
        };
        if changed || desired != self.ge.audition {
            self.ge.audition = desired;
            let model: Box<dyn FtmModel> = match desired {
                Some(i) if i < ig.components.len() => Box::new(isolate(&ig, i)),
                _ => self.models[self.selected].box_clone(),
            };
            let _ = self.tx.send(crate::studio::Command::SetModel(model));
        }
    }
}

/// Build the isolation graph that auditions component `i` alone: a pure sine at
/// the played pitch drives just that component into a mixer. The component keeps
/// its own key-mapping (remapped onto its new index). An exciter ignores the
/// sine and emits its own sound; a resonator resonates the pure tone.
fn isolate(ig: &InstrumentGraph, i: usize) -> InstrumentGraph {
    let key_map = ig
        .key_map
        .iter()
        .filter(|k| k.component == i)
        .map(|k| KeyBinding { component: 1, map: k.map.clone() })
        .collect();
    InstrumentGraph {
        components: vec![Comp::Sine { level: 1.0 }, ig.components[i].clone(), Comp::Mix],
        edges: vec![Edge { from: 0, to: 1, gain: 1.0 }, Edge { from: 1, to: 2, gain: 1.0 }],
        output: 2,
        key_map,
    }
}

/// Draw the canvas + the selected-component panel. Returns `true` if the graph
/// changed (so the caller reships it to the audio thread).
fn draw_editor(ui: &mut egui::Ui, ig: &mut InstrumentGraph, ge: &mut GeState) -> bool {
    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("Drag a box to move it · drag an output port (right) to an input port (left) to connect · click a node/edge to select and audition it")
                .weak()
                .small(),
        );
    });

    // Add-component menu.
    ui.horizontal_wrapped(|ui| {
        ui.label("Add:");
        for (category, items) in add_menu() {
            ui.menu_button(format!("{category} ▾"), |ui| {
                for (label, mk) in items {
                    if ui.button(label).clicked() {
                        ig.components.push(mk());
                        ge.layout.push([20.0, 20.0]);
                        ge.sel = Some(GeSel::Node(ig.components.len() - 1));
                        changed = true;
                        ui.close_menu();
                    }
                }
            });
        }
    });
    ui.separator();

    // --- Canvas ---
    // Give the canvas a bit over half the height and leave the rest for the
    // component-parameter panel, so its controls need far less scrolling.
    let avail = ui.available_size_before_wrap();
    let canvas_h = (avail.y * 0.55).clamp(200.0, avail.y - 200.0);
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(400.0), canvas_h), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(24));

    let n = ig.components.len();
    let node_rect = |i: usize| -> egui::Rect {
        let p = ge.layout.get(i).copied().unwrap_or([20.0, 20.0]);
        egui::Rect::from_min_size(rect.min + egui::vec2(p[0], p[1]), egui::vec2(NODE_W, NODE_H))
    };
    let out_port = |i: usize| node_rect(i).right_center();
    let in_port = |i: usize| node_rect(i).left_center();

    // Edges (drawn under the nodes).
    for (ei, e) in ig.edges.iter().enumerate() {
        if e.from >= n || e.to >= n {
            continue;
        }
        let (a, b) = (out_port(e.from), in_port(e.to));
        let selected = ge.sel == Some(GeSel::Edge(ei));
        let w = 1.0 + 3.0 * e.gain.clamp(0.0, 1.5);
        let col = if selected {
            egui::Color32::from_rgb(240, 220, 120)
        } else {
            egui::Color32::from_gray(150)
        };
        painter.line_segment([a, b], egui::Stroke::new(w, col));
        // gain label at the midpoint
        painter.text(
            a.lerp(b, 0.5),
            egui::Align2::CENTER_BOTTOM,
            format!("{:.2}", e.gain),
            egui::FontId::proportional(10.0),
            egui::Color32::from_gray(180),
        );
    }

    // In-progress wire.
    if let Some(f) = ge.wire_from {
        if let Some(p) = resp.hover_pos().or_else(|| resp.interact_pointer_pos()) {
            painter.line_segment([out_port(f), p], egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(240, 220, 120)));
        }
    }

    // Nodes.
    let labels = ig.labels();
    for i in 0..n {
        let nr = node_rect(i);
        let selected = ge.sel == Some(GeSel::Node(i));
        let is_out = ig.output == i;
        painter.rect_filled(nr, 6.0, node_color(&ig.components[i]));
        let stroke = if selected {
            egui::Stroke::new(2.5_f32, egui::Color32::WHITE)
        } else if is_out {
            egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(240, 220, 120))
        } else {
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(40))
        };
        painter.rect_stroke(nr, 6.0, stroke);
        painter.text(
            nr.center() - egui::vec2(0.0, 6.0),
            egui::Align2::CENTER_CENTER,
            format!("{i}"),
            egui::FontId::monospace(11.0),
            egui::Color32::from_gray(30),
        );
        painter.text(
            nr.center() + egui::vec2(0.0, 8.0),
            egui::Align2::CENTER_CENTER,
            labels[i],
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(245),
        );
        if is_out {
            painter.text(
                nr.right_top() + egui::vec2(-2.0, 1.0),
                egui::Align2::RIGHT_TOP,
                "OUT",
                egui::FontId::monospace(9.0),
                egui::Color32::from_rgb(240, 220, 120),
            );
        }
        // Ports (skip an input port on a pure source exciter — cosmetic).
        painter.circle_filled(out_port(i), PORT_R, egui::Color32::from_gray(210));
        if !ig.is_exciter_at(i) {
            painter.circle_filled(in_port(i), PORT_R, egui::Color32::from_gray(210));
        }
    }

    handle_canvas_interaction(&resp, rect, ig, ge, &mut changed);

    // --- Selected-component / edge panel ---
    ui.separator();
    changed |= selection_panel(ui, ig, ge);

    changed
}

/// Drag/click behaviour on the canvas.
fn handle_canvas_interaction(
    resp: &egui::Response,
    rect: egui::Rect,
    ig: &mut InstrumentGraph,
    ge: &mut GeState,
    changed: &mut bool,
) {
    let n = ig.components.len();
    let node_rect = |i: usize, ge: &GeState| -> egui::Rect {
        let p = ge.layout.get(i).copied().unwrap_or([20.0, 20.0]);
        egui::Rect::from_min_size(rect.min + egui::vec2(p[0], p[1]), egui::vec2(NODE_W, NODE_H))
    };
    let out_port = |i: usize, ge: &GeState| node_rect(i, ge).right_center();
    let in_port = |i: usize, ge: &GeState| node_rect(i, ge).left_center();

    // Start of a drag: decide whether it grabs an output port (→ wire) or a node
    // body (→ move). Use the press origin so the few-px drag threshold doesn't
    // shift the grab.
    if resp.drag_started() {
        if let Some(p) = resp.interact_pointer_pos() {
            ge.drag = None;
            ge.wire_from = None;
            // Output port first (they sit on the node's right edge).
            for i in 0..n {
                if p.distance(out_port(i, ge)) <= PORT_R + 4.0 {
                    ge.wire_from = Some(i);
                    break;
                }
            }
            if ge.wire_from.is_none() {
                for i in (0..n).rev() {
                    if node_rect(i, ge).contains(p) {
                        ge.drag = Some(i);
                        ge.sel = Some(GeSel::Node(i));
                        break;
                    }
                }
            }
        }
    }

    if resp.dragged() {
        if let Some(i) = ge.drag {
            let d = resp.drag_delta();
            if let Some(pos) = ge.layout.get_mut(i) {
                pos[0] = (pos[0] + d.x).max(0.0);
                pos[1] = (pos[1] + d.y).max(0.0);
            }
        }
    }

    if resp.drag_stopped() {
        if let Some(f) = ge.wire_from.take() {
            if let Some(p) = resp.interact_pointer_pos() {
                // Dropped on an input port → connect.
                for j in 0..n {
                    if j != f && !ig.is_exciter_at(j) && p.distance(in_port(j, ge)) <= PORT_R + 8.0 {
                        if !ig.edges.iter().any(|e| e.from == f && e.to == j) {
                            ig.edges.push(Edge { from: f, to: j, gain: 1.0 });
                            ge.sel = Some(GeSel::Edge(ig.edges.len() - 1));
                            *changed = true;
                        }
                        break;
                    }
                }
            }
        }
        ge.drag = None;
    }

    // Click (no drag) → select node or edge under the pointer, else clear.
    if resp.clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            let mut hit = None;
            for i in (0..n).rev() {
                if node_rect(i, ge).contains(p) {
                    hit = Some(GeSel::Node(i));
                    break;
                }
            }
            if hit.is_none() {
                for (ei, e) in ig.edges.iter().enumerate() {
                    if e.from < n && e.to < n {
                        let d = dist_to_segment(p, out_port(e.from, ge), in_port(e.to, ge));
                        if d < 6.0 {
                            hit = Some(GeSel::Edge(ei));
                            break;
                        }
                    }
                }
            }
            ge.sel = hit;
        }
    }
}

/// The panel under the canvas showing the selected node's params (and audition
/// hint) or the selected edge's gain, with delete / set-output actions.
fn selection_panel(ui: &mut egui::Ui, ig: &mut InstrumentGraph, ge: &mut GeState) -> bool {
    let mut changed = false;
    match ge.sel {
        Some(GeSel::Node(i)) if i < ig.components.len() => {
            ui.horizontal(|ui| {
                ui.strong(format!("Node {i}: {}", ig.labels()[i]));
                ui.label(
                    egui::RichText::new("▶ play a key to hear this component")
                        .color(egui::Color32::from_rgb(240, 220, 120))
                        .small(),
                );
            });
            ui.horizontal(|ui| {
                if ig.output != i && ui.button("Set as output").clicked() {
                    ig.output = i;
                    changed = true;
                }
                if ig.components.len() > 1 && ui.button("🗑 Delete node").clicked() {
                    ig.remove_component(i);
                    ge.layout.remove(i.min(ge.layout.len().saturating_sub(1)));
                    ge.sel = None;
                    changed = true;
                }
            });
            if ge.sel.is_some() {
                ui.separator();
                // Fill the panel's remaining height, and make the controls larger
                // (wider sliders, taller rows) so they are easy to play with.
                let h = ui.available_height().max(200.0);
                egui::ScrollArea::vertical().max_height(h).show(ui, |ui| {
                    ui.spacing_mut().slider_width = 260.0;
                    ui.spacing_mut().interact_size.y = 24.0;
                    ui.spacing_mut().item_spacing.y = 8.0;
                    changed |= ig.component_params_ui(i, ui);
                });
            }
        }
        Some(GeSel::Edge(e)) if e < ig.edges.len() => {
            let (from, to) = (ig.edges[e].from, ig.edges[e].to);
            let labels = ig.labels();
            ui.strong(format!(
                "Edge: {} → {}",
                labels.get(from).copied().unwrap_or("?"),
                labels.get(to).copied().unwrap_or("?")
            ));
            changed |= ui
                .add(crate::models::unbounded_slider(&mut ig.edges[e].gain, 0.0..=1.5, "Coupling gain"))
                .changed();
            if ui.button("🗑 Delete edge").clicked() {
                ig.edges.remove(e);
                ge.sel = None;
                changed = true;
            }
        }
        _ => {
            ui.label(egui::RichText::new("Select a component or edge to edit and audition it.").weak());
        }
    }
    changed
}

/// Which components the Add menu offers, grouped into categories, each with a
/// constructor that makes one at a sane default.
fn add_menu() -> Vec<(&'static str, Vec<(&'static str, fn() -> Comp)>)> {
    vec![
        (
            "Exciter",
            vec![
                ("Strike", || Comp::Strike),
                ("Hammer", || Comp::Hammer { hardness: 0.6, felt: 2.5 }),
                ("Reed / lip", || Comp::Reed { pressure: 0.9, stiffness: 1.0, freq_hz: 150.0 }),
                ("Reed + bore (coupled)", || Comp::ReedBore { pressure: 0.9, stiffness: 1.0, length: 0.6555, tone: 1.0, register: 0.0 }),
                ("Breath", || Comp::Breath { level: 0.15, tone: 1.0 }),
                ("Bow", || Comp::Bow { speed: 1.2, force: 0.6 }),
                ("Voice", || Comp::Voice { open_quotient: 0.6, level: 0.5 }),
                ("Sine (tone)", || Comp::Sine { level: 1.0 }),
            ],
        ),
        (
            "Resonator",
            vec![
                ("String", || Comp::String(PureString::default())),
                ("Musical string", || Comp::MusicalString(MusicalString::default())),
                ("Membrane", || Comp::Membrane(DrumMembrane::default())),
                ("Plate", || Comp::Plate(PurePlate::default())),
                ("Bell / cowbell", || Comp::Bell(MetalBell::default())),
                ("Cymbal / gong", || Comp::Cymbal(Cymbal::default())),
                ("Air column (horn)", || Comp::Horn(WebsterHorn::default())),
                ("Body / cavity", || Comp::Body { cavity_litres: 15.0, soundhole_cm: 9.0, top_hz: 195.0, decay_s: 0.18 }),
                ("Snare wires", || Comp::Wires { level: 0.6, tone: 1.0 }),
            ],
        ),
        (
            "Waveguide",
            vec![
                ("Reed pipe", || Comp::ReedPipe { pressure: 0.9, stiffness: 1.0, tone: 1.0 }),
                ("Bowed string", || Comp::BowedString { speed: 1.2, force: 0.6 }),
                ("Air column (bore)", || Comp::Bore { tone: 1.0, length: 0.0 }),
                ("Waveguide horn", || Comp::WaveguideHorn { r1: 0.0073, r2: 0.0, r3: 0.002, length: 0.334, segments: 18, tone: 1.0 }),
            ],
        ),
        ("Output", vec![("Mix", || Comp::Mix)]),
    ]
}

fn dist_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() < 1e-6 {
        0.0
    } else {
        ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0)
    };
    p.distance(a + ab * t)
}

//! The arrangement editor: a wrapping multi-lane timeline UI and its contextual
//! controls. Extracted verbatim from `main.rs`; these are `App` methods that
//! `tracks_panel` (still in `main.rs`) calls into.

use eframe::egui;

use crate::studio::{ClipView, Command, TrackView};
use crate::{App, Snap, Target};

/// What a clip drag is doing. `Move` is the top-middle handle; the edges resize;
/// a drag on the body selects a time range for the Crop/Delete/Loop/Reverse ops.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DragKind {
    Move,
    ResizeR,
    ResizeL,
    Select,
}

impl App {
    /// The arrangement editor: a wrapping multi-lane timeline. Time flows left to
    /// right and wraps to stacked blocks (like a score wrapping systems); each
    /// block shows every track's lane for that time window. Clips can be dragged
    /// to move, edge-dragged to resize, duplicated and deleted; click empty to
    /// seek, double-click an empty lane to place a clip.
    pub(crate) fn arrangement_editor(
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
    pub(crate) fn contextual_panel(
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
}

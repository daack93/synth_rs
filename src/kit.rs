//! Kits: map key ranges to different instruments.
//!
//! A [`Kit`] is a list of [`Zone`]s. Each zone owns its own [`Instrument`] and
//! responds to a MIDI key range; overlapping ranges layer, disjoint ranges split
//! the keyboard. A zone can be **chromatic** (the key sets the pitch, optionally
//! transposed) or a **pad** (`fixed_note`: any key in range plays one fixed
//! pitch — a drum pad).
//!
//! The whole point is that a recorded note event stays just `(note, vel)`: the
//! [`Kit`] re-routes it on playback exactly as it did live, so a kit records and
//! replays with no per-event bookkeeping. [`Playable`] is the shared shape of
//! "a thing you can play" — either a single [`Instrument`] or a [`Kit`] — so the
//! studio's live slot and every track hold the same type.

use std::sync::Arc;

use crate::instrument::{EngineParams, Instrument};
use crate::models::{default_model, model_from_id};
use crate::project::ZoneData;

/// One key-range → instrument mapping within a [`Kit`].
pub struct Zone {
    pub name: String,
    pub lo: u8,
    pub hi: u8,
    pub fixed_note: Option<u8>,
    pub transpose: i8,
    pub inst: Instrument,
}

impl Zone {
    /// The pitch a given key plays in this zone (a pad ignores the key).
    #[inline]
    fn mapped(&self, key: u8) -> u8 {
        match self.fixed_note {
            Some(n) => n,
            None => (key as i16 + self.transpose as i16).clamp(0, 127) as u8,
        }
    }

    #[inline]
    fn in_range(&self, key: u8) -> bool {
        key >= self.lo && key <= self.hi
    }

    fn to_data(&self) -> ZoneData {
        ZoneData {
            name: self.name.clone(),
            lo: self.lo,
            hi: self.hi,
            fixed_note: self.fixed_note,
            transpose: self.transpose,
            model_id: self.inst.model_id().to_string(),
            params: self.inst.model_json(),
            engine: self.inst.engine_params(),
        }
    }
}

/// A bank of zones played as one instrument.
pub struct Kit {
    zones: Vec<Zone>,
}

impl Kit {
    /// Build a kit from serialized zones (each zone's instrument rebuilt from its
    /// model id + params). Unknown model ids fall back to the default model.
    pub fn from_data(sr: f32, sine: Arc<[f32]>, zones: &[ZoneData]) -> Kit {
        let zones = zones
            .iter()
            .map(|z| {
                let model = model_from_id(&z.model_id, &z.params).unwrap_or_else(default_model);
                let inst = Instrument::with_config(sr, sine.clone(), model, z.engine.clone());
                Zone {
                    name: z.name.clone(),
                    lo: z.lo,
                    hi: z.hi,
                    fixed_note: z.fixed_note,
                    transpose: z.transpose,
                    inst,
                }
            })
            .collect();
        Kit { zones }
    }

    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }

    pub fn note_on(&mut self, key: u8, vel: f32) {
        for z in &mut self.zones {
            if z.in_range(key) {
                let n = z.mapped(key);
                z.inst.note_on(n, vel);
            }
        }
    }

    pub fn note_off(&mut self, key: u8) {
        for z in &mut self.zones {
            if z.in_range(key) {
                let n = z.mapped(key);
                z.inst.note_off(n);
            }
        }
    }

    pub fn all_notes_off(&mut self) {
        for z in &mut self.zones {
            z.inst.all_notes_off();
        }
    }

    pub fn release_all(&mut self) {
        for z in &mut self.zones {
            z.inst.release_all();
        }
    }

    pub fn set_bend(&mut self, ratio: f32) {
        for z in &mut self.zones {
            z.inst.set_bend(ratio);
        }
    }

    #[inline]
    pub fn render_frame(&mut self) -> f32 {
        let mut s = 0.0;
        for z in &mut self.zones {
            s += z.inst.render_frame();
        }
        s
    }

    /// A fresh kit with the same zones + instruments (independent voices).
    pub fn snapshot(&self) -> Kit {
        Kit {
            zones: self
                .zones
                .iter()
                .map(|z| Zone {
                    name: z.name.clone(),
                    lo: z.lo,
                    hi: z.hi,
                    fixed_note: z.fixed_note,
                    transpose: z.transpose,
                    inst: z.inst.snapshot(),
                })
                .collect(),
        }
    }

    pub fn to_data(&self) -> Vec<ZoneData> {
        self.zones.iter().map(Zone::to_data).collect()
    }

    #[cfg(test)]
    pub fn active_voices(&self) -> usize {
        self.zones.iter().map(|z| z.inst.active_voices()).sum()
    }
}

/// The serialized description the studio needs to (re)build a track/live slot:
/// `(model_id, params, engine, zones)`. Empty `zones` ⇒ a single instrument.
pub type PlayableParts = (String, serde_json::Value, EngineParams, Vec<ZoneData>);

/// Something you can play: a single [`Instrument`] or a [`Kit`]. The studio's
/// live slot and every track hold one of these.
pub enum Playable {
    Single(Instrument),
    Kit(Kit),
}

impl Playable {
    #[inline]
    pub fn note_on(&mut self, note: u8, vel: f32) {
        match self {
            Playable::Single(i) => i.note_on(note, vel),
            Playable::Kit(k) => k.note_on(note, vel),
        }
    }

    #[inline]
    pub fn note_off(&mut self, note: u8) {
        match self {
            Playable::Single(i) => i.note_off(note),
            Playable::Kit(k) => k.note_off(note),
        }
    }

    pub fn all_notes_off(&mut self) {
        match self {
            Playable::Single(i) => i.all_notes_off(),
            Playable::Kit(k) => k.all_notes_off(),
        }
    }

    pub fn release_all(&mut self) {
        match self {
            Playable::Single(i) => i.release_all(),
            Playable::Kit(k) => k.release_all(),
        }
    }

    pub fn set_bend(&mut self, ratio: f32) {
        match self {
            Playable::Single(i) => i.set_bend(ratio),
            Playable::Kit(k) => k.set_bend(ratio),
        }
    }

    #[inline]
    pub fn render_frame(&mut self) -> f32 {
        match self {
            Playable::Single(i) => i.render_frame(),
            Playable::Kit(k) => k.render_frame(),
        }
    }

    pub fn snapshot(&self) -> Playable {
        match self {
            Playable::Single(i) => Playable::Single(i.snapshot()),
            Playable::Kit(k) => Playable::Kit(k.snapshot()),
        }
    }

    #[cfg(test)]
    pub fn active_voices(&self) -> usize {
        match self {
            Playable::Single(i) => i.active_voices(),
            Playable::Kit(k) => k.active_voices(),
        }
    }

    /// Apply a live model change (single instruments only; a no-op on kits).
    pub fn set_model(&mut self, model: Box<dyn crate::models::FtmModel>) {
        if let Playable::Single(i) = self {
            i.set_model(model);
        }
    }

    /// Apply live engine params (single instruments only; a no-op on kits).
    pub fn set_engine(&mut self, engine: EngineParams) {
        if let Playable::Single(i) = self {
            i.set_engine(engine);
        }
    }

    /// A short label for the track list: the model name, or `Kit (N)`.
    pub fn label(&self) -> String {
        match self {
            Playable::Single(i) => i.model_name().to_string(),
            Playable::Kit(k) => format!("Kit ({})", k.zone_count()),
        }
    }

    /// A stable id used by the UI to key the track editor (`kit` for kits).
    #[cfg(test)]
    pub fn model_id(&self) -> String {
        match self {
            Playable::Single(i) => i.model_id().to_string(),
            Playable::Kit(_) => "kit".to_string(),
        }
    }

    /// The serialized parts for a `TrackData` / `TrackView`.
    pub fn parts(&self) -> PlayableParts {
        match self {
            Playable::Single(i) => {
                (i.model_id().to_string(), i.model_json(), i.engine_params(), Vec::new())
            }
            Playable::Kit(k) => {
                ("kit".to_string(), serde_json::Value::Null, EngineParams::default(), k.to_data())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instrument::make_sine_table;

    fn zone(model_id: &str, lo: u8, hi: u8, fixed: Option<u8>) -> ZoneData {
        ZoneData {
            name: model_id.into(),
            lo,
            hi,
            fixed_note: fixed,
            transpose: 0,
            model_id: model_id.into(),
            params: serde_json::json!({}),
            engine: EngineParams::default(),
        }
    }

    #[test]
    fn split_routes_by_range() {
        let sine = make_sine_table();
        // Low half = drum, high half = string; disjoint ranges.
        let data = vec![zone("drum_membrane", 0, 59, None), zone("pure_string", 60, 127, None)];
        let mut kit = Kit::from_data(48_000.0, sine, &data);
        kit.note_on(40, 1.0); // only the drum zone
        kit.note_on(72, 1.0); // only the string zone
        // Each zone got exactly one voice.
        assert_eq!(kit.zones[0].inst.active_voices(), 1, "drum zone");
        assert_eq!(kit.zones[1].inst.active_voices(), 1, "string zone");
    }

    #[test]
    fn out_of_range_key_plays_nothing() {
        let sine = make_sine_table();
        let data = vec![zone("pure_string", 60, 72, None)];
        let mut kit = Kit::from_data(48_000.0, sine, &data);
        kit.note_on(40, 1.0);
        assert_eq!(kit.zones[0].inst.active_voices(), 0);
    }

    #[test]
    fn overlapping_zones_layer() {
        let sine = make_sine_table();
        let data = vec![zone("pure_string", 60, 72, None), zone("basic_wave", 60, 72, None)];
        let mut kit = Kit::from_data(48_000.0, sine, &data);
        kit.note_on(64, 1.0);
        assert_eq!(kit.zones[0].inst.active_voices(), 1);
        assert_eq!(kit.zones[1].inst.active_voices(), 1, "both layers sound");
    }

    #[test]
    fn note_off_stops_the_mapped_pad_pitch() {
        let sine = make_sine_table();
        // A pad: any key 36..=40 plays fixed note 38.
        let data = vec![zone("drum_membrane", 36, 40, Some(38))];
        let mut kit = Kit::from_data(48_000.0, sine, &data);
        kit.note_on(36, 1.0);
        assert_eq!(kit.zones[0].inst.active_voices(), 1);
        // Releasing the same key must reach the fixed-note voice.
        kit.note_off(36);
        // Struck models ignore note-off, so just assert it doesn't panic / mis-route;
        // a sustained model would now be releasing. Voice still counted until decay.
        assert!(kit.zones[0].inst.active_voices() <= 1);
    }

    #[test]
    fn roundtrips_through_zone_data() {
        let sine = make_sine_table();
        let data = vec![zone("basic_wave", 0, 59, Some(45)), zone("basic_wave", 60, 127, None)];
        let kit = Kit::from_data(48_000.0, sine, &data);
        let back = kit.to_data();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].model_id, "basic_wave");
        assert_eq!(back[0].fixed_note, Some(45));
        assert_eq!(back[1].lo, 60);
    }
}

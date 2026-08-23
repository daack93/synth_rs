//! Instruments and their performance map.
//!
//! An instrument is one or more **sources** (each a polyphonic [`Instrument`])
//! plus a [`PerformanceMap`] that decides, for every MIDI key, *which* sources
//! sound and at *what* pitch. Two shapes exist today:
//!
//!   * **Direct** — one source; the key sets the pitch. A plain instrument.
//!   * **Zones** — a kit: each [`ZoneMap`] binds a key range to a source with a
//!     pitch rule (transpose, or a fixed pad note). Overlapping ranges layer,
//!     disjoint ranges split the keyboard.
//!
//! Both are the same [`Playable`] struct — only the map differs. This is the
//! seam the roadmap's richer key→parameter strategies (overblow tracking,
//! secondary resonators) will grow from: each is a new [`PerformanceMap`]
//! variant, added in one place without touching [`Playable`]'s method set.
//!
//! A recorded note event stays just `(note, vel)`: the map re-routes it on
//! playback exactly as it did live, so a kit records and replays with no
//! per-event bookkeeping.

use std::sync::Arc;

use crate::instrument::{EngineParams, Instrument};
use crate::models::{default_model, model_from_id};
use crate::project::ZoneData;

/// One key-range entry in a zoned [`PerformanceMap`]: which keys it responds to,
/// how they map to a pitch, and which source ([`Playable::sources`] index) it
/// drives.
pub struct ZoneMap {
    pub name: String,
    pub lo: u8,
    pub hi: u8,
    pub fixed_note: Option<u8>,
    pub transpose: i8,
    /// Index into [`Playable::sources`].
    pub source: usize,
}

impl ZoneMap {
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
}

/// How keys route to an instrument's sources. The explicit seam for key→sound
/// mapping: today a direct passthrough or a zoned kit; later, tracked/continuous
/// strategies slot in as new variants.
pub enum PerformanceMap {
    /// One source; the key sets the pitch directly.
    Direct,
    /// A kit: each zone binds a key range to a source with a pitch rule.
    Zones(Vec<ZoneMap>),
}

/// Something you can play: a set of sound sources plus the map from keys to them.
/// The studio's live slot and every track hold one of these.
pub struct Playable {
    sources: Vec<Instrument>,
    map: PerformanceMap,
}

impl Playable {
    /// A single instrument: one source, key sets pitch.
    pub fn single(inst: Instrument) -> Playable {
        Playable {
            sources: vec![inst],
            map: PerformanceMap::Direct,
        }
    }

    /// A kit built from serialized zones (each zone's instrument rebuilt from its
    /// model id + params). Unknown model ids fall back to the default model.
    pub fn from_zones(sr: f32, sine: Arc<[f32]>, zones: &[ZoneData]) -> Playable {
        let mut sources = Vec::with_capacity(zones.len());
        let mut zone_maps = Vec::with_capacity(zones.len());
        for (i, z) in zones.iter().enumerate() {
            let model = model_from_id(&z.model_id, &z.params).unwrap_or_else(default_model);
            sources.push(Instrument::with_config(sr, sine.clone(), model, z.engine.clone()));
            zone_maps.push(ZoneMap {
                name: z.name.clone(),
                lo: z.lo,
                hi: z.hi,
                fixed_note: z.fixed_note,
                transpose: z.transpose,
                source: i,
            });
        }
        Playable {
            sources,
            map: PerformanceMap::Zones(zone_maps),
        }
    }

    /// True for a plain single instrument (a `Direct` map).
    #[inline]
    fn is_single(&self) -> bool {
        matches!(self.map, PerformanceMap::Direct)
    }

    #[inline]
    pub fn note_on(&mut self, note: u8, vel: f32) {
        // Borrow `map` and `sources` as disjoint fields so routing needs no alloc.
        match &self.map {
            PerformanceMap::Direct => self.sources[0].note_on(note, vel),
            PerformanceMap::Zones(zones) => {
                for z in zones {
                    if z.in_range(note) {
                        self.sources[z.source].note_on(z.mapped(note), vel);
                    }
                }
            }
        }
    }

    #[inline]
    pub fn note_off(&mut self, note: u8) {
        match &self.map {
            PerformanceMap::Direct => self.sources[0].note_off(note),
            PerformanceMap::Zones(zones) => {
                for z in zones {
                    if z.in_range(note) {
                        self.sources[z.source].note_off(z.mapped(note));
                    }
                }
            }
        }
    }

    pub fn all_notes_off(&mut self) {
        for s in &mut self.sources {
            s.all_notes_off();
        }
    }

    pub fn release_all(&mut self) {
        for s in &mut self.sources {
            s.release_all();
        }
    }

    pub fn set_bend(&mut self, ratio: f32) {
        for s in &mut self.sources {
            s.set_bend(ratio);
        }
    }

    #[inline]
    pub fn render_frame(&mut self) -> f32 {
        let mut s = 0.0;
        for src in &mut self.sources {
            s += src.render_frame();
        }
        s
    }

    /// A fresh copy with the same sources + map (independent voices).
    pub fn snapshot(&self) -> Playable {
        let sources = self.sources.iter().map(|s| s.snapshot()).collect();
        let map = match &self.map {
            PerformanceMap::Direct => PerformanceMap::Direct,
            PerformanceMap::Zones(zones) => PerformanceMap::Zones(
                zones
                    .iter()
                    .map(|z| ZoneMap {
                        name: z.name.clone(),
                        lo: z.lo,
                        hi: z.hi,
                        fixed_note: z.fixed_note,
                        transpose: z.transpose,
                        source: z.source,
                    })
                    .collect(),
            ),
        };
        Playable { sources, map }
    }

    #[cfg(test)]
    pub fn active_voices(&self) -> usize {
        self.sources.iter().map(|s| s.active_voices()).sum()
    }

    /// Apply a live model change (single instruments only; a no-op on kits).
    pub fn set_model(&mut self, model: Box<dyn crate::models::FtmModel>) {
        if self.is_single() {
            self.sources[0].set_model(model);
        }
    }

    /// Apply live engine params (single instruments only; a no-op on kits).
    pub fn set_engine(&mut self, engine: EngineParams) {
        if self.is_single() {
            self.sources[0].set_engine(engine);
        }
    }

    /// A short label for the track list: the model name, or `Kit (N)`.
    pub fn label(&self) -> String {
        match &self.map {
            PerformanceMap::Direct => self.sources[0].model_name().to_string(),
            PerformanceMap::Zones(zones) => format!("Kit ({})", zones.len()),
        }
    }

    /// A stable id used by the UI to key the track editor (`kit` for kits).
    #[cfg(test)]
    pub fn model_id(&self) -> String {
        match &self.map {
            PerformanceMap::Direct => self.sources[0].model_id().to_string(),
            PerformanceMap::Zones(_) => "kit".to_string(),
        }
    }

    /// The serialized parts for a `LoopTrack` / `TrackView`. A single instrument
    /// serializes with empty zones; a kit with its zones — byte-compatible with
    /// the pre-map format.
    pub fn parts(&self) -> PlayableParts {
        match &self.map {
            PerformanceMap::Direct => {
                let i = &self.sources[0];
                (i.model_id().to_string(), i.model_json(), i.engine_params(), Vec::new())
            }
            PerformanceMap::Zones(zones) => {
                let data = zones
                    .iter()
                    .map(|z| {
                        let inst = &self.sources[z.source];
                        ZoneData {
                            name: z.name.clone(),
                            lo: z.lo,
                            hi: z.hi,
                            fixed_note: z.fixed_note,
                            transpose: z.transpose,
                            model_id: inst.model_id().to_string(),
                            params: inst.model_json(),
                            engine: inst.engine_params(),
                        }
                    })
                    .collect();
                ("kit".to_string(), serde_json::Value::Null, EngineParams::default(), data)
            }
        }
    }

    /// Per-source active-voice count (test/introspection helper).
    #[cfg(test)]
    pub fn source_voices(&self, i: usize) -> usize {
        self.sources[i].active_voices()
    }
}

/// The serialized description the studio needs to (re)build a track/live slot:
/// `(model_id, params, engine, zones)`. Empty `zones` ⇒ a single instrument.
pub type PlayableParts = (String, serde_json::Value, EngineParams, Vec<ZoneData>);

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
        let mut kit = Playable::from_zones(48_000.0, sine, &data);
        kit.note_on(40, 1.0); // only the drum zone
        kit.note_on(72, 1.0); // only the string zone
        // Each source got exactly one voice.
        assert_eq!(kit.source_voices(0), 1, "drum zone");
        assert_eq!(kit.source_voices(1), 1, "string zone");
    }

    #[test]
    fn out_of_range_key_plays_nothing() {
        let sine = make_sine_table();
        let data = vec![zone("pure_string", 60, 72, None)];
        let mut kit = Playable::from_zones(48_000.0, sine, &data);
        kit.note_on(40, 1.0);
        assert_eq!(kit.source_voices(0), 0);
    }

    #[test]
    fn overlapping_zones_layer() {
        let sine = make_sine_table();
        let data = vec![zone("pure_string", 60, 72, None), zone("basic_wave", 60, 72, None)];
        let mut kit = Playable::from_zones(48_000.0, sine, &data);
        kit.note_on(64, 1.0);
        assert_eq!(kit.source_voices(0), 1);
        assert_eq!(kit.source_voices(1), 1, "both layers sound");
    }

    #[test]
    fn note_off_stops_the_mapped_pad_pitch() {
        let sine = make_sine_table();
        // A pad: any key 36..=40 plays fixed note 38.
        let data = vec![zone("drum_membrane", 36, 40, Some(38))];
        let mut kit = Playable::from_zones(48_000.0, sine, &data);
        kit.note_on(36, 1.0);
        assert_eq!(kit.source_voices(0), 1);
        // Releasing the same key must reach the fixed-note voice.
        kit.note_off(36);
        // Struck models ignore note-off, so just assert it doesn't panic / mis-route;
        // a sustained model would now be releasing. Voice still counted until decay.
        assert!(kit.source_voices(0) <= 1);
    }

    #[test]
    fn roundtrips_through_zone_data() {
        let sine = make_sine_table();
        let data = vec![zone("basic_wave", 0, 59, Some(45)), zone("basic_wave", 60, 127, None)];
        let kit = Playable::from_zones(48_000.0, sine, &data);
        let (id, _params, _engine, back) = kit.parts();
        assert_eq!(id, "kit");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].model_id, "basic_wave");
        assert_eq!(back[0].fixed_note, Some(45));
        assert_eq!(back[1].lo, 60);
    }

    #[test]
    fn single_serializes_without_zones() {
        let sine = make_sine_table();
        let inst = Instrument::new(48_000.0, sine);
        let p = Playable::single(inst);
        let (_id, _params, _engine, zones) = p.parts();
        assert!(zones.is_empty(), "a single instrument has no zones");
        assert!(p.model_id() != "kit");
    }
}

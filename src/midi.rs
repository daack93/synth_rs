//! MIDI input via `midir`. Connects to a chosen port and forwards note-on /
//! note-off messages into the audio command channel.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

use midir::{MidiInput, MidiInputConnection};

use crate::studio::Command;

/// Pitch-wheel bend range in semitones (the usual default).
const PITCH_WHEEL_RANGE: f32 = 2.0;

/// A lock-free "last note played" slot the UI can poll for MIDI-learn — every
/// note-on (from MIDI, the on-screen piano, or the computer keyboard) bumps a
/// generation counter and records the note, so a UI field armed for learn can
/// notice the next note and capture it. Packs `gen` (high 24 bits) + `note`.
#[derive(Clone, Default)]
pub struct NoteMonitor(Arc<AtomicU32>);

impl NoteMonitor {
    pub fn record(&self, note: u8) {
        let gen = (self.0.load(Ordering::Relaxed) >> 8).wrapping_add(1);
        self.0.store((gen << 8) | note as u32, Ordering::Relaxed);
    }
    /// `(generation, note)` — the generation changes on every recorded note.
    pub fn latest(&self) -> (u32, u8) {
        let v = self.0.load(Ordering::Relaxed);
        (v >> 8, (v & 0xff) as u8)
    }
}

pub struct MidiInputHandle {
    _conn: MidiInputConnection<()>,
    pub port_name: String,
}

/// List the names of available MIDI input ports.
pub fn list_ports() -> Vec<String> {
    let Ok(midi_in) = MidiInput::new("ftm_synth-list") else {
        return Vec::new();
    };
    midi_in
        .ports()
        .iter()
        .map(|p| midi_in.port_name(p).unwrap_or_else(|_| "<unknown>".into()))
        .collect()
}

/// Connect to the input port at `index`, forwarding events to `tx`.
pub fn connect(index: usize, tx: Sender<Command>, monitor: NoteMonitor) -> Result<MidiInputHandle, String> {
    let mut midi_in = MidiInput::new("ftm_synth").map_err(|e| e.to_string())?;
    midi_in.ignore(midir::Ignore::None);
    let ports = midi_in.ports();
    let port = ports
        .get(index)
        .ok_or_else(|| format!("no MIDI port at index {index}"))?;
    let port_name = midi_in.port_name(port).unwrap_or_else(|_| "<unknown>".into());

    let conn = midi_in
        .connect(
            port,
            "ftm_synth-in",
            move |_stamp, message, _| {
                handle_message(message, &tx, &monitor);
            },
            (),
        )
        .map_err(|e| e.to_string())?;

    Ok(MidiInputHandle {
        _conn: conn,
        port_name,
    })
}

fn handle_message(message: &[u8], tx: &Sender<Command>, monitor: &NoteMonitor) {
    if message.len() < 3 {
        return;
    }
    let status = message[0] & 0xF0;
    let note = message[1];
    let data2 = message[2];
    match status {
        0x90 => {
            if data2 == 0 {
                // note-on with velocity 0 == note-off
                let _ = tx.send(Command::NoteOff { note });
            } else {
                let vel = data2 as f32 / 127.0;
                monitor.record(note); // let the UI's MIDI-learn see it
                let _ = tx.send(Command::NoteOn { note, vel });
            }
        }
        0x80 => {
            let _ = tx.send(Command::NoteOff { note });
        }
        0xB0 if note == 123 => {
            // All notes off (CC 123)
            let _ = tx.send(Command::AllNotesOff);
        }
        0xE0 => {
            // Pitch wheel: 14-bit (LSB, MSB), centre 8192. Map to ±2 semitones.
            let value = ((data2 as i32) << 7) | note as i32;
            let semitones = (value - 8192) as f32 / 8192.0 * PITCH_WHEEL_RANGE;
            let _ = tx.send(Command::SetBend(semitones));
        }
        _ => {}
    }
}

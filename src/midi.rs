//! MIDI input via `midir`. Connects to a chosen port and forwards note-on /
//! note-off messages into the audio command channel.

use std::sync::mpsc::Sender;

use midir::{MidiInput, MidiInputConnection};

use crate::studio::Command;

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
pub fn connect(index: usize, tx: Sender<Command>) -> Result<MidiInputHandle, String> {
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
                handle_message(message, &tx);
            },
            (),
        )
        .map_err(|e| e.to_string())?;

    Ok(MidiInputHandle {
        _conn: conn,
        port_name,
    })
}

fn handle_message(message: &[u8], tx: &Sender<Command>) {
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
        _ => {}
    }
}

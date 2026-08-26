//! MIDI input via `midir`. Connects to a chosen port and forwards note-on /
//! note-off messages into the audio command channel.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

use midir::{MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};

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

/// A small shared ring of recent Launchkey DAW-port messages, so the UI can show
/// them (no terminal needed) — used to identify the transport/encoder controls.
#[derive(Clone, Default)]
pub struct DawMonitor(Arc<std::sync::Mutex<std::collections::VecDeque<String>>>);

impl DawMonitor {
    fn push(&self, line: String) {
        if let Ok(mut q) = self.0.lock() {
            q.push_back(line);
            while q.len() > 40 {
                q.pop_front();
            }
        }
    }
    /// Recent messages, newest last.
    pub fn lines(&self) -> Vec<String> {
        self.0
            .lock()
            .map(|q| q.iter().cloned().collect())
            .unwrap_or_default()
    }
    pub fn clear(&self) {
        if let Ok(mut q) = self.0.lock() {
            q.clear();
        }
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
        0xB0 if note == 1 || note == 2 || note == 11 => {
            // Mod wheel (CC1), breath (CC2), or expression (CC11) → mouth pressure.
            // Rest (0) = nominal; pushing up blows harder (louder + sharper).
            let _ = tx.send(Command::SetBreath(1.0 + (data2 as f32 / 127.0) * 0.6));
        }
        0xD0 => {
            // Channel pressure (aftertouch) → mouth pressure. Press a held key
            // harder to swell and bend the note, the way a player leans on it.
            let _ = tx.send(Command::SetBreath(1.0 + (note as f32 / 127.0) * 0.6));
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


// ============================================================================
// Novation Launchkey Mk4 — DAW-mode control-surface support.
//
// The Launchkey exposes TWO USB-MIDI port pairs: a "MIDI" pair (keys, wheels,
// pads/pots in performance modes) and a "DAW" pair (the control surface —
// transport, encoders, pads, and feedback to the LEDs + screen). We connect
// BOTH inputs (keys → notes as usual; DAW → transport/controls) and open the DAW
// OUTPUT so we can send the handshakes and, later, LED/screen feedback.
//
// Protocol (from the Mk4 Programmer's Reference, regular-SKU header 00 20 29
// 02 14): enter DAW mode with Note-On ch16 note 0x0C vel 0x7F; enable the
// feature controls (tempo/LED/screen settings) with Note-On ch16 note 0x0B vel
// 0x7F. Transport buttons report as CC on channel 16; encoders (Plugin/Mixer/
// Sends) as CC 0x15–0x1C ch16; the Transport encoder mode as relative CC
// 0x55–0x5C ch16; DAW pads as notes on ch1.
// ============================================================================

/// SysEx header for regular-SKU Launchkeys (the 49 Mk4 is a regular SKU).
#[allow(dead_code)]
pub const LK_SYSEX_HEADER: [u8; 6] = [0xF0, 0x00, 0x20, 0x29, 0x02, 0x14];

const LK_DAW_MODE_ON: [u8; 3] = [0x9F, 0x0C, 0x7F];
const LK_DAW_MODE_OFF: [u8; 3] = [0x9F, 0x0C, 0x00];
const LK_FEATURE_ON: [u8; 3] = [0x9F, 0x0B, 0x7F];

/// A live connection to a Launchkey in DAW mode. Holds the keys input, the DAW
/// input, and the DAW output (kept open for LED/screen feedback). Dropping it
/// returns the device to standalone mode.
pub struct LaunchkeyHandle {
    _keys_in: MidiInputConnection<()>,
    _daw_in: MidiInputConnection<()>,
    daw_out: MidiOutputConnection,
    pub keys_port: String,
    pub daw_port: String,
}

impl Drop for LaunchkeyHandle {
    fn drop(&mut self) {
        // Return the Launchkey to standalone (MIDI) mode.
        let _ = self.daw_out.send(&LK_DAW_MODE_OFF);
    }
}

impl LaunchkeyHandle {
    /// Send raw bytes to the DAW port (for LED colour / screen feedback).
    #[allow(dead_code)]
    pub fn send(&mut self, bytes: &[u8]) {
        let _ = self.daw_out.send(bytes);
    }
}

/// Find the (keys, daw) input port names for a connected Launchkey. Returns the
/// two port names if a Launchkey is present. The DAW port's name contains "DAW".
pub fn find_launchkey_ports() -> Option<(String, String)> {
    let midi_in = MidiInput::new("ftm_synth-lk-scan").ok()?;
    let names: Vec<String> = midi_in
        .ports()
        .iter()
        .map(|p| midi_in.port_name(p).unwrap_or_default())
        .collect();
    let is_lk = |n: &str| {
        let n = n.to_lowercase();
        n.contains("launchkey") || n.contains("launch key")
    };
    let daw = names.iter().find(|n| is_lk(n) && n.to_lowercase().contains("daw"))?;
    // The keys port is the other Launchkey port (not the DAW one).
    let keys = names
        .iter()
        .find(|n| is_lk(n) && !n.to_lowercase().contains("daw"))
        .unwrap_or(daw);
    Some((keys.clone(), daw.clone()))
}

fn port_by_name<T: midir::MidiIO>(io: &T, name: &str) -> Option<T::Port> {
    io.ports()
        .into_iter()
        .find(|p| io.port_name(p).as_deref() == Ok(name))
}

/// Connect to a Launchkey in DAW mode: open both inputs and the DAW output,
/// send the DAW-mode + feature-control handshakes, and forward keys as notes and
/// DAW-surface events as transport commands (logging the rest so the exact
/// button CCs can be confirmed against the hardware).
pub fn connect_launchkey(
    tx: Sender<Command>,
    monitor: NoteMonitor,
    log: DawMonitor,
) -> Result<LaunchkeyHandle, String> {
    let (keys_name, daw_name) = find_launchkey_ports().ok_or("no Launchkey found")?;

    // --- keys input (notes / wheels, as usual) ---
    let mut keys_in = MidiInput::new("ftm_synth-lk-keys").map_err(|e| e.to_string())?;
    keys_in.ignore(midir::Ignore::None);
    let kp = port_by_name(&keys_in, &keys_name).ok_or("keys port vanished")?;
    let tx_keys = tx.clone();
    let mon = monitor.clone();
    let keys_conn = keys_in
        .connect(&kp, "ftm_synth-lk-keys", move |_, m, _| handle_message(m, &tx_keys, &mon), ())
        .map_err(|e| e.to_string())?;

    // --- DAW input (transport / encoders / pads) ---
    let mut daw_in = MidiInput::new("ftm_synth-lk-daw").map_err(|e| e.to_string())?;
    daw_in.ignore(midir::Ignore::None);
    let dp = port_by_name(&daw_in, &daw_name).ok_or("DAW port vanished")?;
    let tx_daw = tx.clone();
    let log_daw = log.clone();
    let daw_conn = daw_in
        .connect(&dp, "ftm_synth-lk-daw", move |_, m, _| handle_daw_message(m, &tx_daw, &log_daw), ())
        .map_err(|e| e.to_string())?;

    // --- DAW output (handshakes + feedback) ---
    let daw_out_io = MidiOutput::new("ftm_synth-lk-out").map_err(|e| e.to_string())?;
    let op = port_by_name(&daw_out_io, &daw_name).ok_or("DAW out port vanished")?;
    let mut daw_out = daw_out_io.connect(&op, "ftm_synth-lk-out").map_err(|e| e.to_string())?;
    daw_out.send(&LK_DAW_MODE_ON).map_err(|e| e.to_string())?;
    daw_out.send(&LK_FEATURE_ON).map_err(|e| e.to_string())?;

    Ok(LaunchkeyHandle {
        _keys_in: keys_conn,
        _daw_in: daw_conn,
        daw_out,
        keys_port: keys_name,
        daw_port: daw_name,
    })
}

/// Handle a message on the Launchkey DAW port. Transport buttons map to the
/// arrangement transport; everything else is logged so the button/encoder CCs
/// can be confirmed against the hardware.
fn handle_daw_message(message: &[u8], tx: &Sender<Command>, log: &DawMonitor) {
    if message.len() < 3 {
        return;
    }
    let status = message[0];
    let d1 = message[1];
    let d2 = message[2];
    // Transport buttons report as CC on channel 16 (BFh). The exact CC numbers
    // are in the reference's surface figures; these are the Mk3-lineage defaults
    // and are logged so we can confirm/correct them from live hardware.
    if status == 0xBF {
        match d1 {
            0x73 if d2 > 0 => {
                let _ = tx.send(Command::Stop);
            }
            0x74 if d2 > 0 => {
                let _ = tx.send(Command::Play);
            }
            0x75 if d2 > 0 => {
                let _ = tx.send(Command::Record);
            }
            _ => {}
        }
    }
    // Record every DAW-port message so the UI can show it (identify the controls).
    let kind = match status & 0xF0 {
        0x90 => "note-on",
        0x80 => "note-off",
        0xB0 => "CC",
        0xA0 => "aftertouch",
        _ => "?",
    };
    let ch = (status & 0x0F) + 1;
    log.push(format!("ch{ch:<2} {kind:<9} num={d1:>3} (0x{d1:02X})  val={d2:>3}"));
}

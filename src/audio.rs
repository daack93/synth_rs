//! cpal audio-output wiring.
//!
//! The [`Studio`] is the synth/sequencer state. It is *moved into* the audio
//! callback so it lives on the real-time thread — deliberately, not by
//! accident. The callback mutates it every buffer (advancing the transport,
//! firing notes, summing voices), and doing that from a real-time thread must be
//! lock-free: it can't block on a mutex the UI might be holding without risking
//! xruns. So the two threads don't *share* the Studio — they pass messages:
//!
//!   * the UI / MIDI threads send [`Command`]s down an mpsc channel, drained at
//!     the top of each callback;
//!   * the UI reads back a [`SharedView`] (atomics / lock-free snapshot) for
//!     meters and playhead.
//!
//! So conceptually the app owns the Studio's *control surface* (the command
//! sender + the view) while the real-time thread owns the *live instance*. The
//! audio module is the thin layer that owns the device and pumps that instance;
//! it is not "above" the studio in the architecture.

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::studio::{Command, SharedView, Studio};

pub struct AudioEngine {
    _stream: cpal::Stream,
    pub sample_rate: f32,
    /// Transport / track state the UI reads.
    pub view: Arc<SharedView>,
}

impl AudioEngine {
    /// Build and start the output stream. `rx` delivers note/transport commands.
    pub fn start(rx: Receiver<Command>) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no default output device".to_string())?;
        let config = device
            .default_output_config()
            .map_err(|e| format!("default output config: {e}"))?;

        let sample_rate = config.sample_rate().0 as f32;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();
        let stream_config: cpal::StreamConfig = config.into();

        let mut studio = Studio::new(sample_rate);
        let view = studio.view();
        let err_fn = |e| eprintln!("audio stream error: {e}");

        // Shared render closure: drain commands, then fill the buffer.
        macro_rules! build {
            ($t:ty, $convert:expr) => {{
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                        while let Ok(cmd) = rx.try_recv() {
                            studio.handle(cmd);
                        }
                        let mut scratch = vec![0.0f32; data.len()];
                        studio.render(&mut scratch, channels);
                        for (o, s) in data.iter_mut().zip(scratch.iter()) {
                            *o = $convert(*s);
                        }
                    },
                    err_fn,
                    None,
                )
            }};
        }

        let stream = match sample_format {
            cpal::SampleFormat::F32 => build!(f32, |s: f32| s),
            cpal::SampleFormat::I16 => build!(i16, |s: f32| (s.clamp(-1.0, 1.0) * 32767.0) as i16),
            cpal::SampleFormat::U16 => {
                build!(u16, |s: f32| (((s.clamp(-1.0, 1.0) * 0.5) + 0.5) * 65535.0) as u16)
            }
            other => return Err(format!("unsupported sample format: {other:?}")),
        }
        .map_err(|e| format!("build stream: {e}"))?;

        stream.play().map_err(|e| format!("play: {e}"))?;

        Ok(AudioEngine {
            _stream: stream,
            sample_rate,
            view,
        })
    }
}

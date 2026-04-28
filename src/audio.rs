use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use crossbeam_channel::Sender;
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use std::path::Path;
use tracing::{info, warn};

pub const WHISPER_SR: u32 = 16_000;

pub fn list_input_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("Input devices:");
    for dev in host.input_devices()? {
        let name = dev.name().unwrap_or_else(|_| "<unknown>".into());
        let cfg = dev
            .default_input_config()
            .map(|c| format!("{} ch, {} Hz, {:?}", c.channels(), c.sample_rate().0, c.sample_format()))
            .unwrap_or_else(|e| format!("<no default cfg: {e}>"));
        println!("  - {name}  [{cfg}]");
    }
    Ok(())
}

pub fn input_device_names() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|iter| iter.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

pub fn pick_input_device(name: Option<&str>) -> Result<Device> {
    let host = cpal::default_host();
    match name {
        Some(n) => host
            .input_devices()?
            .find(|d| d.name().ok().as_deref() == Some(n))
            .ok_or_else(|| anyhow!("input device '{n}' not found")),
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device")),
    }
}

/// Open the input stream briefly to verify we actually receive non-zero
/// samples. On macOS, when the microphone TCC permission is missing, cpal
/// happily streams silence (zeroed buffers) without raising an error — the
/// only way to detect this is to peek at the data.
///
/// Returns Ok(()) on real audio, Err with a message hinting at the macOS
/// privacy pane on silence. Blocks for up to `wait_ms` milliseconds.
pub fn check_microphone_permission(device: &Device, wait_ms: u64) -> Result<()> {
    use cpal::traits::StreamTrait;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    let supported = device.default_input_config()?;
    let config: StreamConfig = supported.clone().into();
    let peak_bits = Arc::new(AtomicU32::new(0));
    let peak_clone = peak_bits.clone();

    let update_peak = move |samples: &[f32]| {
        let mut local = 0.0f32;
        for &s in samples {
            let a = s.abs();
            if a > local {
                local = a;
            }
        }
        // Compare-and-swap monotonic max
        loop {
            let cur_bits = peak_clone.load(Ordering::Relaxed);
            let cur = f32::from_bits(cur_bits);
            if local <= cur {
                break;
            }
            if peak_clone
                .compare_exchange(cur_bits, local.to_bits(), Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    };

    let err_fn = |_| {};
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => {
            let mut up = update_peak;
            device.build_input_stream(&config, move |d: &[f32], _| up(d), err_fn, None)?
        }
        cpal::SampleFormat::I16 => {
            let mut up = update_peak;
            device.build_input_stream(
                &config,
                move |d: &[i16], _| {
                    let f: Vec<f32> = d.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                    up(&f);
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let mut up = update_peak;
            device.build_input_stream(
                &config,
                move |d: &[u16], _| {
                    let f: Vec<f32> = d.iter().map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0).collect();
                    up(&f);
                },
                err_fn,
                None,
            )?
        }
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    };

    stream.play()?;
    std::thread::sleep(std::time::Duration::from_millis(wait_ms));
    drop(stream);

    let peak = f32::from_bits(peak_bits.load(Ordering::Relaxed));
    if peak < 1e-6 {
        Err(anyhow!(
            "microphone returned only silence over {}ms — permission likely denied. \
             On macOS: System Settings → Privacy & Security → Microphone, enable this binary, \
             or rebuild after `tccutil reset Microphone`.",
            wait_ms
        ))
    } else {
        info!("microphone OK (peak={:.4} over {}ms)", peak, wait_ms);
        Ok(())
    }
}

pub struct CaptureHandle {
    _stream: Option<Stream>,
    pub input_sr: u32,
    pub input_channels: u16,
}

/// Read a WAV file and stream mono samples through `tx`. Blocks on a worker
/// thread until the file is fully consumed, then drops the sender so the
/// receiver sees `Disconnected` and exits cleanly. Used by `--input-file`
/// for offline tests; converts non-mono to mono by averaging channels.
pub fn start_file_capture(path: &Path, tx: Sender<Vec<f32>>) -> Result<CaptureHandle> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| anyhow!("open wav '{}': {}", path.display(), e))?;
    let spec = reader.spec();
    let input_sr = spec.sample_rate;
    let input_channels = spec.channels;

    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i16::MAX as f32)
            .collect(),
        (hound::SampleFormat::Int, 24) | (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i32::MAX as f32)
            .collect(),
        (fmt, bits) => return Err(anyhow!("unsupported wav format: {bits}-bit {fmt:?}")),
    };

    let mono = if input_channels == 1 {
        interleaved
    } else {
        to_mono(&interleaved, input_channels)
    };

    info!(
        "file capture: {} '{}' @ {} Hz, {} ch ({} mono samples, {:.2}s)",
        if input_channels == 1 { "mono" } else { "stereo->mono" },
        path.display(),
        input_sr,
        input_channels,
        mono.len(),
        mono.len() as f32 / input_sr as f32,
    );

    let chunk_samples = (input_sr as usize / 50).max(1); // ~20ms mono
    std::thread::Builder::new()
        .name("pico-file-capture".into())
        .spawn(move || {
            for chunk in mono.chunks(chunk_samples) {
                if tx.send(chunk.to_vec()).is_err() {
                    break;
                }
            }
        })?;

    Ok(CaptureHandle {
        _stream: None,
        input_sr,
        input_channels: 1,
    })
}

pub fn start_capture(device: &Device, tx: Sender<Vec<f32>>) -> Result<CaptureHandle> {
    let supported = device.default_input_config()?;
    info!(
        "capture: device={}, sr={}, channels={}, format={:?}",
        device.name().unwrap_or_default(),
        supported.sample_rate().0,
        supported.channels(),
        supported.sample_format()
    );

    let input_sr = supported.sample_rate().0;
    let input_channels = supported.channels();
    let config: StreamConfig = supported.clone().into();

    let err_fn = |e| warn!("cpal stream error: {e}");

    // Diagnostic: log raw input level once per second so we can tell whether
    // cpal is delivering silence (mic permission denied / muted device) vs
    // real audio that the VAD is rejecting.
    let mut last_log = std::time::Instant::now();
    let mut peak_since_log: f32 = 0.0;
    let log_raw = move |data: &[f32]| {
        for &s in data {
            let a = s.abs();
            if a > peak_since_log {
                peak_since_log = a;
            }
        }
        if last_log.elapsed() >= std::time::Duration::from_secs(1) {
            info!("[mic raw] peak={:.4} samples={}", peak_since_log, data.len());
            peak_since_log = 0.0;
            last_log = std::time::Instant::now();
        }
    };

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => {
            let mut log_raw = log_raw;
            device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                log_raw(data);
                let mono = to_mono(data, input_channels);
                let _ = tx.try_send(mono);
            },
            err_fn,
            None,
        )?
        },
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                let mono = to_mono(&f, input_channels);
                let _ = tx.try_send(mono);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _| {
                let f: Vec<f32> = data.iter().map(|s| (*s as f32 / u16::MAX as f32) * 2.0 - 1.0).collect();
                let mono = to_mono(&f, input_channels);
                let _ = tx.try_send(mono);
            },
            err_fn,
            None,
        )?,
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    };

    stream.play()?;
    Ok(CaptureHandle { _stream: Some(stream), input_sr, input_channels })
}

fn to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return interleaved.to_vec();
    }
    let ch = channels as usize;
    interleaved
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

pub struct MonoResampler {
    inner: Option<SincFixedIn<f32>>,
    input_sr: u32,
    output_sr: u32,
    chunk: usize,
    buffer: Vec<f32>,
}

impl MonoResampler {
    pub fn new(input_sr: u32, output_sr: u32) -> Result<Self> {
        if input_sr == output_sr {
            return Ok(Self { inner: None, input_sr, output_sr, chunk: 0, buffer: Vec::new() });
        }
        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };
        let chunk = 1024;
        let ratio = output_sr as f64 / input_sr as f64;
        let r = SincFixedIn::<f32>::new(ratio, 1.0, params, chunk, 1)?;
        Ok(Self { inner: Some(r), input_sr, output_sr, chunk, buffer: Vec::with_capacity(chunk * 4) })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        if self.inner.is_none() {
            return Ok(samples.to_vec());
        }
        self.buffer.extend_from_slice(samples);
        let mut out = Vec::new();
        let r = self.inner.as_mut().unwrap();
        while self.buffer.len() >= self.chunk {
            let input: Vec<f32> = self.buffer.drain(..self.chunk).collect();
            let processed = r.process(&[input], None)?;
            out.extend_from_slice(&processed[0]);
        }
        Ok(out)
    }
}

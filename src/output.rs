use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapCons, HeapProd, HeapRb,
};
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

pub fn list_output_devices() -> Result<()> {
    let host = cpal::default_host();
    println!("Output devices:");
    for dev in host.output_devices()? {
        let name = dev.name().unwrap_or_else(|_| "<unknown>".into());
        let cfg = dev
            .default_output_config()
            .map(|c| {
                format!(
                    "{} ch, {} Hz, {:?}",
                    c.channels(),
                    c.sample_rate().0,
                    c.sample_format()
                )
            })
            .unwrap_or_else(|e| format!("<no default cfg: {e}>"));
        println!("  - {name}  [{cfg}]");
    }
    Ok(())
}

pub fn output_device_names() -> Vec<String> {
    let host = cpal::default_host();
    host.output_devices()
        .map(|iter| iter.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

/// Send-safe push side of the output: holds the ringbuffer producers and
/// the resampling parameters. The worker thread keeps this and writes
/// synthesized PCM into it. Separated from `MultichannelOutput` so the
/// !Send `cpal::Stream` stays on the main thread.
pub struct OutputSink {
    producers: Vec<HeapProd<f32>>,
    source_sr: u32,
    sample_rate: u32,
}

impl OutputSink {
    /// Push synthesized PCM to lane. Resamples to device rate, then writes
    /// into the lock-free ringbuffer. When the buffer is full (very long
    /// utterances queued behind a still-playing one) we wait for the audio
    /// callback to drain some space rather than dropping audio outright —
    /// this is backpressure, the worker pauses momentarily until the
    /// device catches up. Hard ceiling at `max_wait_ms` so a stalled
    /// device can't deadlock the worker.
    pub fn push_pcm(&mut self, lane: usize, samples: &[f32]) -> Result<usize> {
        if lane >= self.producers.len() {
            return Err(anyhow!("lane {} out of range", lane));
        }
        let upsampled = resample_linear(samples, self.source_sr, self.sample_rate);
        const MAX_WAIT_MS: u64 = 5_000;
        let start = std::time::Instant::now();
        let mut written = 0;
        while written < upsampled.len() {
            let n = self.producers[lane].push_slice(&upsampled[written..]);
            written += n;
            if written == upsampled.len() {
                break;
            }
            if start.elapsed().as_millis() as u64 >= MAX_WAIT_MS {
                warn!(
                    "lane {}: output buffer full for {}ms — dropped {} / {} samples",
                    lane,
                    MAX_WAIT_MS,
                    upsampled.len() - written,
                    upsampled.len()
                );
                break;
            }
            // Audio callback drains the ring at sample-rate; a 5-10ms
            // sleep gives it room without burning CPU.
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        Ok(written)
    }
}

pub struct MultichannelOutput {
    _stream: Stream,
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
    /// Shared with the audio callback: atomic channel-start index per lane.
    pub channel_starts: Arc<Vec<AtomicU16>>,
}

impl MultichannelOutput {
    /// Open the device and spawn an output stream driven by the given atomic
    /// channel-start handles. The atomics are captured by the audio callback
    /// so mutation from anywhere (e.g. GUI) takes effect on the next frame.
    pub fn open(
        device_name: Option<&str>,
        channel_starts: Arc<Vec<AtomicU16>>,
        volumes: Arc<Vec<AtomicU32>>,
        output_peaks: Arc<Vec<AtomicU32>>,
        source_sr: u32,
    ) -> Result<(Self, OutputSink)> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => host
                .output_devices()?
                .find(|d| d.name().ok().as_deref() == Some(name))
                .ok_or_else(|| anyhow!("output device '{}' not found", name))?,
            None => host
                .default_output_device()
                .ok_or_else(|| anyhow!("no default output device"))?,
        };

        let actual_name = device.name().unwrap_or_else(|_| "<unknown>".into());
        let supported = device.default_output_config()?;
        let sample_rate = supported.sample_rate().0;
        let channels = supported.channels();
        info!(
            "output device '{}': sr={}, ch={}, format={:?}",
            actual_name,
            sample_rate,
            channels,
            supported.sample_format()
        );

        // Big ring buffers: up to ~60s of output at device rate. Kokoro can
        // produce 10+s utterances in one synth and they queue up while a
        // previous TTS is still playing back; smaller buffers were
        // dropping the head of long sentences.
        let buf_capacity = (sample_rate as usize * 60).max(48_000);
        let mut producers = Vec::with_capacity(channel_starts.len());
        let mut consumers: Vec<HeapCons<f32>> = Vec::with_capacity(channel_starts.len());
        for _ in 0..channel_starts.len() {
            let rb: HeapRb<f32> = HeapRb::new(buf_capacity);
            let (prod, cons) = rb.split();
            producers.push(prod);
            consumers.push(cons);
        }

        let total_channels = channels as usize;
        let channel_starts_cb = channel_starts.clone();
        let volumes_cb = volumes.clone();
        let peaks_cb = output_peaks.clone();
        let n_lanes = consumers.len();

        let err_fn = |e| warn!("cpal output stream error: {e}");
        let config = supported.clone().config();

        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    let frames = data.len() / total_channels;
                    // Per-lane peak for this callback. Rendered as a VU meter
                    // by the GUI; reset per callback so the meter reacts in
                    // ~10ms (one buffer @ 48kHz). Decay is the GUI's job.
                    let mut local_peaks = vec![0.0f32; n_lanes];
                    for f in 0..frames {
                        let off = f * total_channels;
                        for c in 0..total_channels {
                            data[off + c] = 0.0;
                        }
                        for (lane_idx, cons) in consumers.iter_mut().enumerate() {
                            let raw = cons.try_pop().unwrap_or(0.0);
                            let vol = f32::from_bits(volumes_cb[lane_idx].load(Ordering::Relaxed));
                            let s = raw * vol;
                            let a = s.abs();
                            if a > local_peaks[lane_idx] {
                                local_peaks[lane_idx] = a;
                            }
                            let ch = channel_starts_cb[lane_idx].load(Ordering::Relaxed) as usize;
                            if ch < total_channels {
                                data[off + ch] = s;
                            }
                            if ch + 1 < total_channels {
                                data[off + ch + 1] = s;
                            }
                        }
                    }
                    for (lane_idx, peak) in local_peaks.iter().enumerate() {
                        peaks_cb[lane_idx].store(peak.to_bits(), Ordering::Relaxed);
                    }
                },
                err_fn,
                None,
            )?,
            other => return Err(anyhow!("unsupported output format {other:?}")),
        };

        stream.play()?;

        let sink = OutputSink {
            producers,
            source_sr,
            sample_rate,
        };
        let device = Self {
            _stream: stream,
            device_name: actual_name,
            sample_rate,
            channels,
            channel_starts,
        };
        Ok((device, sink))
    }
}

fn resample_linear(samples: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    if in_sr == out_sr || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = out_sr as f64 / in_sr as f64;
    let out_len = ((samples.len() as f64) * ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = samples.len() - 1;
    for i in 0..out_len {
        let src_idx_f = i as f64 / ratio;
        let src_idx = src_idx_f as usize;
        let frac = (src_idx_f - src_idx as f64) as f32;
        let a = samples[src_idx.min(last)];
        let b = samples[(src_idx + 1).min(last)];
        out.push(a * (1.0 - frac) + b * frac);
    }
    out
}

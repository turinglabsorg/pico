//! Energy-based VAD with hysteresis. Emits speech utterances when a voiced
//! region is followed by a configurable amount of trailing silence, or when
//! the utterance exceeds `max_speech_samples`.

use tracing::{info, debug};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VadState {
    Silent,
    Speaking,
}

pub struct VadBuffer {
    buf: Vec<f32>,
    cursor: usize,
    state: VadState,
    speech_start: usize,
    silence_len: usize,

    window: usize,
    rms_high: f32,
    rms_low: f32,
    /// Per-window ZCR ceiling. Speech sits ≤ ~0.15 (typical 0.05–0.10);
    /// applause / crowd noise / hiss usually exceeds 0.20. A window above
    /// this is treated as non-speech regardless of energy.
    zcr_max: f32,
    min_silence_samples: usize,
    max_speech_samples: usize,
    preroll_samples: usize,
}

#[derive(Clone, Copy)]
pub struct VadConfig {
    pub sample_rate: u32,
    pub window_ms: u32,          // 20-40 ms typical
    pub rms_high: f32,           // above this → speech starts
    pub rms_low: f32,            // below this → silence candidate
    pub zcr_max: f32,            // above this → treated as non-speech (noise/applause)
    pub min_silence_ms: u32,     // trailing silence required to close utterance
    pub max_speech_ms: u32,      // force-flush if no silence
    pub preroll_ms: u32,         // samples kept before speech_start to give STT context
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            window_ms: 30,
            rms_high: 0.0035,
            rms_low: 0.0018,
            // overridden by main below — kept here for tests/standalone
            zcr_max: 0.18,
            min_silence_ms: 600,
            max_speech_ms: 8_000,
            preroll_ms: 200,
        }
    }
}

impl VadBuffer {
    pub fn new(cfg: VadConfig) -> Self {
        let ms_to_samples = |ms: u32| (ms as usize * cfg.sample_rate as usize) / 1000;
        info!("VAD initialized: sr={}, window_ms={}, rms_high={}, rms_low={}, zcr_max={}, min_silence_ms={}, max_speech_ms={}",
            cfg.sample_rate, cfg.window_ms, cfg.rms_high, cfg.rms_low, cfg.zcr_max, cfg.min_silence_ms, cfg.max_speech_ms);
        Self {
            buf: Vec::with_capacity(cfg.sample_rate as usize * 2),
            cursor: 0,
            state: VadState::Silent,
            speech_start: 0,
            silence_len: 0,
            window: ms_to_samples(cfg.window_ms).max(160),
            rms_high: cfg.rms_high,
            rms_low: cfg.rms_low,
            zcr_max: cfg.zcr_max,
            min_silence_samples: ms_to_samples(cfg.min_silence_ms),
            max_speech_samples: ms_to_samples(cfg.max_speech_ms),
            preroll_samples: ms_to_samples(cfg.preroll_ms),
        }
    }

    pub fn reset(&mut self) {
        debug!("VAD reset");
        self.buf.clear();
        self.cursor = 0;
        self.state = VadState::Silent;
        self.speech_start = 0;
        self.silence_len = 0;
    }

    /// Push mono PCM at the configured rate. Returns zero or more completed
    /// utterances ready for STT. `rms_high` and `rms_low` are read live so
    /// the GUI can tune them on the fly; pass the current Controls values.
    /// `last_rms` is filled with the RMS of the most recent window so the
    /// caller can publish a live mic-level meter.
    pub fn push(
        &mut self,
        samples: &[f32],
        rms_high: f32,
        rms_low: f32,
        last_rms: &mut f32,
    ) -> Vec<Vec<f32>> {
        self.buf.extend_from_slice(samples);
        let mut emitted = Vec::new();

        while self.cursor + self.window <= self.buf.len() {
            let block_end = self.cursor + self.window;
            let block = &self.buf[self.cursor..block_end];
            let r = rms(block);
            let z = zcr(block);
            *last_rms = r;
            let is_voiced = z <= self.zcr_max;

            match self.state {
                VadState::Silent => {
                    if r > rms_high && is_voiced {
                        debug!("VAD: Silent → Speaking at sample {} (rms={:.4}, zcr={:.3})", self.cursor, r, z);
                        self.state = VadState::Speaking;
                        self.speech_start = self.cursor.saturating_sub(self.preroll_samples);
                        self.silence_len = 0;
                    }
                }
                VadState::Speaking => {
                    if r < rms_low || !is_voiced {
                        self.silence_len += self.window;
                    } else {
                        self.silence_len = 0;
                    }
                    let speech_len = block_end - self.speech_start;
                    let flush = self.silence_len >= self.min_silence_samples
                        || speech_len >= self.max_speech_samples;
                    if flush {
                        let utterance = self.buf[self.speech_start..block_end].to_vec();
                        debug!("VAD: Speaking → Silent, flushed {} samples", utterance.len());
                        emitted.push(utterance);
                        self.buf.drain(..block_end);
                        self.cursor = 0;
                        self.state = VadState::Silent;
                        self.silence_len = 0;
                        self.speech_start = 0;
                        continue;
                    }
                }
            }
            self.cursor = block_end;
        }

        // In Silent state, evict everything except the preroll window.
        if self.state == VadState::Silent {
            if self.cursor > self.preroll_samples {
                let drop = self.cursor - self.preroll_samples;
                self.buf.drain(..drop);
                self.cursor -= drop;
            }
        }

        emitted
    }

    pub fn get_state(&self) -> VadState {
        self.state
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Zero-crossing rate normalized to [0, 1] — fraction of adjacent sample
/// pairs that change sign. Voiced speech: ~0.02-0.10. Unvoiced fricatives:
/// up to ~0.20. Crowd/applause/white noise: 0.30+.
fn zcr(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

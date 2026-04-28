use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::mt::Lang;

/// Speaker gender mode set from the GUI: 0 = Auto (pitch tracker),
/// 1 = force Male, 2 = force Female.
pub const GENDER_MODE_AUTO: u8 = 0;
pub const GENDER_MODE_MALE: u8 = 1;
pub const GENDER_MODE_FEMALE: u8 = 2;

/// Lock-free controls shared between pipeline and GUI.
#[derive(Clone)]
pub struct Controls {
    pub active: Arc<AtomicBool>,
    /// Source (input) language — index into Lang::ALL.
    pub source: Arc<AtomicU8>,
    /// Target activation per Lang::ALL slot. `true` = translate to this lang.
    /// Source lang is automatically skipped regardless of this flag.
    pub target_enabled: Arc<Vec<AtomicBool>>,
    /// Output channel start per Lang::ALL slot.
    pub channel_starts: Arc<Vec<AtomicU16>>,
    /// Per-lane linear gain (bits of f32, accessed via `volume`/`set_volume`).
    /// Applied in the audio callback right before writing to the output frame.
    pub volumes: Arc<Vec<AtomicU32>>,
    /// Gender mode for TTS voice selection (see GENDER_MODE_*).
    pub gender_mode: Arc<AtomicU8>,

    /// Current input RMS (bits of f32). Updated by the VAD on every window
    /// so the GUI can show a live mic-level meter without locking.
    pub input_rms: Arc<AtomicU32>,
    /// Live VAD thresholds (bits of f32). GUI sliders write here; the VAD
    /// reads on every window so changes take effect immediately.
    pub vad_rms_high: Arc<AtomicU32>,
    pub vad_rms_low: Arc<AtomicU32>,
    /// Per-lane peak amplitude observed in the most recent output callback
    /// (bits of f32, post-volume). Updated by the audio thread, read by the
    /// GUI for VU meters. One slot per Lang::ALL entry, even when the lane
    /// is silent or the language is disabled.
    pub output_peaks: Arc<Vec<AtomicU32>>,
    /// Per-lane voice override (Kokoro voice name). `None` means follow the
    /// session gender (Auto/Male/Female). One slot per Lang::ALL.
    pub voice_overrides: Arc<Vec<Mutex<Option<String>>>>,
    /// Bumped whenever the user changes a "session-defining" setting
    /// (source language, "Reset" button). The worker compares this against
    /// its own copy and resets per-session state (gender tracker) when it
    /// changes.
    pub session_token: Arc<AtomicU32>,

    /// GUI writes a new input device name here; pipeline consumes on restart.
    pub requested_input: Arc<Mutex<Option<String>>>,
    /// GUI writes a new output device name here; pipeline consumes on restart.
    pub requested_output: Arc<Mutex<Option<String>>>,
    /// Pipeline checks this each inner iteration; GUI sets true to force reopen.
    pub restart_flag: Arc<AtomicBool>,

    /// Populated at startup (and optionally refreshed). GUI reads for dropdowns.
    pub input_devices: Arc<Mutex<Vec<String>>>,
    pub output_devices: Arc<Mutex<Vec<String>>>,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            active: Arc::new(AtomicBool::new(true)),
            source: Arc::new(AtomicU8::new(0)),
            target_enabled: Arc::new(
                Lang::ALL.iter().map(|_| AtomicBool::new(false)).collect(),
            ),
            channel_starts: Arc::new(
                (0..Lang::ALL.len())
                    .map(|i| AtomicU16::new((i as u16) * 2))
                    .collect(),
            ),
            volumes: Arc::new(
                (0..Lang::ALL.len())
                    .map(|_| AtomicU32::new(1.0_f32.to_bits()))
                    .collect(),
            ),
            requested_input: Arc::new(Mutex::new(None)),
            requested_output: Arc::new(Mutex::new(None)),
            restart_flag: Arc::new(AtomicBool::new(false)),
            input_devices: Arc::new(Mutex::new(Vec::new())),
            output_devices: Arc::new(Mutex::new(Vec::new())),
            gender_mode: Arc::new(AtomicU8::new(GENDER_MODE_AUTO)),
            input_rms: Arc::new(AtomicU32::new(0)),
            vad_rms_high: Arc::new(AtomicU32::new(0.0035_f32.to_bits())),
            vad_rms_low: Arc::new(AtomicU32::new(0.0018_f32.to_bits())),
            output_peaks: Arc::new(
                (0..Lang::ALL.len())
                    .map(|_| AtomicU32::new(0))
                    .collect(),
            ),
            voice_overrides: Arc::new(
                (0..Lang::ALL.len())
                    .map(|_| Mutex::new(None))
                    .collect(),
            ),
            session_token: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl Controls {
    pub fn get_source(&self) -> Lang {
        Lang::from_index(self.source.load(Ordering::Relaxed)).unwrap_or(Lang::En)
    }

    pub fn is_target_enabled(&self, lang: Lang) -> bool {
        self.target_enabled[lang.index()].load(Ordering::Relaxed)
    }

    pub fn volume(&self, lang: Lang) -> f32 {
        f32::from_bits(self.volumes[lang.index()].load(Ordering::Relaxed))
    }

    pub fn set_volume(&self, lang: Lang, v: f32) {
        self.volumes[lang.index()].store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn input_rms(&self) -> f32 {
        f32::from_bits(self.input_rms.load(Ordering::Relaxed))
    }

    pub fn set_input_rms(&self, v: f32) {
        self.input_rms.store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn vad_rms_high(&self) -> f32 {
        f32::from_bits(self.vad_rms_high.load(Ordering::Relaxed))
    }

    pub fn set_vad_rms_high(&self, v: f32) {
        self.vad_rms_high.store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn vad_rms_low(&self) -> f32 {
        f32::from_bits(self.vad_rms_low.load(Ordering::Relaxed))
    }

    pub fn set_vad_rms_low(&self, v: f32) {
        self.vad_rms_low.store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn output_peak(&self, lang: Lang) -> f32 {
        f32::from_bits(self.output_peaks[lang.index()].load(Ordering::Relaxed))
    }

    pub fn voice_override(&self, lang: Lang) -> Option<String> {
        self.voice_overrides[lang.index()].lock().ok().and_then(|g| g.clone())
    }

    pub fn set_voice_override(&self, lang: Lang, voice: Option<String>) {
        if let Ok(mut g) = self.voice_overrides[lang.index()].lock() {
            *g = voice;
        }
    }
}

#[derive(Default)]
pub struct SharedState {
    pub listening: bool,
    pub input_device: String,
    pub output_device: String,
    pub output_channels: u16,
    pub ollama_model: String,

    pub src_latest: String,
    pub tr_latest_en: String,
    pub tr_latest_fr: String,
    pub tr_latest_es: String,
    pub tr_latest_it: String,

    pub stt_ms: u128,
    pub mt_ms_en: u128,
    pub mt_ms_fr: u128,
    pub mt_ms_es: u128,
    pub mt_ms_it: u128,
    pub tts_ms_en: u128,
    pub tts_ms_fr: u128,
    pub tts_ms_es: u128,
    pub tts_ms_it: u128,

    pub chunks_processed: u64,
    pub last_activity: Option<Instant>,

    pub history: VecDeque<HistoryEntry>,
    /// Ring buffer of debug log lines, newest first. Capped at 200 entries.
    pub debug_log: VecDeque<String>,
    /// Set when the input device delivers only silence (typically a missing
    /// macOS microphone permission). The GUI shows a banner pointing to the
    /// privacy pane.
    pub mic_permission_warning: Option<String>,
    /// Absolute path of the session-recording directory, when archiving is
    /// enabled. None when --no-save was passed.
    pub session_dir: Option<std::path::PathBuf>,

    pub controls: Controls,
}

pub struct HistoryEntry {
    pub src_code: &'static str,
    pub src_text: String,
    pub translations: Vec<(Lang, String)>,
}

impl SharedState {
    pub fn set_src(&mut self, text: &str) {
        self.src_latest = text.to_string();
    }

    pub fn set_translation(&mut self, lang: Lang, text: &str, mt_ms: u128) {
        match lang {
            Lang::En => {
                self.tr_latest_en = text.to_string();
                self.mt_ms_en = mt_ms;
            }
            Lang::Fr => {
                self.tr_latest_fr = text.to_string();
                self.mt_ms_fr = mt_ms;
            }
            Lang::Es => {
                self.tr_latest_es = text.to_string();
                self.mt_ms_es = mt_ms;
            }
            Lang::It => {
                self.tr_latest_it = text.to_string();
                self.mt_ms_it = mt_ms;
            }
        }
    }

    pub fn set_tts_ms(&mut self, lang: Lang, ms: u128) {
        match lang {
            Lang::En => self.tts_ms_en = ms,
            Lang::Fr => self.tts_ms_fr = ms,
            Lang::Es => self.tts_ms_es = ms,
            Lang::It => self.tts_ms_it = ms,
        }
    }

    pub fn translation_for(&self, lang: Lang) -> &str {
        match lang {
            Lang::En => &self.tr_latest_en,
            Lang::Fr => &self.tr_latest_fr,
            Lang::Es => &self.tr_latest_es,
            Lang::It => &self.tr_latest_it,
        }
    }

    pub fn mt_ms_for(&self, lang: Lang) -> u128 {
        match lang {
            Lang::En => self.mt_ms_en,
            Lang::Fr => self.mt_ms_fr,
            Lang::Es => self.mt_ms_es,
            Lang::It => self.mt_ms_it,
        }
    }

    pub fn tts_ms_for(&self, lang: Lang) -> u128 {
        match lang {
            Lang::En => self.tts_ms_en,
            Lang::Fr => self.tts_ms_fr,
            Lang::Es => self.tts_ms_es,
            Lang::It => self.tts_ms_it,
        }
    }

    /// Push a debug-log line. Keeps the most recent 200 entries.
    pub fn log(&mut self, msg: impl Into<String>) {
        self.debug_log.push_front(msg.into());
        if self.debug_log.len() > 200 {
            self.debug_log.pop_back();
        }
    }

    /// Wipe everything the GUI shows: current source/translation, history,
    /// debug log, and timing counters. The pipeline keeps running.
    pub fn clear_all(&mut self) {
        self.src_latest.clear();
        self.tr_latest_en.clear();
        self.tr_latest_fr.clear();
        self.tr_latest_es.clear();
        self.tr_latest_it.clear();
        self.history.clear();
        self.debug_log.clear();
        self.stt_ms = 0;
        self.mt_ms_en = 0;
        self.mt_ms_fr = 0;
        self.mt_ms_es = 0;
        self.mt_ms_it = 0;
        self.tts_ms_en = 0;
        self.tts_ms_fr = 0;
        self.tts_ms_es = 0;
        self.tts_ms_it = 0;
        self.chunks_processed = 0;
    }

    pub fn push_history(&mut self) {
        if self.src_latest.is_empty() {
            return;
        }
        let source = self.controls.get_source();
        let mut translations = Vec::new();
        for &lang in Lang::ALL {
            if lang == source {
                continue;
            }
            let t = self.translation_for(lang);
            if !t.is_empty() {
                translations.push((lang, t.to_string()));
            }
        }
        self.history.push_front(HistoryEntry {
            src_code: source.code(),
            src_text: self.src_latest.clone(),
            translations,
        });
        if self.history.len() > 20 {
            self.history.pop_back();
        }
    }
}

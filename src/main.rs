mod archive;
mod audio;
mod gui;
#[cfg(target_os = "macos")]
mod macos_perm;
mod mt;
mod mt_cache;
mod output;
mod pitch;
mod state;
mod stt;
mod tts;
mod vad;

use anyhow::Result;
use clap::Parser;
use cpal::traits::DeviceTrait;
use crossbeam_channel::{bounded, RecvTimeoutError};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};

use crate::archive::Archive;
use crate::audio::{
    input_device_names, list_input_devices, pick_input_device,
    start_capture, start_file_capture, MonoResampler, WHISPER_SR,
};
use crate::mt::{Lang, OllamaTranslator};
use crate::mt_cache::MtCache;
use crate::output::{list_output_devices, output_device_names, MultichannelOutput, OutputSink};
use crate::state::{Controls, SharedState};
use crate::stt::Stt;
use crate::vad::{VadBuffer, VadConfig};

#[derive(Parser, Clone)]
#[command(name = "pico", about = "Real-time multilingual speech translator")]
struct Args {
    #[arg(long, default_value = "models/ggml-base.en.bin")]
    model: PathBuf,

    #[arg(long, default_value = "models/kokoro")]
    kokoro_dir: PathBuf,

    #[arg(long)]
    device: Option<String>,

    /// Read audio from a WAV file instead of the microphone (offline / test mode).
    #[arg(long)]
    input_file: Option<PathBuf>,

    #[arg(long)]
    list_devices: bool,

    #[arg(long)]
    list_outputs: bool,

    #[arg(long, default_value = "en")]
    source_lang: String,

    #[arg(long, default_value = "fr,es,it")]
    langs: String,

    #[arg(long, default_value = "https://ollama.com/v1", env = "OLLAMA_BASE_URL")]
    ollama_url: String,

    #[arg(long, default_value = "gemma3:12b")]
    ollama_model: String,

    #[arg(long)]
    no_mt: bool,

    /// Directory for the per-session recording (input.wav + per-language
    /// out-XX.wav, all uncompressed PCM). Defaults to
    /// $HOME/Documents/pico-sessions/<timestamp> when omitted.
    #[arg(long)]
    save: Option<PathBuf>,

    /// Skip session recording entirely.
    #[arg(long)]
    no_save: bool,

    #[arg(long)]
    output_device: Option<String>,

    #[arg(long, default_value_t = 0)]
    ch_en: u16,
    #[arg(long, default_value_t = 2)]
    ch_fr: u16,
    #[arg(long, default_value_t = 4)]
    ch_es: u16,
    #[arg(long, default_value_t = 6)]
    ch_it: u16,

    #[arg(long)]
    no_tts: bool,

    #[arg(long)]
    voice_en: Option<String>,
    #[arg(long)]
    voice_fr: Option<String>,
    #[arg(long)]
    voice_es: Option<String>,
    #[arg(long)]
    voice_it: Option<String>,

    #[arg(long)]
    gui: bool,

    /// Force-flush an utterance after this many seconds without a silence
    /// gap. Lower = more frequent chunks = lower end-to-end latency at the
    /// cost of less context per Whisper call.
    #[arg(long, default_value_t = 8.0)]
    max_utterance_s: f32,
    /// Trailing silence required to close an utterance. Lower = the
    /// pipeline reacts sooner after you stop talking; too low and a
    /// single breath mid-sentence will fragment the translation.
    #[arg(long, default_value_t = 600)]
    vad_silence_ms: u32,
    #[arg(long, default_value_t = 0.0035)]
    vad_rms_high: f32,
    #[arg(long, default_value_t = 0.0018)]
    vad_rms_low: f32,
    #[arg(long, default_value_t = 0.18)]
    vad_zcr_max: f32,
}

impl Args {
    /// Pick the TTS voice for `lang`. Order of precedence:
    ///   1. CLI override (`--voice-en`, etc.)
    ///   2. Voice matching the detected speaker `gender` for that language
    fn voice_for(&self, lang: Lang, gender: pitch::Gender) -> &str {
        let override_voice = match lang {
            Lang::En => self.voice_en.as_deref(),
            Lang::Fr => self.voice_fr.as_deref(),
            Lang::Es => self.voice_es.as_deref(),
            Lang::It => self.voice_it.as_deref(),
        };
        override_voice.unwrap_or_else(|| tts::voice_for_gender(lang, gender))
    }

    fn channel_start_for(&self, lang: Lang) -> u16 {
        match lang {
            Lang::En => self.ch_en,
            Lang::Fr => self.ch_fr,
            Lang::Es => self.ch_es,
            Lang::It => self.ch_it,
        }
    }
}

fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    if args.list_devices {
        return list_input_devices();
    }
    if args.list_outputs {
        return list_output_devices();
    }

    // macOS only: explicitly request microphone access via AVFoundation.
    // cpal does not trigger the consent prompt on its own — without this
    // call the OS silently hands us zeroed buffers.
    #[cfg(target_os = "macos")]
    {
        let status = macos_perm::ensure_microphone_access(std::time::Duration::from_secs(60));
        info!("macOS microphone authorization: {:?}", status);
    }

    if args.gui {
        run_gui(args)
    } else {
        run_cli(args, None)
    }
}

fn run_gui(args: Args) -> Result<()> {
    let state = Arc::new(Mutex::new(SharedState::default()));
    {
        let mut s = state.lock().unwrap();
        s.ollama_model = args.ollama_model.clone();
    }

    let state_worker = state.clone();
    let args_worker = args.clone();
    let _worker = std::thread::Builder::new()
        .name("pico-pipeline".into())
        .spawn(move || {
            if let Err(e) = run_cli(args_worker, Some(state_worker.clone())) {
                warn!("pipeline thread exited with error: {}", e);
                if let Ok(mut s) = state_worker.lock() {
                    s.listening = false;
                }
            }
        })?;

    let app_state = state.clone();
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("pico")
            .with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "pico",
        native_options,
        Box::new(move |_cc| Ok(Box::new(gui::PicoApp::new(app_state)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {}", e))?;

    Ok(())
}

fn run_cli(args: Args, state: Option<Arc<Mutex<SharedState>>>) -> Result<()> {
    let source_lang = Lang::from_code(&args.source_lang)
        .ok_or_else(|| anyhow::anyhow!("unknown source lang '{}'", args.source_lang))?;
    let initial_targets = Lang::parse_csv(&args.langs)?;

    info!(
        "source: {}, initial targets: {:?}",
        source_lang.code(),
        initial_targets.iter().map(|l| l.code()).collect::<Vec<_>>()
    );

    let archive: Option<Arc<Archive>> = if args.no_save {
        None
    } else {
        let dir = match args.save.as_ref() {
            Some(d) => d.clone(),
            None => default_session_dir(),
        };
        let a = Arc::new(Archive::new(dir.clone())?);
        info!("archive dir: {}", a.dir.display());
        if let Some(st) = state.as_ref() {
            st.lock().unwrap().session_dir = Some(dir);
        }
        Some(a)
    };

    let tts_active = !args.no_tts;

    // Channel-start atomics are created ONCE and shared across all output
    // stream re-opens + GUI. Atomic writes from GUI affect next audio frame.
    let channel_atomics: Arc<Vec<AtomicU16>> = Arc::new(
        Lang::ALL
            .iter()
            .map(|l| AtomicU16::new(args.channel_start_for(*l)))
            .collect(),
    );
    // Per-lane volume atomics (bits of f32). Default: unity gain.
    let volume_atomics: Arc<Vec<AtomicU32>> = Arc::new(
        (0..Lang::ALL.len())
            .map(|_| AtomicU32::new(1.0_f32.to_bits()))
            .collect(),
    );
    // Per-lane output peak (post-volume), updated by the audio callback for
    // VU meters in the GUI. One slot per Lang::ALL.
    let output_peaks: Arc<Vec<AtomicU32>> = Arc::new(
        (0..Lang::ALL.len()).map(|_| AtomicU32::new(0)).collect(),
    );

    let target_enabled: Vec<AtomicBool> = Lang::ALL
        .iter()
        .map(|l| AtomicBool::new(initial_targets.contains(l)))
        .collect();

    let controls = Controls {
        // Start in stopped state — the user clicks "Start" in the GUI to
        // begin transcribing. Avoids surprise audio capture on launch.
        active: Arc::new(AtomicBool::new(false)),
        source: Arc::new(AtomicU8::new(source_lang.index() as u8)),
        target_enabled: Arc::new(target_enabled),
        channel_starts: channel_atomics.clone(),
        volumes: volume_atomics.clone(),
        requested_input: Arc::new(Mutex::new(args.device.clone())),
        requested_output: Arc::new(Mutex::new(args.output_device.clone())),
        restart_flag: Arc::new(AtomicBool::new(false)),
        input_devices: Arc::new(Mutex::new(input_device_names())),
        output_devices: Arc::new(Mutex::new(output_device_names())),
        gender_mode: Arc::new(AtomicU8::new(state::GENDER_MODE_AUTO)),
        input_rms: Arc::new(AtomicU32::new(0)),
        vad_rms_high: Arc::new(AtomicU32::new(args.vad_rms_high.to_bits())),
        vad_rms_low: Arc::new(AtomicU32::new(args.vad_rms_low.to_bits())),
        output_peaks: output_peaks.clone(),
        voice_overrides: Arc::new(
            Lang::ALL
                .iter()
                .map(|l| {
                    // Seed from CLI flags if present.
                    let cli_voice = match l {
                        Lang::En => args.voice_en.clone(),
                        Lang::Fr => args.voice_fr.clone(),
                        Lang::Es => args.voice_es.clone(),
                        Lang::It => args.voice_it.clone(),
                    };
                    Mutex::new(cli_voice)
                })
                .collect(),
        ),
        session_token: Arc::new(AtomicU32::new(0)),
    };

    if let Some(st) = &state {
        let mut s = st.lock().unwrap();
        s.controls = controls.clone();
    }

    let kokoro = if tts_active {
        info!(
            "loading Kokoro pool ({} engines, all langs) from {}",
            Lang::ALL.len(),
            args.kokoro_dir.display()
        );
        let t0 = Instant::now();
        let pool = Arc::new(tts::KokoroPool::load(&args.kokoro_dir, Lang::ALL)?);
        info!("Kokoro pool loaded in {}ms", t0.elapsed().as_millis());
        Some(pool)
    } else {
        None
    };

    info!("loading whisper model: {}", args.model.display());
    let stt = Stt::load(&args.model)?;

    let translator = if args.no_mt {
        None
    } else {
        let api_key = std::env::var("OLLAMA_API_KEY").ok();
        let is_cloud = args.ollama_url.contains("ollama.com");
        if is_cloud && api_key.is_none() {
            warn!("ollama url looks like cloud but OLLAMA_API_KEY is not set");
        }
        info!(
            "ollama: url={} model={} auth={}",
            args.ollama_url,
            args.ollama_model,
            if api_key.is_some() { "bearer" } else { "none" }
        );
        let ollama = OllamaTranslator::new(args.ollama_url.clone(), args.ollama_model.clone(), api_key)?;
        Some(Arc::new(MtCache::new(ollama)))
    };

    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()?,
    );

    // Sink is the Send half of the output: producers + sample-rate, shared
    // with the worker for push_pcm. The cpal Stream lives in
    // `output_device` on the main thread (it is !Send).
    let output_handle: Arc<Mutex<Option<OutputSink>>> = Arc::new(Mutex::new(None));
    let mut output_device: Option<MultichannelOutput> = None;

    // Wrap the long-lived dependencies in Arc up-front so the worker thread
    // can clone them once. Stt and Kokoro are themselves Send+Sync.
    let args_arc = Arc::new(args.clone());
    let stt_arc = Arc::new(stt);

    // Worker channel. The capture/VAD loop pushes completed utterances
    // here; the worker pops and runs STT + MT + TTS without blocking the
    // audio path. Bound is generous so a momentary worker stall (a slow MT
    // call) doesn't drop utterances — instead it backpressures.
    let (utt_tx, utt_rx) = bounded::<Vec<f32>>(64);

    // Spawn the worker. Lives for the entire run_cli call.
    let worker_ctx = WorkerCtx {
        args: args_arc.clone(),
        stt: stt_arc.clone(),
        translator: translator.clone(),
        kokoro: kokoro.clone(),
        output: output_handle.clone(),
        archive: archive.clone(),
        state: state.clone(),
        controls: controls.clone(),
        rt: rt.clone(),
    };
    let worker_handle = std::thread::Builder::new()
        .name("pico-worker".into())
        .spawn(move || {
            let mut gender_tracker = pitch::GenderTracker::new();
            let mut last_token = worker_ctx.controls.session_token.load(Ordering::Relaxed);
            while let Ok(chunk) = utt_rx.recv() {
                // Reset per-session state when the GUI bumps the token —
                // happens on source-language change, "Reset" button, etc.
                let cur = worker_ctx.controls.session_token.load(Ordering::Relaxed);
                if cur != last_token {
                    info!("session token changed ({} -> {}) — resetting gender tracker", last_token, cur);
                    gender_tracker = pitch::GenderTracker::new();
                    last_token = cur;
                }
                if let Err(e) = worker_ctx.process_utterance(chunk, &mut gender_tracker) {
                    warn!("process_utterance: {}", e);
                }
            }
            info!("worker thread exiting");
        })?;

    // OUTER RESTART LOOP. Each iteration opens fresh capture + output streams
    // using the currently-requested device names, then runs the inner loop
    // until restart_flag is set (then rebuilds) or the channel closes.
    loop {
        let input_name = controls.requested_input.lock().unwrap().clone();
        let output_name = controls.requested_output.lock().unwrap().clone();

        let (tx, rx) = bounded::<Vec<f32>>(128);
        let (capture, device_name) = if let Some(file) = args.input_file.as_ref() {
            let cap = start_file_capture(file, tx)?;
            (cap, format!("file:{}", file.display()))
        } else {
            let device = match pick_input_device(input_name.as_deref()) {
                Ok(d) => d,
                Err(e) => {
                    warn!("input device error: {} — falling back to default", e);
                    pick_input_device(None)?
                }
            };
            let name = device.name().ok().unwrap_or_else(|| "<unknown>".into());
            // We deliberately skip a startup mic check: the user might
            // launch with the pipeline stopped, and probing in that state
            // produces a bogus "no audio" warning. The VAD loop now
            // detects silent capture itself (only while active) and sets
            // mic_permission_warning when no audio shows up for a while.
            let cap = start_capture(&device, tx)?;
            (cap, name)
        };

        // Open (or reopen) the output. Stream stays on this thread; sink
        // goes into the shared handle so the worker can push_pcm.
        output_device = None;
        *output_handle.lock().unwrap() = None;
        if tts_active {
            let opened = MultichannelOutput::open(
                output_name.as_deref(),
                channel_atomics.clone(),
                volume_atomics.clone(),
                output_peaks.clone(),
                tts::KOKORO_SAMPLE_RATE,
            )
            .or_else(|e| {
                warn!("output device error: {} — opening default", e);
                MultichannelOutput::open(
                    None,
                    channel_atomics.clone(),
                    volume_atomics.clone(),
                    output_peaks.clone(),
                    tts::KOKORO_SAMPLE_RATE,
                )
            })?;
            let (dev, sink) = opened;
            *output_handle.lock().unwrap() = Some(sink);
            output_device = Some(dev);
        }

        if let Some(st) = &state {
            let mut s = st.lock().unwrap();
            s.input_device = device_name.clone();
            s.output_device = output_device
                .as_ref()
                .map(|o| o.device_name.clone())
                .unwrap_or_else(|| "<none>".into());
            s.output_channels = output_device.as_ref().map(|o| o.channels).unwrap_or(0);
            s.listening = true;
        }

        let mut resampler = MonoResampler::new(capture.input_sr, WHISPER_SR)?;
        let vad_cfg = VadConfig {
            sample_rate: WHISPER_SR,
            window_ms: 30,
            rms_high: args.vad_rms_high,
            rms_low: args.vad_rms_low,
            zcr_max: args.vad_zcr_max,
            min_silence_ms: args.vad_silence_ms,
            max_speech_ms: (args.max_utterance_s * 1000.0) as u32,
            preroll_ms: 200,
        };
        let mut vad_buf = VadBuffer::new(vad_cfg);

        info!(
            "capturing @ {} Hz ({}), resampling to {} Hz. VAD silence_ms={}, rms_high={:.3}, rms_low={:.3}",
            capture.input_sr,
            device_name,
            WHISPER_SR,
            args.vad_silence_ms,
            args.vad_rms_high,
            args.vad_rms_low,
        );

        controls.restart_flag.store(false, Ordering::Relaxed);

        let outcome = run_inner(
            archive.as_ref(),
            state.as_ref(),
            &controls,
            rx,
            &mut resampler,
            &mut vad_buf,
            &utt_tx,
        )?;

        // Close the audio streams before opening new ones on next iter.
        *output_handle.lock().unwrap() = None;
        output_device = None;
        drop(capture);

        // File mode is one-shot: exit on EOF, ignore restart signals.
        if args.input_file.is_some() {
            break;
        }

        match outcome {
            LoopOutcome::Restart => {
                info!("restarting pipeline with new device selection");
                continue;
            }
            LoopOutcome::Eof => break,
        }
    }

    // Drop our send side so the worker's recv loop exits cleanly, then join.
    drop(utt_tx);
    if let Err(e) = worker_handle.join() {
        warn!("worker thread panicked: {:?}", e);
    }

    if let Some(a) = &archive {
        if let Err(e) = a.finalize_input() {
            warn!("failed to finalize input.wav: {}", e);
        }
    }
    if let Some(st) = &state {
        let mut s = st.lock().unwrap();
        s.listening = false;
    }

    Ok(())
}

enum LoopOutcome {
    Restart,
    Eof,
}

/// Where to put a session when --save is not specified. Picks
/// `$HOME/Documents/pico-sessions/<YYYY-MM-DD_HH-MM-SS>` so multiple runs
/// don't collide and the user can find them later.
fn default_session_dir() -> PathBuf {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Cheap timestamp without pulling in a date crate: epoch seconds is
    // fine; the GUI can render something nicer on top.
    let stamp = format!("{}", now);
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join("Documents")
        .join("pico-sessions")
        .join(stamp)
}

/// Everything the processing worker needs. The capture/VAD loop sends a
/// completed utterance through a channel; the worker pulls it and runs
/// STT → MT → TTS without blocking the audio thread.
struct WorkerCtx {
    args: Arc<Args>,
    stt: Arc<Stt>,
    translator: Option<Arc<MtCache>>,
    kokoro: Option<Arc<tts::KokoroPool>>,
    output: Arc<Mutex<Option<OutputSink>>>,
    archive: Option<Arc<Archive>>,
    state: Option<Arc<Mutex<SharedState>>>,
    controls: Controls,
    rt: Arc<tokio::runtime::Runtime>,
}

fn run_inner(
    archive: Option<&Arc<Archive>>,
    state: Option<&Arc<Mutex<SharedState>>>,
    controls: &Controls,
    rx: crossbeam_channel::Receiver<Vec<f32>>,
    resampler: &mut MonoResampler,
    vad_buf: &mut VadBuffer,
    utt_tx: &crossbeam_channel::Sender<Vec<f32>>,
) -> Result<LoopOutcome> {
    // Tracks the last time we observed any non-trivial mic energy. Used
    // to surface a "microphone blocked" banner only AFTER the user has
    // actually started the pipeline and we've waited long enough for
    // real audio to show up.
    let mut last_audio_at = Instant::now();
    let mut active_since: Option<Instant> = None;
    loop {
        if controls.restart_flag.load(Ordering::Relaxed) {
            return Ok(LoopOutcome::Restart);
        }

        let mono = match rx.recv_timeout(Duration::from_millis(150)) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return Ok(LoopOutcome::Eof),
        };
        let resampled = resampler.push(&mono)?;

        let active = controls.active.load(Ordering::Relaxed);
        if !active {
            vad_buf.reset();
            // The banner is only meaningful while we're actively listening;
            // clear it when the user pauses so it doesn't linger.
            active_since = None;
            if let Some(st) = state.as_ref() {
                let mut s = st.lock().unwrap();
                if s.mic_permission_warning.is_some() {
                    s.mic_permission_warning = None;
                }
            }
            continue;
        }
        if active_since.is_none() {
            active_since = Some(Instant::now());
            last_audio_at = Instant::now();
        }

        if let Some(a) = archive {
            a.write_input_f32(&resampled)?;
        }

        let rms_high = f32::from_bits(controls.vad_rms_high.load(Ordering::Relaxed));
        let rms_low = f32::from_bits(controls.vad_rms_low.load(Ordering::Relaxed));
        let mut last_rms = 0.0f32;
        let utterances = vad_buf.push(&resampled, rms_high, rms_low, &mut last_rms);
        controls.input_rms.store(last_rms.to_bits(), Ordering::Relaxed);

        // Surface (or clear) the "microphone blocked" banner based on
        // observed audio. Threshold and grace window are tuned so that
        // brief silences inside speech don't trigger it.
        const SILENT_GRACE: Duration = Duration::from_secs(4);
        const RMS_PRESENT: f32 = 1e-4;
        if last_rms > RMS_PRESENT {
            last_audio_at = Instant::now();
            if let Some(st) = state.as_ref() {
                let mut s = st.lock().unwrap();
                if s.mic_permission_warning.is_some() {
                    s.mic_permission_warning = None;
                }
            }
        } else if let Some(started) = active_since {
            let idle = last_audio_at.elapsed();
            // Wait at least SILENT_GRACE since the last sound AND since
            // the user pressed Start, so we don't shout immediately.
            if idle > SILENT_GRACE && started.elapsed() > SILENT_GRACE {
                if let Some(st) = state.as_ref() {
                    let mut s = st.lock().unwrap();
                    if s.mic_permission_warning.is_none() {
                        s.mic_permission_warning = Some(
                            "no audio detected on the input device — check that the right \
                             microphone is selected and that pico has microphone permission \
                             in System Settings → Privacy & Security."
                                .to_string(),
                        );
                    }
                }
            }
        }
        for utterance in utterances {
            // Hand the utterance to the worker thread and keep going. If
            // the worker is busy and the queue is full, log it and drop
            // this one — the alternative is blocking the audio path, which
            // would lose subsequent input entirely.
            match utt_tx.try_send(utterance) {
                Ok(()) => {}
                Err(crossbeam_channel::TrySendError::Full(_)) => {
                    warn!("worker queue full — dropped utterance");
                    if let Some(st) = state {
                        st.lock().unwrap().log("[worker] queue full, dropped utterance".to_string());
                    }
                }
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                    return Ok(LoopOutcome::Eof);
                }
            }
        }
    }
}

impl WorkerCtx {
    fn process_utterance(
        &self,
        chunk: Vec<f32>,
        gender_tracker: &mut pitch::GenderTracker,
    ) -> Result<()> {
        process_utterance_inner(self, chunk, gender_tracker)
    }
}

fn process_utterance_inner(
    ctx: &WorkerCtx,
    chunk: Vec<f32>,
    gender_tracker: &mut pitch::GenderTracker,
) -> Result<()> {
    let args = &*ctx.args;
    let stt = &*ctx.stt;
    let translator = &ctx.translator;
    let kokoro = &ctx.kokoro;
    let archive = ctx.archive.as_ref();
    let state = ctx.state.as_ref();
    let controls = &ctx.controls;
    let rt = &*ctx.rt;
    let source = controls.get_source();

    // Speaker gender for TTS voice selection. The GUI can override (Male /
    // Female); when set to Auto, we estimate F0 on this chunk and feed a
    // running majority tracker so the voice stays stable across utterances
    // even when individual readings flip on short or noisy clips.
    let utt_gender = pitch::estimate_gender(&chunk);
    gender_tracker.observe(utt_gender);
    let gender = match controls.gender_mode.load(Ordering::Relaxed) {
        state::GENDER_MODE_MALE => pitch::Gender::Male,
        state::GENDER_MODE_FEMALE => pitch::Gender::Female,
        _ => gender_tracker.current(),
    };

    let t0 = Instant::now();
    let text = stt.transcribe(&chunk, source)?;
    let stt_ms = t0.elapsed().as_millis();

    let is_silence = text.is_empty()
        || (text.starts_with('[') && text.ends_with(']'))
        || text.chars().all(|c| c.is_ascii_punctuation() || c.is_whitespace());

    // Drop classic Whisper hallucinations ("Grazie", "Thanks for watching",
    // "Subscribe to my channel" etc.) that the model invents from
    // keyboard noise, applause, or near-silence.
    let is_hallucinated = !is_silence && stt::is_hallucination(&text);

    if is_silence || is_hallucinated {
        info!("[stt  {:>4}ms] <{}: {:?}>", stt_ms, if is_hallucinated { "hallucination" } else { "silence" }, text);
        if let Some(st) = state {
            st.lock().unwrap().log(format!(
                "[stt {}ms] <{}>",
                stt_ms,
                if is_hallucinated { "hallucination" } else { "silence" }
            ));
        }
        return Ok(());
    }
    println!("[stt  {:>4}ms] {}: {}", stt_ms, source.code().to_uppercase(), text);

    // Persist the source transcription line. Failures are non-fatal.
    if let Some(a) = archive {
        if let Err(e) = a.append_transcript(source, &text) {
            warn!("transcript append failed for {}: {}", source.code(), e);
        }
    }

    if let Some(st) = state {
        let mut s = st.lock().unwrap();
        s.push_history();
        s.set_src(&text);
        s.tr_latest_en.clear();
        s.tr_latest_fr.clear();
        s.tr_latest_es.clear();
        s.tr_latest_it.clear();
        s.stt_ms = stt_ms;
        s.last_activity = Some(Instant::now());
        s.log(format!("[stt {}ms] {}: {}", stt_ms, source.code().to_uppercase(), text));
        s.log(format!("[gender utt={:?} session={:?}]", utt_gender, gender));
    }

    let targets: Vec<Lang> = Lang::ALL
        .iter()
        .copied()
        .filter(|l| *l != source && controls.is_target_enabled(*l))
        .collect();

    if targets.is_empty() {
        return Ok(());
    }

    let Some(tr) = translator.clone() else { return Ok(()); };
    let text_owned = text.clone();
    let targets_for_mt = targets.clone();
    let mt_results = rt.block_on(async move {
        let futs = targets_for_mt.iter().map(|&l| {
            let tr = tr.clone();
            let t = text_owned.clone();
            async move {
                let t0 = Instant::now();
                let r = tr.translate(&t, source, l).await;
                (l, t0.elapsed().as_millis(), r)
            }
        });
        futures::future::join_all(futs).await
    });

    let mut translated: Vec<(Lang, String)> = Vec::new();
    for (lang, ms, res) in &mt_results {
        match res {
            Ok(t) => {
                println!("[mt   {:>4}ms] {}: {}", ms, lang.code().to_uppercase(), t);
                if let Some(st) = state {
                    let mut s = st.lock().unwrap();
                    s.set_translation(*lang, t, *ms);
                    s.log(format!("[mt {}ms] {}: {}", ms, lang.code().to_uppercase(), t));
                }
                if let Some(a) = archive {
                    if let Err(e) = a.append_transcript(*lang, t) {
                        warn!("transcript append failed for {}: {}", lang.code(), e);
                    }
                }
                if !t.is_empty() {
                    translated.push((*lang, t.clone()));
                }
            }
            Err(e) => {
                warn!("[mt   {:>4}ms] {}: error: {}", ms, lang.code(), e);
                if let Some(st) = state {
                    st.lock().unwrap().log(format!("[mt {}ms] {} ERROR: {}", ms, lang.code(), e));
                }
            }
        }
    }

    let Some(pool) = kokoro.clone() else { return Ok(()); };
    if translated.is_empty() {
        return Ok(());
    }

    let voices: Vec<(Lang, String, String)> = translated
        .iter()
        .map(|(lang, text)| {
            // Per-lane override from the GUI takes precedence over the
            // gender-driven default.
            let voice = controls
                .voice_overrides[lang.index()]
                .lock()
                .ok()
                .and_then(|g| g.clone())
                .unwrap_or_else(|| args.voice_for(*lang, gender).to_string());
            (*lang, voice, text.clone())
        })
        .collect();
    info!(
        "[gender utt={:?} session={:?}] voices: {:?}",
        utt_gender,
        gender,
        voices.iter().map(|(l, v, _)| format!("{}={}", l.code(), v)).collect::<Vec<_>>()
    );

    let tts_t0 = Instant::now();
    let synth_results: Vec<(Lang, u128, Result<Vec<f32>>)> = rt.block_on(async move {
        let handles: Vec<_> = voices
            .into_iter()
            .map(|(lang, voice, text)| {
                let pool = pool.clone();
                tokio::task::spawn_blocking(move || {
                    let t0 = Instant::now();
                    let r = pool.synthesize(lang, &text, &voice);
                    (lang, t0.elapsed().as_millis(), r)
                })
            })
            .collect();
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            if let Ok(v) = h.await {
                out.push(v);
            }
        }
        out
    });

    for (lang, synth_ms, res) in &synth_results {
        match res {
            Ok(pcm) => {
                if let Some(a) = archive {
                    let tmp = a.temp_tts_path(*lang);
                    if let Err(e) = archive::write_pcm_f32_wav(&tmp, tts::KOKORO_SAMPLE_RATE, pcm) {
                        warn!("archive write failed for {}: {}", lang.code(), e);
                    } else if let Err(e) = a.append_tts(*lang, &tmp) {
                        warn!("archive append failed for {}: {}", lang.code(), e);
                    }
                }
                {
                    let mut g = ctx.output.lock().unwrap();
                    if let Some(out) = g.as_mut() {
                        let lane = lang.index();
                        if let Err(e) = out.push_pcm(lane, pcm) {
                            warn!("output push failed for {}: {}", lang.code(), e);
                        }
                    }
                }
                if let Some(st) = state {
                    st.lock().unwrap().set_tts_ms(*lang, *synth_ms);
                }
                info!(
                    "[tts  {:>4}ms] {}: {} samples ({:.2}s audio)",
                    synth_ms,
                    lang.code().to_uppercase(),
                    pcm.len(),
                    pcm.len() as f32 / tts::KOKORO_SAMPLE_RATE as f32
                );
            }
            Err(e) => warn!("[tts  {:>4}ms] {}: error: {}", synth_ms, lang.code(), e),
        }
    }
    info!("[tts total {}ms (parallel)]", tts_t0.elapsed().as_millis());

    if let Some(st) = state {
        st.lock().unwrap().chunks_processed += 1;
    }

    Ok(())
}

use anyhow::Result;
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::mt::Lang;

/// Recognize transcripts that Whisper produces from non-speech audio
/// (silence, keyboard clicks, ambient noise). Returns true when the text
/// looks like one of the known training-data phrases — the caller should
/// drop the segment as if it were silence. Comparison is whitespace- and
/// punctuation-insensitive.
pub fn is_hallucination(text: &str) -> bool {
    let norm: String = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if norm.is_empty() {
        return true;
    }
    // Common Whisper hallucinations sourced from public benchmarks +
    // observed runs. Mostly closing remarks from YouTube tutorials, news
    // sign-offs, and applause artifacts.
    const PATTERNS: &[&str] = &[
        // Italian
        "grazie",
        "grazie a tutti",
        "grazie per l attenzione",
        "grazie per aver guardato",
        "iscrivetevi al canale",
        "iscriviti al canale",
        "ci vediamo nel prossimo video",
        // English
        "thank you",
        "thanks for watching",
        "thanks for watching the video",
        "please subscribe",
        "subscribe to my channel",
        "see you in the next video",
        "you",
        "bye",
        // Spanish
        "gracias",
        "gracias por ver",
        "suscribete",
        "muchas gracias",
        // French
        "merci",
        "merci d avoir regarde",
        "abonnez vous",
        // Generic
        "music",
        "applause",
    ];
    PATTERNS.iter().any(|p| *p == norm)
}

pub struct Stt {
    ctx: WhisperContext,
}

impl Stt {
    pub fn load(model_path: &Path) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(
            model_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("model path is not valid utf-8"))?,
            WhisperContextParameters::default(),
        )?;
        Ok(Self { ctx })
    }

    pub fn transcribe(&self, pcm_16k_mono: &[f32], source: Lang) -> Result<String> {
        let mut state = self.ctx.create_state()?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(source.whisper_code()));
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_n_threads(
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4),
        );
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_suppress_blank(true);

        // Hallucination suppression. Without these Whisper happily transcribes
        // keyboard clicks / silence as "Grazie", "Thanks for watching",
        // "Subscribe to my channel" — phrases overrepresented in its training
        // data. Tighter thresholds + post-filter (see is_hallucination)
        // catch the rest.
        params.set_temperature(0.0);
        params.set_no_speech_thold(0.7);
        params.set_logprob_thold(-0.5);
        params.set_suppress_nst(true);

        state.full(params, pcm_16k_mono)?;

        let n = state.full_n_segments()?;
        let mut out = String::new();
        for i in 0..n {
            let seg = state.full_get_segment_text(i)?;
            out.push_str(&seg);
        }
        Ok(out.trim().to_string())
    }
}

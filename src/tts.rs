use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tts_rs::{
    engines::kokoro::{KokoroEngine, KokoroInferenceParams},
    SynthesisEngine,
};

use crate::mt::Lang;
use crate::pitch::Gender;

pub const KOKORO_SAMPLE_RATE: u32 = 24_000;

pub fn default_voice_for(lang: Lang) -> &'static str {
    voice_for_gender(lang, Gender::Unknown)
}

/// Pick a Kokoro voice that matches the speaker gender for the target
/// language. Falls back to whatever single-gender voice exists for the
/// language when there is no same-gender option (Kokoro has no male French
/// voice, for example).
pub fn voice_for_gender(lang: Lang, gender: Gender) -> &'static str {
    match (lang, gender) {
        (Lang::En, Gender::Male) => "am_michael",
        (Lang::En, _) => "af_heart",

        // Kokoro ships only a female French voice.
        (Lang::Fr, _) => "ff_siwis",

        (Lang::Es, Gender::Male) => "em_alex",
        (Lang::Es, _) => "ef_dora",

        (Lang::It, Gender::Male) => "im_nicola",
        (Lang::It, _) => "if_sara",
    }
}

/// Kokoro voices filtered by language and gender. Returns the slice the
/// GUI shows in the per-lane voice picker. Empty for `Gender::Unknown` —
/// in Auto mode we don't show a dropdown at all and the pitch tracker
/// decides the voice per utterance.
pub fn voices_for_gender(lang: Lang, gender: Gender) -> &'static [&'static str] {
    match (lang, gender) {
        (Lang::En, Gender::Female) => &[
            "af_heart",
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
        ],
        (Lang::En, Gender::Male) => &[
            "am_michael",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_onyx",
            "am_puck",
            "am_santa",
        ],
        // Kokoro ships only a female French voice — return it for both
        // genders so the GUI always has something to pick.
        (Lang::Fr, _) => &["ff_siwis"],
        (Lang::Es, Gender::Female) => &["ef_dora"],
        (Lang::Es, Gender::Male) => &["em_alex", "em_santa"],
        (Lang::It, Gender::Female) => &["if_sara"],
        (Lang::It, Gender::Male) => &["im_nicola"],
        _ => &[],
    }
}

/// Human-readable name for a Kokoro voice id. Drops the
/// `<lang><gender>_` prefix and capitalizes — `af_heart` → `Heart`.
pub fn voice_display_name(voice: &str) -> String {
    let bare = voice.split_once('_').map(|(_, n)| n).unwrap_or(voice);
    let mut chars = bare.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => bare.to_string(),
    }
}

pub struct KokoroPool {
    engines: HashMap<Lang, Arc<Mutex<KokoroEngine>>>,
}

impl KokoroPool {
    pub fn load(model_dir: &Path, langs: &[Lang]) -> Result<Self> {
        let mut engines = HashMap::with_capacity(langs.len());
        for &lang in langs {
            let mut engine = KokoroEngine::new();
            engine
                .load_model(model_dir)
                .map_err(|e| anyhow!("load kokoro for {}: {}", lang.code(), e))?;
            engines.insert(lang, Arc::new(Mutex::new(engine)));
        }
        Ok(Self { engines })
    }

    pub fn synthesize(&self, lang: Lang, text: &str, voice: &str) -> Result<Vec<f32>> {
        let handle = self
            .engines
            .get(&lang)
            .ok_or_else(|| anyhow!("no kokoro engine loaded for {}", lang.code()))?;
        let params = KokoroInferenceParams {
            voice: voice.to_string(),
            speed: 1.0,
            ..Default::default()
        };
        let mut eng = handle.lock().map_err(|_| anyhow!("kokoro engine mutex poisoned"))?;
        let result = eng
            .synthesize(text, Some(params))
            .map_err(|e| anyhow!("kokoro synthesize: {}", e))?;
        Ok(result.samples)
    }
}

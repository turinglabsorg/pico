//! End-to-end translation tests: run the full STT + MT chain on an English
//! fixture, capture the target-language translations, and check that the
//! pipeline produces sensible Spanish (or other) output.
//!
//! Requires Ollama (cloud or local). The binary picks up OLLAMA_API_KEY and
//! OLLAMA_BASE_URL from `.env` via dotenvy. Marked `#[ignore]` like the STT
//! smoke tests. Run with:
//!
//!     cargo test --test translate_smoke -- --ignored --nocapture

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn model(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

struct Pair {
    en: String,
    translation: String,
    mt_ms: u64,
}

struct Parsed {
    /// 1-to-1 alignment of source utterance and its translation. Built by
    /// pairing each `[stt ...] EN:` line with the next-emitted `[mt ...] <TGT>:`.
    pairs: Vec<Pair>,
    stt_total_ms: u64,
    mt_total_ms: u64,
}

fn parse_pipeline(stdout: &str, target_lang_upper: &str) -> Parsed {
    let mt_prefix = format!("] {}: ", target_lang_upper);
    let mut pending_en: Option<String> = None;
    let mut pairs: Vec<Pair> = Vec::new();
    let mut stt_total: u64 = 0;
    let mut mt_total: u64 = 0;
    let mut last_mt_ms: u64 = 0;

    for line in stdout.lines() {
        if line.starts_with("[stt") {
            // [stt   76ms] EN: text
            if let (Some(open), Some(close)) = (line.find('['), line.find(']')) {
                if let Some(ms_pos) = line[open..close].find("ms") {
                    let n: String = line[open + 1..open + ms_pos]
                        .chars()
                        .filter(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(v) = n.parse::<u64>() {
                        stt_total += v;
                    }
                }
                if let Some(idx) = line.find("] EN: ") {
                    let text = line[idx + 6..].trim().to_string();
                    pending_en = Some(text);
                }
            }
        } else if line.starts_with("[mt") {
            if let (Some(open), Some(close)) = (line.find('['), line.find(']')) {
                if let Some(ms_pos) = line[open..close].find("ms") {
                    let n: String = line[open + 1..open + ms_pos]
                        .chars()
                        .filter(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(v) = n.parse::<u64>() {
                        last_mt_ms = v;
                        mt_total += v;
                    }
                }
                if let Some(idx) = line.find(&mt_prefix) {
                    let translation = line[idx + mt_prefix.len()..].trim().to_string();
                    if let Some(en) = pending_en.take() {
                        pairs.push(Pair { en, translation, mt_ms: last_mt_ms });
                    }
                }
            }
        }
    }
    Parsed { pairs, stt_total_ms: stt_total, mt_total_ms: mt_total }
}

fn run_translate(audio: &str, whisper_model: &str, target: &str) -> Parsed {
    if std::env::var("OLLAMA_API_KEY").is_err() {
        // dotenvy is loaded by the binary, but we also check here so the
        // test fails loudly with a clear message instead of timing out on
        // unauthorized requests.
        if !std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".env").exists() {
            panic!("OLLAMA_API_KEY not set and no .env in project root");
        }
    }

    let bin = env!("CARGO_BIN_EXE_pico");
    let audio_path = fixture(audio);
    let model_path = model(whisper_model);
    assert!(audio_path.exists(), "missing fixture: {}", audio_path.display());
    assert!(model_path.exists(), "missing model: {}", model_path.display());

    let wall_t0 = std::time::Instant::now();
    let output = Command::new(bin)
        .arg("--input-file").arg(&audio_path)
        .arg("--model").arg(&model_path)
        .arg("--source-lang").arg("en")
        .arg("--langs").arg(target)
        .arg("--no-tts")
        .output()
        .expect("pico binary should run");
    let wall_s = wall_t0.elapsed().as_secs_f64();

    if !output.status.success() {
        eprintln!("--- stderr ---\n{}", String::from_utf8_lossy(&output.stderr));
        panic!("pico exited with {:?}", output.status);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = parse_pipeline(&stdout, &target.to_uppercase());

    let audio_s = {
        let r = hound::WavReader::open(&audio_path).expect("open wav");
        r.duration() as f64 / r.spec().sample_rate as f64
    };
    let mt_avg = if parsed.pairs.is_empty() { 0 } else { parsed.mt_total_ms / parsed.pairs.len() as u64 };
    eprintln!(
        "\naudio={:.1}s pairs={} stt={:.2}s mt_total={:.2}s mt_avg={}ms wall={:.2}s rtf_wall={:.3}",
        audio_s,
        parsed.pairs.len(),
        parsed.stt_total_ms as f64 / 1000.0,
        parsed.mt_total_ms as f64 / 1000.0,
        mt_avg,
        wall_s,
        wall_s / audio_s,
    );
    parsed
}

/// Heuristic: a non-empty Spanish translation should contain Spanish-typical
/// markers (accented vowels, ñ, ¿, ¡) or Spanish stop words anywhere in the
/// text. Catches the case where the LLM echoes English back verbatim.
/// Tokens are matched whole-word so English "el" / "la" don't false-positive.
fn looks_spanish(text: &str) -> bool {
    let t = text.to_lowercase();
    if t.chars().any(|c| matches!(c, 'á' | 'é' | 'í' | 'ó' | 'ú' | 'ñ' | '¿' | '¡')) {
        return true;
    }
    const STOPS: &[&str] = &[
        "el", "la", "los", "las", "que", "de", "y", "en", "no", "un", "una",
        "su", "es", "al", "del", "para", "por", "con", "como", "mi", "se",
        "le", "lo", "te", "ya", "más", "este", "esta", "esto", "uno",
        "ese", "esa", "eso", "pero", "sino", "donde", "cuando",
    ];
    t.split(|c: char| !c.is_alphabetic())
        .any(|w| !w.is_empty() && STOPS.contains(&w))
}

#[test]
#[ignore]
fn reagan_en_to_es() {
    let p = run_translate("reagan_berlin.wav", "ggml-base.en.bin", "es");

    assert!(!p.pairs.is_empty(), "no STT/MT pairs produced");

    let mut empty_translations = 0;
    let mut english_echoes = 0;
    for pair in &p.pairs {
        if pair.translation.trim().is_empty() {
            empty_translations += 1;
        } else if !looks_spanish(&pair.translation) && pair.translation.len() > 8 {
            english_echoes += 1;
            eprintln!(
                "  suspect non-Spanish output:\n    EN: {}\n    ES: {}",
                pair.en, pair.translation
            );
        }
    }
    eprintln!(
        "translations: {} total, {} empty, {} non-Spanish",
        p.pairs.len(),
        empty_translations,
        english_echoes
    );

    // Quality assertion runs BEFORE the snapshot so a drift (cloud LLM is
    // not perfectly deterministic even at temperature 0) still lets us see
    // whether the pipeline is healthy. Snapshot drift = `cargo insta review`.
    let bad = empty_translations + english_echoes;
    let max_bad = (p.pairs.len() as f32 * 0.10).ceil() as usize;
    assert!(
        bad <= max_bad,
        "{} of {} translations are empty or non-Spanish (max allowed: {})",
        bad, p.pairs.len(), max_bad
    );

    let joined: String = p.pairs.iter().map(|p| p.translation.as_str()).collect::<Vec<_>>().join(" ");
    insta::with_settings!({ snapshot_suffix => "reagan_es" }, {
        insta::assert_snapshot!(joined);
    });
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn looks_spanish_positive_words() {
        assert!(looks_spanish("No preguntes que puede hacer tu pais por ti"));
    }

    #[test]
    fn looks_spanish_positive_accents() {
        assert!(looks_spanish("también"));
    }

    #[test]
    fn looks_spanish_negative_english() {
        assert!(!looks_spanish("the quick brown fox"));
    }

    #[test]
    fn parse_pipeline_pairs() {
        let stdout = "\
[stt   76ms] EN: Hello world
[mt   123ms] ES: Hola mundo
[stt   80ms] EN: How are you?
[mt   145ms] ES: ¿Cómo estás?
";
        let p = parse_pipeline(stdout, "ES");
        assert_eq!(p.pairs.len(), 2);
        assert_eq!(p.pairs[0].en, "Hello world");
        assert_eq!(p.pairs[0].translation, "Hola mundo");
        assert_eq!(p.pairs[0].mt_ms, 123);
        assert_eq!(p.stt_total_ms, 156);
        assert_eq!(p.mt_total_ms, 268);
    }
}

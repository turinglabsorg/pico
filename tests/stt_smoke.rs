//! End-to-end STT tests: feed a known speech recording through the binary's
//! `--input-file` path and check the transcript.
//!
//! Two layers of assertion per fixture:
//!   1. **Snapshot** (`insta`): catches any change in segmentation/wording
//!      (regression detection). Review with `cargo insta review`.
//!   2. **WER vs reference**: checks transcription *quality* against a
//!      canonical reference text. Fails on real degradation.
//!
//! Marked `#[ignore]` because they need local models and audio fixtures.
//! Run with:
//!     ./scripts/download-fixtures.sh
//!     ./scripts/download-models.sh
//!     cargo test --test stt_smoke -- --ignored --nocapture

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn fixture(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

fn reference(name: &str) -> PathBuf {
    fixtures_dir().join("transcripts").join(name)
}

fn model(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

/// Tokenize for comparison: lowercase, strip punctuation, split on whitespace.
fn tokens(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|w| w.to_string())
        .collect()
}

/// Word Error Rate via Levenshtein distance on token sequences.
/// WER = (substitutions + insertions + deletions) / reference_length.
/// 0.0 = perfect; 1.0 = every word wrong.
fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r = tokens(reference);
    let h = tokens(hypothesis);
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let n = r.len();
    let m = h.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if r[i - 1] == h[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[n][m] as f64 / n as f64
}

struct Parsed {
    transcript: String,
    /// Total whisper time across all utterances, in milliseconds.
    stt_total_ms: u64,
    /// Number of utterances produced.
    utterance_count: usize,
}

/// Parse pico stdout. Each utterance line is `[stt   <ms>ms] <LANG>: <text>`.
/// We extract the text (drop non-deterministic timing for snapshots) and
/// separately sum the per-utterance ms for an aggregate timing report.
fn parse_stdout(stdout: &str) -> Parsed {
    let mut utterances: Vec<String> = Vec::new();
    let mut total_ms: u64 = 0;
    for line in stdout.lines() {
        if !line.starts_with("[stt") {
            continue;
        }
        // Inside the brackets: `stt   123ms`. Skip it and parse the ms.
        if let (Some(open), Some(close)) = (line.find('['), line.find(']')) {
            if let Some(ms_pos) = line[open..close].find("ms") {
                let num_str: String = line[open + 1..open + ms_pos]
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(n) = num_str.parse::<u64>() {
                    total_ms += n;
                }
            }
            let after = &line[close + 1..];
            if let Some(colon) = after.find(": ") {
                utterances.push(after[colon + 2..].trim().to_string());
            }
        }
    }
    Parsed {
        transcript: utterances.join(" "),
        stt_total_ms: total_ms,
        utterance_count: utterances.len(),
    }
}

/// Audio duration in seconds from a WAV header.
fn wav_duration_s(path: &PathBuf) -> f64 {
    let reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    reader.duration() as f64 / spec.sample_rate as f64
}

struct Fixture {
    name: &'static str,
    audio: &'static str,
    reference_txt: &'static str,
    model_file: &'static str,
    source_lang: &'static str,
    /// Maximum acceptable WER (0.0 = perfect). Tuned per-fixture: small
    /// models + degraded historical audio = looser bound.
    max_wer: f64,
    /// Maximum acceptable real-time factor (stt_time / audio_duration).
    /// 1.0 = exactly real-time; the live translator needs RTF < 1.0 to keep
    /// up with a microphone. Set generously here to avoid CI flakiness; the
    /// printed value is the signal that matters.
    max_rtf: f64,
}

fn run_fixture(f: &Fixture) {
    let bin = env!("CARGO_BIN_EXE_pico");
    let audio_path = fixture(f.audio);
    let model_path = model(f.model_file);
    let ref_path = reference(f.reference_txt);

    assert!(audio_path.exists(), "missing fixture: {} (run scripts/download-fixtures.sh)", audio_path.display());
    assert!(model_path.exists(), "missing model: {} (run scripts/download-models.sh)", model_path.display());

    let audio_s = wav_duration_s(&audio_path);
    let wall_t0 = std::time::Instant::now();

    let output = Command::new(bin)
        .arg("--input-file").arg(&audio_path)
        .arg("--model").arg(&model_path)
        .arg("--source-lang").arg(f.source_lang)
        .arg("--no-mt")
        .arg("--no-tts")
        .output()
        .expect("pico binary should run");

    let wall_s = wall_t0.elapsed().as_secs_f64();
    assert!(output.status.success(), "pico exited with {:?}\n{}", output.status, String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = parse_stdout(&stdout);
    assert!(!parsed.transcript.is_empty(), "empty transcript — pipeline produced no STT output");

    let stt_s = parsed.stt_total_ms as f64 / 1000.0;
    let rtf_stt = stt_s / audio_s;
    let rtf_wall = wall_s / audio_s;

    // Reference is optional: when missing, we still snapshot + report timing
    // but skip the WER check. Useful when adding new fixtures incrementally.
    let wer_opt = if ref_path.exists() {
        let reference = fs::read_to_string(&ref_path).expect("read reference transcript");
        Some(word_error_rate(&reference, &parsed.transcript))
    } else {
        None
    };

    let wer_str = wer_opt.map(|w| format!("{:.3}", w)).unwrap_or_else(|| "n/a".into());
    eprintln!(
        "\n[{name}] audio={audio:.1}s utterances={n} stt={stt:.2}s wall={wall:.2}s \
         rtf_stt={rtf_s:.3} rtf_wall={rtf_w:.3} wer={wer} (max_wer={mw:.2}, max_rtf={mr:.2})",
        name = f.name,
        audio = audio_s,
        n = parsed.utterance_count,
        stt = stt_s,
        wall = wall_s,
        rtf_s = rtf_stt,
        rtf_w = rtf_wall,
        wer = wer_str,
        mw = f.max_wer,
        mr = f.max_rtf,
    );

    insta::with_settings!({ snapshot_suffix => f.name }, {
        insta::assert_snapshot!(parsed.transcript);
    });

    if let Some(wer) = wer_opt {
        assert!(
            wer <= f.max_wer,
            "WER for {} = {:.3} exceeds threshold {:.3}",
            f.name, wer, f.max_wer
        );
    }
    assert!(
        rtf_stt <= f.max_rtf,
        "STT RTF for {} = {:.3} exceeds threshold {:.3} (audio={:.1}s, stt={:.2}s)",
        f.name, rtf_stt, f.max_rtf, audio_s, stt_s
    );
}

// WER thresholds are calibrated on first-run measurements + headroom.
// Tighten as references improve or models change. Fixtures without a
// reference file still validate the snapshot and RTF.

#[test]
#[ignore]
fn jfk_inaugural() {
    run_fixture(&Fixture {
        name: "jfk_inaugural",
        audio: "jfk_inaugural.wav",
        reference_txt: "jfk_inaugural.txt",
        model_file: "ggml-base.en.bin",
        source_lang: "en",
        max_wer: 0.20,
        max_rtf: 0.5,
    });
}

#[test]
#[ignore]
fn reagan_berlin_wall() {
    // Audio is the post-Venice-summit address (June 1987), which mentions
    // the Berlin Wall — NOT the Brandenburg Gate speech. No canonical
    // reference yet → snapshot + RTF only.
    run_fixture(&Fixture {
        name: "reagan_berlin",
        audio: "reagan_berlin.wav",
        reference_txt: "reagan_berlin.txt",
        model_file: "ggml-base.en.bin",
        source_lang: "en",
        max_wer: 1.0,
        max_rtf: 0.5,
    });
}

#[test]
#[ignore]
fn fdr_declaration_of_war() {
    // Audio includes the ceremonial introduction by the Speaker before FDR
    // begins; reference text mirrors that.
    run_fixture(&Fixture {
        name: "fdr_war",
        audio: "fdr_war.wav",
        reference_txt: "fdr_war.txt",
        model_file: "ggml-base.en.bin",
        source_lang: "en",
        max_wer: 0.40,
        max_rtf: 0.5,
    });
}

#[test]
#[ignore]
fn mussolini_declaration_italian() {
    // Old recording with continuous crowd noise — Whisper hallucinates
    // "Grazie a tutti" repeatedly during applause. Threshold reflects the
    // realistic baseline; tighten if we trim the audio or denoise it.
    run_fixture(&Fixture {
        name: "mussolini_war",
        audio: "mussolini_war.wav",
        reference_txt: "mussolini_war.txt",
        model_file: "ggml-large-v3-turbo.bin",
        source_lang: "it",
        max_wer: 0.55,
        max_rtf: 0.5,
    });
}

#[test]
#[ignore]
fn evita_ultimo_discurso() {
    // No canonical Spanish reference yet → snapshot + RTF only.
    run_fixture(&Fixture {
        name: "evita_ultimo_discurso",
        audio: "evita_ultimo_discurso.wav",
        reference_txt: "evita_ultimo_discurso.txt",
        model_file: "ggml-large-v3-turbo.bin",
        source_lang: "es",
        max_wer: 1.0,
        max_rtf: 0.5,
    });
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn wer_perfect_match() {
        assert_eq!(word_error_rate("hello world", "hello world"), 0.0);
    }

    #[test]
    fn wer_one_substitution() {
        // 1 sub on 2 ref words = 0.5
        assert!((word_error_rate("hello world", "hello there") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn wer_punctuation_normalized() {
        assert_eq!(word_error_rate("Hello, World!", "hello world"), 0.0);
    }

    #[test]
    fn parse_stdout_extracts_text_and_timing() {
        let stdout = "\
some other log
[stt   76ms] EN: Hello there.
INFO: ignore me
[stt   80ms] EN: How are you
";
        let p = parse_stdout(stdout);
        assert_eq!(p.transcript, "Hello there. How are you");
        assert_eq!(p.stt_total_ms, 156);
        assert_eq!(p.utterance_count, 2);
    }
}

//! MT model benchmark: translate the same fixed English transcript through
//! several Ollama models and compare quality via chrF (character n-gram
//! F-score, the standard MT metric for systems where the canonical
//! reference may itself be approximate).
//!
//! The reference model is the strongest in the lineup; chrF measures how
//! close each candidate gets to that ceiling. Use to decide which model
//! tier (and therefore which device) is sufficient for production.
//!
//! Run with:
//!     cargo test --test translate_bench -- --ignored --nocapture
//!
//! Cost note: every call hits Ollama Cloud. ~30 utterances × N models per
//! language. Trim CANDIDATES if you're tight on quota.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const REFERENCE_MODEL: &str = "deepseek-v3.1:671b";

const CANDIDATES: &[&str] = &[
    "ministral-3:3b",
    "gemma3:12b",
    "gpt-oss:20b",
    "gemma4:31b",
    "gpt-oss:120b",
    "deepseek-v4-flash",
];

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    temperature: f32,
    stream: bool,
    max_tokens: u32,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

fn lang_name(code: &str) -> &'static str {
    match code {
        "fr" => "French",
        "es" => "Spanish",
        "it" => "Italian",
        _ => "English",
    }
}

async fn translate_once(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    text: &str,
    source: &str,
    target: &str,
) -> Result<String> {
    let system = format!(
        "Translate the user message from {src} to {tgt}.\n\
         Output ONLY the {tgt} translation. No explanations, no labels, no quotes.\n\
         Preserve proper names, numbers, and punctuation.",
        src = lang_name(source),
        tgt = lang_name(target),
    );
    let req = ChatRequest {
        model,
        messages: vec![
            Message { role: "system", content: &system },
            Message { role: "user", content: text },
        ],
        temperature: 0.0,
        stream: false,
        max_tokens: 1024,
    };
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&req)
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("{}: {} on {}: {}", model, status, url, body);
    }
    let parsed: ChatResponse = resp.json().await?;
    Ok(parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_default()
        .trim()
        .to_string())
}

async fn translate(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    text: &str,
    source: &str,
    target: &str,
) -> Result<String> {
    // One retry on transient network errors. Ollama Cloud occasionally
    // drops connections under burst load.
    match translate_once(client, base_url, api_key, model, text, source, target).await {
        Ok(s) => Ok(s),
        Err(e) => {
            tokio::time::sleep(Duration::from_millis(500)).await;
            translate_once(client, base_url, api_key, model, text, source, target)
                .await
                .map_err(|e2| anyhow::anyhow!("first: {}; retry: {}", e, e2))
        }
    }
}

/// Translate an entire batch of utterances sequentially. Returns translations
/// + total elapsed time. Sequential keeps the per-utterance latency
/// realistic (Ollama Cloud sometimes slows down under burst).
async fn translate_batch(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    utterances: &[String],
    source: &str,
    target: &str,
) -> (Vec<String>, Duration) {
    let t0 = Instant::now();
    let mut out = Vec::with_capacity(utterances.len());
    for (i, u) in utterances.iter().enumerate() {
        match translate(client, base_url, api_key, model, u, source, target).await {
            Ok(t) => out.push(t),
            Err(e) => {
                eprintln!("    [{}] utterance {}: {}", model, i, e);
                out.push(String::new());
            }
        }
    }
    (out, t0.elapsed())
}

/// chrF score (Popović 2015). Average F1 of character n-grams up to order 6,
/// with β=2 (recall weighted 4× precision — standard MT setting). Returns
/// 0..1; higher is better.
fn chrf(reference: &str, candidate: &str) -> f64 {
    const MAX_N: usize = 6;
    const BETA: f64 = 2.0;
    let r: Vec<char> = reference.to_lowercase().chars().filter(|c| !c.is_whitespace()).collect();
    let c: Vec<char> = candidate.to_lowercase().chars().filter(|c| !c.is_whitespace()).collect();
    if r.is_empty() || c.is_empty() {
        return 0.0;
    }
    let mut sum_f = 0.0;
    let mut counted = 0;
    for n in 1..=MAX_N {
        if r.len() < n || c.len() < n {
            continue;
        }
        let r_grams = ngrams(&r, n);
        let c_grams = ngrams(&c, n);
        let mut overlap = 0usize;
        for (g, count) in &c_grams {
            let r_count = r_grams.get(g).copied().unwrap_or(0);
            overlap += count.min(&r_count);
        }
        let total_c: usize = c_grams.values().sum();
        let total_r: usize = r_grams.values().sum();
        let p = overlap as f64 / total_c as f64;
        let recall = overlap as f64 / total_r as f64;
        let f = if p + recall == 0.0 {
            0.0
        } else {
            (1.0 + BETA * BETA) * p * recall / (BETA * BETA * p + recall)
        };
        sum_f += f;
        counted += 1;
    }
    if counted == 0 { 0.0 } else { sum_f / counted as f64 }
}

fn ngrams(s: &[char], n: usize) -> HashMap<String, usize> {
    let mut map: HashMap<String, usize> = HashMap::new();
    if s.len() < n {
        return map;
    }
    for i in 0..=s.len() - n {
        let g: String = s[i..i + n].iter().collect();
        *map.entry(g).or_insert(0) += 1;
    }
    map
}

fn load_fixture_lines(name: &str) -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/transcripts")
        .join(name);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e))
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn run_benchmark(source: &str, target: &str, utterances: &[String]) {
    // dotenvy in the test loop too — env is not pre-loaded outside the binary.
    let _ = dotenvy::dotenv_iter().map(|i| i.flatten().for_each(|(k, v)| std::env::set_var(k, v)));
    let _ = dotenvy::from_path(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env"),
    );

    let api_key = std::env::var("OLLAMA_API_KEY").expect("OLLAMA_API_KEY");
    let base_url = std::env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| "https://ollama.com/v1".to_string());

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();

    eprintln!("\n========================================");
    eprintln!("  Benchmark: {} → {} ({} utterances)", source.to_uppercase(), target.to_uppercase(), utterances.len());
    eprintln!("  Reference: {}", REFERENCE_MODEL);
    eprintln!("========================================");

    eprintln!("\n[1/{}] {} (reference)", CANDIDATES.len() + 1, REFERENCE_MODEL);
    let (ref_trans, ref_dur) = rt.block_on(translate_batch(
        &client, &base_url, &api_key, REFERENCE_MODEL, utterances, source, target,
    ));
    let ref_avg_ms = ref_dur.as_millis() / utterances.len().max(1) as u128;
    let ref_joined = ref_trans.join(" ");
    eprintln!("  total={:.1}s  avg/utt={}ms  empty={}",
        ref_dur.as_secs_f64(),
        ref_avg_ms,
        ref_trans.iter().filter(|t| t.is_empty()).count()
    );

    let mut results: Vec<(String, f64, u128, Vec<String>)> = Vec::new();
    results.push((REFERENCE_MODEL.to_string(), 1.0, ref_avg_ms, ref_trans.clone()));

    for (i, model) in CANDIDATES.iter().enumerate() {
        eprintln!("\n[{}/{}] {}", i + 2, CANDIDATES.len() + 1, model);
        let (trans, dur) = rt.block_on(translate_batch(
            &client, &base_url, &api_key, model, utterances, source, target,
        ));
        let avg_ms = dur.as_millis() / utterances.len().max(1) as u128;
        let joined = trans.join(" ");
        let score = chrf(&ref_joined, &joined);
        let empty = trans.iter().filter(|t| t.is_empty()).count();
        eprintln!("  total={:.1}s  avg/utt={}ms  empty={}  chrF={:.3}",
            dur.as_secs_f64(), avg_ms, empty, score);
        results.push((model.to_string(), score, avg_ms, trans));
    }

    // Final ranked table.
    println!("\n=== Results: {} → {} (chrF vs {}) ===", source.to_uppercase(), target.to_uppercase(), REFERENCE_MODEL);
    println!("{:<22} {:>8} {:>10}", "model", "chrF", "ms/utt");
    println!("{}", "-".repeat(44));
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (m, s, ms, _) in &sorted {
        let marker = if m == REFERENCE_MODEL { " (ref)" } else { "" };
        println!("{:<22} {:>8.3} {:>10}{}", m, s, ms, marker);
    }

    // Spot-check sample line — pick utterance 3 (usually a meaty middle one).
    let sample_idx = (utterances.len() / 3).min(utterances.len().saturating_sub(1));
    println!("\n--- Sample (utterance {}) ---", sample_idx);
    println!("EN: {}", utterances[sample_idx]);
    for (m, _, _, trans) in &results {
        if let Some(t) = trans.get(sample_idx) {
            println!("[{:<20}] {}", m, t);
        }
    }
}

#[test]
#[ignore]
fn bench_en_to_es() {
    let utts = load_fixture_lines("reagan_stt_en.txt");
    run_benchmark("en", "es", &utts);
}

#[test]
#[ignore]
fn bench_en_to_fr() {
    let utts = load_fixture_lines("reagan_stt_en.txt");
    run_benchmark("en", "fr", &utts);
}

#[test]
#[ignore]
fn bench_en_to_it() {
    let utts = load_fixture_lines("reagan_stt_en.txt");
    run_benchmark("en", "it", &utts);
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn chrf_self_is_one() {
        let s = "Hola mundo, esto es una prueba.";
        assert!((chrf(s, s) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chrf_disjoint_is_low() {
        let r = "Hola mundo";
        let c = "xyz qwerty";
        assert!(chrf(r, c) < 0.10);
    }

    #[test]
    fn chrf_close_is_high() {
        // Single-word substitution in 7-word sentence drops chrF noticeably
        // because the metric is character-level. ~0.6 is the realistic floor
        // for "close but not identical".
        let r = "El gato camina por la calle";
        let c = "El gato anda por la calle";
        let s = chrf(r, c);
        assert!(s > 0.5, "expected close translation chrF > 0.5, got {}", s);
    }
}

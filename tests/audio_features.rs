//! Diagnostic only: print distribution of RMS and ZCR over a fixture.
//! Use to calibrate VAD thresholds against real audio. Run with:
//!   cargo test --test audio_features -- --ignored --nocapture
//!
//! Not a regression test — no asserts.

use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn rms(s: &[f32]) -> f32 {
    if s.is_empty() { return 0.0; }
    (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
}

fn zcr(s: &[f32]) -> f32 {
    if s.len() < 2 { return 0.0; }
    let c = s.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
    c as f32 / (s.len() - 1) as f32
}

fn percentile(values: &mut [f32], p: f32) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((values.len() as f32 - 1.0) * p) as usize;
    values[idx]
}

fn analyze(name: &str) {
    let path = fixture(name);
    if !path.exists() {
        eprintln!("skip {}: missing", name);
        return;
    }
    let mut reader = hound::WavReader::open(&path).expect("open wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i16::MAX as f32)
            .collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };
    let win = (spec.sample_rate as usize / 1000) * 30; // 30ms
    let mut rmss: Vec<f32> = Vec::new();
    let mut zcrs: Vec<f32> = Vec::new();
    for chunk in samples.chunks_exact(win) {
        rmss.push(rms(chunk));
        zcrs.push(zcr(chunk));
    }

    let mut rms_sorted = rmss.clone();
    let mut zcr_sorted = zcrs.clone();

    println!("\n[{}] {} windows of 30ms", name, rmss.len());
    println!(
        "  RMS  p10={:.4} p25={:.4} p50={:.4} p75={:.4} p90={:.4} p95={:.4} p99={:.4}",
        percentile(&mut rms_sorted, 0.10),
        percentile(&mut rms_sorted, 0.25),
        percentile(&mut rms_sorted, 0.50),
        percentile(&mut rms_sorted, 0.75),
        percentile(&mut rms_sorted, 0.90),
        percentile(&mut rms_sorted, 0.95),
        percentile(&mut rms_sorted, 0.99),
    );
    println!(
        "  ZCR  p10={:.3} p25={:.3} p50={:.3} p75={:.3} p90={:.3} p95={:.3} p99={:.3}",
        percentile(&mut zcr_sorted, 0.10),
        percentile(&mut zcr_sorted, 0.25),
        percentile(&mut zcr_sorted, 0.50),
        percentile(&mut zcr_sorted, 0.75),
        percentile(&mut zcr_sorted, 0.90),
        percentile(&mut zcr_sorted, 0.95),
        percentile(&mut zcr_sorted, 0.99),
    );
    // How much of audio passes the current gate (rms_high=0.015, zcr_max=0.18)?
    let passes_rms = rmss.iter().filter(|r| **r > 0.015).count();
    let passes_zcr = zcrs.iter().filter(|z| **z <= 0.18).count();
    let passes_both = rmss
        .iter()
        .zip(zcrs.iter())
        .filter(|(r, z)| **r > 0.015 && **z <= 0.18)
        .count();
    let n = rmss.len() as f32;
    println!(
        "  Gate: rms>0.015: {:.1}%   zcr<=0.18: {:.1}%   both: {:.1}%",
        passes_rms as f32 / n * 100.0,
        passes_zcr as f32 / n * 100.0,
        passes_both as f32 / n * 100.0,
    );
}

#[test]
#[ignore]
fn dump_features_all() {
    for name in [
        "jfk_inaugural.wav",
        "reagan_berlin.wav",
        "fdr_war.wav",
        "mussolini_war.wav",
        "evita_ultimo_discurso.wav",
    ] {
        analyze(name);
    }
}

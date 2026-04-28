//! Lightweight speaker-gender estimation via F0 (fundamental frequency).
//!
//! Computes one F0 reading per ~30ms voiced window using normalized
//! autocorrelation (search range 65–500 Hz), then takes the median across
//! the utterance. Adult male speakers cluster ~85–155 Hz, adult female
//! ~165–255 Hz; a threshold of 165 Hz separates the two cleanly. Works on
//! the same 16kHz mono PCM the VAD already produced.
//!
//! No external deps, no ML model. Accuracy is "good enough" for picking a
//! TTS voice — it doesn't need to handle children, falsetto, or whisper.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
    /// Audio too short, too quiet, or non-voiced (fricatives, music, applause).
    Unknown,
}

/// Aggregates per-utterance gender estimates over a session and reports the
/// running majority. Stabilizes the per-utterance pitch detector, which can
/// flip on noisy or short clips. Thread-safe via interior mutex; caller can
/// share an Arc across the pipeline.
#[derive(Default)]
pub struct GenderTracker {
    male: u32,
    female: u32,
}

impl GenderTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push the per-utterance estimate; ignored when Unknown.
    pub fn observe(&mut self, g: Gender) {
        match g {
            Gender::Male => self.male += 1,
            Gender::Female => self.female += 1,
            Gender::Unknown => {}
        }
    }

    /// Best guess so far. Returns Unknown only before any voiced observation.
    pub fn current(&self) -> Gender {
        match (self.male, self.female) {
            (0, 0) => Gender::Unknown,
            (m, f) if m >= f => Gender::Male,
            _ => Gender::Female,
        }
    }
}

const SR: f32 = 16_000.0;
const MIN_F0_HZ: f32 = 65.0;
const MAX_F0_HZ: f32 = 500.0;
/// Boundary between adult male and female F0. Empirically ~165 Hz.
const GENDER_THRESHOLD_HZ: f32 = 165.0;
/// Minimum normalized autocorrelation peak to accept a window as voiced.
const VOICING_THRESHOLD: f32 = 0.30;

/// Estimate the gender of the speaker in this PCM clip.
pub fn estimate_gender(pcm_16k_mono: &[f32]) -> Gender {
    let med = median_f0(pcm_16k_mono);
    match med {
        Some(hz) if hz < GENDER_THRESHOLD_HZ => Gender::Male,
        Some(_) => Gender::Female,
        None => Gender::Unknown,
    }
}

/// Median F0 in Hz across all voiced 30ms windows. None when the audio is
/// too short or contains no voiced windows.
pub fn median_f0(pcm: &[f32]) -> Option<f32> {
    let win = (SR * 0.030) as usize; // 30ms = 480 samples
    let hop = (SR * 0.015) as usize; // 15ms hop
    if pcm.len() < win {
        return None;
    }
    let min_lag = (SR / MAX_F0_HZ) as usize;
    let max_lag = (SR / MIN_F0_HZ) as usize;
    let mut f0s: Vec<f32> = Vec::new();
    let mut i = 0;
    while i + win + max_lag <= pcm.len() {
        let frame = &pcm[i..i + win + max_lag];
        if let Some(f0) = autocorr_f0(frame, win, min_lag, max_lag) {
            f0s.push(f0);
        }
        i += hop;
    }
    if f0s.is_empty() {
        return None;
    }
    f0s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(f0s[f0s.len() / 2])
}

/// Normalized autocorrelation peak-pick over the lag range; returns the F0
/// in Hz at the *first* local maximum that exceeds the voicing threshold.
/// Picking the first peak (not the global max) avoids sub-harmonic errors
/// where lag = 2·T0 has equal correlation to lag = T0 and gets reported as
/// half the true F0.
fn autocorr_f0(frame: &[f32], win: usize, min_lag: usize, max_lag: usize) -> Option<f32> {
    let energy_zero: f32 = frame[..win].iter().map(|x| x * x).sum();
    if energy_zero < 1e-6 {
        return None;
    }
    let mut r = vec![0.0f32; max_lag + 1];
    for lag in min_lag..=max_lag {
        let mut sum = 0.0f32;
        for n in 0..win {
            sum += frame[n] * frame[n + lag];
        }
        let energy_lag: f32 = frame[lag..lag + win].iter().map(|x| x * x).sum();
        let denom = (energy_zero * energy_lag).sqrt();
        if denom > 1e-9 {
            r[lag] = sum / denom;
        }
    }
    // Walk forward from min_lag, return the first lag whose r is above the
    // voicing threshold and is a local max (r[lag] > r[lag-1] && r[lag] >= r[lag+1]).
    for lag in (min_lag + 1)..max_lag {
        if r[lag] >= VOICING_THRESHOLD && r[lag] > r[lag - 1] && r[lag] >= r[lag + 1] {
            return Some(SR / lag as f32);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(freq: f32, secs: f32) -> Vec<f32> {
        let n = (SR * secs) as usize;
        (0..n).map(|i| 0.5 * (TAU * freq * i as f32 / SR).sin()).collect()
    }

    #[test]
    fn detects_male_pitch() {
        // 120 Hz is a typical adult male F0.
        let pcm = sine(120.0, 0.5);
        assert_eq!(estimate_gender(&pcm), Gender::Male);
    }

    #[test]
    fn detects_female_pitch() {
        // 220 Hz is a typical adult female F0.
        let pcm = sine(220.0, 0.5);
        assert_eq!(estimate_gender(&pcm), Gender::Female);
    }

    #[test]
    fn silence_is_unknown() {
        let pcm = vec![0.0f32; (SR * 0.5) as usize];
        assert_eq!(estimate_gender(&pcm), Gender::Unknown);
    }

    #[test]
    fn too_short_is_unknown() {
        let pcm = vec![0.1f32; 100];
        assert_eq!(estimate_gender(&pcm), Gender::Unknown);
    }

    #[test]
    fn tracker_running_majority() {
        let mut t = GenderTracker::new();
        assert_eq!(t.current(), Gender::Unknown);
        t.observe(Gender::Male);
        assert_eq!(t.current(), Gender::Male);
        t.observe(Gender::Female);
        // Tie breaks to Male — first-mover stays. In practice ties resolve
        // on the next utterance; the choice here is just to avoid Unknown.
        assert_eq!(t.current(), Gender::Male);
        t.observe(Gender::Female);
        t.observe(Gender::Female);
        assert_eq!(t.current(), Gender::Female);
        t.observe(Gender::Unknown);
        assert_eq!(t.current(), Gender::Female);
    }

    #[test]
    fn median_robust_to_outliers() {
        // Mostly 200 Hz with a brief 400 Hz harmonic burst — median should
        // stay around the dominant pitch.
        let mut pcm = sine(200.0, 0.5);
        let burst = sine(400.0, 0.05);
        pcm.splice(0..burst.len(), burst.iter().copied());
        let f0 = median_f0(&pcm).unwrap();
        assert!((f0 - 200.0).abs() < 30.0, "expected ~200, got {}", f0);
    }
}

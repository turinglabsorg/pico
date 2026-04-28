use anyhow::Result;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use crate::mt::Lang;

/// Write mono PCM f32 samples as 16-bit signed WAV at `sr` Hz.
pub fn write_pcm_f32_wav(path: &Path, sr: u32, samples: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut w = WavWriter::create(path, spec)?;
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

pub struct Archive {
    pub dir: PathBuf,
    input_writer: Mutex<Option<WavWriter<BufWriter<File>>>>,
    /// One append-mode text file per language (source + every target).
    /// Opened lazily so a session that never produces a given language
    /// doesn't leave an empty file behind.
    transcript_writers: Mutex<HashMap<Lang, BufWriter<File>>>,
}

impl Archive {
    pub fn new(dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&dir)?;
        let input_spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let input_path = dir.join("input.wav");
        let writer = WavWriter::create(&input_path, input_spec)?;
        Ok(Self {
            dir,
            input_writer: Mutex::new(Some(writer)),
            transcript_writers: Mutex::new(HashMap::new()),
        })
    }

    pub fn input_path(&self) -> PathBuf {
        self.dir.join("input.wav")
    }

    pub fn output_path(&self, lang: Lang) -> PathBuf {
        self.dir.join(format!("out-{}.wav", lang.code()))
    }

    pub fn temp_tts_path(&self, lang: Lang) -> PathBuf {
        self.dir.join(format!(".tmp-{}.wav", lang.code()))
    }

    pub fn write_input_f32(&self, samples: &[f32]) -> Result<()> {
        let mut guard = self.input_writer.lock().unwrap();
        if let Some(w) = guard.as_mut() {
            for s in samples {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                w.write_sample(v)?;
            }
            w.flush()?;
        }
        Ok(())
    }

    /// Append the samples from `src` into the per-language master WAV for `lang`.
    /// Creates the master file on first call.
    pub fn append_tts(&self, lang: Lang, src: &Path) -> Result<()> {
        let dst = self.output_path(lang);
        let mut reader = WavReader::open(src)?;
        let src_spec = reader.spec();

        if !dst.exists() {
            let mut writer = WavWriter::create(&dst, src_spec)?;
            for s in reader.samples::<i16>() {
                writer.write_sample(s?)?;
            }
            writer.finalize()?;
        } else {
            let mut writer = WavWriter::append(&dst)?;
            for s in reader.samples::<i16>() {
                writer.write_sample(s?)?;
            }
            writer.finalize()?;
        }
        Ok(())
    }

    pub fn finalize_input(&self) -> Result<()> {
        let mut guard = self.input_writer.lock().unwrap();
        if let Some(w) = guard.take() {
            w.finalize()?;
        }
        // Flush all transcript files. They stay opened so subsequent
        // appends (rare after finalize, but possible) still work.
        if let Ok(mut writers) = self.transcript_writers.lock() {
            for w in writers.values_mut() {
                let _ = w.flush();
            }
        }
        Ok(())
    }

    pub fn transcript_path(&self, lang: Lang) -> PathBuf {
        self.dir.join(format!("transcript-{}.txt", lang.code()))
    }

    /// Append one line `[YYYY-MM-DD HH:MM:SS] <text>` to the per-language
    /// transcript file. Creates the file on first call. Errors are logged
    /// at the call site — failing to write a transcript line should never
    /// stop the pipeline.
    pub fn append_transcript(&self, lang: Lang, text: &str) -> Result<()> {
        let mut writers = self.transcript_writers.lock().unwrap();
        let writer = match writers.get_mut(&lang) {
            Some(w) => w,
            None => {
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(self.transcript_path(lang))?;
                writers.insert(lang, BufWriter::new(f));
                writers.get_mut(&lang).unwrap()
            }
        };
        let stamp = format_timestamp(SystemTime::now());
        writeln!(writer, "[{}] {}", stamp, text)?;
        writer.flush()?;
        Ok(())
    }
}

/// Format SystemTime as local-ish `YYYY-MM-DD HH:MM:SS`. Uses UTC seconds
/// since epoch + a manual decomposition so we don't pull in chrono just
/// for one transcript line. Sub-second precision is dropped — the caller
/// can layer it on if needed.
fn format_timestamp(t: SystemTime) -> String {
    let secs = t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (y, mo, d, h, mi, s) = epoch_to_civil(secs);
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", y, mo, d, h, mi, s)
}

/// Howard Hinnant's days_from_civil inverse — converts UNIX seconds into
/// (year, month, day, hour, minute, second) in UTC. ~10 lines, no deps.
fn epoch_to_civil(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let h = (rem / 3600) as u32;
    let mi = ((rem % 3600) / 60) as u32;
    let s = (rem % 60) as u32;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let mo = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d, h, mi, s)
}

# pico

Real-time speech translation in Rust. Speak into the mic, hear the translation come out — locally on your Mac.

- **Speech-to-text**: [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via `whisper-rs` (Metal on Apple Silicon)
- **Machine translation**: any [Ollama](https://ollama.com) model — local or cloud
- **Text-to-speech**: [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) via `tts-rs` (ONNX Runtime, on-device)
- **Source languages**: EN · FR · ES · IT
- **Target languages**: EN · FR · ES · IT (same set, any combination)

The full pipeline runs offline by default. The only network hop is the MT model when pointed at Ollama Cloud.

## Requirements

- macOS 11+ on Apple Silicon (Intel Macs work but no Metal acceleration)
- Rust 1.75+
- `ffmpeg` (for the test-fixture script)
- ~2 GB free for models

Linux is best-effort: capture/output via cpal works, the macOS-specific microphone-permission code is conditionally compiled out, and `make app` is a no-op.

## Quick start

```sh
# 1. Install Rust if you don't have it
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Models (English-only Whisper + multilingual Kokoro)
./scripts/download-models.sh base.en
./scripts/download-models.sh large-v3-turbo
./scripts/download-kokoro.sh

# 3. Translation backend — pick one:
#    a) Ollama Cloud (default; requires an API key from ollama.com)
echo 'OLLAMA_API_KEY=your_key_here' > .env
echo 'OLLAMA_BASE_URL=https://ollama.com/v1' >> .env
#    b) Local Ollama
ollama pull gemma3:12b
echo 'OLLAMA_BASE_URL=http://localhost:11434/v1' > .env

# 4. Build the macOS app bundle and launch
make run
```

On first launch macOS will prompt for microphone access — the app bundle declares `NSMicrophoneUsageDescription` and an audio-input entitlement so the standard system dialog appears.

## Usage

The CLI runs the full pipeline against the default mic + speakers:

```sh
target/release/pico --gui --langs es,it
```

Common flags:

| flag | default | meaning |
|------|---------|---------|
| `--gui` | off | open the egui control panel (otherwise CLI-only) |
| `--source-lang` | `en` | source language (`en`/`fr`/`es`/`it`) |
| `--langs` | `fr,es,it` | comma-separated target languages |
| `--model` | `models/ggml-base.en.bin` | path to a `whisper.cpp` `.bin` model |
| `--ollama-model` | `gemma3:12b` | Ollama model name |
| `--input-file PATH` | — | offline mode: read audio from a WAV file instead of the mic |
| `--save DIR` | auto | session recording directory (defaults to `~/Documents/pico-sessions/<timestamp>`) |
| `--no-save` | off | disable session recording |
| `--list-devices` / `--list-outputs` | — | print available audio devices and exit |

Run `pico --help` for the full list (VAD thresholds, per-channel routing, etc.).

## How it works

```
mic ──► capture ──► VAD ──► utterance ──► STT ──► MT ──► TTS ──► multichannel out
                                            │       │       │
                                          base.en  Ollama  Kokoro
                                          (or       (cloud  (local
                                          large-v3) /local) ONNX)
```

- **VAD** (Voice Activity Detection) is a hand-rolled RMS+ZCR gate — no extra ML model needed. It chunks continuous speech into utterances on a configurable silence threshold, with a hard ceiling at `--max-utterance-s` to keep latency bounded.
- **Worker thread** runs STT/MT/TTS off the audio path. Capture and VAD never block, so a slow MT call doesn't drop incoming audio. Backpressure on the output ringbuffer prevents head-of-line drops on long sentences.
- **Speaker-gender detection** uses autocorrelation F0 estimation (no ML model). Pico picks a same-gender Kokoro voice automatically; a manual override is available in the GUI.
- **Hallucination filter** post-processes Whisper output: it drops the classic "Grazie", "Thanks for watching", "Subscribe" phrases that the model invents from keyboard noise or near-silence.

## Session output

Each run writes a directory under `~/Documents/pico-sessions/<timestamp>/`:

```
<timestamp>/
├── input.wav              # the source audio (16 kHz mono, 16-bit PCM)
├── out-en.wav             # synthesized English (24 kHz mono, 16-bit PCM)
├── out-es.wav             # synthesized Spanish
├── transcript-it.txt      # source transcription, timestamped per line
├── transcript-en.txt      # English translations
└── transcript-es.txt      # Spanish translations
```

Files are uncompressed PCM — fine to import in any DAW or feed into another tool. `--no-save` skips this entirely.

## Translation-quality benchmark

`tests/translate_bench.rs` measures chrF (character F-score) of several Ollama models against a strong reference, on 29 fixed English utterances translated to Spanish.

Sample results (Reagan 1987 Berlin Wall speech, EN→ES, reference is `deepseek-v3.1:671b`):

| model | chrF | ms/utt |
|-------|-----:|-------:|
| `deepseek-v4-flash` | 0.897 | 3895 |
| `gemma4:31b` | 0.892 | 4871 |
| `gpt-oss:120b` | 0.885 | 2777 |
| `gpt-oss:20b` | 0.864 | 1801 |
| `gemma3:12b` | **0.857** | **810** |
| `ministral-3:3b` | 0.789 | 815 |

`gemma3:12b` is pico's default — best quality-per-millisecond on this benchmark. Run the benchmark yourself with:

```sh
make bench
```

## Tests

- `make test` — unit tests only (no fixtures, no Ollama). Always runs in CI.
- `make test-full` — full STT smoke + benchmark; requires fixtures, models, and an Ollama key.

To prepare the fixtures the first time:

```sh
./scripts/download-fixtures.sh
```

This pulls public-domain speech recordings (JFK, Reagan, FDR, Mussolini, Eva Perón) from archive.org and converts them to 16 kHz mono WAV.

## Project layout

```
src/
├── main.rs           CLI entry point + capture/VAD loop + worker spawn
├── audio.rs          cpal capture, mono resampler, file-mode capture
├── output.rs         multichannel output stream + send-safe push sink
├── vad.rs            RMS + ZCR voice activity detector
├── pitch.rs          F0 autocorrelation, gender tracker
├── stt.rs            whisper.cpp wrapper + hallucination filter
├── mt.rs             Ollama chat translator
├── mt_cache.rs       LRU+TTL translation cache (decorator pattern)
├── tts.rs            Kokoro synthesis + per-language voice picker
├── archive.rs        WAV + transcript writers
├── state.rs          Lock-free Controls + GUI SharedState
├── gui.rs            egui control panel
├── macos_perm.rs     AVFoundation microphone consent (mac-only)
└── output.rs         …

scripts/
├── download-models.sh     fetch a Whisper ggml model
├── download-kokoro.sh     fetch the Kokoro ONNX + voices bundle
├── download-fixtures.sh   fetch public-domain speech for tests
├── codesign-macos.sh      ad-hoc-sign the binary with mic entitlement
└── bundle-macos.sh        package as Pico.app for the system mic prompt

tests/
├── stt_smoke.rs           end-to-end STT against fixtures (WER + RTF)
├── translate_smoke.rs     full pipeline EN→ES on Reagan
├── translate_bench.rs     multi-model chrF benchmark
└── audio_features.rs      VAD threshold calibration helper
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Pico stands on the shoulders of these projects:
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for portable Whisper inference
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for the small, expressive TTS model
- [Ollama](https://ollama.com) for making LLM hosting boring
- [`cpal`](https://github.com/RustAudio/cpal), [`whisper-rs`](https://github.com/tazz4843/whisper-rs), [`tts-rs`](https://github.com/dnaka91/tts-rs), [`ort`](https://github.com/pykeio/ort), [`egui`](https://github.com/emilk/egui)

Speech fixtures used in tests are sourced from public-domain recordings on [archive.org](https://archive.org).

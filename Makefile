# Pico — convenience targets
.PHONY: build app run test bench clean

# Plain release build (CLI / file mode).
build:
	cargo build --release --bin pico

# macOS .app bundle (signed + microphone entitlement). Required for the
# microphone TCC prompt to appear; the raw binary will silently receive
# zeroed buffers.
app: build
	bash scripts/bundle-macos.sh

# Build everything and launch the GUI as a proper app bundle.
run: app
	open target/release/Pico.app --args --gui --langs es,it

# Unit tests only — no fixtures, no Ollama.
test:
	cargo test --release

# Full STT + MT smoke (needs fixtures + models + OLLAMA_API_KEY).
test-full:
	cargo test --release -- --ignored --nocapture

# Translation-quality benchmark across Ollama models.
bench:
	cargo test --release --test translate_bench -- bench_en_to_es --ignored --nocapture

clean:
	cargo clean

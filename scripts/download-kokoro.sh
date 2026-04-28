#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models/kokoro

BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

for f in "kokoro-v1.0.int8.onnx" "voices-v1.0.bin"; do
  out="models/kokoro/$f"
  if [ -f "$out" ]; then
    echo "$out exists, skipping"
    continue
  fi
  echo "Downloading $f ..."
  curl -L --progress-bar -o "$out.part" "$BASE/$f"
  mv "$out.part" "$out"
done

echo "Done: $(ls -lh models/kokoro)"

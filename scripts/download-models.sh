#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

MODEL="${1:-small.en}"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
OUT="models/ggml-${MODEL}.bin"

if [ -f "$OUT" ]; then
  echo "$OUT already exists"
  exit 0
fi

echo "Downloading $URL"
curl -L --progress-bar -o "$OUT.part" "$URL"
mv "$OUT.part" "$OUT"
echo "Saved to $OUT"

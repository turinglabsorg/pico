#!/usr/bin/env bash
# Download public-domain speech fixtures for integration tests.
# Converts each MP3 to 16 kHz mono WAV (whisper's native rate).
# Idempotent: skips files that already exist.
set -euo pipefail

cd "$(dirname "$0")/.."
DEST="tests/fixtures"
mkdir -p "$DEST"

if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg required: brew install ffmpeg" >&2
  exit 1
fi

# Format: <out_basename>|<source_url>
FIXTURES=(
  "jfk_inaugural|https://archive.org/download/GreatestSpeechesBabbleLabs/Inaugural%20Address%20-%20John%20F.%20Kennedy%20(1961).mp3"
  "reagan_berlin|https://archive.org/download/GreatestSpeechesBabbleLabs/Address%20to%20the%20Nation%20on%20the%20Berlin%20Wall%20-%20Ronald%20Reagan%20(1987).mp3"
  "fdr_war|https://archive.org/download/GreatestSpeechesBabbleLabs/DeclarationofWarAgainstJapan.mp3"
  "mussolini_war|https://archive.org/download/MussoliniDeclarationOfWar/Benito%20Mussolini%20Reading%20Italy%2527s%20Declaration%20of%20War%2C%2006%3A10%3A1940.mp3"
  "evita_ultimo_discurso|https://archive.org/download/UltimoDiscursoDeEvaPeron.mp3_283/UltimoDiscursoDeEvaPeron.mp3"
)

for entry in "${FIXTURES[@]}"; do
  base="${entry%%|*}"
  url="${entry##*|}"
  wav="$DEST/${base}.wav"

  if [ -f "$wav" ]; then
    echo "skip ${base}.wav (exists)"
    continue
  fi

  tmp_mp3="$(mktemp -t "${base}.XXXXXX.mp3")"
  trap 'rm -f "$tmp_mp3"' EXIT

  echo "download $base"
  curl -sSL --fail "$url" -o "$tmp_mp3"

  echo "convert  $base -> 16kHz mono WAV"
  ffmpeg -hide_banner -loglevel error -y -i "$tmp_mp3" -ac 1 -ar 16000 -sample_fmt s16 "$wav"

  rm -f "$tmp_mp3"
  trap - EXIT
done

echo
echo "fixtures ready in $DEST/"
ls -lh "$DEST"

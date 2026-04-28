#!/usr/bin/env bash
# macOS-only: ad-hoc codesign the release binary with the microphone
# entitlement so the OS shows the standard "Allow access" prompt instead of
# silently feeding zeroed buffers.
#
# Re-run after every cargo build (signature is on the binary content, not
# the path). Idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS only — skipping" >&2
  exit 0
fi

BIN="${1:-target/release/pico}"
if [[ ! -f "$BIN" ]]; then
  echo "binary not found: $BIN" >&2
  exit 1
fi

ENT="$(mktemp -t pico-entitlements.XXXXXX.plist)"
trap 'rm -f "$ENT"' EXIT

cat > "$ENT" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.device.audio-input</key>
    <true/>
</dict>
</plist>
PLIST

codesign --force --sign - --entitlements "$ENT" "$BIN"
echo "signed: $BIN"
codesign -d --entitlements - "$BIN" 2>&1 | grep -E "audio-input" >/dev/null \
  && echo "entitlement OK" \
  || { echo "WARN: entitlement not found in signature"; exit 1; }

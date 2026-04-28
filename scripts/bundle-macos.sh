#!/usr/bin/env bash
# Create a minimal macOS .app bundle around the release binary so the OS
# tracks it as a proper application. This is the only reliable way to make
# the microphone permission prompt appear and persist across rebuilds.
#
# Run after `cargo build --release`. Idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS only — skipping" >&2
  exit 0
fi

BIN="target/release/pico"
APP="target/release/Pico.app"

if [[ ! -f "$BIN" ]]; then
  echo "binary not found: $BIN — run cargo build --release first" >&2
  exit 1
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Pico</string>
    <key>CFBundleDisplayName</key>
    <string>Pico</string>
    <key>CFBundleIdentifier</key>
    <string>org.turinglabs.pico</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>pico</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Pico needs the microphone to translate your speech in real time.</string>
</dict>
</plist>
PLIST

# Inner binary keeps the cargo name so debugging stack traces look familiar.
cp "$BIN" "$APP/Contents/MacOS/pico-bin"

# Wrapper script becomes the bundle's CFBundleExecutable. It chdirs into the
# project root (so relative `models/...` paths still resolve when launched
# via `open Pico.app`) and tees stdout/stderr to /tmp/pico-gui.log so we
# can read what's happening without reaching into Console.app.
PROJECT_ROOT="$(pwd)"
cat > "$APP/Contents/MacOS/pico" <<WRAP
#!/usr/bin/env bash
cd "$PROJECT_ROOT"
exec "\$(dirname "\$0")/pico-bin" "\$@" 2>&1 | tee /tmp/pico-gui.log
WRAP
chmod +x "$APP/Contents/MacOS/pico"

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

codesign --force --sign - --entitlements "$ENT" "$APP/Contents/MacOS/pico-bin"
codesign --force --sign - --entitlements "$ENT" "$APP"

echo "bundle ready: $APP"
echo
echo "Launch with:"
echo "  open $APP --args --gui --langs es,it"
echo
echo "On first launch macOS will prompt for microphone access."

#!/bin/sh
# Registers the native host with every Chromium-family browser on this machine.
# Provisions the voice first if needed. Re-run after moving this folder.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
NAME=com.leovicenzi.read_aloud
# Fixed by the "key" in extension/manifest.json, so the unpacked extension keeps this id.
EXTENSION_ID=joemjhblemclfnkalfhmfnpopadodhmn
PY="$ROOT/.venv-tts/bin/python"
[ -x "$PY" ] || "$ROOT/scripts/setup-tts.sh"

# Chromium starts the host without arguments or a shell, so it gets a wrapper.
HOST="$ROOT/host/read-aloud-host"
cat > "$HOST" <<WRAPPER
#!/bin/sh
exec "$PY" -I "$ROOT/host/native_host.py" "\$@"
WRAPPER
chmod +x "$HOST"

case "$(uname -s)" in
    Darwin)
        BASE="$HOME/Library/Application Support"
        BROWSERS="Chromium|Google/Chrome|Google/Chrome Beta|Google/Chrome Canary|BraveSoftware/Brave-Browser|Microsoft Edge|Vivaldi"
        ;;
    *)
        BASE="${XDG_CONFIG_HOME:-$HOME/.config}"
        BROWSERS="chromium|google-chrome|google-chrome-beta|google-chrome-unstable|BraveSoftware/Brave-Browser|microsoft-edge|vivaldi"
        ;;
esac

found=0
IFS='|'
for browser in $BROWSERS; do
    [ -d "$BASE/$browser" ] || continue
    mkdir -p "$BASE/$browser/NativeMessagingHosts"
    "$PY" -c 'import json, sys; json.dump({"name": sys.argv[1], "description": "Local read-aloud (Kokoro)", "path": sys.argv[2], "type": "stdio", "allowed_origins": [f"chrome-extension://{sys.argv[3]}/"]}, open(sys.argv[4], "w"), indent=2)' \
        "$NAME" "$HOST" "$EXTENSION_ID" "$BASE/$browser/NativeMessagingHosts/$NAME.json"
    echo "Registered with $browser"
    found=1
done
[ "$found" = 1 ] || { echo "No Chromium-family browser profile found under $BASE" >&2; exit 1; }
echo "Now load $ROOT/extension as an unpacked extension (chrome://extensions, Developer mode)."

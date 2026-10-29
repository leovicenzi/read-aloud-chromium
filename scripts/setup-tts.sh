#!/bin/sh
# Explicit, one-time provisioning. Playback itself is offline and installs nothing.
# Apple Silicon macOS gets MLX; Linux gets PyTorch (CUDA build, CPU fallback at runtime).
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY="$ROOT/.venv-tts/bin/python"
SPACY_EN='https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)
        command -v uv >/dev/null || { echo 'Install uv first: brew install uv' >&2; exit 1; }
        command -v ffmpeg >/dev/null || { echo 'Install FFmpeg first: brew install ffmpeg' >&2; exit 1; }
        uv venv --python 3.11 --allow-existing "$ROOT/.venv-tts"
        uv pip install --python "$PY" 'mlx-audio==0.6.1' 'misaki[en]==0.9.4' "$SPACY_EN"
        ;;
    Linux/x86_64)
        command -v uv >/dev/null || { echo 'Install uv first (https://docs.astral.sh/uv/)' >&2; exit 1; }
        command -v ffmpeg >/dev/null || { echo 'Install FFmpeg first' >&2; exit 1; }
        # kokoro supports Python <3.13. TTS_TORCH_INDEX=cpu skips the multi-GB CUDA wheels.
        INDEX="https://download.pytorch.org/whl/${TTS_TORCH_INDEX:-cu130}"
        uv venv --python 3.12 --allow-existing "$ROOT/.venv-tts"
        uv pip install --python "$PY" --index-url "$INDEX" 'torch==2.14.1'
        uv pip install --python "$PY" 'kokoro==0.9.4' 'misaki[en]==0.9.4' \
            'sounddevice==0.5.6' "$SPACY_EN"
        ;;
    *)
        echo 'Bundled read-aloud supports Apple Silicon macOS and x86_64 Linux; other platforms are not supported.' >&2
        exit 1
        ;;
esac
"$PY" - "$ROOT/host" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import read_aloud
voices = read_aloud.Voices()
# Warm every language's voice without playing sound. Caches weights, voices, language data.
for lang, text in [('en', 'Local reading is ready.'), ('pt', 'A leitura local está pronta.'),
                   ('es', 'La lectura local está lista.'), ('fr', 'La lecture locale est prête.'),
                   ('it', 'La lettura locale è pronta.')]:
    for result in voices.pipeline(lang)(text, voice=read_aloud.voice_for(lang)):
        assert result.audio is not None and len(result.audio) > 0
backend = 'MLX' if read_aloud.use_mlx() else f'PyTorch/{getattr(voices.model, "device", "?")}'
print(f'Kokoro ({backend}) ready in English, Portuguese, Spanish, French and Italian. Playback is offline; use the toolbar button in Chromium.')
PY

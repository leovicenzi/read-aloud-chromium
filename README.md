# Read Aloud for Chromium

A Chromium extension that reads web articles, or the text you select, aloud with a local
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) voice. It highlights each word as it
is spoken. Text never leaves your machine: synthesis runs on your GPU (CUDA on Linux,
MLX on Apple Silicon) or CPU.

It reads English, Brazilian Portuguese, Spanish, French and Italian, choosing the language
per paragraph, so a page that mixes languages switches voice as it goes.

The voice engine is the read-aloud helper from
[leovicenzi/herdr-file-viewer](https://github.com/leovicenzi/herdr-file-viewer), unchanged.

## Install

Requirements: Linux x86_64 or Apple Silicon macOS, [uv](https://docs.astral.sh/uv/),
FFmpeg, and a Chromium-family browser (Chromium, Chrome, Brave, Edge, Vivaldi; 116 or later).

```sh
git clone https://github.com/leovicenzi/read-aloud-chromium
cd read-aloud-chromium
./scripts/install.sh
```

`install.sh` runs `scripts/setup-tts.sh` the first time. That one-time setup downloads
the model, the voices and the language data (several GB on Linux, where it includes the CUDA
build of PyTorch; set `TTS_TORCH_INDEX=cpu` for a much smaller CPU-only install). It then registers the
native host with every Chromium-family browser it finds.

Then load the extension:

1. Open `chrome://extensions` and turn on **Developer mode**.
2. Click **Load unpacked** and pick the `extension` folder.

If you move the folder, run `./scripts/install.sh` again.

## Use

- **Toolbar button** or **Alt+Shift+R**: reads your selection if there is one, otherwise
  the page's article. While reading, the same button pauses and resumes.
- **Right-click**: "Read selection aloud" or "Read this page aloud".

While a reading session is open, a status bar sits at the bottom of the page and these keys
belong to it (typing in a text field is not affected):

| Key | Action |
| --- | --- |
| `Space` | Pause / resume |
| `←` / `→` | Previous / next phrase (`←` more than 2 s into a phrase restarts it) |
| `↑` / `↓` | Previous / next paragraph |
| `Enter` | Read from the selected text, or from the top of the visible page |
| `=` or `+` / `-` | Faster / slower (0.5×–2.5×, remembered) |
| `F` | Toggle scrolling to follow the reading |
| `Esc` | Stop and close |

## How it works

```
page ── content script ── service worker ── Native Messaging ── host/native_host.py ── host/read_aloud.py
```

- **Content script** (`extension/content.js`) finds the article (`<article>`/`<main>`, or
  the element whose paragraphs hold the most text), skipping navigation, sidebars and
  footers. It splits the text into sentences of up to 240 characters and detects each
  paragraph's language with Chromium's built-in detector. Short paragraphs take a
  neighbour's language, and code blocks are read in English. Highlighting uses the
  CSS Custom Highlight API, so the page's HTML is never modified.
- **Service worker** (`extension/background.js`) injects the reader only when you ask
  (`activeTab`; no access to pages otherwise) and relays commands to the native host. One
  tab reads at a time.
- **Native host** (`host/native_host.py`) converts Chromium's length-prefixed messages
  to the helper's JSON lines. The helper loads the model once, synthesizes two phrases
  ahead, and plays through PortAudio. Its log is `~/.cache/read-aloud-chromium/helper.log`.

### Memory

Each loaded helper uses about 2 GB of RAM, plus about 0.6–0.9 GB of GPU memory on CUDA.
When nothing has been loading or playing for **5 minutes**, the extension disconnects and
Chromium stops the helper, freeing all of it. A paused session survives this: resuming
starts a new helper from the current phrase, after a few seconds to reload the model.
Closing or leaving the tab stops the reading at once.

## Limitations

- Chromium's built-in PDF viewer, `chrome://` pages and the Web Store do not allow
  extensions to read them; the toolbar icon shows `×` there.
- Article detection is heuristic. On pages where it picks the wrong part, select the text
  you want and read the selection.
- Text that the page replaces while it is being read can no longer be highlighted.

## Development

```sh
node --test tests/                              # extension logic (no browser)
cd host && python3 -m unittest test_native_host # host framing and lifecycle (no model)
cd host && ../.venv-tts/bin/python -m unittest test_read_aloud  # the helper's own tests
```

`host/read_aloud.py` and `host/test_read_aloud.py` are copies of `scripts/` in
herdr-file-viewer (commit `4a94323`). Change them there first, then copy them over.

The `key` in `extension/manifest.json` pins the unpacked extension's id
(`joemjhblemclfnkalfhmfnpopadodhmn`), which the native host manifest allows.

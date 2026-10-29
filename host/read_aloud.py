#!/usr/bin/env python3
"""Local Kokoro helper. stdin/stdout: JSON lines; stdout is protocol only.

MLX runs inference on Apple Silicon; elsewhere PyTorch does, on CUDA when present.
Audio and input text stay in memory. A single synthesis thread owns the model, keeps it
warm, and prefetches at most two phrases. The audio callback never synthesizes
or launches processes. Progress uses PortAudio's output DAC clock, not a text timer.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from collections import OrderedDict, deque

PROTOCOL_OUT = sys.stdout
MODEL = "mlx-community/Kokoro-82M-bf16"
TORCH_MODEL = "hexgrad/Kokoro-82M"
# Protocol language -> (Kokoro lang_code, voice). en uses Misaki; the others use espeak-ng.
LANGUAGES = {"en": ("a", "af_heart"), "pt": ("p", "pf_dora"), "es": ("e", "ef_dora"),
             "fr": ("f", "ff_siwis"), "it": ("i", "if_sara")}
SAMPLE_RATE = 24000
MAX_REQUEST = 2 * 1024 * 1024
MAX_TEXT_BYTES = 256 * 1024
MAX_CACHE_BYTES = 16 * 1024 * 1024
MAX_CACHE_PHRASES = 16
# MLX otherwise allows each helper to retain nearly a machine's worth of GPU buffers.
MLX_CACHE_BYTES = 64 * 1024 * 1024
# PyTorch's fixed-size CUDA segments fragment across phrase lengths; growable ones reuse space.
CUDA_ALLOC_CONF = "expandable_segments:True"
# Buffers queued ahead of the DAC: ~1 on CoreAudio, 2+ on ALSA/PipeWire at 20 ms blocks.
MAX_QUEUED_BUFFERS = 32


def emit(event, ident, **fields):
    print(json.dumps(dict(event=event, id=ident, **fields)), file=PROTOCOL_OUT, flush=True)


def speed_value(value):
    value = float(value)
    return min(2.5, max(0.5, value)) if math.isfinite(value) else 1.0


def word_marks(text, tokens):
    """Map Kokoro token timestamps to original UTF-8 bytes, conservatively.

    Never invent coordinates when the phonemizer normalizes a token differently.
    Those passages retain the phrase highlight instead of guessing a word.
    """
    marks, cursor = [], 0
    for token in tokens or []:
        value = token.text
        start, end = token.start_ts, token.end_ts
        if not value or start is None or end is None:
            continue
        at = text.find(value, cursor)
        if at < 0:
            continue
        cursor = at + len(value)
        marks.append((float(start), float(end), len(text[:at].encode()),
                      len(text[:cursor].encode())))
    return marks


def stretch(audio, speed, np):
    """Pitch-preserving time stretch, bounded to one short phrase, no temp files."""
    if abs(speed - 1.0) < 0.001 or len(audio) == 0:
        return audio
    factors = [2.0, speed / 2.0] if speed > 2.0 else [speed]
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
         "-af", ",".join(f"atempo={factor}" for factor in factors),
         "-f", "f32le", "pipe:1"],
        input=audio.astype("<f4", copy=False).tobytes(), capture_output=True,
        timeout=5, check=True, cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


@dataclass
class Generated:
    ident: int
    index: int
    audio: object = None
    marks: object = None
    error: str | None = None
    done: bool = False


class AudioCache:
    """Document-scoped LRU of original-speed PCM + alignment, never disk-backed.

    Reset is called by the command thread; synthesis owns gets/puts. A canceled
    old-document generation cannot reinsert its audio after reset.
    """
    def __init__(self, max_bytes=MAX_CACHE_BYTES, max_phrases=MAX_CACHE_PHRASES):
        self.lock = threading.Lock()
        self.document = None
        self.entries = OrderedDict()
        self.bytes = 0
        self.max_bytes, self.max_phrases = max_bytes, max_phrases

    def reset(self, document):
        with self.lock:
            if document is None or document != self.document:
                self.document = document
                self.entries.clear()
                self.bytes = 0

    def get(self, document, index, text):
        with self.lock:
            if document != self.document:
                return None
            entry = self.entries.get(index)
            if entry is None or entry[0] != text:
                return None
            self.entries.move_to_end(index)
            return entry[1], entry[2]

    def put(self, document, index, text, audio, marks):
        with self.lock:
            if document != self.document or document is None:
                return
            old = self.entries.pop(index, None)
            if old is not None:
                self.bytes -= old[1].nbytes
            if audio.nbytes > self.max_bytes or self.max_phrases < 1:
                return
            self.entries[index] = (text, audio, marks)
            self.bytes += audio.nbytes
            while self.bytes > self.max_bytes or len(self.entries) > self.max_phrases:
                _, old = self.entries.popitem(last=False)
                self.bytes -= old[1].nbytes


class Control:
    def __init__(self):
        self.ident = 0
        self.latest_id = 0
        self.cache = AudioCache()
        self.paused = False
        self.speed = 1.0
        self.closed = threading.Event()
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=2)
        self.commands = queue.Queue()

    def submit(self, job):
        # Only the newest pending seek matters; never accumulate full text requests.
        while True:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break
        self.jobs.put_nowait(job)

    def publish(self, result):
        while not self.closed.is_set() and self.ident == result.ident:
            try:
                self.results.put(result, timeout=0.05)
                return
            except queue.Full:
                pass


def read_commands(control):
    try:
        while not control.closed.is_set():
            line = sys.stdin.buffer.readline(MAX_REQUEST + 1)
            if not line or len(line) > MAX_REQUEST:
                break
            try:
                command = json.loads(line)
                if isinstance(command, dict):
                    control.commands.put(command)
            except (ValueError, UnicodeDecodeError):
                continue
    finally:
        control.closed.set()


def use_mlx():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def torch_device(torch):
    """HERDR_TTS_DEVICE wins; otherwise CUDA when present, else CPU."""
    requested = os.environ.get("HERDR_TTS_DEVICE", "").strip()
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# espeak-ng reads the separator in 3.5 / 3,5 as punctuation (a sentence pause) and 1.000
# digit by digit, so numbers are spelled out for it. Misaki already reads English numbers.
NUMBER_WORDS = {"pt": ("ponto", "vírgula"), "es": ("punto", "coma"),
                "fr": ("point", "virgule"), "it": ("punto", "virgola")}
GROUPED = re.compile(r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b")   # 1.000 / 1.000.000 / 1.000,50
SEPARATED = re.compile(r"\b\d+(?:[.,]\d+)+\b")               # 3.5 / 3,5 / 1.18.0


def speakable_numbers(lang, text):
    """Synthesis-only rewrite; offsets reported to the viewer still index the original text."""
    if lang not in NUMBER_WORDS:
        return text
    point, comma = NUMBER_WORDS[lang]
    text = GROUPED.sub(lambda m: m.group().replace(".", "").replace(",", f" {comma} "), text)
    return SEPARATED.sub(lambda m: m.group().replace(".", f" {point} ")
                         .replace(",", f" {comma} "), text)


def voice_for(lang):
    if lang == "en":
        return os.environ.get("HERDR_TTS_VOICE", "af_heart")
    return LANGUAGES[lang][1]


def load_model():
    """One model for every language; the per-language pipelines only add a phonemizer."""
    if use_mlx():
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model as load_mlx
        mx.set_cache_limit(MLX_CACHE_BYTES)
        return load_mlx(MODEL)
    # Read once when CUDA initializes; an explicit user setting wins.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", CUDA_ALLOC_CONF)
    import torch
    from kokoro import KModel
    return KModel(repo_id=TORCH_MODEL).to(torch_device(torch)).eval()


def load_pipeline(lang, model):
    lang_code = LANGUAGES[lang][0]
    if lang == "en":
        # Provisioning must remain explicit, never an implicit Misaki download.
        from importlib.metadata import version, PackageNotFoundError
        try:
            version("en-core-web-sm")
        except PackageNotFoundError:
            raise RuntimeError("English language data is missing; run scripts/setup-tts.sh")
    if use_mlx():
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline
        return KokoroPipeline(lang_code=lang_code, model=model, repo_id=MODEL)
    # Same misaki-based pipeline (graphemes + token timestamps), on PyTorch.
    from kokoro import KPipeline
    return KPipeline(lang_code=lang_code, repo_id=TORCH_MODEL, model=model)


class Voices:
    """Lazily built pipelines sharing one warm model; owned by the synthesis thread."""
    def __init__(self):
        self.model = None
        self.pipelines = {}

    def pipeline(self, lang):
        if self.model is None:
            self.model = load_model()
        if lang not in self.pipelines:
            self.pipelines[lang] = load_pipeline(lang, self.model)
        return self.pipelines[lang]

    def release_cache(self):
        """Drop unused GPU buffers, not the warm model or live pipeline arrays."""
        if use_mlx() and self.model is not None:
            import mlx.core as mx
            mx.clear_cache()
        elif self.model is not None and "torch" in sys.modules:
            torch = sys.modules["torch"]
            # is_initialized: a CPU helper must not create a CUDA context just to free nothing.
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def duration_marks(text, phonemes, durations, vocab, g2p, lang="en"):
    """Word timing for espeak languages, which Kokoro returns without token timestamps.

    Kokoro predicts one duration (in 1/40 s frames) per phoneme it kept, plus <bos>/<eos>.
    Phoneme words are space-separated; each text word is phonemized alone to learn how many
    phoneme words it became ("1990" becomes several). Any disagreement keeps only the phrase
    highlight, never a guessed word.
    """
    if durations is None or not phonemes or not vocab:
        return []
    durations = [int(d) for d in durations]
    kept = [p for p in phonemes if p in vocab]
    if len(durations) != len(kept) + 2:
        return []
    groups, current = [], None
    # Same leading-silence convention as Kokoro's English timestamps.
    frame = max(0, durations[0] - 3)
    for p, d in zip(kept, durations[1:]):
        if p.isspace():
            if current:
                groups.append(current)
            current = None
        else:
            if current is None:
                current = [frame, frame]
            current[1] = frame + d
        frame += d
    if current:
        groups.append(current)
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    counts = [len((g2p(speakable_numbers(lang, text[s:e]))[0] or "").split()) for s, e in words]
    if sum(counts) != len(groups):
        return []
    marks, group = [], 0
    for (s, e), n in zip(words, counts):
        if n:
            marks.append((groups[group][0] / 40, groups[group + n - 1][1] / 40,
                          len(text[:s].encode()), len(text[:e].encode())))
        group += n
    return marks


def result_marks(text, result, pipeline, lang="en"):
    if result.tokens:
        return word_marks(text, result.tokens)
    output = getattr(result, "output", None)
    durations = getattr(output, "pred_dur", None)
    vocab = getattr(getattr(pipeline, "model", None), "vocab", None)
    g2p = getattr(pipeline, "g2p", None)
    if g2p is None:
        return []
    return duration_marks(text, result.phonemes, durations, vocab, g2p, lang)


def generate_job(control, job, voices, np):
    ident, document, offset = job["id"], job["document"], job["offset"]
    for index, chunk in enumerate(job["chunks"]):
        if ident != control.ident or control.closed.is_set():
            break
        text, lang = chunk["text"], chunk.get("lang", "en")
        key = (lang, text)  # the same text in another language is different audio
        cached = control.cache.get(document, offset + index, key)
        if cached is not None:
            audio, marks = cached
        else:
            audios, marks, seconds, cursor = [], [], 0.0, 0
            pipeline = voices.pipeline(lang)
            spoken = speakable_numbers(lang, text)
            for result in pipeline(spoken, voice=voice_for(lang), speed=1.0):
                if ident != control.ident or control.closed.is_set():
                    break
                audio = result.audio
                if hasattr(audio, "cpu"):  # PyTorch tensor; MLX arrays convert directly
                    audio = audio.detach().cpu()
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                segment = result.graphemes
                if spoken != text:
                    # Rewritten numbers: marks map back to the original words only when one
                    # result covers the whole phrase, as it does for these bounded phrases.
                    if segment.strip() == spoken.strip():
                        marks.extend(result_marks(text, result, pipeline, lang))
                    audios.append(audio)
                    seconds += len(audio) / SAMPLE_RATE
                    continue
                at = text.find(segment, cursor)
                if at >= 0:
                    byte_offset = len(text[:at].encode())
                    marks.extend((a+seconds, b+seconds, s+byte_offset, e+byte_offset)
                                 for a,b,s,e in result_marks(segment, result, pipeline, lang))
                    cursor = at + len(segment)
                audios.append(audio)
                seconds += len(audio) / SAMPLE_RATE
            # Never cache a partial phrase when generation was superseded mid-result.
            if ident != control.ident or control.closed.is_set():
                break
            if not audios:
                continue
            audio = np.concatenate(audios)
            control.cache.put(document, offset + index, key, audio, marks)
        control.publish(Generated(ident, index, audio, marks))
    if ident == control.ident:
        control.publish(Generated(ident, -1, done=True))


def synthesize(control):
    voices = Voices()
    while not control.closed.is_set():
        try:
            job = control.jobs.get(timeout=0.1)
        except queue.Empty:
            continue
        ident = job["id"]
        if ident != control.ident:
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                import numpy as np
                generate_job(control, job, voices, np)
        except Exception as exc:
            control.publish(Generated(ident, -1, error=f"{type(exc).__name__}: {exc}"))
        finally:
            # Also release scratch buffers after canceled or failed synthesis.
            voices.release_cache()


class Player:
    """Short callback buffers + DAC-clock cursor. Expensive work runs in main()."""
    def __init__(self, np):
        self.np = np
        self.lock = threading.Lock()
        self.audio = np.empty(0, dtype=np.float32)
        self.position = 0
        self.speed = 1.0
        self.origin = 0.0
        self.elapsed_origin = 0.0
        self.paused = False
        self.clocks = deque(maxlen=MAX_QUEUED_BUFFERS)
        self.finished_at = None

    def callback(self, out, frames, timing, status):
        out.fill(0)
        with self.lock:
            if self.paused or self.position >= len(self.audio):
                return
            count = min(frames, len(self.audio) - self.position)
            out[:count, 0] = self.audio[self.position:self.position+count]
            self.clocks.append((timing.outputBufferDacTime,
                          self.origin + self.position / SAMPLE_RATE * self.speed,
                          self.speed, count / SAMPLE_RATE,
                          self.elapsed_origin + self.position / SAMPLE_RATE))
            self.position += count
            if self.position >= len(self.audio):
                self.finished_at = timing.outputBufferDacTime + count / SAMPLE_RATE

    def load(self, audio, speed, origin=0.0, elapsed_origin=0.0):
        with self.lock:
            self.audio = audio
            self.position = 0
            self.speed = speed
            self.origin = origin
            self.elapsed_origin = elapsed_origin
            self.clocks.clear()
            self.finished_at = None

    def clear(self):
        self.load(self.np.empty(0, dtype=self.np.float32), 1.0)

    def _audible(self, now):
        """Newest buffer already at the DAC; later ones are still queued ahead of it."""
        for clock in reversed(self.clocks):
            if clock[0] <= now:
                return clock
        return None

    def time_in_phrase(self, now):
        with self.lock:
            clock = self._audible(now)
            if clock is None:
                return None
            dac, start, speed, duration, _ = clock
            return start + min(now - dac, duration) * speed

    def elapsed_in_phrase(self, now):
        with self.lock:
            clock = self._audible(now)
            if clock is None:
                return None
            dac, _, _, duration, elapsed = clock
            return elapsed + min(now - dac, duration)

    def next_elapsed_time(self):
        with self.lock:
            return self.elapsed_origin + self.position / SAMPLE_RATE

    def next_source_time(self):
        with self.lock:
            return self.origin + self.position / SAMPLE_RATE * self.speed

    def done(self, now):
        with self.lock:
            return len(self.audio) == 0 or (self.finished_at is not None and now >= self.finished_at)


def drain(items):
    while True:
        try:
            items.get_nowait()
        except queue.Empty:
            return


def apply_command(control, command, player=None):
    """Apply seek/pause atomically before any new phrase can enter the DAC queue."""
    cmd, ident = command.get("cmd"), command.get("id")
    if type(ident) is not int or ident < 1:
        return None
    if cmd == "speak":
        chunks = command.get("chunks")
        document, offset = command.get("document", ident), command.get("offset", 0)
        paused = command.get("paused", False)
        valid = (isinstance(chunks, list) and 0 < len(chunks) <= 2048
                 and type(document) is int and document > 0
                 and type(offset) is int and offset >= 0 and type(paused) is bool)
        if valid:
            valid = all(isinstance(c, dict) and isinstance(c.get("text"), str)
                        and 0 < len(c["text"]) <= 1024
                        and c.get("lang", "en") in LANGUAGES for c in chunks)
        if valid:
            valid = sum(len(c["text"].encode()) for c in chunks) <= MAX_TEXT_BYTES
        try:
            speed = speed_value(command.get("speed", 1.0))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            emit("error", ident, message="Invalid or oversized read request")
            return None
        control.ident = control.latest_id = ident
        control.cache.reset(document)
        control.speed, control.paused = speed, paused
        if player:
            player.clear()
        drain(control.results)
        control.submit(dict(command, document=document, offset=offset))
    elif ident == control.latest_id:
        if cmd == "stop":
            control.ident = 0
            control.paused = False
            control.cache.reset(None)
            drain(control.jobs)
            drain(control.results)
            if player:
                player.clear()
        elif cmd == "pause":
            control.paused = bool(command.get("paused"))
        elif cmd == "speed":
            try:
                control.speed = speed_value(command.get("speed", 1.0))
            except (TypeError, ValueError):
                return None
        else:
            return None
    else:
        return None
    return cmd


def main():
    control = Control()
    threading.Thread(target=read_commands, args=(control,), daemon=True).start()
    threading.Thread(target=synthesize, args=(control,), daemon=True).start()
    player = stream = current = None
    last_mark = None
    pending = None
    try:
        while not control.closed.is_set():
            while True:
                try:
                    command = control.commands.get_nowait()
                except queue.Empty:
                    break
                action = apply_command(control, command, player)
                if action in ("speak", "stop"):
                    current = pending = last_mark = None
            if player:
                with player.lock:
                    player.paused = control.paused
            if current and not control.paused:
                if abs(player.speed - control.speed) > 0.001:
                    # Stretch only the not-yet-queued suffix; the model stays warm
                    # and speed changes do not require regenerating speech.
                    origin = player.next_source_time()
                    offset = min(len(current.audio), round(origin * SAMPLE_RATE))
                    elapsed = player.next_elapsed_time()
                    player.load(stretch(current.audio[offset:], control.speed, np), control.speed, origin, elapsed)
                now = stream.time
                point = player.time_in_phrase(now)
                if point is not None:
                    mark = next(((s,e) for a,b,s,e in current.marks if a <= point < b), None)
                    elapsed = player.elapsed_in_phrase(now)
                    key = (current.index, mark, elapsed is not None and elapsed >= 2.0)
                    if key != last_mark:
                        fields = dict(chunk=current.index, elapsed=elapsed)
                        if mark is not None:
                            fields.update(start=mark[0], end=mark[1])
                        emit("progress", current.ident, **fields)
                        last_mark = key
                if player.done(now):
                    current = None
            if current is None and not control.paused:
                if pending is None:
                    try:
                        pending = control.results.get_nowait()
                    except queue.Empty:
                        pass
                if pending is not None:
                    result, pending = pending, None
                    if result.ident == control.ident:
                        if result.error:
                            emit("error", result.ident, message=result.error)
                            control.ident = 0
                        elif result.done:
                            emit("done", result.ident)
                            control.ident = 0
                        else:
                            if player is None:
                                import numpy as np
                                import sounddevice as sd
                                player = Player(np)
                                stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                                    dtype="float32", blocksize=480, latency="low", callback=player.callback)
                                stream.start()
                            player.load(stretch(result.audio, control.speed, np), control.speed)
                            current = result
                            last_mark = None
            time.sleep(0.01)
    except Exception as exc:
        emit("error", control.ident or 0, message=f"{type(exc).__name__}: {exc}")
    finally:
        control.closed.set()
        if stream:
            stream.abort()
            stream.close()


if __name__ == "__main__":
    # Nothing is downloaded automatically at playback: setup-tts.sh provisions the
    # model, voices, and language data once. Offline documents never go to a server.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

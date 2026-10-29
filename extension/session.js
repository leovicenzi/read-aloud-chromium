// Reading session state, ported from ReadAloud in herdr-file-viewer's src/speech.rs.
// No DOM or chrome.* access: commands go to an injected `send`, time to an injected `now`.
(function (root) {
  const MIN_SPEED = 0.5;
  const MAX_SPEED = 2.5;
  // A helper holds its model in RAM (and VRAM on CUDA) while it lives.
  const HELPER_IDLE_MS = 5 * 60 * 1000;
  // The helper's per-request limits.
  const MAX_CHUNKS = 2048;
  const MAX_BYTES = 256 * 1024;
  const encoder = new TextEncoder();

  function clampSpeed(speed) {
    return Number.isFinite(speed) ? Math.min(MAX_SPEED, Math.max(MIN_SPEED, speed)) : 1;
  }

  // The helper reports word positions as UTF-8 byte offsets; pages index UTF-16 units.
  function byteToIndex(text, byte) {
    let bytes = 0;
    for (let i = 0; i < text.length; i++) {
      if (bytes >= byte) return i;
      const code = text.codePointAt(i);
      bytes += code < 0x80 ? 1 : code < 0x800 ? 2 : code < 0x10000 ? 3 : 4;
      if (code > 0xffff) i++;
    }
    return text.length;
  }

  class Session {
    // `base` separates this tab's ids from other tabs' sessions on the shared helper.
    constructor(send, { now = () => Date.now(), base = 1, speed = 1 } = {}) {
      this.sendCommand = send;
      this.now = now;
      this.id = base;
      this.speed = clampSpeed(speed);
      this.phase = "idle"; // idle | loading | speaking | paused | failed
      this.chunks = []; // [{text, lang, block}]
      this.current = 0;
      this.windowBase = 0;
      this.windowEnd = 0;
      this.elapsed = 0;
      this.document = 0;
      this.highlight = null; // {chunk, word: [start, end] | null} in UTF-16 units
      this.error = null;
      this.follow = true;
      this.busyAt = now();
      this.released = true; // no helper holds this session: only a speak can reach one
    }

    inSession() {
      return this.chunks.length > 0 || this.phase === "failed";
    }

    active() {
      return ["loading", "speaking", "paused"].includes(this.phase);
    }

    send(command) {
      // A fresh helper has never seen this id; starting one to pause or stop is waste.
      if (this.released && command.cmd !== "speak") return;
      this.busyAt = this.now();
      try {
        this.sendCommand(command);
        this.released = false;
      } catch (error) {
        this.phase = "failed";
        this.highlight = null;
        this.error = String((error && error.message) || error);
      }
    }

    start(chunks, at = 0) {
      const paused = this.phase === "paused";
      this.stop();
      if (at >= chunks.length) return;
      this.chunks = chunks;
      this.document = this.id;
      this.playFrom(at, paused);
    }

    playFrom(at, paused) {
      this.id++; // a seek invalidates every queued event from the old position
      this.current = at;
      this.windowBase = at;
      this.windowEnd = at;
      this.elapsed = 0;
      let bytes = 0;
      for (let i = at; i < this.chunks.length && i - at < MAX_CHUNKS; i++) {
        const size = encoder.encode(this.chunks[i].text).length;
        if (bytes + size > MAX_BYTES) break;
        bytes += size;
        this.windowEnd++;
      }
      this.highlight = { chunk: at, word: null };
      this.error = null;
      this.phase = paused ? "paused" : "loading";
      this.send({
        cmd: "speak",
        id: this.id,
        document: this.document,
        offset: at,
        paused,
        chunks: this.chunks.slice(at, this.windowEnd).map((c) => ({ text: c.text, lang: c.lang })),
        speed: this.speed,
      });
    }

    seek(at) {
      if (at < 0 || at >= this.chunks.length) return false;
      this.playFrom(at, this.phase === "paused");
      return true;
    }

    nextPhrase() {
      return this.seek(this.current + 1);
    }

    previousPhrase() {
      // Two seconds into a phrase, "previous" restarts it, like a media player.
      return this.seek(this.elapsed >= 2 ? this.current : Math.max(0, this.current - 1));
    }

    nextParagraph() {
      const here = this.chunks[this.current];
      if (!here) return false;
      for (let i = this.current + 1; i < this.chunks.length; i++) {
        if (this.chunks[i].block !== here.block) return this.seek(i);
      }
      return false;
    }

    previousParagraph() {
      const here = this.chunks[this.current];
      if (!here) return false;
      let i = this.current - 1;
      while (i >= 0 && this.chunks[i].block === here.block) i--;
      if (i < 0) return this.seek(0);
      const block = this.chunks[i].block;
      while (i > 0 && this.chunks[i - 1].block === block) i--;
      return this.seek(i);
    }

    toggleFollow() {
      this.follow = !this.follow;
    }

    togglePause() {
      if (!this.active()) return;
      const paused = this.phase !== "paused";
      if (!paused && this.released) {
        // The idle helper was stopped while paused: resume from the current phrase.
        this.playFrom(this.current, false);
        return;
      }
      this.phase = paused ? "paused" : "loading";
      this.send({ cmd: "pause", id: this.id, paused });
    }

    stop() {
      if (this.active() || this.chunks.length) this.send({ cmd: "stop", id: this.id });
      this.reset();
    }

    // Another tab took the helper: forget this session without addressing the helper.
    abandon() {
      this.released = true;
      this.reset();
    }

    reset() {
      this.id++; // invalidates queued events from the old session
      this.phase = "idle";
      this.highlight = null;
      this.error = null;
      this.chunks = [];
      this.document = 0;
      this.current = 0;
      this.elapsed = 0;
      this.windowBase = 0;
      this.windowEnd = 0;
    }

    adjustSpeed(delta) {
      this.speed = clampSpeed(Math.round((this.speed + delta) * 10) / 10);
      if (this.active()) this.send({ cmd: "speed", id: this.id, speed: this.speed });
    }

    // Applies one helper event; true when what is shown changed.
    onEvent(event) {
      if (event.event === "released") {
        this.released = true;
        return false;
      }
      if (!this.active() || (event.id !== this.id && event.id !== 0)) return false;
      if (event.event === "progress" && event.id === this.id) {
        const at = this.windowBase + event.chunk;
        if (!Number.isInteger(event.chunk) || at >= this.windowEnd) return false;
        const text = this.chunks[at].text;
        this.current = at;
        this.elapsed = Number.isFinite(event.elapsed) && event.elapsed >= 0 ? event.elapsed : 0;
        let word = null;
        if (Number.isInteger(event.start) && Number.isInteger(event.end)) {
          const start = byteToIndex(text, event.start);
          const end = byteToIndex(text, event.end);
          if (start < end) word = [start, end];
        }
        this.highlight = { chunk: at, word };
        if (this.phase !== "paused") this.phase = "speaking";
        return true;
      }
      if (event.event === "done") {
        if (this.windowEnd < this.chunks.length) {
          this.playFrom(this.windowEnd, this.phase === "paused");
        } else {
          this.phase = "idle";
          this.highlight = null;
          // Keep the position so ← can revisit finished audio.
          this.current = this.chunks.length - 1;
        }
        return true;
      }
      if (event.event === "error") {
        this.phase = "failed";
        this.highlight = null;
        this.error = event.message || "Read-aloud helper exited";
        return true;
      }
      return false;
    }

    // True once the helper has been neither loading nor playing for HELPER_IDLE_MS: the
    // caller then releases it. A paused or finished session survives; the next speak
    // starts a fresh helper.
    releaseIfIdle(now) {
      if (this.phase === "loading" || this.phase === "speaking") {
        this.busyAt = now;
        return false;
      }
      if (this.released || now - this.busyAt < HELPER_IDLE_MS) return false;
      this.released = true;
      return true;
    }
  }

  const api = { Session, HELPER_IDLE_MS, MIN_SPEED, MAX_SPEED, clampSpeed, byteToIndex };
  root.ReadAloudSession = api;
  if (typeof module === "object") module.exports = api;
})(globalThis);

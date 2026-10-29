// Injected on demand (toolbar, shortcut, or context menu). Finds the article or selection,
// splits it into phrases, and drives one reading Session through the service worker.
(function () {
  if (globalThis.__readAloud) return; // already injected: the listener below handles it
  globalThis.__readAloud = true;
  const { normalize, phrases, pickLanguage, pageLanguage, assignLanguages } = ReadAloudText;
  const { Session } = ReadAloudSession;

  const BLOCK = "p,h1,h2,h3,h4,h5,h6,li,blockquote,pre,figcaption,dt,dd,td,th,caption,summary";
  // Text grouped under the nearest of these when no BLOCK element holds it.
  const CONTAINER = `${BLOCK},div,section,article,main,body`;
  const NEVER = "script,style,noscript,template,svg,math,textarea,select,button,[aria-hidden=true]";
  // Page furniture that is not part of an article. A selection reads whatever it covers.
  const FURNITURE = "nav,aside,footer,[role=navigation],[role=complementary],.mw-editsection,sup.reference";
  const LABELS = { en: "EN", pt: "PT", es: "ES", fr: "FR", it: "IT" };

  // ── Extraction ───────────────────────────────────────────────────────────────────────

  function textLength(el) {
    return (el.innerText || "").replace(/\s+/g, " ").length;
  }

  function visible(el) {
    return el.checkVisibility ? el.checkVisibility() : el.offsetParent !== null;
  }

  // <article> or <main> when the page marks one up; otherwise the element whose
  // paragraphs hold the most text (parents get the full score, grandparents half).
  function articleRoot() {
    const marked = [...document.querySelectorAll("article, main, [role=main]")]
      .filter((el) => visible(el) && textLength(el) > 500)
      .sort((a, b) => textLength(b) - textLength(a))[0];
    if (marked) return marked;
    const scores = new Map();
    for (const p of document.querySelectorAll("p")) {
      const length = textLength(p);
      if (length < 40 || !visible(p) || p.closest(FURNITURE)) continue;
      const parent = p.parentElement;
      if (!parent) continue;
      scores.set(parent, (scores.get(parent) || 0) + length);
      if (parent.parentElement) {
        const grand = parent.parentElement;
        scores.set(grand, (scores.get(grand) || 0) + length / 2);
      }
    }
    let best = document.body;
    let top = 0;
    for (const [el, score] of scores) {
      if (score > top) [best, top] = [el, score];
    }
    return best;
  }

  function kindOf(el) {
    if (el.closest("pre")) return "code";
    return /^H[1-6]$/.test(el.tagName) ? "heading" : "prose";
  }

  // Blocks of page text in reading order, each a normalized string with a map back to the
  // text nodes. A range clips the text to a selection.
  function collectBlocks(root, range) {
    const skip = range ? NEVER : `${NEVER},${FURNITURE}`;
    const shown = new Map();
    const isShown = (el) => {
      if (!shown.has(el)) shown.set(el, !el.closest(skip) && visible(el));
      return shown.get(el);
    };
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const blocks = [];
    const byElement = new Map();
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const parent = node.parentElement;
      if (!parent || !isShown(parent)) continue;
      if (range && !range.intersectsNode(node)) continue;
      const start = range && node === range.startContainer ? range.startOffset : 0;
      const end = range && node === range.endContainer ? range.endOffset : node.data.length;
      if (start >= end) continue;
      let el = parent.closest(CONTAINER);
      if (!el || !root.contains(el)) el = root.nodeType === 1 ? root : parent;
      let block = byElement.get(el);
      if (!block) {
        block = { el, kind: kindOf(el), pieces: [] };
        byElement.set(el, block);
        blocks.push(block);
      }
      block.pieces.push({ node, start, text: node.data.slice(start, end) });
    }
    return blocks
      .map((b) => Object.assign(b, normalize(b.pieces)))
      .filter((b) => /[\p{L}\p{N}]/u.test(b.text));
  }

  async function detect(text) {
    try {
      return await chrome.i18n.detectLanguage(text.slice(0, 2000));
    } catch {
      return null;
    }
  }

  async function buildChunks(mode) {
    const selection = getSelection();
    const useSelection = mode !== "page" && selection && !selection.isCollapsed;
    let blocks = [];
    if (useSelection) {
      const range = selection.getRangeAt(0);
      let root = range.commonAncestorContainer;
      if (root.nodeType !== 1) root = root.parentElement;
      blocks = collectBlocks(root, range);
    }
    if (!blocks.length) blocks = collectBlocks(articleRoot(), null);
    const fallback = pageLanguage(document.documentElement.lang);
    const detected = await Promise.all(blocks.map(async (b) =>
      b.kind === "code" ? null : pickLanguage(await detect(b.text), b.text)));
    const langs = assignLanguages(
      blocks.map((b, i) => ({ kind: b.kind, detected: detected[i] })), fallback);
    const chunks = [];
    blocks.forEach((block, index) => {
      for (const [start, end] of phrases(block.text, langs[index])) {
        chunks.push({ text: block.text.slice(start, end), lang: langs[index], block: index, start, end });
      }
    });
    return { blocks, chunks };
  }

  // ── Page state ─────────────────────────────────────────────────────────────────────────

  let blocks = [];
  let port = null;
  let overlay = null;
  let lastScroll = 0;
  const session = new Session(send, {
    // Separate this tab's ids from other tabs sharing the helper.
    base: 1 + Math.floor(Math.random() * 1e9) * 1000,
  });
  chrome.storage.local.get("speed").then(({ speed }) => {
    if (speed) session.speed = ReadAloudSession.clampSpeed(speed);
  }).catch(() => {});

  function connect() {
    port = chrome.runtime.connect({ name: "read-aloud" });
    port.onMessage.addListener((event) => {
      if (event.event === "superseded") {
        session.abandon();
        render();
        return;
      }
      if (session.onEvent(event)) render();
    });
    // The service worker went away, and the helper with it.
    port.onDisconnect.addListener(() => {
      port = null;
      session.onEvent({ event: "released" });
    });
  }

  function send(command) {
    if (!port) connect();
    port.postMessage({ type: "command", command });
  }

  function rangeFor(block, start, end) {
    const [ps, os] = block.map[start];
    const [pe, oe] = block.map[end - 1];
    const range = document.createRange();
    range.setStart(block.pieces[ps].node, block.pieces[ps].start + os);
    range.setEnd(block.pieces[pe].node, block.pieces[pe].start + oe + 1);
    return range;
  }

  function chunkRange(index, word) {
    const chunk = session.chunks[index];
    const block = blocks[chunk.block];
    try {
      if (word) return rangeFor(block, chunk.start + word[0], chunk.start + word[1]);
      return rangeFor(block, chunk.start, chunk.end);
    } catch {
      return null; // the page removed this text since it was read
    }
  }

  function ensureStyle() {
    if (document.getElementById("read-aloud-style")) return;
    const style = document.createElement("style");
    style.id = "read-aloud-style";
    style.textContent = `
      ::highlight(read-aloud-phrase) { background-color: rgba(255, 213, 79, 0.35); }
      ::highlight(read-aloud-word) { background-color: #ffd54f; color: #111; }`;
    document.documentElement.append(style);
  }

  function paint() {
    CSS.highlights.delete("read-aloud-phrase");
    CSS.highlights.delete("read-aloud-word");
    const mark = session.highlight;
    if (!mark) return;
    ensureStyle();
    const phrase = chunkRange(mark.chunk, null);
    if (!phrase) return;
    CSS.highlights.set("read-aloud-phrase", new Highlight(phrase));
    const word = mark.word && chunkRange(mark.chunk, mark.word);
    if (word) CSS.highlights.set("read-aloud-word", new Highlight(word));
    if (session.follow && mark.chunk !== lastScroll) {
      lastScroll = mark.chunk;
      const rect = phrase.getBoundingClientRect();
      if (rect.top < 60 || rect.bottom > innerHeight - 80) {
        const el = phrase.startContainer.parentElement;
        if (el) el.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    }
  }

  function statusText() {
    const n = session.chunks.length;
    const at = `${Math.min(session.current + 1, n)}/${n}`;
    const chunk = session.chunks[session.current];
    const lang = chunk ? LABELS[chunk.lang] : "";
    const speed = `${session.speed.toFixed(1)}×`;
    const keys = `Space ${session.phase === "paused" ? "resume" : "pause"} · ←→ phrase · ↑↓ paragraph · −= speed · F follow ${session.follow ? "on" : "off"} · Esc stop`;
    switch (session.phase) {
      case "loading": return `Loading voice… ${at} · ${lang} · ${speed} — Esc stop`;
      case "speaking": return `Reading ${at} · ${lang} · ${speed} — ${keys}`;
      case "paused": return `Paused ${at} · ${lang} · ${speed} — ${keys}`;
      case "failed": return `${session.error} — Esc close`;
      default: return `Finished — ← go back · Esc close`;
    }
  }

  function showOverlay() {
    if (!overlay) {
      const host = document.createElement("read-aloud-status");
      const shadow = host.attachShadow({ mode: "closed" });
      shadow.innerHTML = `<style>
        div { position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%);
          z-index: 2147483647; max-width: calc(100vw - 32px); box-sizing: border-box;
          padding: 8px 14px; border-radius: 8px; background: rgba(20, 20, 20, 0.92);
          color: #f5f5f5; font: 13px/1.4 system-ui, sans-serif; box-shadow: 0 2px 12px rgba(0,0,0,.3);
          pointer-events: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        </style><div></div>`;
      overlay = { host, line: shadow.querySelector("div") };
      document.documentElement.append(host);
    }
    overlay.line.textContent = statusText();
  }

  function render() {
    paint();
    if (session.inSession()) {
      showOverlay();
      addEventListener("keydown", onKey, true);
    } else {
      overlay?.host.remove();
      overlay = null;
      removeEventListener("keydown", onKey, true);
      lastScroll = -1;
    }
  }

  // First phrase at or after a page point: the selection, else the top of the viewport.
  function chunkFromHere() {
    const selection = getSelection();
    for (let i = 0; i < session.chunks.length; i++) {
      const range = chunkRange(i, null);
      if (!range) continue;
      if (selection && !selection.isCollapsed) {
        if (range.compareBoundaryPoints(Range.END_TO_START, selection.getRangeAt(0)) <= 0 &&
            range.compareBoundaryPoints(Range.START_TO_END, selection.getRangeAt(0)) > 0) return i;
        continue;
      }
      if (range.getBoundingClientRect().top >= 0) return i;
    }
    return -1;
  }

  function editable(target) {
    return target instanceof Element &&
      (target.closest("input,textarea,select,[contenteditable=''],[contenteditable=true]") !== null);
  }

  // Reading mode: while a session exists these keys belong to it, like the viewer.
  function onKey(event) {
    if (event.ctrlKey || event.metaKey || event.altKey || editable(event.target)) return;
    const actions = {
      " ": () => session.togglePause(),
      ArrowRight: () => session.nextPhrase(),
      ArrowLeft: () => session.previousPhrase(),
      ArrowDown: () => session.nextParagraph(),
      ArrowUp: () => session.previousParagraph(),
      Escape: () => session.stop(),
      "=": () => changeSpeed(0.1),
      "+": () => changeSpeed(0.1),
      "-": () => changeSpeed(-0.1),
      f: () => session.toggleFollow(),
      F: () => session.toggleFollow(),
      Enter: () => {
        const at = chunkFromHere();
        if (at >= 0) session.seek(at);
      },
    };
    const action = actions[event.key];
    if (!action) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    action();
    render();
  }

  function changeSpeed(delta) {
    session.adjustSpeed(delta);
    chrome.storage.local.set({ speed: session.speed }).catch(() => {});
  }

  async function activate(mode) {
    // The toolbar button doubles as play/pause while a session is active.
    if (session.active() && mode === "auto") {
      session.togglePause();
      render();
      return;
    }
    const built = await buildChunks(mode);
    if (!built.chunks.length) {
      session.stop();
      session.phase = "failed";
      session.error = "No readable text found";
      render();
      return;
    }
    blocks = built.blocks;
    lastScroll = -1;
    session.start(built.chunks, 0);
    render();
  }

  // Release an idle helper (see Session.releaseIfIdle).
  setInterval(() => {
    if (session.releaseIfIdle(Date.now()) && port) port.postMessage({ type: "release" });
  }, 30 * 1000);

  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "activate") activate(message.mode);
  });
})();

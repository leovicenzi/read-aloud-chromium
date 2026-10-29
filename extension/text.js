// Pure text handling: whitespace normalization with a map back to page text, phrase
// splitting, and per-block language choice. No DOM access, so Node can test it.
(function (root) {
  const LANGS = ["en", "pt", "es", "fr", "it"];
  const MAX_PHRASE = 240; // matches the helper's phrase size in the herdr viewer
  const MIN_DETECT_WORDS = 6; // shorter blocks take a neighbour's language

  // pieces: [{text}] in reading order. Whitespace runs (including across pieces) become one
  // space; map[i] = [piece, offset] locates output unit i in the original text.
  function normalize(pieces) {
    let text = "";
    const map = [];
    let space = true; // drops leading whitespace
    pieces.forEach((piece, p) => {
      for (let i = 0; i < piece.text.length; i++) {
        if (/\s/.test(piece.text[i])) {
          if (!space) {
            text += " ";
            map.push([p, i]);
            space = true;
          }
        } else {
          text += piece.text[i];
          map.push([p, i]);
          space = false;
        }
      }
    });
    if (text.endsWith(" ")) {
      text = text.slice(0, -1);
      map.pop();
    }
    return { text, map };
  }

  function trim(text, a, b) {
    while (a < b && text[a] === " ") a++;
    while (b > a && text[b - 1] === " ") b--;
    return [a, b];
  }

  // Never cut between the halves of a surrogate pair.
  function safeCut(text, at) {
    const code = text.charCodeAt(at - 1);
    return code >= 0xd800 && code <= 0xdbff ? at - 1 : at;
  }

  // An oversized sentence prefers its last comma within the limit, then a word boundary.
  function splitLong(text, a, b, out) {
    while (b - a > MAX_PHRASE) {
      const window = text.slice(a, a + MAX_PHRASE);
      let cut = window.lastIndexOf(", ");
      if (cut > 0) cut += 1;
      else {
        cut = window.lastIndexOf(" ");
        if (cut <= 0) cut = safeCut(text, a + MAX_PHRASE) - a;
      }
      const [s, e] = trim(text, a, a + cut);
      if (s < e) out.push([s, e]);
      a = trim(text, a + cut, b)[0];
    }
    if (a < b) out.push([a, b]);
  }

  // Sentence ranges [start, end) in UTF-16 units. Intl.Segmenter keeps 3.5 and most
  // abbreviations whole.
  function phrases(text, lang) {
    const out = [];
    const segmenter = new Intl.Segmenter(lang || "en", { granularity: "sentence" });
    for (const s of segmenter.segment(text)) {
      const [a, b] = trim(text, s.index, s.index + s.segment.length);
      if (a < b) splitLong(text, a, b, out);
    }
    return out;
  }

  function words(text) {
    return (text.match(/\S+/g) || []).length;
  }

  // Chromium's detector result → an allowlisted language, or null when unsure.
  function pickLanguage(result, text) {
    if (!result || words(text) < MIN_DETECT_WORDS) return null;
    const top = [...(result.languages || [])].sort((a, b) => b.percentage - a.percentage)[0];
    if (!top) return null;
    const lang = top.language.toLowerCase().split(/[-_]/)[0];
    return LANGS.includes(lang) ? lang : null;
  }

  function pageLanguage(tag) {
    const lang = (tag || "").toLowerCase().split(/[-_]/)[0];
    return LANGS.includes(lang) ? lang : "en";
  }

  // blocks: [{kind: "prose"|"heading"|"code", detected: lang|null}]. Code is read as English;
  // an undetected heading takes the next block's language, anything else the previous one's.
  function assignLanguages(blocks, fallback) {
    const langs = blocks.map((b) => (b.kind === "code" ? "en" : b.detected));
    let previous = null;
    blocks.forEach((b, i) => {
      if (b.kind === "code") return;
      if (langs[i]) previous = langs[i];
      else if (b.kind !== "heading" && previous) langs[i] = previous;
    });
    let next = null;
    for (let i = blocks.length - 1; i >= 0; i--) {
      if (blocks[i].kind === "code") continue;
      if (langs[i]) next = langs[i];
      else if (next) langs[i] = next;
    }
    return langs.map((l) => l || fallback);
  }

  const api = { LANGS, MAX_PHRASE, MIN_DETECT_WORDS, normalize, phrases, pickLanguage,
    pageLanguage, assignLanguages };
  root.ReadAloudText = api;
  if (typeof module === "object") module.exports = api;
})(globalThis);

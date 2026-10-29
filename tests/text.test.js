const test = require("node:test");
const assert = require("node:assert/strict");
const T = require("../extension/text.js");

test("normalize collapses whitespace across pieces and maps back to source offsets", () => {
  const pieces = [{ text: "  Olá\n  " }, { text: "mundo" }, { text: " !  " }];
  const { text, map } = T.normalize(pieces);
  assert.equal(text, "Olá mundo !");
  assert.equal(map.length, text.length);
  assert.deepEqual(map[0], [0, 2]); // "O"
  assert.deepEqual(map[3], [0, 5]); // the run "\n  " became one space at its first unit
  assert.deepEqual(map[4], [1, 0]); // "m"
  assert.deepEqual(map[10], [2, 1]); // "!"
});

test("phrases split sentences but keep decimals and stay within the phrase limit", () => {
  const text = "The value is 3.5 today. It rose again! Was it 1.000?";
  const got = T.phrases(text, "en").map(([a, b]) => text.slice(a, b));
  assert.deepEqual(got, ["The value is 3.5 today.", "It rose again!", "Was it 1.000?"]);
  const long = `${"palavra ".repeat(20)}vírgula, ${"mais ".repeat(60)}fim.`;
  const parts = T.phrases(long, "pt").map(([a, b]) => long.slice(a, b));
  assert.ok(parts.every((p) => p.length <= T.MAX_PHRASE && p === p.trim()));
  assert.ok(parts[0].endsWith("vírgula,"), "an oversized sentence prefers its last comma");
  assert.equal(parts.join(" "), long);
});

test("an unbroken run is cut at the limit without splitting a surrogate pair", () => {
  const text = "a".repeat(T.MAX_PHRASE - 1) + "😀" + "b".repeat(10);
  const parts = T.phrases(text, "en");
  assert.equal(parts[0][1], T.MAX_PHRASE - 1);
  assert.equal(parts.map(([a, b]) => text.slice(a, b)).join(""), text);
});

test("detector results map onto the allowlist only when the block is long enough", () => {
  const six = "uma frase com seis palavras aqui";
  const result = { isReliable: true, languages: [{ language: "pt-BR", percentage: 97 }] };
  assert.equal(T.pickLanguage(result, six), "pt");
  assert.equal(T.pickLanguage(result, "só três palavras"), null);
  assert.equal(T.pickLanguage({ languages: [{ language: "gl", percentage: 90 }] }, six), null);
  assert.equal(T.pickLanguage(null, six), null);
  assert.equal(T.pageLanguage("es-MX"), "es");
  assert.equal(T.pageLanguage("de"), "en");
});

test("headings take the next block's language, others the previous; code is English", () => {
  const blocks = [
    { kind: "heading", detected: null },
    { kind: "prose", detected: "pt" },
    { kind: "code", detected: null },
    { kind: "prose", detected: null },
    { kind: "heading", detected: null },
    { kind: "prose", detected: "fr" },
  ];
  assert.deepEqual(T.assignLanguages(blocks, "en"), ["pt", "pt", "en", "pt", "fr", "fr"]);
  assert.deepEqual(T.assignLanguages([{ kind: "prose", detected: null }], "it"), ["it"]);
});

const test = require("node:test");
const assert = require("node:assert/strict");
const { Session, HELPER_IDLE_MS, byteToIndex } = require("../extension/session.js");

function make(chunks = [chunk("One.", 0), chunk("Two.", 0), chunk("Três.", 1), chunk("Four.", 2)]) {
  const sent = [];
  const session = new Session((c) => sent.push(c), { now: () => 0, base: 100 });
  session.start(chunks);
  return { session, sent, last: () => sent[sent.length - 1] };
}

function chunk(text, block, lang = "en") {
  return { text, block, lang };
}

test("start speaks the window with its language, and progress maps UTF-8 words to UTF-16", () => {
  const { session, last } = make();
  assert.deepEqual(last().chunks[2], { text: "Três.", lang: "en" });
  assert.equal(last().offset, 0);
  assert.equal(session.phase, "loading");
  const id = last().id;
  assert.equal(session.onEvent({ event: "progress", id, chunk: 2, start: 0, end: 5 }), true);
  assert.deepEqual(session.highlight, { chunk: 2, word: [0, 4] }); // "Três" is 5 bytes, 4 units
  assert.equal(session.phase, "speaking");
  assert.equal(session.onEvent({ event: "progress", id: id - 1, chunk: 0 }), false, "stale id");
  assert.equal(session.onEvent({ event: "progress", id, chunk: 9 }), false, "outside window");
});

test("byteToIndex handles astral characters", () => {
  assert.equal(byteToIndex("a😀b", 5), 3);
  assert.equal(byteToIndex("a😀b", 99), 4);
});

test("navigation seeks by phrase and paragraph and keeps pause", () => {
  const { session, last } = make();
  session.togglePause();
  assert.equal(last().cmd, "pause");
  assert.equal(session.nextParagraph(), true);
  assert.deepEqual([last().cmd, last().offset, last().paused], ["speak", 2, true]);
  session.nextPhrase();
  assert.equal(last().offset, 3);
  session.previousParagraph();
  assert.equal(last().offset, 2);
  session.previousParagraph();
  assert.equal(last().offset, 0, "back to the start of the previous block");
  session.elapsed = 3;
  session.previousPhrase();
  assert.equal(last().offset, 0, "restarts a phrase two seconds in");
  assert.equal(session.nextParagraph() && session.nextParagraph() && session.nextParagraph(), false);
});

test("done finishes the session but keeps it for revisiting; Esc-stop ends it", () => {
  const { session, sent, last } = make();
  session.onEvent({ event: "done", id: last().id });
  assert.equal(session.phase, "idle");
  assert.equal(session.inSession(), true);
  assert.equal(session.current, 3);
  session.previousPhrase();
  assert.deepEqual([last().cmd, last().offset, last().paused], ["speak", 2, false]);
  session.stop();
  assert.equal(last().cmd, "stop");
  assert.equal(session.inSession(), false);
  assert.ok(sent.every((c) => c.id > 100));
});

test("errors with id 0 come from the host and fail the session", () => {
  const { session } = make();
  assert.equal(session.onEvent({ event: "error", id: 0, message: "Host gone" }), true);
  assert.equal(session.phase, "failed");
  assert.equal(session.error, "Host gone");
});

test("the helper is released only after the idle period, and a paused session resumes", () => {
  const { session, sent, last } = make();
  assert.equal(session.releaseIfIdle(HELPER_IDLE_MS * 9), false, "loading is busy");
  session.togglePause();
  const pausedAt = session.busyAt;
  assert.equal(session.releaseIfIdle(pausedAt + HELPER_IDLE_MS - 1), false);
  assert.equal(session.releaseIfIdle(pausedAt + HELPER_IDLE_MS), true);
  assert.equal(session.releaseIfIdle(pausedAt + HELPER_IDLE_MS * 2), false, "once");
  const count = sent.length;
  session.adjustSpeed(0.2);
  assert.equal(sent.length, count, "no helper is started to change speed");
  session.togglePause();
  assert.deepEqual([last().cmd, last().offset, last().paused, last().speed], ["speak", 0, false, 1.2]);
  session.togglePause();
  assert.equal(session.releaseIfIdle(session.busyAt + HELPER_IDLE_MS), true);
  session.stop();
  assert.equal(sent.length, count + 2, "no helper is started to stop");
});

test("a released event (host gone) and abandon (another tab) make the next command a speak", () => {
  const { session, sent, last } = make();
  session.onEvent({ event: "released" });
  const count = sent.length;
  session.togglePause(); // pause: nothing to address
  assert.equal(sent.length, count);
  session.togglePause(); // resume: fresh speak
  assert.equal(last().cmd, "speak");
  session.abandon();
  assert.equal(session.inSession(), false);
  assert.equal(sent.length, count + 1, "abandon never sends a stop");
});

test("windows respect the helper's byte limit and continue on done", () => {
  const big = Array.from({ length: 1200 }, (_, i) => chunk("x".repeat(239) + ".", i));
  const { session, last } = make(big);
  assert.ok(last().chunks.length < 1200);
  const first = last().chunks.length;
  session.onEvent({ event: "done", id: last().id });
  assert.equal(last().cmd, "speak");
  assert.equal(last().offset, first);
});

test("speed is clamped and rounded", () => {
  const { session } = make();
  for (let i = 0; i < 40; i++) session.adjustSpeed(0.1);
  assert.equal(session.speed, 2.5);
  for (let i = 0; i < 40; i++) session.adjustSpeed(-0.1);
  assert.equal(session.speed, 0.5);
});

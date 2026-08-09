import { test } from "node:test"
import assert from "node:assert/strict"
import { segmentInline, segmentRich } from "./markdown.ts"

/**
 * The instrument rule reaches the chat: financial figures inside an answer
 * set in tabular mono, so -$55.62 reads as a readout, not prose.
 */

test("currency, percents and SOL amounts become num segments", () => {
  assert.deepEqual(segmentRich("PnL of -$55.62 and -35.3% over 47.25 SOL"), [
    { kind: "text", text: "PnL of " },
    { kind: "num", text: "-$55.62" },
    { kind: "text", text: " and " },
    { kind: "num", text: "-35.3%" },
    { kind: "text", text: " over " },
    { kind: "num", text: "47.25 SOL" },
  ])
})

test("bare counts are num segments, but numbers inside hyphenated words stay text", () => {
  const segs = segmentRich("across 460 labeled decisions in the 30-minute window")
  assert.deepEqual(segs[1], { kind: "num", text: "460" })
  assert.ok(segs.some((s) => s.kind === "text" && s.text.includes("30-minute")))
})

test("numbers inside strong spans are still set as numerals", () => {
  const segs = segmentRich("**-$77.27 across 9 closes**")
  assert.deepEqual(segs[0], { kind: "strong-num", text: "-$77.27" })
  assert.ok(segs.some((s) => s.kind === "strong" && s.text.includes("across")))
  assert.deepEqual(
    segs.filter((s) => s.kind === "strong-num").map((s) => s.text),
    ["-$77.27", "9"],
  )
})

test("code spans are never re-segmented", () => {
  assert.deepEqual(segmentRich("run `git log -3` now"), [
    { kind: "text", text: "run " },
    { kind: "code", text: "git log -3" },
    { kind: "text", text: " now" },
  ])
})

test("plain text passes through untouched", () => {
  assert.deepEqual(segmentInline("hello there"), [{ kind: "text", text: "hello there" }])
})

test("bold spans become strong segments with surrounding text intact", () => {
  assert.deepEqual(segmentInline("a **bold** b"), [
    { kind: "text", text: "a " },
    { kind: "strong", text: "bold" },
    { kind: "text", text: " b" },
  ])
})

test("inline code becomes a code segment", () => {
  assert.deepEqual(segmentInline("run `npm test` now"), [
    { kind: "text", text: "run " },
    { kind: "code", text: "npm test" },
    { kind: "text", text: " now" },
  ])
})

test("an unclosed marker stays literal, which keeps a mid-stream chunk honest", () => {
  assert.deepEqual(segmentInline("upper **bounds"), [{ kind: "text", text: "upper **bounds" }])
  assert.deepEqual(segmentInline("tick `incomplete"), [{ kind: "text", text: "tick `incomplete" }])
})

test("several spans in one line all resolve", () => {
  assert.deepEqual(segmentInline("**a** and **b**"), [
    { kind: "strong", text: "a" },
    { kind: "text", text: " and " },
    { kind: "strong", text: "b" },
  ])
})

test("empty input yields no segments", () => {
  assert.deepEqual(segmentInline(""), [])
})

import { test } from "node:test"
import assert from "node:assert/strict"
import { segmentInline } from "./markdown.ts"

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

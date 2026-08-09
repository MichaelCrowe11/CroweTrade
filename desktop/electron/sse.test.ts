import { test } from "node:test"
import assert from "node:assert/strict"
import { createSseParser } from "./sse.ts"

test("one complete event parses", () => {
  const p = createSseParser()
  const out = p.push('data: {"type":"response.output_text.delta","delta":"Hi"}\n\n')
  assert.equal(out.length, 1)
  assert.equal(out[0]?.type, "response.output_text.delta")
  assert.equal(out[0]?.["delta"], "Hi")
})

test("an event split mid-JSON across chunks arrives once, whole", () => {
  const p = createSseParser()
  const first = p.push('data: {"type":"response.completed","resp')
  assert.equal(first.length, 0)
  const second = p.push('onse":{"status":"completed"}}\n\n')
  assert.equal(second.length, 1)
  assert.equal(second[0]?.type, "response.completed")
})

test("several events in one chunk all parse, in order", () => {
  const p = createSseParser()
  const out = p.push(
    'data: {"type":"a"}\n\ndata: {"type":"b"}\n\ndata: {"type":"c"}\n\n',
  )
  assert.deepEqual(
    out.map((e) => e.type),
    ["a", "b", "c"],
  )
})

test("CRLF separators are handled, including a CRLF split across chunks", () => {
  const p = createSseParser()
  const first = p.push('data: {"type":"a"}\r')
  assert.equal(first.length, 0)
  const second = p.push('\n\r\ndata: {"type":"b"}\r\n\r\n')
  assert.deepEqual(
    second.map((e) => e.type),
    ["a", "b"],
  )
})

test("[DONE], comments, event: lines and blank data are ignored", () => {
  const p = createSseParser()
  const out = p.push(
    ': keep-alive\n\nevent: message\ndata: {"type":"a"}\n\ndata: [DONE]\n\ndata: \n\n',
  )
  assert.deepEqual(
    out.map((e) => e.type),
    ["a"],
  )
})

test("unparseable payloads and typeless objects are dropped, later events survive", () => {
  const p = createSseParser()
  const out = p.push('data: not json{\n\ndata: {"no":"type"}\n\ndata: {"type":"ok"}\n\n')
  assert.deepEqual(
    out.map((e) => e.type),
    ["ok"],
  )
})

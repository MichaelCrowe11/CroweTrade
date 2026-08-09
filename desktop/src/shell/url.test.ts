import { test } from "node:test"
import assert from "node:assert/strict"
import { normalizeUrl } from "./url.ts"

test("bare hostname gets https", () => {
  assert.equal(normalizeUrl("solscan.io"), "https://solscan.io/")
})

test("bare host with path gets https and keeps the path", () => {
  assert.equal(
    normalizeUrl("solscan.io/token/So11111111111111111111111111111111111111112"),
    "https://solscan.io/token/So11111111111111111111111111111111111111112",
  )
})

test("explicit https is preserved", () => {
  assert.equal(normalizeUrl("https://dexscreener.com/solana"), "https://dexscreener.com/solana")
})

test("explicit http is allowed", () => {
  assert.equal(normalizeUrl("http://example.com/"), "http://example.com/")
})

test("surrounding whitespace is trimmed", () => {
  assert.equal(normalizeUrl("  solscan.io  "), "https://solscan.io/")
})

test("javascript scheme is refused", () => {
  assert.equal(normalizeUrl("javascript:alert(1)"), null)
})

test("file scheme is refused", () => {
  assert.equal(normalizeUrl("file:///etc/passwd"), null)
})

test("empty input is refused", () => {
  assert.equal(normalizeUrl(""), null)
  assert.equal(normalizeUrl("   "), null)
})

test("unparseable input is refused", () => {
  assert.equal(normalizeUrl("not a url at all"), null)
})

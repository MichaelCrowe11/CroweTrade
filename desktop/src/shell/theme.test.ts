import { test } from "node:test"
import assert from "node:assert/strict"
import { parseTheme, nextTheme, applyTheme, type Theme } from "./theme.ts"

/** A stand-in for documentElement that records what was done to it. */
function fakeRoot() {
  const attrs = new Map<string, string>()
  return {
    attrs,
    setAttribute: (k: string, v: string) => void attrs.set(k, v),
    removeAttribute: (k: string) => void attrs.delete(k),
  }
}

test("anything unrecognised resolves to dark, never throws", () => {
  assert.equal(parseTheme("light"), "light")
  assert.equal(parseTheme("dark"), "dark")
  assert.equal(parseTheme(null), "dark")
  assert.equal(parseTheme(undefined), "dark")
  assert.equal(parseTheme(""), "dark")
  assert.equal(parseTheme("{corrupt json"), "dark")
  assert.equal(parseTheme("LIGHT"), "dark") // case-sensitive on purpose
})

test("toggling round-trips", () => {
  const t: Theme = "dark"
  assert.equal(nextTheme(t), "light")
  assert.equal(nextTheme(nextTheme(t)), t)
})

test("dark writes NO attribute: the default path stays stateless", () => {
  const r = fakeRoot()
  applyTheme(r, "light")
  assert.equal(r.attrs.get("data-theme"), "light")
  applyTheme(r, "dark")
  assert.equal(r.attrs.has("data-theme"), false)
})

import { test } from "node:test"
import assert from "node:assert/strict"
import { standingOf, countdown, pct, gapPt } from "./standing.ts"

const BUDGET = {
  spentTodaySol: 1.5,
  dailyCapSol: 50,
  remainingSol: 48.5,
  openSlots: 8,
  canEnter: true,
  breaker: { open: false, until: null as string | null },
}

test("kill wins over everything, including an open breaker", () => {
  const s = standingOf({
    killed: true,
    budget: { ...BUDGET, canEnter: false, breaker: { open: true, until: "2026-08-09T04:23:22.996Z" } },
  })
  assert.equal(s.state, "killed")
})

test("open breaker names itself and carries the parsed deadline", () => {
  const s = standingOf({
    killed: false,
    budget: { ...BUDGET, canEnter: false, breaker: { open: true, until: "2026-08-09T04:23:22.996Z" } },
  })
  assert.equal(s.state, "breaker")
  assert.equal(s.state === "breaker" && s.untilMs, Date.parse("2026-08-09T04:23:22.996Z"))
})

test("open breaker with an unparseable deadline still reports the breaker", () => {
  const s = standingOf({
    killed: false,
    budget: { ...BUDGET, canEnter: false, breaker: { open: true, until: null } },
  })
  assert.equal(s.state, "breaker")
  assert.equal(s.state === "breaker" && s.untilMs, null)
})

test("blocked entry with the day budget gone reads as the cap", () => {
  const s = standingOf({
    killed: false,
    budget: { ...BUDGET, spentTodaySol: 50, remainingSol: 0, canEnter: false },
  })
  assert.equal(s.state, "cap")
})

test("blocked entry with no free slots reads as slots", () => {
  const s = standingOf({
    killed: false,
    budget: { ...BUDGET, openSlots: 0, canEnter: false },
  })
  assert.equal(s.state, "slots")
})

test("blocked entry with no visible cause is PAUSED, never trading", () => {
  const s = standingOf({ killed: false, budget: { ...BUDGET, canEnter: false } })
  assert.equal(s.state, "paused")
})

test("nothing in the way reads as trading", () => {
  assert.equal(standingOf({ killed: false, budget: BUDGET }).state, "trading")
})

test("a summary without a budget block still reads as trading", () => {
  assert.equal(standingOf({ killed: false }).state, "trading")
})

test("countdown formats minutes and seconds to the deadline", () => {
  const now = 1_786_000_000_000
  assert.equal(countdown(now + 12 * 60_000 + 34_000, now), "12:34")
  assert.equal(countdown(now + 5_000, now), "0:05")
})

test("countdown clamps at zero once the deadline passed", () => {
  const now = 1_786_000_000_000
  assert.equal(countdown(now - 10_000, now), "0:00")
})

test("countdown with no deadline is empty", () => {
  assert.equal(countdown(null, 1_786_000_000_000), "")
})

test("gapPt is entered minus refused, in points, and refuses nulls", () => {
  assert.equal(gapPt(-29.1, -30.3), "+1.2pt")
  assert.equal(gapPt(-30.3, -29.1), "-1.2pt")
  assert.equal(gapPt(null, -30.3), "--")
  assert.equal(gapPt(-29.1, null), "--")
})

test("pct renders sign, one decimal, and refuses to invent a number for null", () => {
  assert.equal(pct(-29.130184614746895), "-29.1%")
  assert.equal(pct(1.2), "+1.2%")
  assert.equal(pct(0), "0.0%")
  assert.equal(pct(null), "--")
})

import { test } from "node:test"
import assert from "node:assert/strict"
import { describeEvent } from "./events.ts"

/** Real payload shapes captured from the live engine's events table. */

test("a consecutive-stops breaker trip explains itself and how long it holds", () => {
  const v = describeEvent({
    at: 1_786_245_802_996,
    kind: "breaker",
    data: JSON.stringify({ tripped: true, until: 1_786_249_402_996, afterConsecutiveStops: 4 }),
  })
  assert.equal(v.kind, "breaker")
  assert.equal(v.label, "BREAKER")
  assert.match(v.detail, /4 consecutive stops/)
  assert.match(v.detail, /60m/)
})

test("a loss-velocity breaker trip names its trigger", () => {
  const v = describeEvent({
    at: 1_786_245_802_996,
    kind: "breaker",
    data: JSON.stringify({
      tripped: true,
      kind: "loss-velocity",
      windowLossUsd: -31.4,
      until: 1_786_249_402_996,
    }),
  })
  assert.equal(v.kind, "breaker")
  assert.match(v.detail, /loss velocity/)
})

test("a breaker release reads as released", () => {
  const v = describeEvent({ at: 0, kind: "breaker", data: JSON.stringify({ tripped: false }) })
  assert.equal(v.kind, "breaker")
  assert.match(v.detail, /released/)
})

test("alert_sent shows the subject line", () => {
  const v = describeEvent({
    at: 0,
    kind: "alert_sent",
    data: JSON.stringify({ subject: "CroweTrade: launchpad re-validation ready", labeled: 118 }),
  })
  assert.equal(v.kind, "info")
  assert.equal(v.label, "ALERT")
  assert.match(v.detail, /launchpad re-validation ready/)
})

test("alert_failed is a failure and carries the error", () => {
  const v = describeEvent({
    at: 0,
    kind: "alert_failed",
    data: JSON.stringify({ error: "resend 403" }),
  })
  assert.equal(v.kind, "fail")
  assert.match(v.detail, /resend 403/)
})

test("scan_error is a failure and carries the message", () => {
  const v = describeEvent({
    at: 0,
    kind: "scan_error",
    data: JSON.stringify({ message: "/latest/dex/search -> 429" }),
  })
  assert.equal(v.kind, "fail")
  assert.match(v.detail, /429/)
})

test("entries, exits, skips and kill keep their existing shapes", () => {
  const entry = describeEvent({
    at: 0,
    kind: "entry",
    data: JSON.stringify({ symbol: "YUKI", verdict: "caution" }),
  })
  assert.equal(entry.kind, "entry")
  assert.equal(entry.label, "ENTER YUKI")

  const exitWin = describeEvent({
    at: 0,
    kind: "exit",
    data: JSON.stringify({ symbol: "YUKI", reason: "take-profit", pnlUsd: 12.3 }),
  })
  assert.equal(exitWin.kind, "exit-win")
  assert.match(exitWin.detail, /\+\$12\.30/)

  const exitLoss = describeEvent({
    at: 0,
    kind: "exit",
    data: JSON.stringify({ symbol: "SPAZ", reason: "stop-loss", pnlUsd: -11.46 }),
  })
  assert.equal(exitLoss.kind, "exit-loss")
  assert.match(exitLoss.detail, /-\$11\.46/)

  const skip = describeEvent({
    at: 0,
    kind: "entry_skipped",
    data: JSON.stringify({ symbol: "KEK", reason: "impact above cost hurdle" }),
  })
  assert.equal(skip.kind, "skip")
  assert.match(skip.detail, /impact above cost hurdle/)

  const kill = describeEvent({ at: 0, kind: "kill", data: JSON.stringify({ on: true }) })
  assert.equal(kill.label, "KILL")
  assert.match(kill.detail, /engaged/)
})

test("malformed payloads degrade to the bare kind rather than throwing", () => {
  const v = describeEvent({ at: 0, kind: "exit", data: "not json{" })
  assert.equal(v.kind, "skip")
  assert.equal(v.label, "exit")
  assert.equal(v.detail, "")
})

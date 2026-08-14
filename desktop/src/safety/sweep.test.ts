import { test } from "node:test"
import assert from "node:assert/strict"
import { describeReturns, netOfCost, ROUND_TRIP } from "../../../shared/sweep.ts"

/**
 * These guard the arithmetic behind a recommendation to loosen a gate that
 * spends money. The interesting cases are all about the heavy tail: this
 * market's mean is the portfolio number AND the number one outlier hijacks,
 * so every test here is really about not being fooled by a single token.
 */

test("median averages the middle pair on even counts", () => {
  assert.equal(describeReturns([-50, -10, 10, 50])!.medianPct, 0)
  assert.equal(describeReturns([-50, -10, 50])!.medianPct, -10)
})

test("unsorted input is summarized correctly", () => {
  // The engine hands these over in query order, never sorted.
  const s = describeReturns([500, -90, 12, -40])!
  assert.equal(s.medianPct, -14)
  assert.equal(s.bestPct, 500)
})

test("dropping the best exposes a cell carried by one token", () => {
  // The real case, 2026-08-13: the drift ceiling's 250-and-above cell showed
  // +289% mean on n=5 and was worthless without its single winner. This is
  // the check that refuted the hypothesis the sweep was built to confirm.
  const cell = [1457, -60, -80, -30, -42]
  const s = describeReturns(cell)!
  assert.equal(s.n, 5)
  assert.ok(s.meanPct > 240, `headline mean should look great, got ${s.meanPct}`)
  assert.ok(
    s.meanExBestPct !== null && s.meanExBestPct < 0,
    `without its best token the cell should be negative, got ${s.meanExBestPct}`,
  )
})

test("a genuine edge survives losing its best token", () => {
  // The contrast case: broad-based positive returns stay positive.
  const s = describeReturns([120, 90, 140, 200, 75, 110])!
  assert.ok(s.meanExBestPct !== null && s.meanExBestPct > 90)
})

test("only ONE instance of a tied maximum is dropped", () => {
  // Filtering by value would delete the whole tied cohort and overstate the
  // penalty exactly when the tail is flat rather than spiky.
  const s = describeReturns([100, 100, 100, 100])!
  assert.equal(s.meanExBestPct, 100)
})

test("meanExBest is null at n=1 rather than zero", () => {
  // A missing robustness check must never read as one that passed.
  const s = describeReturns([250])!
  assert.equal(s.n, 1)
  assert.equal(s.meanPct, 250)
  assert.equal(s.meanExBestPct, null)
})

test("an empty set is null, not a zeroed struct", () => {
  // Same distinction the funnel window draws. A band with no candidates must
  // not report 0%, which would read as "admits nothing profitable" rather
  // than "admits nothing".
  assert.equal(describeReturns([]), null)
})

test("share counts use strict thresholds", () => {
  // Exactly zero is not "up"; exactly 100 is not "over 100".
  const s = describeReturns([0, 100, 101, -1])!
  assert.equal(s.upPct, 50)
  assert.equal(s.over100Pct, 25)
})

test("impact is charged as a round trip, not one way", () => {
  // A candidate at 3% impact costs ~6% to get in and out. Scoring it on the
  // raw forward return reads that 6% as free, and the sweep would then
  // recommend admitting everything.
  assert.equal(ROUND_TRIP, 2)
  assert.equal(netOfCost(10, 3), 4)
  assert.equal(netOfCost(-50, 1.5), -53)
})

test("cost can flip a marginally positive candidate negative", () => {
  // The whole point of the hurdle: a token that rose 4% while costing 2.5%
  // each way is a loss, and a sweep ignoring that would call it a win.
  assert.ok(netOfCost(4, 2.5) < 0)
})

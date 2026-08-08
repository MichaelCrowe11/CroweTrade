import { test } from "node:test"
import assert from "node:assert/strict"
import { computeFeatures, type TickSeries } from "../../../shared/features.ts"

/**
 * The features feed the future edge model, so their failure modes matter more
 * than their happy paths: a wrong sign on flow, or an artifact read as signal,
 * poisons every downstream calibration.
 */

const s = (
  prices: number[],
  liquidity: number[],
  buys24h: number[],
  sells24h: number[],
): TickSeries => ({ prices, liquidity, buys24h, sells24h })

test("buy-dominated flow reads positive", () => {
  const f = computeFeatures(s([1, 1.1, 1.2], [5000, 5000, 5000], [100, 130, 160], [50, 55, 60]))
  // 60 new buys vs 10 new sells across the window.
  assert.ok(f.netFlowShare !== null && f.netFlowShare > 0.6)
})

test("sell-dominated flow reads negative", () => {
  const f = computeFeatures(s([1, 0.95, 0.9], [5000, 5000, 5000], [100, 105, 110], [50, 90, 130]))
  assert.ok(f.netFlowShare !== null && f.netFlowShare < -0.6)
})

test("sliding-window counter decreases are artifacts, not negative flow", () => {
  // Cumulative 24h counters DROP when old activity ages out of the window.
  // That must clamp to zero flow, never count as selling.
  const f = computeFeatures(s([1, 1, 1], [5000, 5000, 5000], [100, 90, 80], [50, 45, 40]))
  assert.equal(f.netFlowShare, null) // zero observed flow -> no share at all
})

test("accelerating final minute reads positive flowAccel", () => {
  // Window minutes: 10 then 40 events; last minute is 40 vs avg 25.
  const f = computeFeatures(s([1, 1, 1], [5000, 5000, 5000], [0, 8, 40], [0, 2, 10]))
  assert.ok(f.flowAccel !== null && f.flowAccel > 0.5)
})

test("price progress and liquidity trend carry sign and scale", () => {
  const f = computeFeatures(s([1, 1.5], [4000, 5000], [0, 10], [0, 10]))
  assert.ok(f.priceProgressPct !== null && Math.abs(f.priceProgressPct - 50) < 1e-9)
  assert.ok(f.liqTrendPct !== null && Math.abs(f.liqTrendPct - 25) < 1e-9)
})

test("a single tick yields nulls, never zeros", () => {
  // Zero is a claim ("flat, balanced"); null is honesty ("cannot know yet").
  const f = computeFeatures(s([1], [5000], [10], [5]))
  assert.equal(f.ticks, 1)
  assert.equal(f.netFlowShare, null)
  assert.equal(f.priceProgressPct, null)
  assert.equal(f.liqTrendPct, null)
})

test("mismatched series lengths use the common prefix", () => {
  const f = computeFeatures(s([1, 1.1, 1.2], [5000, 5000], [0, 10], [0, 5]))
  assert.equal(f.ticks, 2)
  assert.ok(f.priceProgressPct !== null && f.priceProgressPct > 9)
})

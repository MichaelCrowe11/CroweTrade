import { test } from "node:test"
import assert from "node:assert/strict"
import {
  fit, score, auc, buildFeatureVector, passesModelGate,
  FEATURE_NAMES, type TrainingRow,
} from "../../../shared/model.ts"
import { ARMED_MODEL } from "../../../shared/armed-model.ts"

/**
 * A model that decides trades has to be tested on the properties that make it
 * trustworthy, not on whether it runs. Chiefly: does it refuse to speak when
 * the data cannot support a claim, and does it rank better than a coin flip
 * ONLY when there is genuine signal.
 */

/** Deterministic pseudo-random so these tests never flake. */
function rng(seed: number) {
  let s = seed
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648
    return s / 2147483648
  }
}

function rows(n: number, signal: boolean, seed = 7): TrainingRow[] {
  const r = rng(seed)
  const out: TrainingRow[] = []
  for (let i = 0; i < n; i++) {
    const flow = r() * 2 - 1
    const noise = [r(), r(), r(), r(), r()]
    // With signal, the label follows flow. Without, it is a coin flip and no
    // model should be able to beat 0.5 on it.
    const label: 0 | 1 = signal ? (flow > 0 ? 1 : 0) : r() > 0.5 ? 1 : 0
    out.push({
      at: 1_700_000_000_000 + i * 60_000,
      features: [flow, ...noise],
      label,
    })
  }
  return out
}

test("refuses to fit below the minimum sample", () => {
  assert.equal(fit(rows(40, true), 80), null)
})

test("keeps a scoring set large enough to mean anything", () => {
  const m = fit(rows(85, true), 80)
  assert.ok(m, "expected a fit at 85 rows")
  assert.ok(m.testN >= 10, `test set too small: ${m.testN}`)
})

test("learns a real signal: AUC well above chance", () => {
  const m = fit(rows(400, true), 80)
  assert.ok(m, "expected a fit")
  assert.ok(m.auc > 0.8, `expected strong AUC, got ${m.auc}`)
})

test("does NOT invent signal in pure noise: AUC near chance", () => {
  const m = fit(rows(400, false, 99), 80)
  assert.ok(m, "expected a fit")
  // The critical property. A model that scores high here is memorising, and
  // would hand the sizing layer confident nonsense.
  assert.ok(m.auc < 0.65, `expected near-chance AUC on noise, got ${m.auc}`)
})

test("splits temporally, never randomly", () => {
  const data = rows(200, true)
  const m = fit(data, 80)
  assert.ok(m)
  // 75/25 by time.
  assert.equal(m.trainN, 150)
  assert.equal(m.testN, 50)
})

test("scoring is bounded to a probability", () => {
  const m = fit(rows(300, true), 80)
  assert.ok(m)
  for (const f of [[-99, 0, 0, 0, 0, 0], [99, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]) {
    const p = score(m, f)
    assert.ok(p >= 0 && p <= 1, `probability out of range: ${p}`)
  }
})

test("auc handles a single-class set without dividing by zero", () => {
  assert.equal(auc([0.1, 0.9, 0.5], [1, 1, 1]), 0.5)
  assert.equal(auc([0.1, 0.9, 0.5], [0, 0, 0]), 0.5)
})

test("reliability bins report observed frequency alongside predicted", () => {
  const m = fit(rows(400, true), 80)
  assert.ok(m)
  assert.ok(m.reliability.length > 0)
  for (const b of m.reliability) {
    assert.ok(b.predicted >= 0 && b.predicted <= 1)
    assert.ok(b.observed >= 0 && b.observed <= 1)
    assert.ok(b.n > 0)
  }
})

// ── The armed gate and its single feature-vector definition ────────────────

test("buildFeatureVector order matches the FEATURE_NAMES contract", () => {
  const v = buildFeatureVector(
    { ticks: 4, netFlowShare: 0.5, flowAccel: 0.2, priceProgressPct: 30, liqTrendPct: -10 },
    9_999, true,
  )
  assert.equal(v.length, FEATURE_NAMES.length)
  assert.equal(v[0], 0.5)              // netFlowShare
  assert.equal(v[1], 0.2)              // flowAccel
  assert.equal(v[2], 30)               // priceProgressPct
  assert.equal(v[3], -10)              // liqTrendPct
  assert.equal(v[4], 4)                // ticksObserved
  assert.equal(v[5], Math.log10(10_000)) // logLiquidityUsd
  assert.equal(v[6], 1)                // liqKnown
  assert.equal(v[7], 1)                // isLaunchpad
})

test("unmeasured liquidity reads as unknown, never as an empty pool", () => {
  const v = buildFeatureVector({ ticks: 3 }, null, false)
  assert.equal(v[5], 0)
  assert.equal(v[6], 0) // liqKnown = 0 is the honest statement; log 0 alone would say "$1 pool"
})

test("artifact-scale progress figures are clamped, honest ones pass through", () => {
  // A tick window spanning the graduated-coin pricing fault carried progress
  // in the millions of percent; the clamp keeps faults from owning the scale.
  const bad = buildFeatureVector({ ticks: 3, priceProgressPct: 2_000_000, liqTrendPct: -99_999 }, 5_000, false)
  assert.equal(bad[2], 500)
  assert.equal(bad[3], -100)
  const ok = buildFeatureVector({ ticks: 3, priceProgressPct: 45, liqTrendPct: -20 }, 5_000, false)
  assert.equal(ok[2], 45)
  assert.equal(ok[3], -20)
})

test("armed gate: unknown never passes, unarmed passes everything", () => {
  assert.equal(passesModelGate(0.2, null), false)   // uncomputable blocks when armed
  assert.equal(passesModelGate(0.2, 0.19), false)
  assert.equal(passesModelGate(0.2, 0.2), true)
  assert.equal(passesModelGate(null, null), true)   // no gate exists
  assert.equal(passesModelGate(null, 0.01), true)
})

test("frozen armed model prefers measured deep liquidity, as fitted", () => {
  const base = { ticks: 3, netFlowShare: 0, flowAccel: 0, priceProgressPct: 10, liqTrendPct: 5 }
  const deep = score(ARMED_MODEL, buildFeatureVector(base, 50_000, true))
  const thin = score(ARMED_MODEL, buildFeatureVector(base, 800, true))
  assert.ok(deep > thin, `expected deep pool to score above thin: ${deep} vs ${thin}`)
  assert.ok(deep >= 0 && deep <= 1)
})

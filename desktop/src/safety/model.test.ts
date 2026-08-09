import { test } from "node:test"
import assert from "node:assert/strict"
import { fit, score, auc, type TrainingRow } from "../../../shared/model.ts"

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

import { test } from "node:test"
import assert from "node:assert/strict"
import { trajectoryConfirms, type Trajectory } from "../../../shared/trajectory.ts"

/**
 * The trajectory gate decides whether OUR OWN observed tape supports an entry.
 * It is the piece of v2 most exposed to off-by-one and empty-data mistakes,
 * and a wrong "true" here buys a token whose liquidity is walking out.
 */

const t = (prices: number[], liquidity: number[]): Trajectory => ({ prices, liquidity })

test("rising price with stable liquidity confirms", () => {
  assert.equal(trajectoryConfirms(t([1, 1.1, 1.3], [5000, 5100, 5050]), 3), true)
})

test("falling price refuses", () => {
  assert.equal(trajectoryConfirms(t([1.3, 1.2, 1.0], [5000, 5000, 5000]), 3), false)
})

test("flat price refuses: confirmation requires progress, not survival", () => {
  assert.equal(trajectoryConfirms(t([1, 1, 1], [5000, 5000, 5000]), 3), false)
})

test("draining liquidity refuses even with rising price", () => {
  // Price up 30% while liquidity drops 40% is a rug pulling in the classic
  // shape: mark it up, walk the floor out.
  assert.equal(trajectoryConfirms(t([1, 1.15, 1.3], [5000, 4000, 3000]), 3), false)
})

test("liquidity dip within 10% tolerance still confirms", () => {
  assert.equal(trajectoryConfirms(t([1, 1.1, 1.2], [5000, 4800, 4600]), 3), true)
})

test("too few observations refuses", () => {
  assert.equal(trajectoryConfirms(t([1, 1.2], [5000, 5000]), 3), false)
})

test("missing trajectory refuses", () => {
  assert.equal(trajectoryConfirms(undefined, 3), false)
})

test("zero starting price or liquidity refuses rather than dividing", () => {
  assert.equal(trajectoryConfirms(t([0, 1, 2], [5000, 5000, 5000]), 3), false)
  assert.equal(trajectoryConfirms(t([1, 1.1, 1.2], [0, 100, 200]), 3), false)
})

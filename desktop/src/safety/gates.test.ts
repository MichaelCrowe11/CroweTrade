import { test } from "node:test"
import assert from "node:assert/strict"
import { combineVerdict, evaluateGates, type TokenSnapshot } from "../../../shared/gates.ts"

/**
 * The verdict policy is the line between observing and trading, so its tests
 * are written against SNAPSHOTS, not hand-built GateResult arrays: if a gate's
 * thresholds change, these tests re-derive the gates the same way production
 * does and the policy is judged on what the operator would actually see.
 */

const LAMPORTS_PER_SOL = 1_000_000_000n

/** A token with everything known and everything healthy. */
function healthy(): TokenSnapshot {
  return {
    mint: "So11111111111111111111111111111111111111112",
    asOf: 1_700_000_000_000,
    launchedAt: 1_699_999_000_000,
    mintAuthority: null,
    freezeAuthority: null,
    lpLockedBps: 10_000,
    topHolderShare: 0.05,
    solReserveLamports: 100n * LAMPORTS_PER_SOL,
    deployerPriorMints: 4,
    deployerPriorRugs: 0,
  }
}

/** A token seconds old: nothing observed yet beyond the pool's existence. */
function newborn(): TokenSnapshot {
  return {
    mint: "So11111111111111111111111111111111111111112",
    asOf: 1_700_000_000_000,
    launchedAt: 1_700_000_000_000,
    mintAuthority: undefined,
    freezeAuthority: undefined,
    lpLockedBps: undefined,
    topHolderShare: undefined,
    solReserveLamports: undefined,
    deployerPriorMints: undefined,
    deployerPriorRugs: undefined,
  }
}

test("fully healthy token clears", () => {
  assert.equal(combineVerdict(evaluateGates(healthy())), "clear")
})

test("retained mint authority blocks regardless of everything else passing", () => {
  const s = healthy()
  s.mintAuthority = "Deploy3r111111111111111111111111111111111111"
  assert.equal(combineVerdict(evaluateGates(s)), "blocked")
})

test("live freeze authority blocks", () => {
  const s = healthy()
  s.freezeAuthority = "Deploy3r111111111111111111111111111111111111"
  assert.equal(combineVerdict(evaluateGates(s)), "blocked")
})

test("unlocked LP blocks", () => {
  const s = healthy()
  s.lpLockedBps = 0
  assert.equal(combineVerdict(evaluateGates(s)), "blocked")
})

test("a newborn with nothing known reads insufficient-data, not pass or fail", () => {
  assert.equal(combineVerdict(evaluateGates(newborn())), "insufficient-data")
})

test("critical unknowns cap at caution even when observed gates pass", () => {
  const s = newborn()
  // Authorities observed and clean; LP still unknown -> caution, never clear.
  s.mintAuthority = null
  s.freezeAuthority = null
  assert.equal(combineVerdict(evaluateGates(s)), "caution")
})

test("one elevated fail degrades to caution", () => {
  const s = healthy()
  s.topHolderShare = 0.4
  assert.equal(combineVerdict(evaluateGates(s)), "caution")
})

test("two elevated fails block", () => {
  const s = healthy()
  s.topHolderShare = 0.4
  s.solReserveLamports = 1n * LAMPORTS_PER_SOL
  assert.equal(combineVerdict(evaluateGates(s)), "blocked")
})

test("deployer with prior rugs plus thin liquidity blocks", () => {
  const s = healthy()
  s.deployerPriorMints = 6
  s.deployerPriorRugs = 3
  s.solReserveLamports = 1n * LAMPORTS_PER_SOL
  assert.equal(combineVerdict(evaluateGates(s)), "blocked")
})

test("unknown never renders as pass in any gate", () => {
  for (const gate of evaluateGates(newborn())) {
    assert.notEqual(gate.state, "pass", `${gate.id} passed on no data`)
  }
})

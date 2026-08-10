import { test } from "node:test"
import assert from "node:assert/strict"
import { validateProposal, PROPOSABLE } from "../../../shared/proposal.ts"
import { PAPER_POLICY } from "../../../shared/policy.ts"

/**
 * The test that justifies the file: a proposal for something already in
 * effect must come back as a no-op. On 2026-08-10 the Analyst recommended
 * dropping the profile discovery feed hours after the allowlist had dropped
 * it, and nothing in the system could tell it so.
 */

test("★ a change already in effect is a NO-OP, not a plan", () => {
  // Exactly the recommendation the Analyst made: drop profile, keep launchpad.
  // The running policy already says this.
  const r = validateProposal(PAPER_POLICY, [
    { path: "entry.allowedOrigins", to: ["launchpad"] },
  ])
  assert.equal(r.entirelyNoop, true)
  assert.equal(r.ok, false)
  assert.equal(r.changes[0]?.noop, true)
  assert.match(r.errors.join(" "), /already in effect/)
})

test("array comparison ignores ORDER, so a reshuffle is still a no-op", () => {
  const policy = { ...PAPER_POLICY, entry: { ...PAPER_POLICY.entry, allowedOrigins: ["launchpad", "boost"] } }
  const r = validateProposal(policy, [{ path: "entry.allowedOrigins", to: ["boost", "launchpad"] }])
  assert.equal(r.changes[0]?.noop, true)
})

test("a real change validates and reports both sides", () => {
  const r = validateProposal(PAPER_POLICY, [
    { path: "entry.maxDriftSinceFirstSightPct", to: 20 },
  ])
  assert.equal(r.ok, true)
  assert.equal(r.entirelyNoop, false)
  assert.equal(r.changes[0]?.from, 30)
  assert.equal(r.changes[0]?.to, 20)
})

test("fields outside the allowlist are refused, including the consent record", () => {
  for (const path of ["product", "signature", "signer", "waiverSha256", "expiresAt"]) {
    const r = validateProposal(PAPER_POLICY, [{ path, to: "anything" }])
    assert.equal(r.ok, false, path)
    assert.match(r.errors.join(" "), /not a proposable field/)
  }
})

test("a nonexistent path is refused rather than silently creating a field", () => {
  // Not in PROPOSABLE, so it is caught there first — which is the point:
  // the allowlist is the primary defence, path existence the secondary.
  const r = validateProposal(PAPER_POLICY, [{ path: "entry.madeUpKnob", to: 1 }])
  assert.equal(r.ok, false)
})

test("type mismatches are malformed proposals, not policy questions", () => {
  const r = validateProposal(PAPER_POLICY, [{ path: "exit.stopLossPct", to: "tighter" }])
  assert.equal(r.ok, false)
  assert.match(r.errors.join(" "), /expects number/)
})

test("negative and non-finite numbers are refused", () => {
  for (const bad of [-1, NaN, Infinity]) {
    const r = validateProposal(PAPER_POLICY, [{ path: "dailyCapSol", to: bad }])
    assert.equal(r.ok, false, String(bad))
  }
})

test("an empty proposal is refused", () => {
  const r = validateProposal(PAPER_POLICY, [])
  assert.equal(r.ok, false)
  assert.match(r.errors.join(" "), /no changes/)
})

test("the same field twice is refused rather than last-write-wins", () => {
  const r = validateProposal(PAPER_POLICY, [
    { path: "dailyCapSol", to: 10 },
    { path: "dailyCapSol", to: 20 },
  ])
  assert.equal(r.ok, false)
  assert.match(r.errors.join(" "), /more than once/)
})

// ── Tighten vs loosen ──────────────────────────────────────────────────────
//
// The governance axis the envelope documents: tighten instantly, loosen with
// delay. The sign is NOT guessable per field, and getting it backwards would
// label a loosening as safe to apply immediately.

test("lowering a CAP tightens; raising it loosens", () => {
  assert.equal(validateProposal(PAPER_POLICY, [{ path: "dailyCapSol", to: 10 }]).changes[0]?.tightens, true)
  assert.equal(validateProposal(PAPER_POLICY, [{ path: "dailyCapSol", to: 99 }]).changes[0]?.tightens, false)
})

test("raising a FLOOR tightens, which is the opposite direction", () => {
  // The case a naive "lower is safer" rule gets wrong.
  assert.equal(
    validateProposal(PAPER_POLICY, [{ path: "entry.minModelProb", to: 0.4 }]).changes[0]?.tightens,
    true,
  )
  assert.equal(
    validateProposal(PAPER_POLICY, [{ path: "entry.minModelProb", to: 0.05 }]).changes[0]?.tightens,
    false,
  )
})

test("removing a discovery source tightens; adding one loosens", () => {
  const two = { ...PAPER_POLICY, entry: { ...PAPER_POLICY.entry, allowedOrigins: ["launchpad", "profile"] } }
  assert.equal(
    validateProposal(two, [{ path: "entry.allowedOrigins", to: ["launchpad"] }]).changes[0]?.tightens,
    true,
  )
  assert.equal(
    validateProposal(PAPER_POLICY, [{ path: "entry.allowedOrigins", to: ["launchpad", "profile"] }])
      .changes[0]?.tightens,
    false,
  )
})

test("requiring a CLEAR verdict tightens", () => {
  assert.equal(
    validateProposal(PAPER_POLICY, [{ path: "entry.minVerdict", to: "clear" }]).changes[0]?.tightens,
    true,
  )
})

test("every proposable path actually exists on the policy", () => {
  // Guards against the allowlist drifting out of step with the envelope shape,
  // which would let an agent propose a field that silently does nothing.
  for (const path of PROPOSABLE) {
    const r = validateProposal(PAPER_POLICY, [{ path, to: null }])
    assert.ok(
      !r.errors.some((e) => e.includes("does not exist")),
      `${path} is proposable but missing from PAPER_POLICY`,
    )
  }
})

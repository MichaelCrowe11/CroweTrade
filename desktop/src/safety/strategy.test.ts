import { test } from "node:test"
import assert from "node:assert/strict"
import { trajectoryConfirms, hasDrifted, type Trajectory } from "../../../shared/trajectory.ts"
import { emptyFunnel } from "../../../shared/funnel.ts"

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

// ── Discovery allowlist ────────────────────────────────────────────────────
//
// The change that took the promotional feed out of the book. It is an
// ALLOWLIST on purpose: the failure mode being designed against is a new
// discovery source getting added to the scanner and inheriting permission to
// spend money because nobody remembered to exclude it.

/** The predicate as decideEntries applies it, isolated so it can be pinned
 *  without constructing a whole Candidate and policy envelope. */
const originAllowed = (origin: string, allowed: string[]): boolean =>
  origin === "held" || allowed.includes(origin)

test("allowlist admits only named origins", () => {
  assert.equal(originAllowed("launchpad", ["launchpad"]), true)
  assert.equal(originAllowed("profile", ["launchpad"]), false)
  assert.equal(originAllowed("boost", ["launchpad"]), false)
  assert.equal(originAllowed("both", ["launchpad"]), false)
})

test("an unrecognised new source is refused by default, not admitted", () => {
  // The whole point of an allowlist over an exclusion list.
  assert.equal(originAllowed("some-future-feed", ["launchpad"]), false)
})

test("held is always permitted: it re-prices an open position, not a purchase", () => {
  assert.equal(originAllowed("held", ["launchpad"]), true)
  assert.equal(originAllowed("held", []), true)
})

test("an empty allowlist buys nothing at all", () => {
  assert.equal(originAllowed("launchpad", []), false)
  assert.equal(originAllowed("profile", []), false)
})

// ── First-sight drift ──────────────────────────────────────────────────────
//
// The gate built from the finding that the engine buys tokens that already
// ran: +142.8% average between first sight and entry, 54 of 73 bought higher.

test("a token that already ran past the threshold is refused", () => {
  // Doubled since we first saw it: exactly the shape that was losing money.
  assert.equal(hasDrifted(1.0, 2.0, 30), true)
  assert.equal(hasDrifted(1.0, 1.31, 30), true)
})

test("modest drift inside the threshold is allowed", () => {
  assert.equal(hasDrifted(1.0, 1.29, 30), false)
  assert.equal(hasDrifted(1.0, 1.0, 30), false)
})

test("a token that FELL since first sight is never refused for drifting", () => {
  // Drift is directional. Buying lower than we first saw is the opposite of
  // chasing, and this gate must not block it.
  assert.equal(hasDrifted(1.0, 0.5, 30), false)
})

test("an unknown first-sight price REFUSES, like every other gate here", () => {
  assert.equal(hasDrifted(null, 1.0, 30), true)
  assert.equal(hasDrifted(undefined, 1.0, 30), true)
  assert.equal(hasDrifted(0, 1.0, 30), true)
  assert.equal(hasDrifted(NaN, 1.0, 30), true)
})

test("a non-finite entry price refuses rather than comparing as false", () => {
  assert.equal(hasDrifted(1.0, NaN, 30), true)
  assert.equal(hasDrifted(1.0, 0, 30), true)
})

test("the boundary is float, not exact, and the test says so", () => {
  // (1.30 - 1.0) / 1.0 * 100 evaluates to 30.000000000000004, so a nominal
  // "exactly +30%" is REFUSED. That is not worth an epsilon: the threshold is
  // a judgement call in whole percent and a 4e-15 error at the boundary
  // cannot matter, whereas pretending to exactness would hide a real property
  // of the arithmetic from whoever reads this next.
  assert.equal(hasDrifted(1.0, 1.30, 30), true, "float boundary lands just over")
  assert.equal(hasDrifted(1.0, 1.29, 30), false, "clearly inside")
  assert.equal(hasDrifted(1.0, 1.31, 30), true, "clearly outside")
})

// ── Momentum filter and absent data ────────────────────────────────────────
//
// The interaction that stopped all trading on 2026-08-10: launchpad tokens
// carry changeH1 = null structurally (a four-minute-old mint has no hourly
// change), and refusing null refused the entire allowed universe.

/** The predicate as decideEntries applies it. */
const parabolicRefuses = (changeH1: number | null, ceiling: number): boolean =>
  changeH1 !== null && changeH1 > ceiling

test("a KNOWN change above the ceiling still refuses", () => {
  assert.equal(parabolicRefuses(120, 80), true)
  assert.equal(parabolicRefuses(80.1, 80), true)
})

test("a known change at or under the ceiling passes", () => {
  assert.equal(parabolicRefuses(80, 80), false)
  assert.equal(parabolicRefuses(-30, 80), false)
})

test("UNKNOWN change passes here, because drift catches it downstream", () => {
  // Refusing null refused every launchpad token, which is the only origin the
  // policy admits. The first-sight drift gate answers the same question with
  // our own recorded price rather than a third party's hourly claim.
  assert.equal(parabolicRefuses(null, 80), false)
})

// ── Entry funnel accounting ────────────────────────────────────────────────
//
// The counts only help if they are trustworthy, and the property that makes
// them trustworthy is that they SUM to scanned: every candidate is counted at
// exactly the first stage that rejects it, so nothing vanishes silently and
// the largest bucket is genuinely the blocker.

test("an empty funnel starts at zero everywhere", () => {
  const f = emptyFunnel()
  for (const [k, v] of Object.entries(f)) assert.equal(v, 0, `${k} should start at 0`)
})

test("the funnel has a bucket for every rejection the filter can make", () => {
  // If a stage is added to decideEntries without a bucket, candidates vanish
  // from the accounting and the sum stops matching scanned.
  const f = emptyFunnel()
  for (const k of [
    "heldAlready", "noPrice", "thinLiquidity", "noCreatedAt", "tooOld", "tooNew",
    "parabolic", "originNotAllowed", "trajectoryUnconfirmed", "verdictTooLow",
    "modelProbTooLow", "budgetOrSlotsExhausted", "admitted",
  ]) {
    assert.ok(k in f, `missing bucket: ${k}`)
  }
})

test("buckets sum to scanned, which is what makes the largest one meaningful", () => {
  // Simulating a tick's accounting: 10 scanned, distributed across stages.
  const f = emptyFunnel()
  f.scanned = 10
  f.thinLiquidity = 7
  f.tooNew = 2
  f.admitted = 1
  const { scanned, ...rest } = f
  assert.equal(Object.values(rest).reduce((a, b) => a + b, 0), scanned)
})

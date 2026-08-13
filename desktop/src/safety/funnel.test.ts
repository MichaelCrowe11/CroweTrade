import { test } from "node:test"
import assert from "node:assert/strict"
import {
  emptyFunnel, packFunnel, unpackFunnel, pushFunnelRing, summarizeFunnelRing,
  FUNNEL_KEYS, EXEC_FUNNEL_KEYS, emptyExecFunnel, packExecFunnel,
  unpackExecFunnel, upsertFunnelRow, type FunnelRingEntry,
} from "../../../shared/funnel.ts"

/**
 * The rolling window exists because the per-tick funnel has a resolution
 * floor: the dominant bucket is readable at n=1, every stage below it sees one
 * or two candidates a tick. These tests are mostly about the two ways an
 * accumulator lies — mixing policy regimes, and mislabelling packed history.
 */

const tick = (over: Partial<ReturnType<typeof emptyFunnel>> = {}) => ({ ...emptyFunnel(), ...over })
const entry = (at: number, h: string, over = {}): FunnelRingEntry =>
  ({ at, h, c: packFunnel(tick(over)) })

test("pack and unpack round-trip every stage", () => {
  const f = emptyFunnel()
  FUNNEL_KEYS.forEach((k, i) => { f[k] = i + 1 })
  assert.deepEqual(unpackFunnel(packFunnel(f)), f)
})

test("a short packed row zero-fills rather than shifting stages", () => {
  // A stage appended after some history was written must not relabel the
  // older rows. Truncated input reads as "that stage was not measured", 0.
  const short = [7, 1]
  const out = unpackFunnel(short)
  assert.equal(out.scanned, 7)
  assert.equal(out.heldAlready, 1)
  assert.equal(out.admitted, 0)
  assert.equal(out.modelProbTooLow, 0)
})

test("the ring keeps the newest entries and drops the oldest", () => {
  let ring: FunnelRingEntry[] = []
  for (let i = 0; i < 5; i++) ring = pushFunnelRing(ring, entry(i, "a"), 3)
  assert.equal(ring.length, 3)
  assert.deepEqual(ring.map((e) => e.at), [2, 3, 4])
})

test("the window sums only ticks from the current policy", () => {
  const ring = [
    entry(1, "old", { scanned: 100, thinLiquidity: 100 }),
    entry(2, "old", { scanned: 100, thinLiquidity: 100 }),
    entry(3, "new", { scanned: 10, thinLiquidity: 8, tooNew: 2 }),
    entry(4, "new", { scanned: 10, thinLiquidity: 7, tooNew: 3 }),
  ]
  const w = summarizeFunnelRing(ring, "new")
  assert.ok(w)
  assert.equal(w.ticks, 2)
  assert.equal(w.counts.scanned, 20)
  assert.equal(w.counts.thinLiquidity, 15)
  assert.equal(w.counts.tooNew, 5)
  // The refused regime must not leak in: 200 scanned under "old" would have
  // buried the 20 that describe what is actually running.
  assert.equal(w.counts.scanned, 20)
})

test("a window holding other policies says so", () => {
  const mixed = summarizeFunnelRing([entry(1, "old"), entry(2, "new")], "new")
  assert.equal(mixed?.mixed, true)
  const clean = summarizeFunnelRing([entry(1, "new"), entry(2, "new")], "new")
  assert.equal(clean?.mixed, false)
})

test("no ticks under the current policy returns null, never a zeroed window", () => {
  // A window of all-zeros reads as "every gate saw nothing and refused
  // nothing", which is a claim. Absence is not that claim.
  assert.equal(summarizeFunnelRing([entry(1, "old")], "new"), null)
  assert.equal(summarizeFunnelRing([], "new"), null)
})

test("window bounds report the counted ticks, not the ring", () => {
  const w = summarizeFunnelRing(
    [entry(1, "old"), entry(50, "new"), entry(90, "new")],
    "new",
  )
  assert.equal(w?.since, 50)
  assert.equal(w?.until, 90)
})

const xtick = (over: Partial<ReturnType<typeof emptyExecFunnel>> = {}) =>
  ({ ...emptyExecFunnel(), ...over })
const xentry = (at: number, h: string, sel = {}, ex = {}): FunnelRingEntry =>
  ({ at, h, c: packFunnel(tick(sel)), x: packExecFunnel(xtick(ex)) })

test("the second write of a tick replaces the first, never appends", () => {
  // A tick writes twice: selection before the entry loop, both halves after.
  // Appending both would double every completed tick and halve the window's
  // real span.
  let ring: FunnelRingEntry[] = []
  ring = upsertFunnelRow(ring, entry(100, "p", { scanned: 9, admitted: 2 }), 120)
  ring = upsertFunnelRow(ring, xentry(100, "p", { scanned: 9, admitted: 2 }, { admitted: 2, entered: 1, drifted: 1 }), 120)
  assert.equal(ring.length, 1)
  assert.equal(ring[0]?.at, 100)
  const w = summarizeFunnelRing(ring, "p")!
  assert.equal(w.ticks, 1)
  assert.equal(w.counts.scanned, 9)
  assert.equal(w.exec?.entered, 1)
  assert.equal(w.execUnmeasuredTicks, 0)
})

test("a tick that dies leaves its selection half and reads as unmeasured", () => {
  // The whole reason the write was split. The engine was resetting partway
  // through the entry loop and a ring recording only completed ticks went
  // silent for exactly the ticks that were failing.
  let ring: FunnelRingEntry[] = []
  ring = upsertFunnelRow(ring, entry(100, "p", { scanned: 9, admitted: 2 }), 120)
  ring = upsertFunnelRow(ring, xentry(160, "p", { scanned: 7 }, { admitted: 1, entered: 1 }), 120)
  const w = summarizeFunnelRing(ring, "p")!
  assert.equal(w.ticks, 2)
  assert.equal(w.counts.scanned, 16)
  assert.equal(w.execUnmeasuredTicks, 1)
  assert.equal(w.exec?.entered, 1)
})

test("a new tick appends and the cap still holds", () => {
  let ring: FunnelRingEntry[] = []
  for (let i = 0; i < 5; i++) {
    ring = upsertFunnelRow(ring, entry(i, "p"), 3)
    ring = upsertFunnelRow(ring, xentry(i, "p"), 3)
  }
  assert.equal(ring.length, 3)
  assert.deepEqual(ring.map((e) => e.at), [2, 3, 4])
})

test("execution counts round-trip and zero-fill a short row", () => {
  const f = emptyExecFunnel()
  EXEC_FUNNEL_KEYS.forEach((k, i) => { f[k] = i + 1 })
  assert.deepEqual(unpackExecFunnel(packExecFunnel(f)), f)
  assert.equal(unpackExecFunnel([3]).admitted, 3)
  assert.equal(unpackExecFunnel([3]).entered, 0)
})

test("the execution chain sums to admitted", () => {
  // The invariant that makes the largest post-admission bucket trustworthy,
  // exactly as buckets-sum-to-scanned does for selection.
  const w = summarizeFunnelRing([
    xentry(1, "p", {}, { admitted: 4, simulationFailed: 2, noRoute: 1, entered: 1 }),
    xentry(2, "p", {}, { admitted: 3, impactAboveHurdle: 2, drifted: 1 }),
  ], "p")!
  const x = w.exec!
  const parts = EXEC_FUNNEL_KEYS.filter((k) => k !== "admitted")
    .reduce((n, k) => n + x[k], 0)
  assert.equal(parts, x.admitted)
  assert.equal(x.admitted, 7)
  assert.equal(x.entered, 1)
  assert.equal(x.simulationFailed, 2)
})

test("ticks with no execution row are reported as unmeasured, not as zero", () => {
  // A pre-upgrade row zero-filled would claim that tick admitted nothing.
  // "We did not look" and "nothing happened" must not render the same.
  const ring = [
    entry(1, "p", { scanned: 10 }),
    xentry(2, "p", { scanned: 10 }, { admitted: 2, entered: 1, noRoute: 1 }),
  ]
  const w = summarizeFunnelRing(ring, "p")!
  assert.equal(w.ticks, 2)
  assert.equal(w.execUnmeasuredTicks, 1)
  assert.equal(w.exec?.admitted, 2)
  assert.equal(w.counts.scanned, 20)
})

test("a window with no execution rows at all returns exec null", () => {
  const w = summarizeFunnelRing([entry(1, "p"), entry(2, "p")], "p")!
  assert.equal(w.exec, null)
  assert.equal(w.execUnmeasuredTicks, 2)
})

test("the two funnels join at admitted", () => {
  // Selection ends where execution begins; if these disagree the ring has
  // paired rows from different ticks.
  const w = summarizeFunnelRing([
    xentry(1, "p", { scanned: 5, thinLiquidity: 3, admitted: 2 }, { admitted: 2, entered: 1, drifted: 1 }),
  ], "p")!
  assert.equal(w.counts.admitted, w.exec?.admitted)
})

test("summing preserves the property that buckets sum to scanned", () => {
  // The one invariant that makes the largest bucket trustworthy.
  const ring = [
    entry(1, "p", { scanned: 6, thinLiquidity: 4, tooNew: 1, admitted: 1 }),
    entry(2, "p", { scanned: 5, thinLiquidity: 3, parabolic: 2 }),
  ]
  const w = summarizeFunnelRing(ring, "p")!
  const stages = FUNNEL_KEYS.filter((k) => k !== "scanned")
    .reduce((n, k) => n + w.counts[k], 0)
  assert.equal(stages, w.counts.scanned)
})

/**
 * A tick where the entry stage never ran.
 *
 * Added 2026-08-13 after the ring went dark for an hour while the breaker was
 * cooling. Both writeFunnelRow calls sat inside `if (!killed && !breakerOpen
 * && !expired)`, so the window returned null — which reads identically to a
 * freshly deployed policy whose first tick has not landed. Two very different
 * situations, one indistinguishable symptom, during exactly the state the
 * funnel exists to explain.
 */
test("a blocked tick keeps the sum-to-scanned invariant", () => {
  // The invariant is what makes "largest bucket is the answer" true. A
  // blocked tick examined nobody, so the whole scan belongs to one bucket
  // rather than being silently dropped.
  const w = summarizeFunnelRing([
    entry(1, "p", { scanned: 251, blockedBreaker: 251 }),
  ], "p")!
  const stages = FUNNEL_KEYS.filter((k) => k !== "scanned")
    .reduce((n, k) => n + w.counts[k], 0)
  assert.equal(stages, w.counts.scanned)
  assert.equal(w.counts.blockedBreaker, 251)
  assert.equal(w.counts.admitted, 0)
})

test("the three blocked reasons stay distinct", () => {
  // They need opposite responses: un-kill, redeploy, or wait. A single
  // "blocked" bucket would have made the breaker look like an outage.
  const w = summarizeFunnelRing([
    entry(1, "p", { scanned: 10, blockedBreaker: 10 }),
    entry(2, "p", { scanned: 20, blockedKilled: 20 }),
    entry(3, "p", { scanned: 30, blockedExpired: 30 }),
  ], "p")!
  assert.equal(w.counts.blockedBreaker, 10)
  assert.equal(w.counts.blockedKilled, 20)
  assert.equal(w.counts.blockedExpired, 30)
  assert.equal(w.counts.scanned, 60)
})

test("rows packed before the blocked stages existed read as zero", () => {
  // The 120 rows already in the live ring were packed at 14 wide. Appending
  // must not relabel them: an old row's `admitted` must stay `admitted`.
  const old = [246, 0, 2, 202, 0, 15, 0, 0, 0, 21, 3, 0, 0, 3]
  const out = unpackFunnel(old)
  assert.equal(out.scanned, 246)
  assert.equal(out.admitted, 3)
  assert.equal(out.blockedKilled, 0)
  assert.equal(out.blockedExpired, 0)
  assert.equal(out.blockedBreaker, 0)
})

test("a blocked tick carries no execution half", () => {
  // Nothing was admitted, so claiming a zeroed exec funnel would assert the
  // entry loop ran and admitted nobody. It never ran at all.
  const w = summarizeFunnelRing([
    entry(1, "p", { scanned: 8, blockedBreaker: 8 }),
  ], "p")!
  assert.equal(w.exec, null)
  assert.equal(w.execUnmeasuredTicks, 1)
})

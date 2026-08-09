import test from "node:test"
import assert from "node:assert/strict"
import { separation, composeBody, READABLE_SAMPLE, type OriginStat } from "./alert.ts"

/**
 * These tests exist because the alert carries a CLAIM, not just data. The whole
 * reason the launchpad thesis had to be retracted the first time is that two
 * percentages were put side by side and read as a finding. The z-test is what
 * makes the second telling trustworthy, so it is the thing under test.
 */

const stat = (origin: string, labeled: number, died: number, e: number | null = null, r: number | null = null): OriginStat =>
  ({ origin, labeled, died, enteredRet: e, refusedRet: r })

test("reproduces the retracted claim at a real sample size", () => {
  // 11/100 versus 45/95: the shape of the original 10.8-vs-52.4 assertion.
  const s = separation(stat("launchpad", 100, 11), stat("profile", 95, 45))
  assert.ok(s)
  assert.equal(Number(s.launchpadRate.toFixed(4)), 0.11)
  assert.equal(Number(s.baselineRate.toFixed(4)), 0.4737)
  // Hand-computed: pooled p = 56/195, se = 0.06482, z = -5.611.
  assert.ok(Math.abs(s.z - -5.611) < 0.02, `z was ${s.z}`)
  assert.equal(s.significant, true)
})

test("does not invent an edge between two identical universes", () => {
  // The expensive direction to get wrong: a false positive here migrates
  // discovery onto a feed that is no better.
  const s = separation(stat("launchpad", 100, 44), stat("profile", 95, 45))
  assert.ok(s)
  assert.equal(s.significant, false)
})

test("refuses to answer on a thin sample instead of answering confidently", () => {
  assert.equal(separation(stat("launchpad", 12, 1), stat("profile", 95, 45)), null)
  assert.equal(separation(stat("launchpad", 100, 11), stat("profile", 4, 2)), null)
})

test("normal CDF is calibrated: a 1.96 sigma gap lands near p=0.05", () => {
  const s = separation(stat("a", 10000, 5000), stat("b", 10000, 5000 + Math.round(1.96 * Math.sqrt(2 * 2500))))
  assert.ok(s)
  assert.ok(Math.abs(s.pValue - 0.05) < 0.012, `p was ${s.pValue}`)
})

test("a healthier launchpad is reported without implying profit", () => {
  const { subject, text } = composeBody({
    launchpad: stat("launchpad", 100, 11, -12.4, -30.1),
    baseline: stat("profile", 95, 45, -29.1, -30.3),
    killed: false, breakerOpen: true, policyHash: "a09c405f",
  })
  assert.match(subject, /launchpad wins/)
  assert.match(text, /does NOT yet say the strategy makes money/)
  assert.match(text, /No capital at risk/)
  // A p-value of 2e-8 must not render as "0.0000", which reads as certainty.
  assert.doesNotMatch(text, /p = 0\.0000/)
})

test("an absent edge says so plainly rather than burying it", () => {
  const { subject, text } = composeBody({
    launchpad: stat("launchpad", 100, 44),
    baseline: stat("profile", 95, 45),
    killed: false, breakerOpen: false, policyHash: null,
  })
  assert.match(subject, /no edge found/)
  assert.match(text, /does NOT survive clean data/)
})

test("a worse launchpad is reported as a reason to stop", () => {
  const { subject, text } = composeBody({
    launchpad: stat("launchpad", 100, 80),
    baseline: stat("profile", 95, 45),
    killed: true, breakerOpen: false, policyHash: null,
  })
  assert.match(subject, /launchpad loses/)
  assert.match(text, /Do not migrate discovery/)
})

test("the trigger threshold is large enough to test against the baseline", () => {
  // separation() needs 30 a side; the trigger must clear that with room, or the
  // alert would fire and then decline to say anything.
  assert.ok(READABLE_SAMPLE >= 100)
})

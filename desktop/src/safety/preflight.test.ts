import { test } from "node:test"
import assert from "node:assert/strict"
import { preflight, liveArmed, MIN_SOL_RESERVE, type TradeIntent, type PreflightContext } from "../../../shared/preflight.ts"
import { PAPER_POLICY } from "../../../shared/policy.ts"

/**
 * This guard is the last thing between a policy and real money, so it is
 * tested for REFUSAL first. Every test below asserts that something is
 * blocked; only one asserts a trade is allowed, and it has to satisfy every
 * condition at once. That asymmetry is deliberate: the expensive failure is a
 * wrongly-permitted trade, never a wrongly-refused one.
 */

const LIVE = {
  ...PAPER_POLICY,
  product: "crowetrade-live" as const,
  expiresAt: "2099-01-01T00:00:00Z",
  signature: "sig",
  signer: "wallet",
}

const OK_INTENT: TradeIntent = {
  mint: "So11111111111111111111111111111111111111112",
  sizeSol: 0.05,
  spentTodaySol: 0,
  openPositions: 0,
  impactPct: 0.4,
  simulationOk: true,
  walletBalanceSol: 1,
}

const OK_CTX: PreflightContext = {
  policy: LIVE,
  nowMs: Date.parse("2026-01-01T00:00:00Z"),
  killed: false,
  liveArmed: true,
  signatureVerified: true,
}

test("a fully compliant trade is allowed", () => {
  assert.equal(preflight(OK_INTENT, OK_CTX), null)
})

test("the kill switch dominates every other condition", () => {
  const r = preflight(OK_INTENT, { ...OK_CTX, killed: true })
  assert.match(r ?? "", /kill switch/)
})

test("live trading must be armed explicitly, never inherited", () => {
  assert.match(preflight(OK_INTENT, { ...OK_CTX, liveArmed: false }) ?? "", /not armed/)
})

test("a PAPER envelope can never move real funds", () => {
  const r = preflight(OK_INTENT, { ...OK_CTX, policy: { ...LIVE, product: "crowetrade-paper" } })
  assert.match(r ?? "", /not a live envelope/)
})

test("expired consent refuses, and an unreadable date counts as expired", () => {
  assert.match(
    preflight(OK_INTENT, { ...OK_CTX, policy: { ...LIVE, expiresAt: "2020-01-01T00:00:00Z" } }) ?? "",
    /expired/,
  )
  assert.match(
    preflight(OK_INTENT, { ...OK_CTX, policy: { ...LIVE, expiresAt: "not a date" } }) ?? "",
    /unreadable/,
  )
})

test("a PRESENT but unverified signature refuses: present is not valid", () => {
  // The defect this closes: any non-empty string used to satisfy the check.
  assert.match(preflight(OK_INTENT, { ...OK_CTX, signatureVerified: false }) ?? "", /did not verify/)
  // Omitted entirely must also refuse, so a caller that forgets fails closed.
  const { signatureVerified: _drop, ...noFlag } = OK_CTX
  assert.match(preflight(OK_INTENT, noFlag) ?? "", /did not verify/)
})

test("an unsigned live envelope refuses: nobody consented to these limits", () => {
  assert.match(
    preflight(OK_INTENT, { ...OK_CTX, policy: { ...LIVE, signature: null } }) ?? "",
    /unsigned/,
  )
  assert.match(
    preflight(OK_INTENT, { ...OK_CTX, policy: { ...LIVE, signer: null } }) ?? "",
    /unsigned/,
  )
})

test("per-trade cap is enforced at the boundary", () => {
  const at = { ...OK_INTENT, sizeSol: LIVE.perTradeCapSol }
  assert.equal(preflight({ ...at, walletBalanceSol: 99 }, OK_CTX), null, "exactly at the cap is allowed")
  const over = { ...OK_INTENT, sizeSol: LIVE.perTradeCapSol + 0.0001, walletBalanceSol: 99 }
  assert.match(preflight(over, OK_CTX) ?? "", /per-trade cap/)
})

test("the daily cap counts what is already spent, not just this trade", () => {
  const intent = {
    ...OK_INTENT,
    sizeSol: 0.5,
    spentTodaySol: LIVE.dailyCapSol - 0.4,
    walletBalanceSol: 99,
  }
  assert.match(preflight(intent, OK_CTX) ?? "", /over the .* cap/)
})

test("position slots are enforced", () => {
  const intent = { ...OK_INTENT, openPositions: LIVE.maxOpenPositions }
  assert.match(preflight(intent, OK_CTX) ?? "", /already holding/)
})

test("the wallet must keep enough to EXIT, not merely to enter", () => {
  // Balance covers the trade exactly, leaving nothing for the sell's fee.
  const intent = { ...OK_INTENT, sizeSol: 0.5, walletBalanceSol: 0.5 }
  assert.match(preflight(intent, OK_CTX) ?? "", /exit reserve/)
  const ok = { ...OK_INTENT, sizeSol: 0.5, walletBalanceSol: 0.5 + MIN_SOL_RESERVE }
  assert.equal(preflight(ok, OK_CTX), null)
})

test("a trade that failed simulation is refused", () => {
  assert.match(preflight({ ...OK_INTENT, simulationOk: false }, OK_CTX) ?? "", /simulation/)
})

test("impact over the cost hurdle is refused", () => {
  const intent = { ...OK_INTENT, impactPct: LIVE.entry.maxEntryImpactPct + 0.01 }
  assert.match(preflight(intent, OK_CTX) ?? "", /impact/)
})

test("non-finite numbers refuse rather than compare as false", () => {
  // NaN fails every > comparison, so a naive guard would PASS it through.
  for (const bad of [NaN, Infinity]) {
    assert.notEqual(preflight({ ...OK_INTENT, sizeSol: bad }, OK_CTX), null, `sizeSol ${bad}`)
    assert.notEqual(preflight({ ...OK_INTENT, impactPct: bad }, OK_CTX), null, `impactPct ${bad}`)
    assert.notEqual(
      preflight({ ...OK_INTENT, walletBalanceSol: bad === Infinity ? NaN : bad }, OK_CTX),
      null,
      `balance ${bad}`,
    )
  }
  assert.notEqual(preflight({ ...OK_INTENT, sizeSol: 0 }, OK_CTX), null, "zero size")
  assert.notEqual(preflight({ ...OK_INTENT, sizeSol: -1 }, OK_CTX), null, "negative size")
})

test("refusal names the FIRST problem, so the operator fixes the right thing", () => {
  // Killed AND expired AND over cap: the kill switch is what it should say.
  const r = preflight(
    { ...OK_INTENT, sizeSol: 999 },
    { ...OK_CTX, killed: true, policy: { ...LIVE, expiresAt: "2020-01-01T00:00:00Z" } },
  )
  assert.match(r ?? "", /kill switch/)
})

// ── Arming ────────────────────────────────────────────────────────────────
//
// The environment half of the lock. The DEFAULT must be inert: a missing
// flag, a missing key, an empty key, or a flag set to anything but the exact
// string "1" all leave the live path disarmed. Anything looser and a stray
// environment variable spends money.

test("live is disarmed by default: an empty environment arms nothing", () => {
  assert.equal(liveArmed({}), false)
})

test("flag and key are BOTH required", () => {
  assert.equal(liveArmed({ LIVE_TRADING: "1" }), false)
  assert.equal(liveArmed({ LIVE_TRADING: "1", TRADING_KEYPAIR: "" }), false)
  assert.equal(liveArmed({ TRADING_KEYPAIR: "abc" }), false)
  assert.equal(liveArmed({ LIVE_TRADING: "1", TRADING_KEYPAIR: "abc" }), true)
})

test("only the exact string '1' arms, so a truthy typo stays inert", () => {
  for (const v of ["true", "yes", "0", "on", " 1", "1 ", "TRUE"]) {
    assert.equal(liveArmed({ LIVE_TRADING: v, TRADING_KEYPAIR: "abc" }), false, `LIVE_TRADING=${v}`)
  }
})

test("a non-string key does not arm", () => {
  assert.equal(liveArmed({ LIVE_TRADING: "1", TRADING_KEYPAIR: 123 }), false)
  assert.equal(liveArmed({ LIVE_TRADING: "1", TRADING_KEYPAIR: null }), false)
})

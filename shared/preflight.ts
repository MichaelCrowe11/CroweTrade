/**
 * The last thing that runs before real money moves.
 *
 * Every check here also exists somewhere upstream. That duplication is the
 * point. Upstream checks decide WHETHER TO WANT a trade and live in code that
 * changes often; this decides whether one is ALLOWED and is meant to change
 * almost never. If a refactor upstream ever drops a cap, this still refuses.
 *
 * Three properties make it worth trusting:
 *
 * 1. PURE. No clock, no network, no database — every input is an argument, so
 *    every branch is reachable from a test. A guard you cannot exhaustively
 *    test is a guard you are hoping about.
 * 2. DENY BY DEFAULT. It returns a reason string on refusal and null only when
 *    every check passed. An unknown state refuses; there is no path where
 *    missing information means proceed.
 * 3. IT DOES NOT SIGN. It returns a verdict. The caller signs. Keeping the
 *    decision separate from the capability means this file can be read and
 *    argued about without reading the crypto.
 *
 * The order matters and is deliberate: kill switch first because it must
 * dominate everything, then consent (expiry/signature), then caps, then the
 * trade-specific facts. A refusal names the FIRST thing wrong, which is the
 * one the operator should fix.
 */

import type { PolicyEnvelope } from "./policy.js"

/**
 * Is the live path armed at all?
 *
 * Lives here rather than beside the sender because it is safety-critical and
 * must be testable, and the sender imports the RPC helper at runtime, which
 * `node --test --experimental-strip-types` cannot resolve through a `.js`
 * specifier. That constraint already shaped trajectory.ts, standing.ts and
 * events.ts in this codebase: anything that must be tested carries no runtime
 * imports.
 *
 * Deliberately strict. Only the exact string "1" arms, so a truthy-looking
 * typo ("true", "yes", " 1") leaves the path inert, and a missing or empty
 * key disarms regardless of the flag. The expensive direction of error here
 * is arming by accident.
 */
export function liveArmed(env: Record<string, unknown>): boolean {
  const flag = env["LIVE_TRADING"]
  const key = env["TRADING_KEYPAIR"]
  return flag === "1" && typeof key === "string" && key.length > 0
}

export interface TradeIntent {
  mint: string
  /** Size of THIS trade in SOL. */
  sizeSol: number
  /** SOL already committed today, excluding this trade. */
  spentTodaySol: number
  /** Positions currently open, excluding this trade. */
  openPositions: number
  /** Quoted one-way price impact, percent. */
  impactPct: number
  /** Did the transaction simulate successfully against live state? */
  simulationOk: boolean
  /** Wallet balance in SOL, so a trade cannot exceed what exists. */
  walletBalanceSol: number
}

export interface PreflightContext {
  policy: PolicyEnvelope
  /** Wall clock, injected so expiry is testable without waiting. */
  nowMs: number
  killed: boolean
  /** True only when the operator has explicitly armed live trading. */
  liveArmed: boolean
}

/**
 * Fee headroom.
 *
 * A trade that spends the entire balance leaves nothing for the network fee of
 * the trade itself, let alone the EXIT. Being unable to sell is the worst
 * failure this system has, worse than any single loss, so the wallet must
 * always retain enough to get out.
 */
export const MIN_SOL_RESERVE = 0.01

/**
 * Returns null when the trade may proceed, or a human-readable reason it may
 * not. The string is shown to the operator and logged, so it names the
 * specific limit rather than saying "policy violation".
 */
export function preflight(intent: TradeIntent, ctx: PreflightContext): string | null {
  const { policy } = ctx

  // 1. Kill switch dominates. It is the one control that must work even when
  //    everything else is misconfigured.
  if (ctx.killed) return "kill switch is on"

  // 2. Live trading is opt-in, never inherited. A paper envelope must not be
  //    able to move real funds just because a key happened to be present.
  if (!ctx.liveArmed) return "live trading is not armed"
  if (policy.product !== "crowetrade-live") {
    return `policy is ${policy.product}, not a live envelope`
  }

  // 3. Consent must be current. An unparseable date counts as expired:
  //    unknown never authorizes anything, and least of all spending.
  const expiresMs = Date.parse(policy.expiresAt)
  if (Number.isNaN(expiresMs)) return "policy expiry is unreadable"
  if (ctx.nowMs >= expiresMs) return `policy expired at ${policy.expiresAt}`

  // A live envelope must carry the wallet's signature over its own hash. This
  // is what makes a fill traceable to a specific consent rather than to a
  // config file someone edited.
  if (!policy.signature || !policy.signer) {
    return "live envelope is unsigned: no wallet has consented to these limits"
  }

  // 4. Caps. Each is checked against THIS trade plus what is already committed.
  if (!Number.isFinite(intent.sizeSol) || intent.sizeSol <= 0) {
    return "trade size is not a positive number"
  }
  if (intent.sizeSol > policy.perTradeCapSol) {
    return `trade ${intent.sizeSol} SOL exceeds per-trade cap ${policy.perTradeCapSol}`
  }
  const dayTotal = intent.spentTodaySol + intent.sizeSol
  if (dayTotal > policy.dailyCapSol) {
    return `would spend ${dayTotal.toFixed(3)} SOL today, over the ${policy.dailyCapSol} cap`
  }
  if (intent.openPositions >= policy.maxOpenPositions) {
    return `already holding ${intent.openPositions} of ${policy.maxOpenPositions} positions`
  }

  // 5. The wallet must be able to afford the trade AND still exit.
  if (!Number.isFinite(intent.walletBalanceSol)) return "wallet balance is unknown"
  if (intent.walletBalanceSol < intent.sizeSol + MIN_SOL_RESERVE) {
    return (
      `balance ${intent.walletBalanceSol.toFixed(4)} SOL cannot cover ` +
      `${intent.sizeSol} plus the ${MIN_SOL_RESERVE} exit reserve`
    )
  }

  // 6. Trade-specific facts last. Simulation failing means the transaction
  //    would have failed on chain, so sending it only buys a fee.
  if (!intent.simulationOk) return "transaction failed simulation"
  if (!Number.isFinite(intent.impactPct)) return "price impact is unknown"
  if (intent.impactPct > policy.entry.maxEntryImpactPct) {
    return `impact ${intent.impactPct.toFixed(2)}% over the ${policy.entry.maxEntryImpactPct}% hurdle`
  }

  return null
}

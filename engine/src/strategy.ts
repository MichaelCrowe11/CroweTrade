/**
 * Strategy v0: the safety-gate survivorship test.
 *
 * Deliberately the simplest honest strategy: enter every token the policy
 * envelope permits, at envelope size, and let the exit rules work. It carries
 * no alpha model on purpose. What it measures is whether the safety gates plus
 * disciplined exits alone clear zero on this market, which is the baseline
 * every real strategy must beat and the first number a funding conversation
 * will ask for. If the gates cannot beat "buy everything that passed them",
 * nothing built on top of them deserves capital.
 */

import type { Candidate } from "../../shared/dexscreener.js"
import { evaluateGates, combineVerdict, type Verdict } from "../../shared/gates.js"
import type { PolicyEnvelope } from "../../shared/policy.js"

export interface OpenPosition {
  id: string
  mint: string
  symbol: string
  entryPriceUsd: number
  sizeSol: number
  sizeUsd: number
  tokenAmount: number
  openedAt: number
  policyHash: string
  verdictAtEntry: Verdict
}

export interface EntryDecision {
  candidate: Candidate
  verdict: Verdict
  sizeSol: number
}

export interface ExitDecision {
  position: OpenPosition
  reason: "take-profit" | "stop-loss" | "time-stop" | "safety-exit" | "veto"
  exitPriceUsd: number
}

/**
 * Slippage model, stated plainly: paper fills are worsened by trade size as a
 * share of pool liquidity, capped at 5%. Real fills will be worse in fast
 * markets; TCA against real fills later replaces this constant with measured
 * truth. Overstating our own fills is the one lie the track record cannot
 * survive, so the model errs pessimistic.
 */
export function slippageBps(tradeUsd: number, liquidityUsd: number): number {
  if (liquidityUsd <= 0) return 500
  return Math.min(500, Math.round((tradeUsd / liquidityUsd) * 10_000))
}

const VERDICT_RANK: Record<Verdict, number> = {
  clear: 3,
  caution: 2,
  "insufficient-data": 1,
  blocked: 0,
}

export { trajectoryConfirms, type Trajectory } from "../../shared/trajectory.js"
import { trajectoryConfirms, type Trajectory } from "../../shared/trajectory.js"

export function decideEntries(
  candidates: Candidate[],
  open: OpenPosition[],
  spentTodaySol: number,
  solUsd: number,
  policy: PolicyEnvelope,
  now: number,
  trajectories: Map<string, Trajectory>,
): EntryDecision[] {
  const held = new Set(open.map((p) => p.mint))
  const minRank = VERDICT_RANK[policy.entry.minVerdict]
  const out: EntryDecision[] = []

  let budgetSol = Math.max(0, policy.dailyCapSol - spentTodaySol)
  let slots = Math.max(0, policy.maxOpenPositions - open.length)

  for (const c of candidates) {
    if (slots === 0 || budgetSol < policy.perTradeCapSol) break
    if (held.has(c.mint)) continue
    if (c.priceUsd === null || c.priceUsd <= 0) continue
    if (c.liquidityUsd === null || c.liquidityUsd < policy.entry.minLiquidityUsd) continue
    if (c.createdAt === null) continue
    const ageMin = (now - c.createdAt) / 60_000
    if (ageMin > policy.entry.maxTokenAgeMinutes) continue
    if (ageMin < policy.entry.minTokenAgeMinutes) continue

    // Momentum-exhaustion filter. Unknown change is treated as disqualifying
    // rather than neutral: we cannot tell a quiet token from a parabolic one,
    // and the parabolic case is the one that has been losing money.
    if (c.changeH1 === null || c.changeH1 > policy.entry.maxChangeH1Pct) continue

    // Paid promotion is a distribution event by definition.
    if (policy.entry.excludeBoosted && c.origin === "boost") continue

    // Our own tape must agree before a listing gets our capital.
    if (!trajectoryConfirms(trajectories.get(c.mint), policy.entry.minObservedTicks)) continue

    const verdict = combineVerdict(evaluateGates(c.snapshot))
    if (VERDICT_RANK[verdict] < minRank) continue

    // Clear tokens get full envelope size, caution gets half: the policy's
    // "buy blind small, never blind big" made arithmetic.
    const sizeSol = verdict === "clear" ? policy.perTradeCapSol : policy.perTradeCapSol / 2
    out.push({ candidate: c, verdict, sizeSol })
    budgetSol -= sizeSol
    slots -= 1
  }
  return out
}

export function decideExits(
  open: OpenPosition[],
  prices: Map<string, { priceUsd: number; verdict: Verdict }>,
  policy: PolicyEnvelope,
  now: number,
): ExitDecision[] {
  const out: ExitDecision[] = []
  for (const p of open) {
    const cur = prices.get(p.mint)
    if (!cur) continue // no fresh price this tick; hold and try next tick
    const pnlPct = ((cur.priceUsd - p.entryPriceUsd) / p.entryPriceUsd) * 100
    const heldMin = (now - p.openedAt) / 60_000

    // Order matters: a token turning BLOCKED exits first regardless of PnL —
    // a live rug signal outranks a profit target.
    if (cur.verdict === "blocked") {
      out.push({ position: p, reason: "safety-exit", exitPriceUsd: cur.priceUsd })
    } else if (pnlPct >= policy.exit.takeProfitPct) {
      out.push({ position: p, reason: "take-profit", exitPriceUsd: cur.priceUsd })
    } else if (pnlPct <= -policy.exit.stopLossPct) {
      out.push({ position: p, reason: "stop-loss", exitPriceUsd: cur.priceUsd })
    } else if (heldMin >= policy.exit.timeStopMinutes) {
      out.push({ position: p, reason: "time-stop", exitPriceUsd: cur.priceUsd })
    }
  }
  return out
}

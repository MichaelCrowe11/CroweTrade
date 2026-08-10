/**
 * Survivability gates.
 *
 * These run BEFORE any statistical sizing, and they are vetoes rather than
 * terms in a score. The reason is a real asymmetry: the existing CroweTrade risk
 * stack sizes positions by variance, and variance cannot see a rug. A token that
 * goes to zero in one block has unremarkable historical sigma right up until the
 * moment it does not. No amount of expected edge should be allowed to size into
 * a mint whose deployer can still print supply.
 *
 * Every gate is computed from OUR decoded state, never from a third party's
 * boolean. An aggregator that tells you "safe: true" is telling you the output
 * of a heuristic you cannot inspect, version, or backtest.
 */

import type { Mint, BlockTime } from "./types.js"

/**
 * Three states, deliberately.
 *
 * "unknown" is not a failure and must never be rendered as one, nor collapsed
 * into a passing default. Early in a token's life most gates are legitimately
 * unknown, and a system that reports unknown as either pass or fail is lying in
 * one direction or the other. The UI gives it its own visual state.
 */
export type GateState = "pass" | "fail" | "unknown"

export interface GateResult {
  id: GateId
  label: string
  state: GateState
  /** Human-readable basis for the verdict. Shown on the panel. */
  detail: string
  /** Whether a fail here is survivable-but-bad vs immediately disqualifying. */
  severity: "critical" | "elevated"
}

export type GateId =
  | "mint-authority"
  | "freeze-authority"
  | "lp-locked"
  | "holder-concentration"
  | "liquidity-depth"
  | "deployer-history"

/** Point-in-time snapshot. Every field carries the blockTime it was true at. */
export interface TokenSnapshot {
  mint: Mint
  asOf: BlockTime
  launchedAt: BlockTime | null
  mintAuthority: string | null | undefined
  freezeAuthority: string | null | undefined
  lpLockedBps: number | undefined
  /** Share of supply held by the largest non-pool holder, 0..1. */
  topHolderShare: number | undefined
  solReserveLamports: bigint | undefined
  /** Prior mints by this deployer and how many rugged. undefined = not indexed. */
  deployerPriorMints: number | undefined
  deployerPriorRugs: number | undefined
}

const LAMPORTS_PER_SOL = 1_000_000_000n

/** Minimum pool depth worth considering, in SOL. Below this, exit slips badly. */
const MIN_LIQUIDITY_SOL = 5n

function gateMintAuthority(s: TokenSnapshot): GateResult {
  const base = { id: "mint-authority" as const, label: "MINT AUTHORITY", severity: "critical" as const }
  if (s.mintAuthority === undefined) {
    return { ...base, state: "unknown", detail: "chain read pending" }
  }
  return s.mintAuthority === null
    ? { ...base, state: "pass", detail: "revoked, supply is fixed" }
    : { ...base, state: "fail", detail: `retained by ${s.mintAuthority.slice(0, 8)}` }
}

function gateFreezeAuthority(s: TokenSnapshot): GateResult {
  const base = { id: "freeze-authority" as const, label: "FREEZE AUTHORITY", severity: "critical" as const }
  if (s.freezeAuthority === undefined) {
    return { ...base, state: "unknown", detail: "chain read pending" }
  }
  return s.freezeAuthority === null
    ? { ...base, state: "pass", detail: "revoked, cannot be frozen" }
    : { ...base, state: "fail", detail: "can freeze your account" }
}

function gateLpLocked(s: TokenSnapshot): GateResult {
  const base = { id: "lp-locked" as const, label: "LP LOCKED", severity: "critical" as const }
  // Pre-graduation pump.fun tokens have no separate LP to lock -- the bonding
  // curve program holds the liquidity -- so this reads unknown for most fresh
  // launches and that is a fact about the venue, not a gap in our data.
  if (s.lpLockedBps === undefined) return { ...base, state: "unknown", detail: "no LP to check yet" }
  if (s.lpLockedBps >= 9_900) return { ...base, state: "pass", detail: "burned, cannot be pulled" }
  return { ...base, state: "fail", detail: `only ${(s.lpLockedBps / 100).toFixed(1)}% locked` }
}

function gateHolderConcentration(s: TokenSnapshot): GateResult {
  const base = { id: "holder-concentration" as const, label: "HOLDER SPREAD", severity: "elevated" as const }
  if (s.topHolderShare === undefined) return { ...base, state: "unknown", detail: "not yet indexed" }
  const pct = (s.topHolderShare * 100).toFixed(1)
  return s.topHolderShare <= 0.15
    ? { ...base, state: "pass", detail: `top holder ${pct}%, well spread` }
    : { ...base, state: "fail", detail: `top holder ${pct}%` }
}

/** Lamports are integers and stay integers; only the DISPLAY converts to float.
 *  Dividing bigints truncates, which turned 9.9 SOL into "9" in the first cut. */
function formatSol(lamports: bigint): string {
  const sol = Number(lamports) / Number(LAMPORTS_PER_SOL)
  if (sol >= 1_000) return `${Math.round(sol).toLocaleString()} SOL`
  return `${sol.toFixed(sol >= 10 ? 1 : 2)} SOL`
}

function gateLiquidityDepth(s: TokenSnapshot): GateResult {
  const base = { id: "liquidity-depth" as const, label: "LIQUIDITY DEPTH", severity: "elevated" as const }
  if (s.solReserveLamports === undefined) return { ...base, state: "unknown", detail: "depth unmeasured" }
  const shown = formatSol(s.solReserveLamports)
  return s.solReserveLamports >= MIN_LIQUIDITY_SOL * LAMPORTS_PER_SOL
    ? { ...base, state: "pass", detail: shown }
    : { ...base, state: "fail", detail: `${shown}, exit will slip` }
}

function gateDeployerHistory(s: TokenSnapshot): GateResult {
  const base = { id: "deployer-history" as const, label: "DEPLOYER HISTORY", severity: "elevated" as const }
  if (s.deployerPriorMints === undefined || s.deployerPriorRugs === undefined) {
    return { ...base, state: "unknown", detail: "corpus not yet built" }
  }
  if (s.deployerPriorMints === 0) return { ...base, state: "unknown", detail: "first mint by this wallet" }
  return s.deployerPriorRugs === 0
    ? { ...base, state: "pass", detail: `${s.deployerPriorMints} prior mints, none rugged` }
    : { ...base, state: "fail", detail: `${s.deployerPriorRugs}/${s.deployerPriorMints} rugged` }
}

/** Evaluates every gate against a point-in-time snapshot. Order is display order. */
export function evaluateGates(s: TokenSnapshot): GateResult[] {
  return [
    gateMintAuthority(s),
    gateFreezeAuthority(s),
    gateLpLocked(s),
    gateHolderConcentration(s),
    gateLiquidityDepth(s),
    gateDeployerHistory(s),
  ]
}

export type Verdict = "clear" | "caution" | "blocked" | "insufficient-data"

/**
 * Combines individual gate results into a single go/no-go verdict.
 *
 * DEFAULT POLICY, written 2026-08-08 with Michael's go-ahead. Every constant
 * here is a risk-appetite dial, not a law of nature; tune freely. The shape of
 * the policy, though, is deliberate and should survive tuning:
 *
 *  1. A CONFIRMED critical fail blocks, full stop. Mint authority retained,
 *     freeze authority live, LP unlocked: each is a mechanism for losing the
 *     entire position in one block. No signal strength overrides these.
 *
 *  2. A critical gate still UNKNOWN does not block on its own, because a
 *     thirty-second-old token has everything unknown and blocking on unknown
 *     forfeits the entire early edge. Instead unknowns cap the verdict at
 *     "caution", and the sizing layer downstream must treat caution as a
 *     fraction of normal size. You may buy blind small; you may not buy blind
 *     big. That is the whole policy in one sentence.
 *
 *  3. Elevated fails (thin liquidity, concentrated holders, deployer with
 *     rugs) individually degrade to caution; two or more together block.
 *     One wound is survivable, a pattern is a verdict.
 *
 *  4. If nothing at all is known, the honest output is insufficient-data,
 *     which the UI renders as its own state and the trader treats as "watch,
 *     do not touch".
 */
export function combineVerdict(gates: GateResult[]): Verdict {
  const criticals = gates.filter((g) => g.severity === "critical")
  const elevated = gates.filter((g) => g.severity === "elevated")

  if (criticals.some((g) => g.state === "fail")) return "blocked"

  const elevatedFails = elevated.filter((g) => g.state === "fail").length
  if (elevatedFails >= 2) return "blocked"

  const known = gates.filter((g) => g.state !== "unknown").length
  if (known === 0) return "insufficient-data"

  const criticalUnknowns = criticals.filter((g) => g.state === "unknown").length
  if (criticalUnknowns > 0 || elevatedFails === 1) return "caution"

  return "clear"
}

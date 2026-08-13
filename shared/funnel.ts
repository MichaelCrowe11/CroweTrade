/**
 * Where candidates die inside decideEntries.
 *
 * Built because three separate blockers hid here in one evening, each costing
 * hours: a momentum filter refusing structurally-absent data, a liquidity
 * floor calibrated for a different venue, and a trade size the venue could not
 * absorb. Every time the engine reported `{scanned: N, entered: 0}` with no
 * entry_skipped events — because entry_skipped is emitted at the QUOTE, and
 * nothing was reaching it.
 *
 * The `decisions` table cannot answer this. It snapshots once per mint at
 * first sight and describes the calibration corpus, not the live entry path,
 * and reading it as a funnel produced a wrong diagnosis twice tonight.
 *
 * Counts are per tick and cumulative down the chain: a candidate is counted
 * at exactly the first stage that rejects it, so the numbers sum to
 * `scanned` and the largest one is the answer.
 */
export interface FunnelCounts {
  scanned: number
  heldAlready: number
  noPrice: number
  thinLiquidity: number
  noCreatedAt: number
  tooOld: number
  tooNew: number
  parabolic: number
  originNotAllowed: number
  trajectoryUnconfirmed: number
  verdictTooLow: number
  modelProbTooLow: number
  budgetOrSlotsExhausted: number
  admitted: number
}

export function emptyFunnel(): FunnelCounts {
  return {
    scanned: 0, heldAlready: 0, noPrice: 0, thinLiquidity: 0, noCreatedAt: 0,
    tooOld: 0, tooNew: 0, parabolic: 0, originNotAllowed: 0,
    trajectoryUnconfirmed: 0, verdictTooLow: 0, modelProbTooLow: 0,
    budgetOrSlotsExhausted: 0, admitted: 0,
  }
}

/**
 * Fixed key order for the packed ring. Append-only: adding a stage at the END
 * keeps older packed rows readable, and `unpackFunnel` zero-fills the tail.
 * Reordering these silently relabels history, which is the exact class of lie
 * this file exists to prevent.
 */
export const FUNNEL_KEYS = [
  "scanned", "heldAlready", "noPrice", "thinLiquidity", "noCreatedAt",
  "tooOld", "tooNew", "parabolic", "originNotAllowed",
  "trajectoryUnconfirmed", "verdictTooLow", "modelProbTooLow",
  "budgetOrSlotsExhausted", "admitted",
] as const satisfies readonly (keyof FunnelCounts)[]

export function packFunnel(f: FunnelCounts): number[] {
  return FUNNEL_KEYS.map((k) => f[k])
}

export function unpackFunnel(a: readonly number[]): FunnelCounts {
  const out = emptyFunnel()
  FUNNEL_KEYS.forEach((k, i) => { out[k] = a[i] ?? 0 })
  return out
}

/**
 * Where an ADMITTED candidate dies between decideEntries and the book.
 *
 * The selection funnel ends at `admitted`, and everything after it — the
 * Jupiter quote, the cost hurdle, the first-sight drift check, the simulation
 * dry run, and on the live path the wallet and the guard — had no tally at
 * all. Two of those stages did not even emit an event: a candidate whose price
 * or liquidity had gone null since the scan, and a fill that produced zero
 * tokens, both fell out on a bare `continue`.
 *
 * That is the same blind spot the selection funnel was built to close, one
 * stage further down, and it became load-bearing the moment the model gate was
 * disarmed on 2026-08-11: the first candidate admitted in two days died at the
 * simulation gate, and a single event was the only trace.
 *
 * Buckets are exclusive and sum to `admitted`, so `entered` plus every refusal
 * accounts for everything decideEntries handed over.
 */
export interface ExecFunnelCounts {
  /** Handed over by decideEntries; the denominator for this stage. */
  admitted: number
  /** Price or liquidity went null between the scan and the quote. */
  missingPrice: number
  /** Jupiter had no route; a token we could not have bought at all. */
  noRoute: number
  impactAboveHurdle: number
  drifted: number
  simulationFailed: number
  /** Live path only: wallet balance unreadable. */
  walletUnreadable: number
  /** Live path only: the pre-send guard or the swap refused. */
  liveRefused: number
  /** Confirmed or quoted, but no tokens arrived. Never write a position we
   *  would then try to sell. */
  noTokens: number
  entered: number
}

export function emptyExecFunnel(): ExecFunnelCounts {
  return {
    admitted: 0, missingPrice: 0, noRoute: 0, impactAboveHurdle: 0, drifted: 0,
    simulationFailed: 0, walletUnreadable: 0, liveRefused: 0, noTokens: 0,
    entered: 0,
  }
}

/** Append-only, for the same reason as FUNNEL_KEYS. */
export const EXEC_FUNNEL_KEYS = [
  "admitted", "missingPrice", "noRoute", "impactAboveHurdle", "drifted",
  "simulationFailed", "walletUnreadable", "liveRefused", "noTokens", "entered",
] as const satisfies readonly (keyof ExecFunnelCounts)[]

export function packExecFunnel(f: ExecFunnelCounts): number[] {
  return EXEC_FUNNEL_KEYS.map((k) => f[k])
}

export function unpackExecFunnel(a: readonly number[]): ExecFunnelCounts {
  const out = emptyExecFunnel()
  EXEC_FUNNEL_KEYS.forEach((k, i) => { out[k] = a[i] ?? 0 })
  return out
}

/** One tick's funnels, tagged with the policy that produced them. */
export interface FunnelRingEntry {
  /** Tick timestamp. */
  at: number
  /** Policy hash in force for this tick. */
  h: string
  /** Selection counts packed in FUNNEL_KEYS order. */
  c: number[]
  /**
   * Execution counts packed in EXEC_FUNNEL_KEYS order. Optional because rows
   * written before this stage was instrumented have none, and zero-filling
   * them would claim those ticks admitted nothing rather than admitting we
   * did not look.
   */
  x?: number[]
}

/**
 * A rolling window of per-tick funnels.
 *
 * The per-tick funnel was built on the reasoning that "a history of funnels
 * would be noise". That was true while one bucket held 99% of `scanned`. It is
 * false for every stage BELOW the dominant one: measured 2026-08-11, only one
 * to three candidates per tick survive the liquidity floor, so the age, drift,
 * verdict and model buckets each see a sample of one or two. A single tick
 * cannot distinguish "this gate refuses everything" from "this gate saw
 * nothing", and those need opposite fixes.
 */
export function pushFunnelRing(
  ring: readonly FunnelRingEntry[],
  entry: FunnelRingEntry,
  cap: number,
): FunnelRingEntry[] {
  const next = [...ring, entry]
  return next.length > cap ? next.slice(next.length - cap) : next
}

/**
 * Append a tick's row, or replace the last one when it is the SAME tick.
 *
 * A tick writes twice: the selection half before the entry loop, then both
 * halves after it. Appending both would double-count every completed tick and
 * halve the window's real span. Keyed on `at` rather than on position because
 * that is what actually identifies a tick.
 */
export function upsertFunnelRow(
  ring: readonly FunnelRingEntry[],
  row: FunnelRingEntry,
  cap: number,
): FunnelRingEntry[] {
  const last = ring[ring.length - 1]
  if (last && last.at === row.at) {
    const next = ring.slice()
    next[next.length - 1] = row
    return next
  }
  return pushFunnelRing(ring, row, cap)
}

export interface FunnelWindow {
  /** Ticks in the window that ran under `policyHash`. */
  ticks: number
  /** Timestamp of the oldest counted tick. */
  since: number
  /** Timestamp of the newest counted tick. */
  until: number
  /**
   * True when the ring also holds ticks from OTHER policy hashes, which are
   * excluded from `counts`. It means the window is shorter than the ring, not
   * that anything is wrong — but a reader comparing this to a wall-clock span
   * would otherwise be misled.
   */
  mixed: boolean
  policyHash: string
  counts: FunnelCounts
  /**
   * The post-admission chain over the same ticks, or null when no counted tick
   * carried it. Null is "not measured here", which is a different statement
   * from a zeroed struct claiming nothing was admitted.
   */
  exec: ExecFunnelCounts | null
  /** Counted ticks written before the execution stages were instrumented. */
  execUnmeasuredTicks: number
}

/**
 * Sum the ring, counting ONLY ticks that ran under the current policy.
 *
 * Segmenting by policy hash is the same discipline the external audit forced
 * on the trade cohorts: a number pooled across parameter regimes describes no
 * regime. Here it matters more, not less — the whole point of the window is to
 * decide whether a gate is misconfigured, and stale ticks from before it was
 * changed argue for the change that was already made.
 */
export function summarizeFunnelRing(
  ring: readonly FunnelRingEntry[],
  policyHash: string,
): FunnelWindow | null {
  const mine = ring.filter((e) => e.h === policyHash)
  if (mine.length === 0) return null
  const counts = emptyFunnel()
  for (const e of mine) {
    const c = unpackFunnel(e.c)
    for (const k of FUNNEL_KEYS) counts[k] += c[k]
  }
  const measured = mine.filter((e) => Array.isArray(e.x))
  let exec: ExecFunnelCounts | null = null
  if (measured.length > 0) {
    exec = emptyExecFunnel()
    for (const e of measured) {
      const x = unpackExecFunnel(e.x!)
      for (const k of EXEC_FUNNEL_KEYS) exec[k] += x[k]
    }
  }
  return {
    ticks: mine.length,
    since: mine[0]!.at,
    until: mine[mine.length - 1]!.at,
    mixed: mine.length !== ring.length,
    policyHash,
    counts,
    exec,
    execUnmeasuredTicks: mine.length - measured.length,
  }
}

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

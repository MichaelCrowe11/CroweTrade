/**
 * Summary statistics for counterfactual sweeps.
 *
 * Split out of ledger.ts because this is where a sweep lies. The engine's
 * sweeps decide whether to loosen a gate that spends money, and the arithmetic
 * that produced the recommendation has to be testable without a Durable
 * Object. No runtime imports, so it runs under strip-types.
 *
 * The statistic that matters here is `meanExBestPct`, and the reason is the
 * shape of this market rather than a general preference for robustness.
 *
 * Launchpad returns are a lottery: the median 30-minute forward return is
 * about -52% and the winners run past +2,000%. Two consequences follow, and
 * they pull in opposite directions.
 *
 * First, the MEAN is the portfolio-relevant number, not the median. Equal-
 * sized bets across a lottery earn the mean, and a rule that improves the
 * median by refusing the tail is the exact error that kept `takeProfitPct` at
 * 120 while winners ran +2,133%.
 *
 * Second, the mean over a heavy tail is hostage to its largest observation. A
 * grid cell can look extraordinary on n=5 because one token did +1,457%.
 *
 * So report both, and read them together. Dropping the single best token is
 * the cheapest honest check that exists: if a rule only wins with its best
 * ticket included, it is a coin flip wearing a decimal point. Measured
 * 2026-08-13, this is what showed the drift ceiling's most attractive cell to
 * be worthless — +289% headline, -2.9% once its one winner was removed — and
 * it refuted the hypothesis the sweep was built to confirm.
 */

export interface ReturnStats {
  n: number
  medianPct: number
  /** Share of tokens with a positive return. */
  upPct: number
  over100Pct: number
  /** Portfolio return per equal-sized bet. */
  meanPct: number
  /**
   * `meanPct` recomputed without the single largest observation. Null at n=1,
   * where "without the best" is not a sample. Never silently zero: a missing
   * robustness check must not read as a robustness check that passed.
   */
  meanExBestPct: number | null
  bestPct: number
}

const round = (x: number, dp: number): number => Number(x.toFixed(dp))

const mean = (xs: readonly number[]): number =>
  xs.reduce((t, x) => t + x, 0) / xs.length

/** Median of a sorted array. Even lengths average the middle pair. */
function medianOfSorted(sorted: readonly number[]): number {
  const n = sorted.length
  return n % 2 === 1
    ? sorted[(n - 1) / 2]!
    : (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2
}

/**
 * Summarize a set of returns, in percent.
 *
 * Returns null for an empty set rather than a zeroed struct, the same
 * distinction the funnel window draws: "nothing to measure here" and "measured
 * and found zero" are different claims, and a grid cell that quietly reports
 * 0% for an empty band would read as a rule that admits nothing profitable
 * rather than a rule that admits nothing at all.
 */
export function describeReturns(rets: readonly number[]): ReturnStats | null {
  if (rets.length === 0) return null
  const sorted = [...rets].sort((a, b) => a - b)
  const n = sorted.length
  const best = sorted[n - 1]!
  // Drops ONE instance of the maximum by position, not every row equal to it.
  // Filtering by value would delete an entire tied cohort and overstate the
  // penalty exactly when the tail is flat.
  const exBest = sorted.slice(0, n - 1)
  return {
    n,
    medianPct: round(medianOfSorted(sorted), 1),
    upPct: round((100 * sorted.filter((r) => r > 0).length) / n, 1),
    over100Pct: round((100 * sorted.filter((r) => r > 100).length) / n, 1),
    meanPct: round(mean(sorted), 1),
    meanExBestPct: exBest.length > 0 ? round(mean(exBest), 1) : null,
    bestPct: round(best, 0),
  }
}

/**
 * Charge a one-way cost as a round trip.
 *
 * Impact is paid going in and again coming out, so a sweep that admits a
 * candidate at 3% impact and scores it on its raw forward return is reading a
 * 6% cost as free. Without this the sweep recommends admitting everything,
 * which is the failure it exists to prevent.
 */
export const ROUND_TRIP = 2

export function netOfCost(retPct: number, oneWayCostPct: number): number {
  return retPct - ROUND_TRIP * oneWayCostPct
}

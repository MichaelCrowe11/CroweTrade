/**
 * Trajectory confirmation: does OUR OWN observed tape support an entry?
 *
 * Standalone and dependency-free so it can be unit-tested under node --test
 * and shared verbatim by the engine. The inputs come from the engine's ticks
 * table: what this system itself watched happen, minute by minute, which no
 * promotional feed can pollute.
 */

/** Observed minute-ticks for one mint, oldest first. */
export interface Trajectory {
  prices: number[]
  liquidity: number[]
}

/**
 * Requires: enough observations, price above the start of our window, and
 * liquidity not draining (>= 90% of window start). A token whose liquidity is
 * walking out the door mid-listing is a rug in progress regardless of price.
 * Flat price refuses too: confirmation requires progress, not survival.
 */
export function trajectoryConfirms(t: Trajectory | undefined, minTicks: number): boolean {
  if (!t || t.prices.length < minTicks) return false
  const p0 = t.prices[0]
  const pN = t.prices[t.prices.length - 1]
  const l0 = t.liquidity[0]
  const lN = t.liquidity[t.liquidity.length - 1]
  if (p0 === undefined || pN === undefined || l0 === undefined || lN === undefined) return false
  if (p0 <= 0 || l0 <= 0) return false
  return pN > p0 && lN >= l0 * 0.9
}

/**
 * Has this token already run since WE first saw it?
 *
 * The gate that the 2026-08-10 measurement demanded. Across 73 live-shaped
 * entries, price moved +142.8% in the 23 minutes between the engine's first
 * sight of a token and its actual entry, and 54 of 73 were bought HIGHER than
 * first sight. The engine was not selecting tokens that pop; it was selecting
 * tokens that had ALREADY popped and buying near the top.
 *
 * The existing parabolic filter cannot catch this. `maxChangeH1Pct` reads the
 * FEED's hourly change, which is a claim about the last hour from a third
 * party. This reads our own recorded first-sight price against the price we
 * are about to pay, which is the only comparison that describes the trade we
 * are actually making.
 *
 * Returns true when the entry should be REFUSED.
 *
 * Unknown refuses, consistent with every other gate here: a first-sight price
 * we cannot read means we cannot tell whether we are chasing, and "we do not
 * know" has never been a reason to spend money in this system.
 */
export function hasDrifted(
  firstSightPrice: number | null | undefined,
  entryPrice: number,
  maxDriftPct: number,
): boolean {
  if (firstSightPrice === null || firstSightPrice === undefined) return true
  if (!Number.isFinite(firstSightPrice) || firstSightPrice <= 0) return true
  if (!Number.isFinite(entryPrice) || entryPrice <= 0) return true
  const driftPct = ((entryPrice - firstSightPrice) / firstSightPrice) * 100
  return driftPct > maxDriftPct
}

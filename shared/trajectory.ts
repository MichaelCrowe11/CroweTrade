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

/**
 * Decision-time features, computed from OUR OWN tick observations.
 *
 * These are the raw material for the calibrated edge model — the open problem
 * the 2026-08-08 repo survey named: nothing anywhere produces a calibrated
 * probability that a launch survives and moves. The two feature families here
 * are the retargeting that survey recommended: order-book imbalance becomes
 * swap-flow imbalance, and price impact becomes a live pool-depth read.
 *
 * Dependency-free on purpose: testable under node --test, shared verbatim by
 * the engine, and free of anything a promotional feed can pollute.
 *
 * Every input series is oldest-first, one entry per observed minute-tick.
 * buys/sells are the CUMULATIVE 24h counters the feed reports; deltas between
 * consecutive ticks recover per-minute flow.
 */

export interface TickSeries {
  prices: number[]
  liquidity: number[]
  buys24h: number[]
  sells24h: number[]
}

export interface DecisionFeatures {
  /** Observations backing these numbers. Few ticks = wide error bars. */
  ticks: number
  /** Net buy share of flow across the window, -1..1. The retargeted book_imbalance. */
  netFlowShare: number | null
  /** Last-minute flow vs window average, as a ratio - 1. Positive = accelerating. */
  flowAccel: number | null
  /** Price change across the window, percent. */
  priceProgressPct: number | null
  /** Liquidity change across the window, percent. Negative = draining. */
  liqTrendPct: number | null
}

/** Per-minute deltas from a cumulative counter, clamped at zero: the 24h
 *  window slides, so the counter can tick DOWN without any new activity, and
 *  a negative "new buys this minute" is an artifact, not information. */
function deltas(cumulative: number[]): number[] {
  const out: number[] = []
  for (let i = 1; i < cumulative.length; i++) {
    const prev = cumulative[i - 1]
    const cur = cumulative[i]
    if (prev === undefined || cur === undefined) continue
    out.push(Math.max(0, cur - prev))
  }
  return out
}

export function computeFeatures(s: TickSeries): DecisionFeatures {
  const n = Math.min(s.prices.length, s.liquidity.length, s.buys24h.length, s.sells24h.length)
  const out: DecisionFeatures = {
    ticks: n,
    netFlowShare: null,
    flowAccel: null,
    priceProgressPct: null,
    liqTrendPct: null,
  }
  if (n < 2) return out

  const buyFlow = deltas(s.buys24h.slice(0, n))
  const sellFlow = deltas(s.sells24h.slice(0, n))
  const totalBuys = buyFlow.reduce((a, b) => a + b, 0)
  const totalSells = sellFlow.reduce((a, b) => a + b, 0)
  const total = totalBuys + totalSells
  if (total > 0) {
    out.netFlowShare = (totalBuys - totalSells) / total

    // Acceleration: the most recent minute's activity against the window's
    // per-minute average. Launch flow decays fast; a token still accelerating
    // minutes in is behaving differently from one bleeding out.
    const lastBuy = buyFlow[buyFlow.length - 1] ?? 0
    const lastSell = sellFlow[sellFlow.length - 1] ?? 0
    const avgPerMinute = total / buyFlow.length
    if (avgPerMinute > 0) out.flowAccel = (lastBuy + lastSell) / avgPerMinute - 1
  }

  const p0 = s.prices[0]
  const pN = s.prices[n - 1]
  if (p0 !== undefined && pN !== undefined && p0 > 0) {
    out.priceProgressPct = ((pN - p0) / p0) * 100
  }
  const l0 = s.liquidity[0]
  const lN = s.liquidity[n - 1]
  if (l0 !== undefined && lN !== undefined && l0 > 0) {
    out.liqTrendPct = ((lN - l0) / l0) * 100
  }
  return out
}

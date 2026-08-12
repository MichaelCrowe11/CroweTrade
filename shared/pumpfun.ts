/**
 * Launchpad discovery: Pump.fun's own chronological coin listing.
 *
 * Added 2026-08-09 because the promotional feed was MEASURED unprofitable.
 * At n=87 labeled decisions, tokens the policy entered returned -29.7% over 30
 * minutes and tokens it refused returned -30.3%. Identical. Selection inside
 * that universe adds nothing, because the universe has no winners to isolate:
 * DexScreener's token-profiles and token-boosts are advertising, surfaced when
 * someone is paying to be seen, which is when they need buyers.
 *
 * This source is different in kind. It is every mint the launchpad created, in
 * creation order, with no placement fee involved. It is not chain-decoded --
 * that remains the eventual pipeline -- but it is the complete launch universe
 * rather than a marketed slice of it, which is the property that failed.
 *
 * It also carries `creator`, the deployer address. That is the input the
 * deployer-history gate has never had, and combined with our own labeled
 * outcomes it makes that gate computable from data we own.
 *
 * Both sources now run side by side, tagged by origin, so the calibration loop
 * decides which is better on evidence instead of on this argument.
 */

import type { Candidate, DiscoveryOrigin } from "./dexscreener.js"

const LISTING = "https://frontend-api-v3.pump.fun/coins"
const LAMPORTS_PER_SOL = 1_000_000_000

interface PumpCoin {
  mint: string
  name: string
  symbol: string
  creator: string
  created_timestamp: number
  complete: boolean
  /** Bonding-curve SOL actually deposited, in lamports. The real floor. */
  real_sol_reserves?: number
  virtual_sol_reserves?: number
  total_supply?: number
  base_decimals?: number
  usd_market_cap?: number
  pool_address?: string
  is_banned?: boolean
  nsfw?: boolean
}

/**
 * Bonding-curve liquidity in USD.
 *
 * Uses REAL reserves, never virtual. Virtual reserves are the curve's pricing
 * fiction and are non-zero the instant a token exists; treating them as
 * liquidity would report depth that cannot be sold into, which is the same
 * class of lie as the slippage model that flattered the record for hours.
 */
function curveLiquidityUsd(c: PumpCoin, solUsd: number): number | null {
  const lamports = c.real_sol_reserves
  if (lamports === undefined || solUsd <= 0) return null
  return (lamports / LAMPORTS_PER_SOL) * solUsd
}

function toCandidate(c: PumpCoin, solUsd: number, now: number): Candidate | null {
  if (!c.mint || c.is_banned) return null
  // Price from market cap over supply, in WHOLE TOKENS.
  //
  // total_supply arrives in BASE UNITS. Dividing market cap by it directly
  // yields a price per base unit -- off by 10^decimals, which for a six-decimal
  // mint is a factor of a million. Later ticks price the same token correctly
  // per whole token, so the ratio between them produced forward returns in the
  // hundreds of millions of percent and corrupted 810 of 874 launchpad rows
  // before the research tool surfaced it on its first query.
  const decimals = c.base_decimals ?? 6
  const supply = c.total_supply !== undefined ? c.total_supply / 10 ** decimals : undefined
  const mcap = c.usd_market_cap
  // Graduated coins get NO price from this listing. After graduation the pool
  // sets the price and the listing's usd_market_cap lags it — measured live: a
  // token that graduated within its first minute listed at an mcap ~2,470x
  // below its pool price, and that first tick manufactured a +248,331%
  // "return" that carried an entire counterfactual mean. Unknown is honest;
  // the Jupiter probe rail prices what the curve math cannot.
  const priceUsd =
    !c.complete && supply && supply > 0 && mcap ? mcap / supply : null

  return {
    mint: c.mint,
    symbol: c.symbol ?? "?",
    name: c.name ?? "",
    dex: c.complete ? "pumpswap" : "pumpfun-curve",
    origin: "launchpad" as DiscoveryOrigin,
    pool: c.pool_address ?? null,
    priceUsd,
    // The listing carries no hourly change. Null is honest; the engine's own
    // tick history supplies trajectory once we have observed the token.
    changeH1: null,
    liquidityUsd: curveLiquidityUsd(c, solUsd),
    volume24h: null,
    buys24h: null,
    sells24h: null,
    createdAt: c.created_timestamp ?? null,
    creator: c.creator ?? null,
    snapshot: {
      mint: c.mint,
      asOf: now,
      launchedAt: c.created_timestamp ?? null,
      // Authorities still come from chain state, resolved by the engine's
      // batched account read. The listing does not assert them and neither
      // do we.
      mintAuthority: undefined,
      freezeAuthority: undefined,
      // Pre-graduation, liquidity lives in the bonding curve and cannot be
      // pulled by the deployer; the program owns it. That is a structural
      // fact about this venue, not an observation, so it is a genuine pass.
      lpLockedBps: c.complete ? undefined : 10_000,
      topHolderShare: undefined,
      solReserveLamports:
        c.real_sol_reserves !== undefined ? BigInt(Math.round(c.real_sol_reserves)) : undefined,
      deployerPriorMints: undefined,
      deployerPriorRugs: undefined,
    },
  }
}

/** The API caps a page at roughly 70; depth comes from pagination. */
const PAGE_SIZE = 70
/** Bound external work even if mint velocity spikes or timestamps are bad. */
const MAX_PAGES = 8
const DEFAULT_POLL_INTERVAL_MS = 60_000

export interface LaunchpadScanRequirements {
  minTokenAgeMinutes: number
  minObservedTicks: number
  pollIntervalMs?: number
  maxPages?: number
}

export interface LaunchpadScanResult {
  candidates: Candidate[]
  /** False means a failed page or the request budget ended before the target. */
  complete: boolean
  pagesAttempted: number
  failedOffsets: number[]
  targetHistoryMs: number
  coveredHistoryMs: number
}

/**
 * History required for a mint to be both old enough and observed often enough.
 *
 * One extra poll interval matters. A mint created just after a scan is first
 * seen almost one interval old; if discovery ends exactly at the age floor,
 * its last visible observation can still be just too young to enter.
 */
export function requiredLaunchpadHistoryMs(
  requirements: LaunchpadScanRequirements,
): number {
  const pollIntervalMs = Math.max(1, requirements.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS)
  const minAgeMs = Math.max(0, requirements.minTokenAgeMinutes) * 60_000
  const observedTicksMs = Math.max(1, Math.ceil(requirements.minObservedTicks)) * pollIntervalMs
  return Math.max(minAgeMs + pollIntervalMs, observedTicksMs)
}

/**
 * Newest launches, newest first, deep enough to satisfy the engine's current
 * observation rules. Page count is deliberately NOT the success criterion:
 * mint velocity changes, while the required time horizon does not.
 */

export async function fetchLaunchpadCandidates(
  solUsd: number,
  signal: AbortSignal,
  requirements: LaunchpadScanRequirements,
): Promise<LaunchpadScanResult> {
  const now = Date.now()
  const targetHistoryMs = requiredLaunchpadHistoryMs(requirements)
  const targetCreatedAt = now - targetHistoryMs
  const requestedMaxPages = requirements.maxPages ?? MAX_PAGES
  const maxPages = Number.isFinite(requestedMaxPages)
    ? Math.max(1, Math.min(MAX_PAGES, Math.floor(requestedMaxPages)))
    : MAX_PAGES
  const seen = new Set<string>()
  const out: Candidate[] = []
  const failedOffsets: number[] = []
  let pagesAttempted = 0
  let oldestCreatedAt: number | null = null
  let reachedTarget = false

  for (let page = 0; page < maxPages; page++) {
    const offset = page * PAGE_SIZE
    pagesAttempted += 1
    try {
      const url =
        `${LISTING}?offset=${offset}&limit=${PAGE_SIZE}&sort=created_timestamp&order=DESC&includeNsfw=false`
      const res = await fetch(url, {
        signal,
        headers: { accept: "application/json", "user-agent": "CroweTrade/1.0" },
      })
      if (!res.ok) {
        failedOffsets.push(offset)
        continue
      }
      const body = await res.json()
      if (!Array.isArray(body) || body.length === 0) {
        failedOffsets.push(offset)
        break
      }
      for (const raw of body) {
        const createdAt = Number((raw as PumpCoin)?.created_timestamp)
        if (Number.isFinite(createdAt) && createdAt > 0 && createdAt <= now) {
          oldestCreatedAt = oldestCreatedAt === null
            ? createdAt
            : Math.min(oldestCreatedAt, createdAt)
        }
        // Pages can overlap as new mints shift the window mid-fetch; first
        // sighting wins so a token is never counted twice in one tick.
        const coin = raw as PumpCoin
        if (!coin?.mint || seen.has(coin.mint)) continue
        seen.add(coin.mint)
        const c = toCandidate(coin, solUsd, now)
        if (c !== null) out.push(c)
      }
      if (oldestCreatedAt !== null && oldestCreatedAt <= targetCreatedAt) {
        reachedTarget = true
        break
      }
    } catch (error) {
      // A tick-wide abort is control flow, not a page failure; do not swallow it.
      if (signal.aborted) throw error
      failedOffsets.push(offset)
    }
  }

  return {
    candidates: out,
    complete: reachedTarget && failedOffsets.length === 0,
    pagesAttempted,
    failedOffsets,
    targetHistoryMs,
    coveredHistoryMs: oldestCreatedAt === null ? 0 : Math.max(0, now - oldestCreatedAt),
  }
}

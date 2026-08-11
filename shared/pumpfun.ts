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

/**
 * Newest launches, newest first, across several pages.
 *
 * Depth is the whole point. The engine cannot enter a token it has not
 * observed for three minutes, and a single page keeps a mint visible for
 * roughly one -- so shallow polling did not filter the universe, it made
 * almost all of it unreachable.
 */
/** The API caps a page at ~70 regardless of what `limit` asks for, so depth
 *  comes from pagination rather than a bigger page. Measured 2026-08-11:
 *  offset 0 reaches ~3.7 minutes back, 70 reaches ~6.9, 140 reaches ~9.9. */
const PAGE = 70
const PAGES = 3

export async function fetchLaunchpadCandidates(
  solUsd: number,
  signal: AbortSignal,
  pages = PAGES,
): Promise<Candidate[]> {
  const now = Date.now()
  const seen = new Set<string>()
  const out: Candidate[] = []

  // Sequential pages, each failure isolated. ONE page is worth ~1 minute of
  // visibility and the engine's own rules require a token to be 3 minutes old
  // with 3 observed ticks before it may be entered -- so a single page made
  // 96.2% of the universe structurally unenterable: mints fell off the listing
  // before they were ever eligible. Measured: average visible span 1 minute,
  // average 2 observations, only 3.8% visible for 3 minutes.
  for (let i = 0; i < pages; i++) {
    try {
      const url =
        `${LISTING}?offset=${i * PAGE}&limit=${PAGE}&sort=created_timestamp&order=DESC&includeNsfw=false`
      const res = await fetch(url, {
        signal,
        headers: { accept: "application/json", "user-agent": "CroweTrade/1.0" },
      })
      if (!res.ok) continue
      const body = (await res.json()) as PumpCoin[]
      if (!Array.isArray(body)) continue
      for (const raw of body) {
        // Pages can overlap as new mints shift the window mid-fetch; first
        // sighting wins so a token is never counted twice in one tick.
        if (!raw?.mint || seen.has(raw.mint)) continue
        seen.add(raw.mint)
        const c = toCandidate(raw, solUsd, now)
        if (c !== null) out.push(c)
      }
    } catch {
      // A failed page costs depth, never the tick. The promotional source and
      // position management continue regardless.
    }
  }
  return out
}

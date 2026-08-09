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
  const supply = c.total_supply
  const mcap = c.usd_market_cap
  // Price from market cap over supply. Absent either, we have no price, and a
  // candidate without a price is not tradeable -- say so with null rather than
  // inventing zero.
  const priceUsd = supply && supply > 0 && mcap ? mcap / supply : null

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
 * Newest launches, newest first.
 *
 * `limit` is deliberately modest: the engine ticks every minute and the
 * launchpad produces a handful of mints per minute, so a large page mostly
 * re-fetches tokens already recorded.
 */
export async function fetchLaunchpadCandidates(
  solUsd: number,
  signal: AbortSignal,
  limit = 50,
): Promise<Candidate[]> {
  try {
    const url =
      `${LISTING}?offset=0&limit=${limit}&sort=created_timestamp&order=DESC&includeNsfw=false`
    const res = await fetch(url, {
      signal,
      headers: { accept: "application/json", "user-agent": "CroweTrade/1.0" },
    })
    if (!res.ok) return []
    const body = (await res.json()) as PumpCoin[]
    if (!Array.isArray(body)) return []
    const now = Date.now()
    return body
      .map((c) => toCandidate(c, solUsd, now))
      .filter((c): c is Candidate => c !== null)
  } catch {
    // A failed launchpad fetch must not take the tick down; the promotional
    // source and position management continue independently.
    return []
  }
}

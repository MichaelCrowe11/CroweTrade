/**
 * Bootstrap feed: DexScreener public API.
 *
 * This is deliberately the WEAK feed, and it is here to make the app runnable
 * today rather than because it is good. It is exactly what every competing
 * terminal reads: derived aggregate numbers, recomputed on someone else's
 * schedule, identical for everyone who asks.
 *
 * It also demonstrates the gap better than any argument could. An aggregator
 * can tell you price, liquidity and volume. It cannot tell you whether the mint
 * authority was revoked, whether LP is actually burned, how concentrated the
 * holders are, or whether this deployer has rugged before. Those come from
 * decoding chain state ourselves, so on this feed those gates report "unknown",
 * which is the honest answer and the reason the pipeline work exists.
 *
 * It sits behind the same boundary the real feeds will use, so replacing it is
 * a swap, not a rewrite.
 */

import type { TokenSnapshot } from "./gates.js"

const PROFILES = "https://api.dexscreener.com/token-profiles/latest/v1"
const BOOSTS = "https://api.dexscreener.com/token-boosts/latest/v1"
const PAIRS = "https://api.dexscreener.com/tokens/v1/solana"
const SEARCH = "https://api.dexscreener.com/latest/dex/search"

/** The tokens endpoint accepts at most 30 comma-separated addresses per call. */
const MAX_ADDRESSES_PER_CALL = 30

/** Only the fields we consume. The response carries considerably more. */
interface DexPair {
  chainId: string
  dexId: string
  pairAddress?: string
  baseToken: { address: string; name: string; symbol: string }
  priceUsd?: string
  priceChange?: { h1?: number; h24?: number }
  liquidity?: { usd?: number }
  volume?: { h24?: number }
  txns?: { h24?: { buys: number; sells: number } }
  pairCreatedAt?: number
}

interface TokenRef {
  chainId: string
  tokenAddress: string
}

/** Where discovery surfaced a mint. "boost" is PAID promotion. */
export type DiscoveryOrigin = "profile" | "boost" | "both" | "held" | "launchpad"

export interface Candidate {
  mint: string
  symbol: string
  name: string
  dex: string
  /**
   * Provenance of the listing itself. Measured 2026-08-08: 61% of entries from
   * the promotional feeds hit a -35% stop with zero recoveries, which is what
   * being someone's exit liquidity looks like. Tagging origin lets the record
   * prove or refute that per cohort instead of arguing about it.
   */
  origin: DiscoveryOrigin
  /** Pool account for the deepest venue; the key into the candle fetch. */
  pool: string | null
  /**
   * Deployer address, when the source provides it. The launchpad listing does;
   * the promotional aggregator does not. This is the input the deployer-history
   * gate has never had.
   */
  creator?: string | null
  priceUsd: number | null
  /** Percent change over the last hour, e.g. -12.4. */
  changeH1: number | null
  liquidityUsd: number | null
  volume24h: number | null
  buys24h: number | null
  sells24h: number | null
  createdAt: number | null
  snapshot: TokenSnapshot
}

const LAMPORTS_PER_SOL = 1_000_000_000

/** Approximate SOL reserve from the USD liquidity figure the aggregator reports.
 *  Pool liquidity is roughly half quote asset, hence the halving. This is an
 *  estimate standing in for a real reserve read, and it is one more reason this
 *  feed is temporary. */
function estimateSolReserve(liquidityUsd: number | undefined, solUsd: number): bigint | undefined {
  if (liquidityUsd === undefined || solUsd <= 0) return undefined
  return BigInt(Math.round((liquidityUsd / 2 / solUsd) * LAMPORTS_PER_SOL))
}

function toCandidate(
  p: DexPair,
  solUsd: number,
  now: number,
  origin: DiscoveryOrigin,
): Candidate {
  const createdAt = p.pairCreatedAt ?? null
  return {
    mint: p.baseToken.address,
    symbol: p.baseToken.symbol,
    name: p.baseToken.name,
    dex: p.dexId,
    origin,
    pool: p.pairAddress ?? null,
    priceUsd: p.priceUsd ? Number(p.priceUsd) : null,
    changeH1: p.priceChange?.h1 ?? null,
    liquidityUsd: p.liquidity?.usd ?? null,
    volume24h: p.volume?.h24 ?? null,
    buys24h: p.txns?.h24?.buys ?? null,
    sells24h: p.txns?.h24?.sells ?? null,
    createdAt,
    snapshot: {
      mint: p.baseToken.address,
      // asOf is the observation time here because the aggregator does not tell
      // us the block its numbers came from. That is itself disqualifying for
      // point-in-time work, and it is why backfill has to come from the ledger.
      asOf: now,
      launchedAt: createdAt,
      // Everything below requires decoding chain state. Undefined means unknown,
      // and unknown must never be rendered as a pass.
      mintAuthority: undefined,
      freezeAuthority: undefined,
      lpLockedBps: undefined,
      topHolderShare: undefined,
      solReserveLamports: estimateSolReserve(p.liquidity?.usd, solUsd),
      deployerPriorMints: undefined,
      deployerPriorRugs: undefined,
    },
  }
}

async function getJson<T>(url: string, signal: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`${new URL(url).pathname} -> ${res.status}`)
  return (await res.json()) as T
}

/** Current SOL price in USD, used to convert reported USD liquidity to SOL. */
export async function fetchSolUsd(signal: AbortSignal): Promise<number> {
  const body = await getJson<{ pairs?: DexPair[] }>(`${SEARCH}?q=SOL%2FUSDC`, signal)
  const quote = body.pairs?.find(
    (p) => p.chainId === "solana" && p.baseToken.symbol === "SOL" && p.priceUsd,
  )
  if (!quote?.priceUsd) throw new Error("no SOL/USDC pair in response")
  return Number(quote.priceUsd)
}

/**
 * Collects candidate mints from the two discovery endpoints.
 *
 * Both are used because they surface different populations: profiles skews to
 * freshly created tokens, boosts to ones someone is paying to promote. Paying
 * for promotion is not a positive signal on its own, but it does concentrate
 * the set of tokens that will actually see volume, which is where the tradeable
 * moves are.
 */
async function discoverMints(signal: AbortSignal): Promise<Map<string, DiscoveryOrigin>> {
  const [profiles, boosts] = await Promise.all([
    getJson<TokenRef[]>(PROFILES, signal).catch(() => [] as TokenRef[]),
    getJson<TokenRef[]>(BOOSTS, signal).catch(() => [] as TokenRef[]),
  ])
  const origin = new Map<string, DiscoveryOrigin>()
  for (const t of profiles) {
    if (t.chainId === "solana" && t.tokenAddress) origin.set(t.tokenAddress, "profile")
  }
  for (const t of boosts) {
    if (t.chainId === "solana" && t.tokenAddress) {
      origin.set(t.tokenAddress, origin.has(t.tokenAddress) ? "both" : "boost")
    }
  }
  return origin
}

export interface Scan {
  candidates: Candidate[]
  /** Spot SOL/USD at scan time; the header pulse and SOL conversions use it. */
  solUsd: number
}

/**
 * Prices an explicit list of mints. The engine needs this for HELD positions:
 * a token drops out of the discovery endpoints within hours, but an open
 * position must keep pricing until it exits, or exits simply stop firing and
 * the ledger rots with phantom positions.
 */
export async function fetchPairsForMints(
  mints: string[],
  solUsd: number,
  signal: AbortSignal,
  origins?: Map<string, DiscoveryOrigin>,
): Promise<Candidate[]> {
  if (mints.length === 0) return []

  const batches: string[][] = []
  for (let i = 0; i < mints.length; i += MAX_ADDRESSES_PER_CALL) {
    batches.push(mints.slice(i, i + MAX_ADDRESSES_PER_CALL))
  }

  // A failed batch drops to empty rather than failing the whole refresh: a
  // partial scan list is far more useful than an error screen.
  const results = await Promise.all(
    batches.map((b) =>
      getJson<DexPair[]>(`${PAIRS}/${b.join(",")}`, signal).catch(() => [] as DexPair[]),
    ),
  )

  const now = Date.now()
  const byMint = new Map<string, Candidate>()

  for (const pair of results.flat()) {
    if (pair.chainId !== "solana") continue
    const candidate = toCandidate(
      pair,
      solUsd,
      now,
      origins?.get(pair.baseToken.address) ?? "held",
    )
    // A token can list on several venues. Keep the deepest pool, since that is
    // the one an exit would actually route through.
    const existing = byMint.get(candidate.mint)
    if (!existing || (candidate.liquidityUsd ?? 0) > (existing.liquidityUsd ?? 0)) {
      byMint.set(candidate.mint, candidate)
    }
  }

  return [...byMint.values()]
}

/**
 * Fetches current Solana meme-coin candidates.
 *
 * Sorted newest first, because on this market age is the dominant variable and
 * a list ordered by anything else buries the only rows that matter.
 */
export async function fetchCandidates(signal: AbortSignal): Promise<Scan> {
  const [solUsd, origins] = await Promise.all([fetchSolUsd(signal), discoverMints(signal)])
  if (origins.size === 0) return { candidates: [], solUsd }

  const priced = await fetchPairsForMints([...origins.keys()], solUsd, signal, origins)
  const candidates = priced
    .sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0))
    .slice(0, 50)
  return { candidates, solUsd }
}

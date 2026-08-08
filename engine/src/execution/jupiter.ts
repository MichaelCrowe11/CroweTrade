/**
 * Real route quotes from Jupiter.
 *
 * This replaces the invented slippage model in strategy.ts, which was measured
 * wrong by ~20x on 2026-08-08: quoting 0.25 SOL into a held token ($18.8K
 * liquidity) returned 2.10% real price impact where the model predicted 0.10%.
 * A paper record priced by that model flatters itself on every single fill,
 * entry and exit, which is the one failure mode that makes a track record
 * worthless.
 *
 * These are the SAME routes live execution will take. Quoting needs no wallet,
 * no funds, and no signature, so the paper phase can price itself honestly at
 * zero risk. When execution arms, the quote here becomes the transaction there.
 */

const JUPITER = "https://lite-api.jup.ag/swap/v1/quote"
export const WSOL = "So11111111111111111111111111111111111111112"
const LAMPORTS_PER_SOL = 1_000_000_000

export interface Quote {
  /** Output in the destination token's base units. */
  outAmount: bigint
  /** Fractional price impact, e.g. 0.021 for 2.1%. */
  priceImpactPct: number
  /** Venue labels along the route, for the record. */
  route: string
  /**
   * The unmodified response body. Jupiter's swap endpoint requires the whole
   * quote echoed back verbatim, so it is carried rather than reconstructed —
   * a re-serialized quote is a different quote.
   */
  raw: unknown
}

interface JupResponse {
  outAmount?: string
  priceImpactPct?: string
  routePlan?: { swapInfo?: { label?: string } }[]
  error?: string
}

async function quote(
  inputMint: string,
  outputMint: string,
  amount: bigint,
  slippageBps: number,
): Promise<Quote | null> {
  const url =
    `${JUPITER}?inputMint=${inputMint}&outputMint=${outputMint}` +
    `&amount=${amount}&slippageBps=${slippageBps}`
  try {
    const res = await fetch(url)
    if (!res.ok) return null
    const body = (await res.json()) as JupResponse
    // Jupiter answers 200 with an error body when no route exists, which for a
    // freshly launched token is normal rather than exceptional.
    if (body.error || !body.outAmount) return null
    return {
      outAmount: BigInt(body.outAmount),
      priceImpactPct: Number(body.priceImpactPct ?? 0),
      route: (body.routePlan ?? [])
        .map((r) => r.swapInfo?.label ?? "?")
        .join(" > "),
      raw: body,
    }
  } catch {
    // Network failure must not be mistaken for "no route": callers treat null
    // as "cannot price this right now" and skip, never as "price is zero".
    return null
  }
}

/** SOL in, token out. Returns real token base units received. */
export function quoteBuy(mint: string, sol: number, slippageBps: number): Promise<Quote | null> {
  const lamports = BigInt(Math.round(sol * LAMPORTS_PER_SOL))
  return quote(WSOL, mint, lamports, slippageBps)
}

/** Token in, SOL out. Returns real lamports received. */
export function quoteSell(
  mint: string,
  tokenBaseUnits: bigint,
  slippageBps: number,
): Promise<Quote | null> {
  return quote(mint, WSOL, tokenBaseUnits, slippageBps)
}

export { LAMPORTS_PER_SOL }

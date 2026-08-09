/**
 * x402 payment gateway: sell the corpus, not the trades.
 *
 * The strategy has not been shown to work. The DATA has: we hold a
 * point-in-time-correct record of token launches with decision-time features
 * and labeled 30-minute outcomes, plus safety gates read from chain rather than
 * from an aggregator's opinion. Nobody sells that, because the people who could
 * are using it. An agent asking "is this mint safe, and what happened to
 * launches that looked like this" is a per-call question with a per-call price,
 * which is exactly the shape x402 was built for.
 *
 * This is the one revenue path whose viability does not depend on us beating
 * the market. It depends on having measured it honestly, which is the thing
 * this system demonstrably does.
 *
 * Wire format follows the x402 v2 specification (coinbase/x402,
 * specs/x402-specification-v2.md + specs/transports-v2/http.md):
 *   402 response  -> PAYMENT-REQUIRED header, base64 JSON PaymentRequired
 *   client retry  -> PAYMENT-SIGNATURE header, base64 JSON PaymentPayload
 *   settlement    -> PAYMENT-RESPONSE header, base64 JSON
 * The body carries the same JSON as the header so a human poking at the
 * endpoint with curl sees something readable instead of an empty 402.
 */

/** Solana mainnet, CAIP-2. From the spec's exact-SVM scheme document. */
export const SOLANA_MAINNET = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"

/** USDC on Solana mainnet. Six decimals, so 1000 atomic units = $0.001. */
export const USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

export interface PricedRoute {
  /** Atomic USDC units. 1_000 = $0.001. */
  amount: string
  description: string
}

/**
 * The price list.
 *
 * Priced per call in fractions of a cent because the buyer is an agent
 * screening hundreds of mints, not a person buying a report. A safety check
 * that costs more than the gas it saves you is not a product.
 */
export const ROUTES: Record<string, PricedRoute> = {
  "/api/v1/safety": {
    amount: "2000", // $0.002
    description:
      "Survivability gates for one Solana mint, read from chain state: mint and freeze authority, LP lock, holder concentration, liquidity depth, deployer history. Returns pass/fail/unknown per gate and a combined verdict. Unknown is never reported as a pass.",
  },
  "/api/v1/corpus": {
    amount: "5000", // $0.005
    description:
      "Outcome statistics from our labeled launch corpus: death rate and 30-minute forward returns, split by discovery origin, with sample sizes. Point-in-time correct.",
  },
}

interface PaymentRequirements {
  scheme: "exact"
  network: string
  amount: string
  asset: string
  payTo: string
  maxTimeoutSeconds: number
  extra: { name: string; decimals: number }
}

function b64(value: unknown): string {
  return btoa(String.fromCharCode(...new TextEncoder().encode(JSON.stringify(value))))
}

function unb64<T>(header: string): T | null {
  try {
    const bytes = Uint8Array.from(atob(header), (c) => c.charCodeAt(0))
    return JSON.parse(new TextDecoder().decode(bytes)) as T
  } catch {
    return null
  }
}

/** Is this route sold, and is the gateway configured to sell it? */
export function priceFor(pathname: string): PricedRoute | null {
  for (const [prefix, route] of Object.entries(ROUTES)) {
    if (pathname === prefix || pathname.startsWith(`${prefix}/`)) return route
  }
  return null
}

/**
 * Builds the 402. Carries the requirements in BOTH the spec header and the
 * body: agents read the header, humans read the body, and an endpoint that
 * returns a bare 402 to a curious developer has failed at being discoverable.
 */
export function paymentRequired(
  req: Request,
  route: PricedRoute,
  payTo: string,
  error = "PAYMENT-SIGNATURE header is required",
): Response {
  const accepts: PaymentRequirements[] = [{
    scheme: "exact",
    network: SOLANA_MAINNET,
    amount: route.amount,
    asset: USDC_MINT,
    payTo,
    maxTimeoutSeconds: 60,
    extra: { name: "USDC", decimals: 6 },
  }]

  const body = {
    x402Version: 2,
    error,
    resource: {
      url: req.url,
      description: route.description,
      mimeType: "application/json",
    },
    accepts,
    extensions: {},
    // Not part of the spec: a hint so a human who hits this by hand knows what
    // they are looking at and what it costs in money rather than atomic units.
    _human: {
      priceUsd: (Number(route.amount) / 1_000_000).toFixed(4),
      docs: "https://x402.org",
      note: "Free endpoints remain open: /api/health, /api/positions, /api/exit-sweep.",
    },
  }

  return new Response(JSON.stringify(body, null, 2), {
    status: 402,
    headers: {
      "Content-Type": "application/json",
      "PAYMENT-REQUIRED": b64(body),
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Expose-Headers": "PAYMENT-REQUIRED, PAYMENT-RESPONSE",
    },
  })
}

export interface SettleResult {
  ok: boolean
  header: string
  payer?: string
  transaction?: string
  errorReason?: string
}

/**
 * Verifies and settles through a facilitator.
 *
 * The facilitator is deliberately a third party: it holds the chain
 * integration, we hold the data. Settlement happens BEFORE the response is
 * served, so a failed settle means no data leaves. Verifying but not settling
 * would hand out paid answers on a promise.
 */
export async function settle(
  paymentSignature: string,
  route: PricedRoute,
  payTo: string,
  facilitator: string,
): Promise<SettleResult> {
  const payload = unb64<Record<string, unknown>>(paymentSignature)
  if (!payload) {
    return { ok: false, header: b64({ success: false, errorReason: "invalid_payload" }), errorReason: "invalid_payload" }
  }

  const requirements = {
    scheme: "exact",
    network: SOLANA_MAINNET,
    amount: route.amount,
    asset: USDC_MINT,
    payTo,
    maxTimeoutSeconds: 60,
    extra: { name: "USDC", decimals: 6 },
  }

  try {
    const res = await fetch(`${facilitator.replace(/\/$/, "")}/settle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x402Version: 2, paymentPayload: payload, paymentRequirements: requirements }),
    })
    const out = (await res.json()) as {
      success?: boolean; transaction?: string; payer?: string; errorReason?: string
    }
    if (!res.ok || !out.success) {
      const reason = out.errorReason ?? `facilitator_${res.status}`
      return { ok: false, header: b64({ success: false, errorReason: reason }), errorReason: reason }
    }
    return {
      ok: true,
      header: b64({
        success: true,
        transaction: out.transaction ?? "",
        network: SOLANA_MAINNET,
        payer: out.payer ?? "",
      }),
      payer: out.payer,
      transaction: out.transaction,
    }
  } catch (e) {
    const reason = e instanceof Error ? e.message : "facilitator_unreachable"
    return { ok: false, header: b64({ success: false, errorReason: reason }), errorReason: reason }
  }
}

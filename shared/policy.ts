/**
 * The policy envelope: the single object a user consents to.
 *
 * This is the consent framework Michael specced on 2026-08-08. One signed
 * object carries the legal waiver (by hash), the exact autonomy limits, and an
 * expiry. The user's wallet signature over this object is simultaneously the
 * legal consent record and the credential the signer service checks before any
 * transaction. Nothing executes outside a live, signed, unexpired envelope.
 *
 * Governance rules the envelope encodes, as agreed:
 *  - tighten instantly, loosen with delay (armAfter on any loosening change)
 *  - kill switch is instant and unconditional
 *  - a veto window follows every autonomous entry: within it, one action
 *    unwinds the position at market (a second trade, never a reversal;
 *    finality is real and the UI must never pretend otherwise)
 *  - every fill records the hash of the envelope that authorized it, so every
 *    trade has provable lineage: this fill, under this policy version, under
 *    this signed consent
 *
 * PAPER PHASE: signature and signer stay null and the engine trades imaginary
 * capital under the same envelope discipline, so the audit trail exists and is
 * demonstrable before real money does. Wallet signing lands with wallet
 * connect in the execution layer.
 */

export interface PolicyEnvelope {
  version: 1
  product: "crowetrade-paper" | "crowetrade-live"
  /** SHA-256 of the exact waiver text consented to (shared/waiver.md). */
  waiverSha256: string

  /** Hard caps, enforced at the signer for live and at the engine for paper. */
  perTradeCapSol: number
  dailyCapSol: number
  maxOpenPositions: number

  entry: {
    /** Minimum verdict allowed to open: "clear" only, or caution-and-better. */
    minVerdict: "clear" | "caution"
    maxTokenAgeMinutes: number
    minLiquidityUsd: number
  }

  exit: {
    takeProfitPct: number
    stopLossPct: number
    /** Flat exit after this long regardless of price. Meme decay is real. */
    timeStopMinutes: number
    /** Human veto window after each autonomous entry. */
    vetoWindowMinutes: number
  }

  /** ISO time after which the envelope is dead and nothing trades. */
  expiresAt: string

  /** Wallet signature over the canonical hash. Null during the paper phase. */
  signature: string | null
  /** The signing wallet. Null during the paper phase. */
  signer: string | null
}

/**
 * Default paper policy, v1. Every number is a dial; the SHAPE (caps + gates +
 * exits + veto) is the contract. Loosening any of these in a live envelope
 * must re-arm with delay and require a fresh signature.
 */
export const PAPER_POLICY: PolicyEnvelope = {
  version: 1,
  product: "crowetrade-paper",
  waiverSha256: "unsigned-paper-phase",
  perTradeCapSol: 0.5,
  dailyCapSol: 10,
  maxOpenPositions: 8,
  entry: {
    minVerdict: "caution",
    maxTokenAgeMinutes: 90,
    minLiquidityUsd: 3_000,
  },
  exit: {
    takeProfitPct: 60,
    stopLossPct: 35,
    timeStopMinutes: 45,
    vetoWindowMinutes: 10,
  },
  expiresAt: "2027-01-01T00:00:00Z",
  signature: null,
  signer: null,
}

/** Stable stringify: keys sorted, so the hash is canonical across runtimes. */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`)
    return `{${entries.join(",")}}`
  }
  return JSON.stringify(value)
}

/**
 * The policy hash stamped on every fill and, for live envelopes, the exact
 * bytes the wallet signs. WebCrypto, so it runs identically in the renderer,
 * in Workers, and under node --test.
 */
export async function policyHash(p: PolicyEnvelope): Promise<string> {
  const unsigned = { ...p, signature: null }
  const bytes = new TextEncoder().encode(canonical(unsigned))
  const digest = await crypto.subtle.digest("SHA-256", bytes)
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("")
}

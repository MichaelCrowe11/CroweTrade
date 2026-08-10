/**
 * The live send path. Nothing else in this codebase can broadcast.
 *
 * It is deliberately a separate module from swap.ts, which builds and
 * simulates and will never send. That separation is the safety model: a reader
 * can answer "can this thing spend money" by asking which modules import THIS
 * file, and today the answer is one explicit, flagged call site.
 *
 * Three locks, all of which must be open:
 *
 *   1. `LIVE_TRADING=1` in the Worker environment. Absent = inert.
 *   2. A funded signer key in the environment. Absent = inert.
 *   3. `preflight()` returns null for the specific trade.
 *
 * Any one missing and this returns a refusal instead of a signature. The
 * checks are re-read on every call rather than cached at module load, so
 * revoking the flag takes effect on the next trade rather than the next
 * deploy.
 *
 * WHAT THIS IS FOR FIRST: a dust test. Twenty dollars is not a bet on the
 * strategy, it is the only way to prove the caps hold when the money is real,
 * that a fill can be confirmed and reconciled, and that the kill switch stops
 * a live path and not just a simulated one. Those are engineering facts and
 * they cannot be established on paper.
 */

import { currentRpc } from "../../../shared/solana.js"
import { preflight, liveArmed, type TradeIntent, type PreflightContext } from "../../../shared/preflight.js"

/** Re-exported so callers import one execution surface; defined in
 *  shared/preflight.ts because it must be unit-testable. */
export { liveArmed }

export interface SendResult {
  ok: boolean
  /** Transaction signature, present only when actually broadcast. */
  signature: string | null
  /** Why it did not send, or why it failed after sending. */
  error: string | null
  /** True when refused BEFORE broadcast: no fee was paid, nothing happened. */
  refusedBeforeSend: boolean
}

function refuse(error: string): SendResult {
  return { ok: false, signature: null, error, refusedBeforeSend: true }
}

/**
 * Sign and broadcast a Jupiter swap transaction.
 *
 * `signTx` is injected rather than imported so this module never itself
 * touches key material in a form that could be logged: the caller supplies a
 * function that turns unsigned transaction bytes into signed ones. A test
 * passes a fake and exercises every branch without a key existing.
 */
export async function sendSwap(
  swapTxBase64: string,
  intent: TradeIntent,
  ctx: PreflightContext,
  signTx: (unsignedBase64: string) => Promise<string>,
): Promise<SendResult> {
  // The guard runs FIRST and its verdict is final. Note that ctx.liveArmed is
  // supplied by the caller from liveArmed(env) above, so the environment lock
  // and the policy lock are checked in one place.
  const refusal = preflight(intent, ctx)
  if (refusal) return refuse(refusal)

  let signed: string
  try {
    signed = await signTx(swapTxBase64)
  } catch (e) {
    return refuse(`signing failed: ${e instanceof Error ? e.message : String(e)}`)
  }

  // From here a fee can be paid, so `refusedBeforeSend` becomes false: the
  // operator needs to know whether a failure cost anything.
  try {
    const res = await fetch(currentRpc(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "sendTransaction",
        params: [
          signed,
          {
            encoding: "base64",
            // Never skip preflight at the RPC: it is a second, independent
            // simulation and the cheapest possible rejection of a bad trade.
            skipPreflight: false,
            // A dropped transaction is better than a duplicate fill.
            maxRetries: 2,
          },
        ],
      }),
    })
    const body = (await res.json()) as { result?: string; error?: { message?: string } }
    if (body.error) {
      return { ok: false, signature: null, error: body.error.message ?? "send rejected", refusedBeforeSend: false }
    }
    if (!body.result) {
      return { ok: false, signature: null, error: "no signature returned", refusedBeforeSend: false }
    }
    return { ok: true, signature: body.result, error: null, refusedBeforeSend: false }
  } catch (e) {
    // A network failure AFTER broadcast is the genuinely dangerous case: the
    // transaction may or may not have landed. Say so plainly rather than
    // reporting a clean failure, because the caller must reconcile before
    // assuming anything.
    return {
      ok: false,
      signature: null,
      error: `send failed in flight, state UNKNOWN, reconcile before retrying: ${e instanceof Error ? e.message : String(e)}`,
      refusedBeforeSend: false,
    }
  }
}

export interface Confirmation {
  confirmed: boolean
  /** Slot the transaction landed in, when known. */
  slot: number | null
  /** On-chain execution error, if it landed but failed. */
  txError: string | null
}

/**
 * Poll until the transaction confirms or the budget runs out.
 *
 * A send that is not confirmed is NOT a completed trade, and treating it as
 * one is how a book drifts from the chain. Timing out returns confirmed:false
 * rather than throwing, because "we do not know yet" is a real state that the
 * reconciler has to handle, not an exception.
 */
export async function confirmSignature(
  signature: string,
  attempts = 20,
  waitMs = 1_500,
  sleep: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
): Promise<Confirmation> {
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(currentRpc(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "getSignatureStatuses",
          params: [[signature], { searchTransactionHistory: true }],
        }),
      })
      const body = (await res.json()) as {
        result?: { value?: ({ slot?: number; err?: unknown; confirmationStatus?: string } | null)[] }
      }
      const st = body.result?.value?.[0]
      if (st) {
        const status = st.confirmationStatus
        if (status === "confirmed" || status === "finalized") {
          return {
            confirmed: true,
            slot: st.slot ?? null,
            txError: st.err ? JSON.stringify(st.err) : null,
          }
        }
      }
    } catch {
      // Transient RPC failure; keep polling. The budget bounds this.
    }
    if (i < attempts - 1) await sleep(waitMs)
  }
  return { confirmed: false, slot: null, txError: null }
}

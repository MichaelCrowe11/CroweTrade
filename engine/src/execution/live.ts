/**
 * The full live round trip, in one place: build, simulate, guard, sign, send,
 * confirm, reconcile.
 *
 * ENTRIES AND EXITS SHARE THIS PATH. That is deliberate and it is the thing
 * that was missing when the signer landed: an engine that can enter live and
 * cannot exit live is strictly worse than one that does neither, because it
 * can acquire a position it has no automated way to close. A swap is a swap;
 * only the direction of the quote differs.
 *
 * The exit differs in exactly one respect, and it is a policy difference
 * rather than a mechanical one: an exit must be attempted even when the
 * envelope would refuse a new trade. The kill switch, the daily cap and the
 * breaker all stop NEW RISK; none of them should trap a position. So exits
 * skip the entry guard and carry their own, much smaller one.
 */

import { currentRpc } from "../../../shared/solana.js"
import { preflight, type TradeIntent, type PreflightContext } from "../../../shared/preflight.js"
import { signTransaction } from "../../../shared/signer.js"
import { realizedFill, type RealizedFill, type TxMeta } from "../../../shared/reconcile.js"

const SWAP = "https://lite-api.jup.ag/swap/v1/swap"

export type Direction = "entry" | "exit"

export interface LiveResult {
  ok: boolean
  signature: string | null
  /** What the chain says happened. Null when we could not reconcile it. */
  fill: RealizedFill | null
  error: string | null
  /** True when nothing was broadcast, so no fee was paid and nothing changed. */
  refusedBeforeSend: boolean
}

function refused(error: string): LiveResult {
  return { ok: false, signature: null, fill: null, error, refusedBeforeSend: true }
}

async function rpc(method: string, params: unknown): Promise<unknown> {
  const res = await fetch(currentRpc(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  })
  return res.json()
}

/** Wallet balance in SOL, or null when it cannot be read. Null must refuse
 *  upstream rather than default to zero or to plenty. */
export async function walletBalanceSol(owner: string): Promise<number | null> {
  try {
    const body = (await rpc("getBalance", [owner])) as { result?: { value?: number } }
    const lamports = body.result?.value
    return typeof lamports === "number" ? lamports / 1e9 : null
  } catch {
    return null
  }
}

/**
 * Execute one swap for real.
 *
 * `quoteResponse` is passed through to Jupiter VERBATIM. Re-serializing it
 * produces a different quote — a lesson already paid for in this codebase —
 * and the transaction that comes back must be the one we simulate and the one
 * we sign, byte for byte.
 */
export async function executeSwap(
  quoteResponse: unknown,
  owner: string,
  keypairJson: string,
  mint: string,
  direction: Direction,
  guard: { intent: TradeIntent; ctx: PreflightContext } | null,
): Promise<LiveResult> {
  // 1. The entry guard. Exits pass null: a position must never be trapped by
  //    a cap or a breaker, which exist to stop new risk, not to prevent
  //    closing risk already taken.
  if (direction === "entry") {
    if (!guard) return refused("entry attempted with no guard context")
    const refusal = preflight(guard.intent, guard.ctx)
    if (refusal) return refused(refusal)
  }

  // 2. Build the real transaction for the real owner.
  let swapTx: string
  try {
    const res = await fetch(SWAP, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quoteResponse, userPublicKey: owner, wrapAndUnwrapSol: true }),
    })
    if (!res.ok) return refused(`swap build ${res.status}`)
    const built = (await res.json()) as { swapTransaction?: string; error?: string }
    if (!built.swapTransaction) return refused(built.error ?? "no transaction returned")
    swapTx = built.swapTransaction
  } catch (e) {
    return refused(`swap build: ${e instanceof Error ? e.message : String(e)}`)
  }

  // 3. Simulate before spending a fee. Skipped for exits ONLY if it fails for
  //    a reason that would also block the sell forever -- see below; we still
  //    run it, but a failed exit simulation is reported rather than fatal,
  //    because a position we cannot simulate is one we still want to try to
  //    close.
  let simErr: string | null = null
  try {
    const body = (await rpc("simulateTransaction", [
      swapTx,
      { encoding: "base64", sigVerify: false, replaceRecentBlockhash: true },
    ])) as { result?: { value?: { err?: unknown } } }
    const err = body.result?.value?.err
    if (err) simErr = JSON.stringify(err)
  } catch (e) {
    simErr = e instanceof Error ? e.message : String(e)
  }
  if (simErr && direction === "entry") return refused(`simulation failed: ${simErr}`)

  // 4. Sign. From here a fee can be paid.
  let signed: string
  try {
    signed = await signTransaction(swapTx, keypairJson)
  } catch (e) {
    return refused(`signing failed: ${e instanceof Error ? e.message : String(e)}`)
  }

  // 5. Broadcast.
  let signature: string
  try {
    const body = (await rpc("sendTransaction", [
      signed,
      { encoding: "base64", skipPreflight: false, maxRetries: 2 },
    ])) as { result?: string; error?: { message?: string } }
    if (body.error) {
      return { ok: false, signature: null, fill: null, error: body.error.message ?? "send rejected", refusedBeforeSend: false }
    }
    if (!body.result) {
      return { ok: false, signature: null, fill: null, error: "no signature returned", refusedBeforeSend: false }
    }
    signature = body.result
  } catch (e) {
    // The genuinely dangerous case: it may or may not have landed. Say UNKNOWN
    // rather than reporting a clean failure, because a retry could double-fill.
    return {
      ok: false, signature: null, fill: null, refusedBeforeSend: false,
      error: `send failed in flight, state UNKNOWN, reconcile before retrying: ${e instanceof Error ? e.message : String(e)}`,
    }
  }

  // 6. Confirm, then reconcile against what the chain actually recorded.
  const fill = await confirmAndReconcile(signature, owner, mint)
  if (!fill) {
    return {
      ok: false, signature, fill: null, refusedBeforeSend: false,
      error: "sent but not reconciled: confirm manually before trading this mint again",
    }
  }
  if (fill.failedOnChain) {
    return { ok: false, signature, fill, error: "transaction landed but failed on chain", refusedBeforeSend: false }
  }
  return { ok: true, signature, fill, error: null, refusedBeforeSend: false }
}

/**
 * Wait for confirmation, then read the realized fill.
 *
 * A send that is not confirmed is NOT a completed trade. Returning null here
 * makes the caller record an unreconciled position rather than a fictional
 * one, which is the difference between a book that drifts from the chain and
 * one that admits it does not know.
 */
export async function confirmAndReconcile(
  signature: string,
  owner: string,
  mint: string,
  attempts = 20,
  waitMs = 1_500,
  sleep: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
): Promise<RealizedFill | null> {
  for (let i = 0; i < attempts; i++) {
    try {
      const body = (await rpc("getTransaction", [
        signature,
        { encoding: "json", commitment: "confirmed", maxSupportedTransactionVersion: 0 },
      ])) as { result?: TxMeta | null }
      if (body.result) {
        const fill = realizedFill(body.result, owner, mint)
        if (fill) return fill
      }
    } catch {
      // Transient RPC failure; the attempt budget bounds this.
    }
    if (i < attempts - 1) await sleep(waitMs)
  }
  return null
}

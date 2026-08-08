/**
 * Dry-run execution: build the real transaction, simulate it against live
 * mainnet state, and never broadcast.
 *
 * There is deliberately NO send path in this module. Safety here is structural,
 * not a flag someone can flip by accident: the code that would call
 * sendTransaction does not exist. When execution arms, it arrives as a new
 * module behind the policy envelope's signature check, and this one keeps
 * working exactly as it does now.
 *
 * Simulating is a stronger entry gate than quoting alone. A quote says a route
 * exists and prices it. A simulation runs the actual instructions against real
 * account state and catches what a quote cannot: missing token accounts,
 * transfer hooks on Token-2022 mints, program errors, compute overruns. A token
 * that quotes cleanly but fails simulation is one you would have paid fees to
 * fail at, and no competitor screens for it because it costs an extra call.
 */

const SWAP = "https://lite-api.jup.ag/swap/v1/swap"
const RPC = "https://api.mainnet-beta.solana.com"

/**
 * Stand-in owner for dry runs.
 *
 * Simulation runs with sigVerify off, so no key is needed and none is used.
 * This is a well-known system account that exists on chain, which is what lets
 * the simulation reach the swap logic instead of failing on a missing account.
 */
const DRY_RUN_OWNER = "11111111111111111111111111111112"

export interface DryRun {
  ok: boolean
  /** Compute units the swap actually consumes. Sizes the CU limit later. */
  unitsConsumed: number | null
  /** Priority fee in lamports Jupiter judged appropriate for current traffic. */
  priorityFeeLamports: number | null
  /** Simulation error, when the transaction would have failed on chain. */
  error: string | null
}

interface SwapResponse {
  swapTransaction?: string
  prioritizationFeeLamports?: number
  error?: string
}

interface SimResponse {
  result?: { value?: { err?: unknown; unitsConsumed?: number; logs?: string[] } }
  error?: { message?: string }
}

/**
 * Builds and simulates the swap described by a Jupiter quote.
 *
 * Returns ok:false for any failure, including network trouble. Callers must
 * treat that as "do not enter": an unverifiable trade is not a safe trade, and
 * declining to enter costs nothing while entering blind costs the position.
 */
export async function dryRunSwap(quoteResponse: unknown): Promise<DryRun> {
  const fail = (error: string): DryRun => ({
    ok: false, unitsConsumed: null, priorityFeeLamports: null, error,
  })

  let built: SwapResponse
  try {
    const res = await fetch(SWAP, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quoteResponse,
        userPublicKey: DRY_RUN_OWNER,
        wrapAndUnwrapSol: true,
      }),
    })
    if (!res.ok) return fail(`swap build ${res.status}`)
    built = (await res.json()) as SwapResponse
  } catch (e) {
    return fail(`swap build: ${e instanceof Error ? e.message : String(e)}`)
  }

  if (!built.swapTransaction) return fail(built.error ?? "no transaction returned")
  const priorityFeeLamports = built.prioritizationFeeLamports ?? null

  try {
    const res = await fetch(RPC, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "simulateTransaction",
        params: [
          built.swapTransaction,
          {
            encoding: "base64",
            // No signatures exist on a dry run, and the blockhash Jupiter
            // embedded may already be stale by the time we simulate.
            sigVerify: false,
            replaceRecentBlockhash: true,
          },
        ],
      }),
    })
    if (!res.ok) return { ...fail(`simulate ${res.status}`), priorityFeeLamports }
    const body = (await res.json()) as SimResponse
    if (body.error) return { ...fail(body.error.message ?? "rpc error"), priorityFeeLamports }

    const value = body.result?.value
    if (value?.err) {
      return {
        ok: false,
        unitsConsumed: value.unitsConsumed ?? null,
        priorityFeeLamports,
        error: JSON.stringify(value.err),
      }
    }
    return {
      ok: true,
      unitsConsumed: value?.unitsConsumed ?? null,
      priorityFeeLamports,
      error: null,
    }
  } catch (e) {
    return { ...fail(`simulate: ${e instanceof Error ? e.message : String(e)}`), priorityFeeLamports }
  }
}

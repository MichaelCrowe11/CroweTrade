/**
 * Direct chain reads.
 *
 * This is the first piece of real in-house data in the app: rather than asking
 * an aggregator whether a token is safe and trusting its boolean, we read the
 * mint account ourselves and decide. It is a small slice of what the full
 * decode pipeline will do, but it is the same principle and it resolves the two
 * highest-consequence gates today.
 *
 * A revoked mint authority means the deployer cannot print supply. A revoked
 * freeze authority means they cannot freeze your token account and strand you
 * in a position. Both are single-field reads, both are free, and neither is
 * something price history can tell you.
 *
 * NOTE: the public endpoint below is rate limited and is not suitable for
 * production. It is the same swap-the-adapter story as the feed: the call shape
 * does not change when the endpoint does.
 */

const RPC = "https://api.mainnet-beta.solana.com"

/** getMultipleAccounts caps at 100 addresses; stay well under to avoid 413s. */
const MAX_ACCOUNTS_PER_CALL = 50

export interface MintFacts {
  mintAuthority: string | null
  freezeAuthority: string | null
  supply: bigint
  decimals: number
}

interface ParsedMintAccount {
  data?: {
    parsed?: {
      info?: {
        mintAuthority?: string | null
        freezeAuthority?: string | null
        supply?: string
        decimals?: number
      }
    }
  }
}

interface RpcResponse<T> {
  result?: { value?: T }
  error?: { message?: string }
}

async function rpc<T>(method: string, params: unknown[], signal: AbortSignal): Promise<T> {
  const res = await fetch(RPC, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
    signal,
  })
  if (!res.ok) throw new Error(`${method} -> ${res.status}`)
  const body = (await res.json()) as RpcResponse<T>
  if (body.error) throw new Error(`${method}: ${body.error.message ?? "rpc error"}`)
  if (body.result?.value === undefined) throw new Error(`${method}: empty result`)
  return body.result.value
}

/**
 * Reads mint authorities for many mints.
 *
 * Absence from the returned map means "we could not read it", which the caller
 * must keep as unknown. It must NOT be collapsed into "revoked", because that
 * would turn a network failure into a safety pass, which is exactly the class
 * of bug that loses money quietly.
 */
export async function fetchMintFacts(
  mints: string[],
  signal: AbortSignal,
): Promise<Map<string, MintFacts>> {
  const out = new Map<string, MintFacts>()
  if (mints.length === 0) return out

  const batches: string[][] = []
  for (let i = 0; i < mints.length; i += MAX_ACCOUNTS_PER_CALL) {
    batches.push(mints.slice(i, i + MAX_ACCOUNTS_PER_CALL))
  }

  for (const batch of batches) {
    let accounts: (ParsedMintAccount | null)[]
    try {
      accounts = await rpc<(ParsedMintAccount | null)[]>(
        "getMultipleAccounts",
        [batch, { encoding: "jsonParsed" }],
        signal,
      )
    } catch {
      // A failed batch leaves those mints unknown. Deliberately not fatal: a
      // partially-resolved panel beats an error screen, and unknown is safe.
      continue
    }

    accounts.forEach((account, i) => {
      const mint = batch[i]
      const info = account?.data?.parsed?.info
      if (!mint || !info || info.supply === undefined || info.decimals === undefined) return
      out.set(mint, {
        // jsonParsed omits these fields entirely when revoked, so undefined and
        // null both mean revoked here. Anything else is a live authority.
        mintAuthority: info.mintAuthority ?? null,
        freezeAuthority: info.freezeAuthority ?? null,
        supply: BigInt(info.supply),
        decimals: info.decimals,
      })
    })
  }

  return out
}

/**
 * What actually happened, read from the chain.
 *
 * A quote is a prediction and a fill is a fact, and the gap between them is
 * the number this project has been wrong about before: the invented slippage
 * model flattered every paper fill by 20x until a real Jupiter quote was
 * measured against it. Once real money moves, the same mistake becomes
 * unfalsifiable unless the book records what the chain says rather than what
 * the router promised.
 *
 * So every live fill is reconciled: after a transaction confirms, read its
 * pre- and post-balances and derive the REALIZED amounts. Those are what get
 * written to `positions`. The quote is kept alongside, purely so the
 * difference can be measured — that difference is real TCA, as opposed to the
 * old repo's version which returned hardcoded constants.
 *
 * No runtime imports, so this is testable under strip-types.
 */

/** Lamports per SOL, as a bigint: dividing bigints truncates, and this project
 *  already shipped "9.9 SOL" rendered as "9" once. Convert only at display. */
export const LAMPORTS_PER_SOL = 1_000_000_000n

export interface TokenBalance {
  accountIndex: number
  mint: string
  owner?: string
  uiTokenAmount: { amount: string; decimals: number }
}

/** The subset of a Solana transaction we need. Everything is optional because
 *  an RPC can return a confirmed transaction with meta missing entirely. */
export interface TxMeta {
  meta?: {
    err?: unknown
    fee?: number
    preBalances?: number[]
    postBalances?: number[]
    preTokenBalances?: TokenBalance[]
    postTokenBalances?: TokenBalance[]
  } | null
}

export interface RealizedFill {
  /** Net lamports the fee payer lost (buy) or gained (sell), fee INCLUDED. */
  solDeltaLamports: bigint
  /** Net base units of the traded mint the owner gained (buy) or lost (sell). */
  tokenDelta: bigint
  /** Decimals of the traded mint, needed to render the token delta. */
  decimals: number
  /** Network + priority fee actually paid, in lamports. */
  feeLamports: bigint
  /** True when the transaction landed but its instructions failed. */
  failedOnChain: boolean
}

/**
 * Sum an owner's balance for one mint across every token account they hold.
 *
 * Per-account rather than per-owner is how the RPC reports it, and a wallet
 * can legitimately hold the same mint in more than one account. Summing is the
 * only correct reading; taking the first would silently undercount.
 */
function ownerTokenTotal(
  balances: TokenBalance[] | undefined,
  owner: string,
  mint: string,
): { amount: bigint; decimals: number | null } {
  let amount = 0n
  let decimals: number | null = null
  for (const b of balances ?? []) {
    if (b.mint !== mint) continue
    // A balance entry without an owner cannot be attributed, and guessing is
    // how a reconciler credits someone else's tokens to us.
    if (b.owner !== owner) continue
    amount += BigInt(b.uiTokenAmount.amount)
    decimals = b.uiTokenAmount.decimals
  }
  return { amount, decimals }
}

/**
 * Derive the realized fill from a confirmed transaction.
 *
 * `feePayerIndex` is 0 for every transaction we build: the signer pays. It is
 * a parameter rather than a constant so the assumption is visible.
 *
 * Returns null when the transaction carries no meta, which is a real state
 * (the RPC has the signature but not the detail yet) and must not be confused
 * with a zero fill. A zero fill would tell the book the trade did nothing.
 */
export function realizedFill(
  tx: TxMeta,
  owner: string,
  mint: string,
  feePayerIndex = 0,
): RealizedFill | null {
  const m = tx.meta
  if (!m) return null
  const pre = m.preBalances
  const post = m.postBalances
  if (!pre || !post || pre.length <= feePayerIndex || post.length <= feePayerIndex) return null

  const preLamports = BigInt(pre[feePayerIndex] as number)
  const postLamports = BigInt(post[feePayerIndex] as number)
  // Positive = SOL left the wallet (a buy). Negative = SOL arrived (a sell).
  const solDeltaLamports = preLamports - postLamports

  const before = ownerTokenTotal(m.preTokenBalances, owner, mint)
  const after = ownerTokenTotal(m.postTokenBalances, owner, mint)
  const decimals = after.decimals ?? before.decimals
  if (decimals === null) return null

  return {
    solDeltaLamports,
    tokenDelta: after.amount - before.amount,
    decimals,
    feeLamports: BigInt(m.fee ?? 0),
    failedOnChain: m.err !== null && m.err !== undefined,
  }
}

export interface Tca {
  /** What the router said we would get, in base units or lamports. */
  quoted: bigint
  /** What we actually got. */
  realized: bigint
  /** Signed percent: negative means the fill was worse than quoted. */
  slippagePct: number
}

/**
 * Quoted against realized. This is the honest TCA the old Python stack only
 * claimed to have.
 *
 * A zero quote returns 0% rather than dividing: an unquotable trade is a
 * different problem, and an Infinity here would poison every aggregate that
 * touched it.
 */
export function tca(quoted: bigint, realized: bigint): Tca {
  const slippagePct =
    quoted === 0n ? 0 : (Number(realized - quoted) / Number(quoted)) * 100
  return { quoted, realized, slippagePct }
}

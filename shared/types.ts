/**
 * The normalized event vocabulary.
 *
 * This is the contract every feed adapter writes to and every consumer reads
 * from. It exists so that the question "where did this data come from" is
 * answered in exactly one place, the adapter, and nowhere else in the system.
 *
 * Three feeds will produce these events:
 *   - historical backfill from the Old Faithful ledger archive
 *   - live streaming (a gRPC transaction stream now, DoubleZero Edge later)
 *   - replay from our own store, for tests and for backtests
 *
 * They MUST produce identical events for identical on-chain activity. If the
 * backfill path derives liquidity one way and the live path derives it another,
 * models train on one distribution and execute against a different one. That is
 * train/serve skew: the backtest looks excellent, live performance is poor, and
 * nothing in the logs explains it. `parity.ts` exists to catch that.
 */

/** Base-58 SPL token mint address. Narrowed for readability, not validated here. */
export type Mint = string

/** Milliseconds since epoch, sourced from block time, never from wall clock. */
export type BlockTime = number

/** Which venue program produced an event. Drives decoder dispatch. */
export type Venue =
  | "pumpfun-curve"
  | "pump-amm"
  | "raydium-cpmm"
  | "raydium-clmm"
  | "meteora-dlmm"
  | "orca-whirlpool"

/**
 * Provenance travels with every event.
 *
 * `source` is what makes a parity test possible at all: to compare a live event
 * against its archival re-derivation you must be able to tell them apart. It is
 * also the honest answer to "how stale is this", since a backfill event may
 * arrive hours after the block it describes.
 */
export interface Provenance {
  source: "live" | "backfill" | "replay"
  /** Solana slot. The real ordering key. Timestamps can tie; slots do not. */
  slot: number
  /** Signature of the transaction this was decoded from. */
  signature: string
  /** When our system observed it. Differs from blockTime, sometimes by hours. */
  observedAt: BlockTime
}

interface EventBase {
  mint: Mint
  venue: Venue
  /** Block time, NOT observation time. Every feature must be computed on this. */
  blockTime: BlockTime
  provenance: Provenance
}

/** A token was created. First sighting. */
export interface TokenLaunched extends EventBase {
  kind: "token.launched"
  deployer: string
  name: string
  symbol: string
  decimals: number
  initialSupply: bigint
}

/** A swap executed against a pool. */
export interface SwapExecuted extends EventBase {
  kind: "swap.executed"
  trader: string
  /** True when the trader received the token, false when they sold it. */
  isBuy: boolean
  /** Lamports of SOL moved. Integer, never float: floats lose lamports. */
  solLamports: bigint
  tokenAmount: bigint
}

/** Pool reserves changed, from any cause. The basis for price and depth. */
export interface LiquidityChanged extends EventBase {
  kind: "liquidity.changed"
  solReserveLamports: bigint
  tokenReserve: bigint
}

/**
 * A mint's authorities changed.
 *
 * `null` means the authority was revoked, which is the safe state. This is one
 * of the highest-signal events in the whole system: an un-revoked mint authority
 * means the deployer can print supply at will, and no amount of price history
 * tells you that.
 */
export interface AuthorityChanged extends EventBase {
  kind: "authority.changed"
  mintAuthority: string | null
  freezeAuthority: string | null
}

/** LP tokens were burned or locked. Signals the deployer cannot pull liquidity. */
export interface LiquidityLocked extends EventBase {
  kind: "liquidity.locked"
  /** Basis points of LP supply burned or locked. 10000 = fully burned. */
  lockedBps: number
  method: "burn" | "lock-contract"
}

export type TokenEvent =
  | TokenLaunched
  | SwapExecuted
  | LiquidityChanged
  | AuthorityChanged
  | LiquidityLocked

export type TokenEventKind = TokenEvent["kind"]

/**
 * A feed of normalized events.
 *
 * Deliberately an async iterable rather than a callback or an EventEmitter:
 * backpressure comes free. If the consumer is slow to process a block, the
 * adapter simply is not asked for the next one. With an EventEmitter a slow
 * consumer silently accumulates an unbounded queue, and under a launch burst
 * that is how you run out of memory at the exact moment you most wanted to be
 * trading.
 */
export interface FeedAdapter {
  readonly name: string
  readonly source: Provenance["source"]
  events(signal: AbortSignal): AsyncIterable<TokenEvent>
  close(): Promise<void>
}

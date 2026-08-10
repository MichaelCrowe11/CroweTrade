import { test } from "node:test"
import assert from "node:assert/strict"
import { realizedFill, tca, type TxMeta } from "../../../shared/reconcile.ts"

/**
 * Reconciliation decides what the book believes actually happened, so the
 * tests care most about the cases where a naive reader would record a
 * confident wrong number: missing meta read as a zero fill, a second token
 * account silently ignored, or someone else's balance credited to us.
 */

const OWNER = "OwnerPubkey"
const MINT = "MintPubkey"

const bal = (accountIndex: number, owner: string, mint: string, amount: string, decimals = 6) =>
  ({ accountIndex, mint, owner, uiTokenAmount: { amount, decimals } })

test("a buy: SOL leaves, tokens arrive", () => {
  const tx: TxMeta = {
    meta: {
      err: null,
      fee: 5_000,
      preBalances: [1_000_000_000, 0],
      postBalances: [900_000_000, 0],
      preTokenBalances: [],
      postTokenBalances: [bal(1, OWNER, MINT, "250000")],
    },
  }
  const f = realizedFill(tx, OWNER, MINT)
  assert.ok(f)
  assert.equal(f.solDeltaLamports, 100_000_000n, "positive means SOL left the wallet")
  assert.equal(f.tokenDelta, 250_000n)
  assert.equal(f.feeLamports, 5_000n)
  assert.equal(f.failedOnChain, false)
})

test("a sell: tokens leave, SOL arrives as a NEGATIVE sol delta", () => {
  const tx: TxMeta = {
    meta: {
      err: null,
      fee: 5_000,
      preBalances: [900_000_000, 0],
      postBalances: [980_000_000, 0],
      preTokenBalances: [bal(1, OWNER, MINT, "250000")],
      postTokenBalances: [bal(1, OWNER, MINT, "0")],
    },
  }
  const f = realizedFill(tx, OWNER, MINT)
  assert.ok(f)
  assert.equal(f.solDeltaLamports, -80_000_000n, "negative means SOL arrived")
  assert.equal(f.tokenDelta, -250_000n)
})

test("MISSING META returns null, never a zero fill", () => {
  // The dangerous case: the RPC knows the signature but not the detail yet.
  // A zero fill would tell the book the trade did nothing, which is a lie.
  assert.equal(realizedFill({ meta: null }, OWNER, MINT), null)
  assert.equal(realizedFill({}, OWNER, MINT), null)
  assert.equal(
    realizedFill({ meta: { preBalances: [1], postBalances: undefined } }, OWNER, MINT),
    null,
  )
})

test("balances belonging to someone else are never credited to us", () => {
  const tx: TxMeta = {
    meta: {
      err: null, fee: 0,
      preBalances: [100, 0], postBalances: [100, 0],
      preTokenBalances: [],
      postTokenBalances: [bal(1, "SomeoneElse", MINT, "999999")],
    },
  }
  // No balance attributable to us, so decimals are unknown and this is null
  // rather than a fabricated zero-token fill.
  assert.equal(realizedFill(tx, OWNER, MINT), null)
})

test("a mint held across TWO token accounts is summed, not sampled", () => {
  const tx: TxMeta = {
    meta: {
      err: null, fee: 0,
      preBalances: [100, 0], postBalances: [100, 0],
      preTokenBalances: [],
      postTokenBalances: [bal(1, OWNER, MINT, "100"), bal(2, OWNER, MINT, "50")],
    },
  }
  const f = realizedFill(tx, OWNER, MINT)
  assert.ok(f)
  assert.equal(f.tokenDelta, 150n, "both accounts must count")
})

test("other mints in the same transaction are ignored", () => {
  const tx: TxMeta = {
    meta: {
      err: null, fee: 0,
      preBalances: [100, 0], postBalances: [100, 0],
      preTokenBalances: [],
      postTokenBalances: [bal(1, OWNER, MINT, "10"), bal(2, OWNER, "OtherMint", "9999")],
    },
  }
  assert.equal(realizedFill(tx, OWNER, MINT)?.tokenDelta, 10n)
})

test("a transaction that landed but FAILED is flagged, not silently accepted", () => {
  const tx: TxMeta = {
    meta: {
      err: { InstructionError: [0, "Custom"] }, fee: 5_000,
      preBalances: [100, 0], postBalances: [95, 0],
      preTokenBalances: [], postTokenBalances: [bal(1, OWNER, MINT, "0")],
    },
  }
  assert.equal(realizedFill(tx, OWNER, MINT)?.failedOnChain, true)
})

test("TCA reports a worse fill as negative slippage", () => {
  const t = tca(1_000n, 950n)
  assert.equal(t.slippagePct, -5)
  assert.equal(tca(1_000n, 1_050n).slippagePct, 5)
})

test("a zero quote yields 0%, never Infinity", () => {
  // An Infinity here would poison every aggregate that touched it.
  assert.equal(tca(0n, 100n).slippagePct, 0)
  assert.ok(Number.isFinite(tca(0n, 0n).slippagePct))
})

test("large base-unit amounts stay exact: bigint, never float", () => {
  const huge = "9007199254740993" // 2^53 + 1, unrepresentable as a JS number
  const tx: TxMeta = {
    meta: {
      err: null, fee: 0,
      preBalances: [0, 0], postBalances: [0, 0],
      preTokenBalances: [],
      postTokenBalances: [bal(1, OWNER, MINT, huge)],
    },
  }
  assert.equal(realizedFill(tx, OWNER, MINT)?.tokenDelta, BigInt(huge))
})

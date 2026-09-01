import test from "node:test"
import assert from "node:assert/strict"
import { decideExits, decideEntries, emptyFunnel, type OpenPosition } from "./strategy.ts"
import { PAPER_POLICY } from "../../shared/policy.ts"
import type { Candidate } from "../../shared/dexscreener.ts"

const NOW = 1_800_000_000_000

function pos(over: Partial<OpenPosition> = {}): OpenPosition {
  return {
    id: "p1", mint: "M", symbol: "M",
    // The 2026-08-31 basis mismatch, verbatim: a fill of 0.75 against a
    // listing mark of 1.0 at the same tick.
    entryPriceUsd: 0.75, entryMarkUsd: 1.0, entrySolUsd: 103,
    sizeSol: 0.1, sizeUsd: 10.3, tokenAmount: 13.7,
    openedAt: NOW - 60_000, policyHash: "h", verdictAtEntry: "caution",
    ...over,
  }
}
const policy = {
  ...PAPER_POLICY,
  exit: { ...PAPER_POLICY.exit, takeProfitPct: 20, stopLossPct: 50, timeStopMinutes: 5 },
}
const prices = (px: number) => new Map([["M", { priceUsd: px, verdict: "caution" as const }]])

test("exits compare the mark against the entry MARK, not the fill", () => {
  // +10% on a consistent basis. On the old basis this read +46.7% and took profit.
  assert.deepEqual(decideExits([pos()], prices(1.1), policy, NOW), [])
  assert.equal(decideExits([pos()], prices(1.21), policy, NOW)[0]?.reason, "take-profit")
  assert.equal(decideExits([pos()], prices(0.49), policy, NOW)[0]?.reason, "stop-loss")
  // -35% on the old basis was a true -52%; on the new basis -35% is held.
  assert.deepEqual(decideExits([pos()], prices(0.65), policy, NOW), [])
  assert.equal(
    decideExits([pos({ openedAt: NOW - 5 * 60_000 })], prices(1.05), policy, NOW)[0]?.reason,
    "time-stop",
  )
})

test("rows written before entry_mark existed fall back to the fill", () => {
  const legacy = pos({ entryMarkUsd: null })
  assert.equal(decideExits([legacy], prices(0.9), policy, NOW)[0]?.reason, "take-profit")
})

function cand(over: Partial<Candidate["snapshot"]> = {}): Candidate {
  return {
    mint: "C", symbol: "C", name: "C", dex: "pumpfun-curve", origin: "launchpad", pool: null,
    priceUsd: 0.000001, changeH1: null, liquidityUsd: 2000, volume24h: null,
    buys24h: null, sells24h: null, createdAt: NOW - 5 * 60_000, creator: "K",
    snapshot: {
      mint: "C", asOf: NOW, launchedAt: NOW - 5 * 60_000,
      mintAuthority: null, freezeAuthority: null, lpLockedBps: 10_000,
      topHolderShare: undefined, solReserveLamports: 20_000_000_000n,
      deployerPriorMints: undefined, deployerPriorRugs: undefined,
      ...over,
    },
  } as Candidate
}
const traj = new Map([["C", { prices: [1, 1.1, 1.2], liquidity: [1000, 1100, 1200] }]])
const admit = (c: Candidate) => {
  const funnel = emptyFunnel()
  const out = decideEntries([c], [], 0, 100, PAPER_POLICY, NOW, traj, undefined, undefined, funnel)
  return { admitted: out.length, refused: funnel.deployerRefused }
}

test("deployer history refuses prior rugs and factories, and passes unknown", () => {
  assert.deepEqual(admit(cand()), { admitted: 1, refused: 0 })
  assert.deepEqual(admit(cand({ deployerPriorMints: 3, deployerPriorRugs: 1 })), { admitted: 0, refused: 1 })
  assert.deepEqual(admit(cand({ deployerPriorMints: 11, deployerPriorRugs: 0 })), { admitted: 0, refused: 1 })
  assert.deepEqual(admit(cand({ deployerPriorMints: 10, deployerPriorRugs: 0 })), { admitted: 1, refused: 0 })
})

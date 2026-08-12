import test, { type TestContext } from "node:test"
import assert from "node:assert/strict"
import {
  fetchLaunchpadCandidates,
  requiredLaunchpadHistoryMs,
} from "../../shared/pumpfun.ts"

const NOW = 2_000_000_000_000

function coin(mint: string, ageMinutes: number) {
  return {
    mint,
    name: mint,
    symbol: mint,
    creator: `creator-${mint}`,
    created_timestamp: NOW - ageMinutes * 60_000,
    complete: false,
    real_sol_reserves: 1_000_000_000,
    total_supply: 1_000_000,
    base_decimals: 6,
    usd_market_cap: 1_000,
  }
}

function offsetOf(input: string | URL | Request): number {
  return Number(new URL(String(input)).searchParams.get("offset"))
}

function installFetch(
  t: TestContext,
  handler: (offset: number) => Response | Promise<Response>,
): number[] {
  const offsets: number[] = []
  const originalFetch = globalThis.fetch
  const originalNow = Date.now
  globalThis.fetch = async (input) => {
    const offset = offsetOf(input)
    offsets.push(offset)
    return handler(offset)
  }
  Date.now = () => NOW
  t.after(() => {
    globalThis.fetch = originalFetch
    Date.now = originalNow
  })
  return offsets
}

const requirements = {
  minTokenAgeMinutes: 3,
  minObservedTicks: 3,
  pollIntervalMs: 60_000,
}

test("derives a four-minute discovery horizon from the entry rules", () => {
  assert.equal(requiredLaunchpadHistoryMs(requirements), 4 * 60_000)
})

test("paginates until timestamps, not a fixed page count, cover the horizon", async (t) => {
  const offsets = installFetch(t, (offset) => {
    const age = offset === 0 ? 1 : offset === 70 ? 3 : 5
    return Response.json([coin(`mint-${offset}`, age)])
  })

  const result = await fetchLaunchpadCandidates(100, new AbortController().signal, requirements)

  assert.deepEqual(offsets, [0, 70, 140])
  assert.equal(result.complete, true)
  assert.equal(result.coveredHistoryMs, 5 * 60_000)
  assert.deepEqual(result.candidates.map((candidate) => candidate.mint), ["mint-0", "mint-70", "mint-140"])
})

test("deduplicates overlapping pages by first sighting", async (t) => {
  installFetch(t, (offset) => Response.json(
    offset === 0
      ? [coin("overlap", 1)]
      : [coin("overlap", 2), coin("older", 5)],
  ))

  const result = await fetchLaunchpadCandidates(100, new AbortController().signal, requirements)

  assert.equal(result.complete, true)
  assert.deepEqual(result.candidates.map((candidate) => candidate.mint), ["overlap", "older"])
})

test("a failed page does not discard healthy pages, but coverage is incomplete", async (t) => {
  const offsets = installFetch(t, (offset) => offset === 0
    ? new Response("upstream failure", { status: 503 })
    : Response.json([coin("healthy", 5)]))

  const result = await fetchLaunchpadCandidates(100, new AbortController().signal, requirements)

  assert.deepEqual(offsets, [0, 70])
  assert.deepEqual(result.candidates.map((candidate) => candidate.mint), ["healthy"])
  assert.equal(result.complete, false)
  assert.deepEqual(result.failedOffsets, [0])
})

test("reports when the page budget ends before the required history", async (t) => {
  const offsets = installFetch(t, (offset) => Response.json([coin(`mint-${offset}`, 1)]))

  const result = await fetchLaunchpadCandidates(100, new AbortController().signal, {
    ...requirements,
    maxPages: 2,
  })

  assert.deepEqual(offsets, [0, 70])
  assert.equal(result.complete, false)
  assert.equal(result.pagesAttempted, 2)
  assert.equal(result.coveredHistoryMs, 60_000)
})

test("does not turn a tick-wide abort into a recoverable page failure", async (t) => {
  const controller = new AbortController()
  installFetch(t, () => {
    controller.abort()
    throw new Error("aborted")
  })

  await assert.rejects(
    fetchLaunchpadCandidates(100, controller.signal, requirements),
    /aborted/,
  )
})

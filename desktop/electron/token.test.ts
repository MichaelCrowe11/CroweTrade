import { test } from "node:test"
import assert from "node:assert/strict"
import { createTokenCache } from "./token.ts"

test("within the ttl the fetcher runs once and the token is reused", async () => {
  let calls = 0
  const cache = createTokenCache(async () => {
    calls++
    return `tok-${calls}`
  }, 10_000)
  assert.equal(await cache.get(1_000), "tok-1")
  assert.equal(await cache.get(5_000), "tok-1")
  assert.equal(calls, 1)
})

test("past the ttl the token refreshes", async () => {
  let calls = 0
  const cache = createTokenCache(async () => {
    calls++
    return `tok-${calls}`
  }, 10_000)
  assert.equal(await cache.get(1_000), "tok-1")
  assert.equal(await cache.get(12_000), "tok-2")
  assert.equal(calls, 2)
})

test("concurrent gets share one in-flight fetch", async () => {
  let calls = 0
  let release: (v: string) => void = () => {}
  const cache = createTokenCache(() => {
    calls++
    return new Promise<string>((res) => {
      release = res
    })
  }, 10_000)
  const a = cache.get(1_000)
  const b = cache.get(1_001)
  release("tok")
  assert.equal(await a, "tok")
  assert.equal(await b, "tok")
  assert.equal(calls, 1)
})

test("a failed fetch does not poison the cache; the next get retries", async () => {
  let calls = 0
  const cache = createTokenCache(async () => {
    calls++
    if (calls === 1) throw new Error("az flaked")
    return "tok-2"
  }, 10_000)
  await assert.rejects(cache.get(1_000), /az flaked/)
  assert.equal(await cache.get(2_000), "tok-2")
  assert.equal(calls, 2)
})

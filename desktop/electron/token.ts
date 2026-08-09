import { execFile } from "node:child_process"

/**
 * Azure token cache for the Analyst and the Orchestrator.
 *
 * The token used to be fetched with a BLOCKING execFileSync on every single
 * ask, which froze the main process (and therefore every native window
 * interaction) for up to a second per question, for a credential that lives
 * about an hour. Now: async, cached for 30 minutes, one in-flight fetch
 * shared by concurrent callers, and a failed fetch clears itself so the next
 * ask retries instead of inheriting a rejection forever.
 *
 * The cache core is injected-clock, injected-fetcher, and tested.
 */

export interface TokenCache {
  get(nowMs: number): Promise<string>
}

export function createTokenCache(fetcher: () => Promise<string>, ttlMs: number): TokenCache {
  let token: string | null = null
  let fetchedAt = 0
  let inflight: Promise<string> | null = null

  return {
    get(nowMs) {
      if (token !== null && nowMs - fetchedAt < ttlMs) return Promise.resolve(token)
      if (inflight) return inflight
      inflight = fetcher()
        .then((t) => {
          token = t
          fetchedAt = nowMs
          inflight = null
          return t
        })
        .catch((e: unknown) => {
          inflight = null
          throw e
        })
      return inflight
    },
  }
}

function fetchAzToken(): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      "az",
      ["account", "get-access-token", "--resource", "https://ai.azure.com", "--query", "accessToken", "-o", "tsv"],
      { encoding: "utf8" },
      (err, stdout) => {
        if (err) reject(err)
        else resolve(stdout.trim())
      },
    )
  })
}

const AZ_TTL_MS = 30 * 60_000
const azCache = createTokenCache(fetchAzToken, AZ_TTL_MS)

/** The shared entry point: cached, async, never blocks the main process. */
export function azToken(): Promise<string> {
  return azCache.get(Date.now())
}

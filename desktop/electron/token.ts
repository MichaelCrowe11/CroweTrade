import { execFile } from "node:child_process"
import { existsSync } from "node:fs"

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

/**
 * Where `az` actually lives, for a GUI process.
 *
 * An app launched from Finder or the Dock inherits launchd's minimal PATH
 * (/usr/bin:/bin:/usr/sbin:/sbin) — NOT the shell's — so Homebrew's
 * /opt/homebrew/bin is absent and a bare execFile("az") fails ENOENT in the
 * installed app while working perfectly from a terminal. Same class as the
 * packaged-resource bugs: what the dev machine supplies, the customer path
 * does not. Candidates are tried in order and the first that exists wins;
 * bare "az" stays last so a PATH that DOES carry it still works.
 */
const AZ_CANDIDATES = [
  "/opt/homebrew/bin/az",
  "/usr/local/bin/az",
  "/usr/bin/az",
  "az",
]

function resolveAz(): string {
  for (const p of AZ_CANDIDATES) {
    if (p !== "az" && existsSync(p)) return p
  }
  return "az"
}

function fetchAzToken(): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      resolveAz(),
      ["account", "get-access-token", "--resource", "https://ai.azure.com", "--query", "accessToken", "-o", "tsv"],
      { encoding: "utf8" },
      (err, stdout) => {
        // Name the real cause. "spawn az ENOENT" tells the operator nothing
        // about which of "not installed" or "not on this process's PATH" they
        // are looking at, and they are different problems with different fixes.
        if (err) {
          const enoent = (err as NodeJS.ErrnoException).code === "ENOENT"
          reject(enoent
            ? new Error(
              "Azure CLI not found by the app. It resolves from a terminal but not " +
              "from a Finder-launched app, which inherits a minimal PATH. Checked: " +
              AZ_CANDIDATES.join(", "),
            )
            : err)
        } else resolve(stdout.trim())
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

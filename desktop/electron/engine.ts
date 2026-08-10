import { app } from "electron"
import * as fs from "node:fs"
import * as path from "node:path"

/**
 * The desktop's connection to its own engine.
 *
 * This file replaced the Azure rail on 2026-08-09, when Azure revoked the
 * Foundry credits and took the Analyst and Orchestrator down with them. Both
 * now reach a model through the engine Worker, which holds the Workers AI
 * binding server-side. The consequence worth stating plainly: THIS APP SHIPS
 * NO MODEL CREDENTIAL. It carries the operator's engine admin token, which it
 * already needed, and nothing else.
 *
 * The old rail shelled out to `az` for a token. That failed in the installed
 * app for a reason worth remembering: an app launched from Finder inherits
 * launchd's minimal PATH, so Homebrew's bin is absent and `az` resolves from a
 * terminal but ENOENTs from the Dock. Shelling out to developer tooling is not
 * a credential strategy for a shipped product.
 */

export const ENGINE = "https://crowetrade-engine.yellow-block-3adc.workers.dev"

let cached: string | null = null

/**
 * Resolve the engine admin token.
 *
 * Order is deliberate: an explicit environment override first (so a test or a
 * second operator can point elsewhere without touching disk), then the app's
 * own userData copy, then the repo file in development. The token is NEVER
 * bundled -- a secret inside a distributed binary is a published secret.
 */
export function adminToken(): string {
  if (cached) return cached
  const fromEnv = process.env.CROWETRADE_ADMIN_TOKEN?.trim()
  if (fromEnv) return (cached = fromEnv)

  const candidates = [
    path.join(app.getPath("userData"), "admin-token"),
    // Development only: the repo checkout beside this file.
    path.join(__dirname, "../../engine/.admin-token"),
  ]
  for (const p of candidates) {
    try {
      const v = fs.readFileSync(p, "utf8").trim()
      if (v) return (cached = v)
    } catch {
      // Missing or unreadable; try the next candidate.
    }
  }
  // Named, actionable failure. "401" alone would send the operator hunting
  // through the engine when the actual problem is a file on this machine.
  throw new Error(
    `No engine admin token. Write it to ${candidates[0]} (chmod 600) ` +
    `or set CROWETRADE_ADMIN_TOKEN.`,
  )
}

export function engineHeaders(): Record<string, string> {
  return { Authorization: `Bearer ${adminToken()}`, "Content-Type": "application/json" }
}

/**
 * Chat-completions SSE deltas, normalised.
 *
 * GLM-5.2 is a reasoning model and emits `reasoning_content` deltas alongside
 * `content`. They are separated here rather than concatenated: reasoning is
 * the model thinking, not the model answering, and splicing the two into one
 * stream would put working-out into the operator's transcript.
 */
export interface Deltas {
  onText: (t: string) => void
  onReasoning?: (t: string) => void
}

export async function streamCompletion(
  res: Response,
  d: Deltas,
): Promise<{ text: string; toolCalls: ToolCall[] }> {
  if (!res.ok || !res.body) {
    throw new Error(`engine ${res.status}: ${(await res.text()).slice(0, 200)}`)
  }
  const decoder = new TextDecoder()
  let buf = ""
  let text = ""
  // Tool calls stream in FRAGMENTS: name arrives on one delta, arguments
  // accumulate across many, keyed by index. Assembling them per index is the
  // whole reason this is not a one-line JSON parse.
  const partial = new Map<number, { id?: string; name: string; args: string }>()

  for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
    buf += decoder.decode(chunk, { stream: true })
    const lines = buf.split("\n")
    buf = lines.pop() ?? ""
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue
      const payload = line.slice(6).trim()
      if (!payload || payload === "[DONE]") continue
      let delta: Record<string, unknown>
      try {
        delta = (JSON.parse(payload) as { choices?: { delta?: Record<string, unknown> }[] })
          .choices?.[0]?.delta ?? {}
      } catch {
        continue // a partial frame; the next chunk completes it
      }
      const c = delta["content"]
      if (typeof c === "string" && c) {
        text += c
        d.onText(c)
      }
      const r = delta["reasoning_content"]
      if (typeof r === "string" && r) d.onReasoning?.(r)

      const calls = delta["tool_calls"] as
        | { index?: number; id?: string; function?: { name?: string; arguments?: string } }[]
        | undefined
      for (const tc of calls ?? []) {
        const i = tc.index ?? 0
        const cur = partial.get(i) ?? { name: "", args: "" }
        if (tc.id) cur.id = tc.id
        if (tc.function?.name) cur.name = tc.function.name
        if (tc.function?.arguments) cur.args += tc.function.arguments
        partial.set(i, cur)
      }
    }
  }
  const toolCalls: ToolCall[] = [...partial.entries()]
    .sort(([a], [b]) => a - b)
    .filter(([, v]) => v.name)
    .map(([, v]) => ({ id: v.id, name: v.name, args: v.args }))
  return { text, toolCalls }
}

export interface ToolCall {
  id?: string
  name: string
  args: string
}

/** One turn through the engine's inference passthrough. */
export function llm(body: {
  messages: unknown[]
  tools?: unknown[]
  stream?: boolean
  max_tokens?: number
}): Promise<Response> {
  return fetch(`${ENGINE}/api/llm`, {
    method: "POST",
    headers: engineHeaders(),
    body: JSON.stringify(body),
  })
}

/**
 * CroweTrade Engine: Worker entry.
 *
 * Static assets (the landing page) serve automatically for matching paths;
 * everything else lands here. The cron drives the Ledger's trading tick.
 *
 * Read endpoints are public JSON: the paper record is the demo. Mutating
 * endpoints (kill, veto) require the admin bearer token, compared through a
 * digest so length never leaks.
 */

import { Ledger } from "./ledger.js"
import { analystStream, ANALYST_MODEL } from "./analyst.js"
import { priceFor, paymentRequired, settle, ROUTES, SOLANA_MAINNET } from "./x402.js"
import { tierSatisfied, type Tier } from "../../shared/auth.js"

export { Ledger }

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Authorization, Content-Type",
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  })
}

/**
 * Two capability tiers, one comparison.
 *
 * "admin" is the operator: kill, veto, tick, raw inference. Anything that
 * changes what the engine does with money, or spends without grounding.
 * "research" is a second person: the corpus, the fitted model, the gates, the
 * Analyst. Everything it reaches only reads, or costs inference that arrives
 * attached to a question about the ledger.
 *
 * The admin token satisfies BOTH tiers, so nothing the operator could already
 * do stops working and the installed terminal keeps its single credential. A
 * research token satisfies only its own tier and lives in its own secret, so
 * revoking a collaborator is one `wrangler secret delete ENGINE_RESEARCH_TOKEN`
 * and never rotates the operator's token or locks them out of their own app.
 *
 * When ENGINE_RESEARCH_TOKEN is unset the whole tier is inert, the same way the
 * live-trading path is inert without its secrets. Absence is the safe state.
 */
async function sameToken(presented: string, expected: string | undefined): Promise<boolean> {
  if (!expected) return false
  const enc = new TextEncoder()
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(presented)),
    crypto.subtle.digest("SHA-256", enc.encode(expected)),
  ])
  // Digests, so the comparison never leaks the length of either string.
  return crypto.subtle.timingSafeEqual(a, b)
}

/**
 * What a refused caller is told.
 *
 * TODO(michael): decide this one. It is a real trade-off, not a formality.
 *
 * Today every refusal is an identical 401 "unauthorized", so a collaborator
 * whose research token hits /api/kill sees exactly what a stranger with no
 * token sees. That is the quiet option: a stolen token learns nothing about
 * what it is or what else exists. It is also the option that will have Dannie
 * convinced his credential is broken while it is working perfectly.
 *
 * The alternative is 403 with a reason ("research token cannot reach this
 * route"), which is honest to a colleague and informative to a prober.
 *
 * Only the refusal path calls this, so recomputing the match here is free.
 * Implement the body you want and swap the four admin guards over to it, or
 * delete this and keep the uniform 401.
 */
// async function refusal(req: Request, env: Env, tier: Tier): Promise<Response> {
//   return json({ error: "unauthorized" }, 401)
// }

async function authorized(req: Request, env: Env, tier: Tier = "admin"): Promise<boolean> {
  const token = req.headers.get("Authorization")?.replace(/^Bearer\s+/i, "")
  if (!token) return false
  // Both comparisons always run rather than short-circuiting on the first
  // match, so the work done does not depend on which secret was presented.
  const [admin, research] = await Promise.all([
    sameToken(token, env.ENGINE_ADMIN_TOKEN),
    sameToken(token, env.ENGINE_RESEARCH_TOKEN),
  ])
  return tierSatisfied(tier, { admin, research })
}

function ledger(env: Env) {
  // One global ledger. Multi-tenant later means one DO per user envelope.
  return env.LEDGER.get(env.LEDGER.idFromName("global"))
}

export default {
  async scheduled(_event, env, _ctx): Promise<void> {
    const stub = ledger(env)
    const result = await stub.tick()
    console.log(JSON.stringify({ msg: "tick", ...result }))
    // Checked every tick, fires at most once. Kept out of tick() so a mail
    // provider outage can never fail a trade; awaited rather than left dangling
    // so its log line lands inside this invocation.
    const alert = await stub.maybeAlert()
    if (alert.sent) console.log(JSON.stringify({ msg: "alert", ...alert }))
    // The daily digest: queued at its hour, sent by the flush below.
    const digest = await stub.maybeDigest()
    if (digest.queued) console.log(JSON.stringify({ msg: "digest", ...digest }))
    // Operational alerts (breaker trips, kill flips, scan outages) queue
    // during the tick and send here, on the same no-mail-inside-trading seam.
    // Hourly archive of refused decision rows past retention, one batch a run.
    const arch = await stub.maybeArchive()
    if (arch.archived > 0 || arch.reason.startsWith("verification")) console.log(JSON.stringify({ msg: "archive", ...arch }))
    const ops = await stub.flushAlerts()
    if (ops.sent > 0 || ops.failed > 0) console.log(JSON.stringify({ msg: "opalerts", ...ops }))
  },

  async fetch(req, env, _ctx): Promise<Response> {
    const url = new URL(req.url)
    if (req.method === "OPTIONS") return new Response(null, { headers: CORS })

    try {
      if (url.pathname === "/api/health") {
        return json({ ok: true, service: "crowetrade-engine", mode: "paper" })
      }
      if (url.pathname === "/api/positions" && req.method === "GET") {
        return json(await ledger(env).summary())
      }
      // ── Paid surface (x402) ────────────────────────────────────────────
      //
      // Free endpoints stay free: the paper record is the demo and putting it
      // behind a paywall would hide the one thing that makes this credible.
      // What is sold is the thing that costs us to produce and that nobody
      // else has: chain-read safety gates and the labeled outcome corpus.
      if (url.pathname === "/api/v1" && req.method === "GET") {
        // Discovery. An agent cannot buy what it cannot find, and a bare 402
        // on an unknown path teaches it nothing.
        return json({
          service: "CroweTrade data API",
          payment: { protocol: "x402", version: 2, network: SOLANA_MAINNET, asset: "USDC" },
          configured: Boolean(env.X402_PAY_TO && env.X402_FACILITATOR),
          endpoints: Object.entries(ROUTES).map(([path, r]) => ({
            path: `${path}/{mint}`.replace("/{mint}", path === "/api/v1/safety" ? "/{mint}" : ""),
            priceUsd: (Number(r.amount) / 1_000_000).toFixed(4),
            description: r.description,
          })),
          free: ["/api/health", "/api/positions", "/api/exit-sweep", "/api/entry-sweep", "/api/train"],
        })
      }

      const priced = priceFor(url.pathname)
      if (priced) {
        const payTo = env.X402_PAY_TO
        const facilitator = env.X402_FACILITATOR
        if (!payTo || !facilitator) {
          // Unconfigured is a server problem, not a client one. Saying "402"
          // here would tell an agent to pay an address that does not exist.
          return json(
            { error: "payments_not_configured", detail: "X402_PAY_TO and X402_FACILITATOR are unset" },
            503,
          )
        }

        const sig = req.headers.get("PAYMENT-SIGNATURE")
        if (!sig) return paymentRequired(req, priced, payTo)

        // Settle BEFORE serving. Verifying without settling would hand out
        // paid answers on a promise.
        const result = await settle(sig, priced, payTo, facilitator)
        if (!result.ok) {
          const res = paymentRequired(req, priced, payTo, result.errorReason ?? "settlement_failed")
          res.headers.set("PAYMENT-RESPONSE", result.header)
          return res
        }

        const body =
          url.pathname.startsWith("/api/v1/safety")
            ? await (async () => {
                const mint = url.pathname.split("/").pop() ?? ""
                if (!/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(mint)) {
                  return { error: "invalid_mint", detail: "expected a base58 Solana mint address" }
                }
                return await ledger(env).safetyFor(mint)
              })()
            : ledger(env).corpusStats()

        const res = json(body)
        res.headers.set("PAYMENT-RESPONSE", result.header)
        return res
      }

      // The agent's research surface: read-only SQL over the corpus. Admin
      // token because an unbounded read of our own data is not something to
      // hand to the open internet, even read-only.
      // The Analyst. Authenticated because inference costs money and an open
      // endpoint is someone else's bill; the model itself is read-only by
      // construction (proposals write nothing an engine consults), so the
      // token guards spend, not capability.
      if (url.pathname === "/api/analyst" && req.method === "POST") {
        if (!(await authorized(req, env, "research"))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => ({}))) as { question?: string }
        const question = (body.question ?? "").trim()
        if (!question) return json({ error: "question required" }, 400)
        const stub = ledger(env)
        const { stream, consulted } = await analystStream(env.AI, question, {
          state: () => stub.summary(),
          exitSweep: () => stub.exitSweep(),
          modelFit: () => stub.trainModel(),
          proposePolicy: (args) => stub.proposePolicy(args),
        })
        return new Response(stream, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            // Grounding, visible to the client without polluting the text
            // stream. Final before the body flows: tool rounds run first.
            "X-Analyst-Tools": consulted.join(","),
            ...CORS,
            "Access-Control-Expose-Headers": "X-Analyst-Tools",
          },
        })
      }

      // Inference passthrough for the ORCHESTRATOR.
      //
      // The Analyst's loop runs server-side above because its tools are engine
      // reads. The Orchestrator's cannot: its tools are the operator's shell,
      // panels and notebooks, which exist only on the operator's machine. So it
      // keeps a client-side loop and borrows the model through here.
      //
      // The point of the hop is that no Cloudflare credential ships inside a
      // distributed desktop binary. The app already holds the admin token, so
      // this adds no new secret to the client. It does mean the admin token
      // unlocks inference spend -- acceptable, because that same token already
      // unlocks kill, veto and tick, so the blast radius does not grow.
      // Gates for the terminal's scan list, computed on the engine's better
      // data (Helius authorities, the creators table, the labeled corpus)
      // instead of recomputed in the app against a weaker feed.
      if (url.pathname === "/api/gates" && req.method === "POST") {
        if (!(await authorized(req, env, "research"))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => null)) as {
          mints?: unknown; detail?: unknown
        } | null
        const mints = Array.isArray(body?.mints)
          ? body.mints.filter((m): m is string => typeof m === "string")
          : []
        if (mints.length === 0) return json({ error: "mints required" }, 400)
        const detail = typeof body?.detail === "string" ? body.detail : undefined
        return json(await ledger(env).gatesFor(mints, detail))
      }

      if (url.pathname === "/api/llm" && req.method === "POST") {
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => null)) as {
          messages?: unknown[]; tools?: unknown[]; stream?: boolean; max_tokens?: number
        } | null
        if (!body?.messages?.length) return json({ error: "messages required" }, 400)
        const out = await env.AI.run(ANALYST_MODEL as keyof AiModels, {
          messages: body.messages,
          ...(body.tools ? { tools: body.tools } : {}),
          stream: body.stream !== false,
          // Reasoning tokens bill against this before any answer appears; see
          // the empty-answer note in analyst.ts.
          max_tokens: body.max_tokens ?? 8000,
        } as never)
        return body.stream === false
          ? json(out)
          : new Response(out as ReadableStream, {
              headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", ...CORS },
            })
      }

      // Proposals an agent has recorded, for the operator to review. Read-only,
      // so it sits at the research tier: it is a queue of suggestions, not a
      // control surface. Acting on one still takes an admin route.
      if (url.pathname === "/api/proposals" && req.method === "GET") {
        if (!(await authorized(req, env, "research"))) return json({ error: "unauthorized" }, 401)
        return json(await ledger(env).listProposals())
      }

      if (url.pathname === "/api/research" && req.method === "POST") {
        if (!(await authorized(req, env, "research"))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => ({}))) as { sql?: string }
        if (!body.sql) return json({ error: "sql required" }, 400)
        return json(await ledger(env).researchQuery(body.sql))
      }
      if (url.pathname === "/api/train" && req.method === "GET") {
        // Research tier: a fit over the corpus is the heaviest read the object
        // serves, and an unauthenticated one was a free reset button.
        if (!(await authorized(req, env, "research"))) return json({ error: "unauthorized" }, 401)
        return json(await ledger(env).trainModel())
      }
      if (url.pathname === "/api/exit-sweep" && req.method === "GET") {
        return json(await ledger(env).exitSweep())
      }
      if (url.pathname === "/api/entry-sweep" && req.method === "GET") {
        return json(await ledger(env).entrySweep())
      }
      if (url.pathname === "/api/archive" && req.method === "GET") {
        if (!(await authorized(req, env, "research"))) return json({ error: "unauthorized" }, 401)
        return json(await ledger(env).archiveStatus())
      }
      if (url.pathname === "/api/archive" && req.method === "POST") {
        // Operator: run one batch now. `retainDays` overrides the window for a
        // verification run; it archives before it deletes either way.
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => ({}))) as { retainDays?: number; resetHistory?: boolean }
        const retainDays = typeof body.retainDays === "number" && body.retainDays >= 1 ? body.retainDays : undefined
        return json(await ledger(env).maybeArchive({ force: true, retainDays, resetHistory: body.resetHistory === true }))
      }
      if (url.pathname === "/api/genesis" && req.method === "POST") {
        // The collector on the Pro posts its daily summary here.
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => null)) as unknown
        if (!body || typeof body !== "object") return json({ error: "json object required" }, 400)
        return json(await ledger(env).setGenesisReport(body))
      }
      if (url.pathname === "/api/genesis" && req.method === "GET") {
        return json(await ledger(env).genesisReport())
      }
      if (url.pathname === "/api/digest" && req.method === "GET") {
        // Read the digest as it would be sent, without sending it.
        if (!(await authorized(req, env, "research"))) return json({ error: "unauthorized" }, 401)
        return json(await ledger(env).composeDigest(Date.now()))
      }
      if (url.pathname === "/api/digest" && req.method === "POST") {
        // Operator: send today's digest now.
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        const queued = await ledger(env).maybeDigest(true)
        const ops = await ledger(env).flushAlerts()
        return json({ ...queued, ...ops })
      }
      if (url.pathname === "/api/kill" && req.method === "POST") {
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => ({}))) as { on?: boolean }
        await ledger(env).setKill(body.on !== false)
        return json({ ok: true, killed: body.on !== false })
      }
      if (url.pathname === "/api/veto" && req.method === "POST") {
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => ({}))) as { id?: string }
        if (!body.id) return json({ error: "id required" }, 400)
        return json(await ledger(env).requestVeto(body.id))
      }
      // Manual tick trigger for verification; same power as waiting a minute,
      // so it carries the same auth as the other mutating endpoints.
      if (url.pathname === "/api/tick" && req.method === "POST") {
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        return json(await ledger(env).tick())
      }
      return json({ error: "not found" }, 404)
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      console.error(JSON.stringify({ msg: "unhandled", error: e instanceof Error ? e.stack : message }))
      // Surface the message to the caller. This is an operator endpoint behind
      // a bearer token, not a public surface, and a bare "internal" cost a full
      // debugging round trip that the message alone would have answered.
      return json({ error: "internal", message }, 500)
    }
  },
} satisfies ExportedHandler<Env>

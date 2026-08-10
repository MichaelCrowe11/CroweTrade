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
import { analystStream } from "./analyst.js"
import { priceFor, paymentRequired, settle, ROUTES, SOLANA_MAINNET } from "./x402.js"

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

async function authorized(req: Request, env: Env): Promise<boolean> {
  const token = req.headers.get("Authorization")?.replace(/^Bearer\s+/i, "")
  if (!token || !env.ENGINE_ADMIN_TOKEN) return false
  const enc = new TextEncoder()
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", enc.encode(token)),
    crypto.subtle.digest("SHA-256", enc.encode(env.ENGINE_ADMIN_TOKEN)),
  ])
  return crypto.subtle.timingSafeEqual(a, b)
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
    // Operational alerts (breaker trips, kill flips, scan outages) queue
    // during the tick and send here, on the same no-mail-inside-trading seam.
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
          free: ["/api/health", "/api/positions", "/api/exit-sweep", "/api/train"],
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
      // construction (its only three tools are reads), so the token guards
      // spend, not capability.
      if (url.pathname === "/api/analyst" && req.method === "POST") {
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => ({}))) as { question?: string }
        const question = (body.question ?? "").trim()
        if (!question) return json({ error: "question required" }, 400)
        const stub = ledger(env)
        const stream = await analystStream(env.AI, question, {
          state: () => stub.summary(),
          exitSweep: () => stub.exitSweep(),
          modelFit: () => stub.trainModel(),
        })
        return new Response(stream, {
          headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", ...CORS },
        })
      }

      if (url.pathname === "/api/research" && req.method === "POST") {
        if (!(await authorized(req, env))) return json({ error: "unauthorized" }, 401)
        const body = (await req.json().catch(() => ({}))) as { sql?: string }
        if (!body.sql) return json({ error: "sql required" }, 400)
        return json(await ledger(env).researchQuery(body.sql))
      }
      if (url.pathname === "/api/train" && req.method === "GET") {
        return json(await ledger(env).trainModel())
      }
      if (url.pathname === "/api/exit-sweep" && req.method === "GET") {
        return json(await ledger(env).exitSweep())
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

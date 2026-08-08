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
    const result = await ledger(env).tick()
    console.log(JSON.stringify({ msg: "tick", ...result }))
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
      console.error(JSON.stringify({ msg: "unhandled", error: e instanceof Error ? e.stack : String(e) }))
      return json({ error: "internal" }, 500)
    }
  },
} satisfies ExportedHandler<Env>

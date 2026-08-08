/**
 * PostHog capture, fire-and-forget.
 *
 * The engine is the product's evidence layer, so every tick, entry, exit,
 * veto and kill lands in PostHog with the policy hash attached. Dashboards
 * over these events ARE the validation record a funding conversation reads.
 *
 * Gated on the POSTHOG_API_KEY secret: absent means analytics silently off,
 * because a missing analytics key must never be able to stop trading logic.
 * CroweTrade gets its OWN PostHog project; reusing another product's key
 * would pollute both streams.
 */

const HOST = "https://us.i.posthog.com"

export function capture(
  env: Env,
  ctx: { waitUntil(p: Promise<unknown>): void },
  event: string,
  properties: Record<string, unknown>,
): void {
  const key = env.POSTHOG_API_KEY
  if (!key) return
  ctx.waitUntil(
    fetch(`${HOST}/i/v0/e/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: key,
        event,
        distinct_id: "crowetrade-engine",
        properties: { ...properties, product: "crowetrade", phase: "paper" },
        timestamp: new Date().toISOString(),
      }),
    }).catch(() => {
      // Analytics loss is acceptable; trading interruption is not.
    }),
  )
}

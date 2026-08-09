/**
 * The engine's outbound alerts.
 *
 * There is exactly one thing worth waking a human for right now: whether the
 * launchpad universe is measurably healthier than the promotional feed. That
 * claim was made once on corrupted data (the base-units price bug), retracted,
 * and has to be re-earned on clean rows. Nobody should have to poll an endpoint
 * for hours waiting for it.
 *
 * Transport is Resend's HTTP API rather than the house SMTP rails, because a
 * Worker has no raw sockets and therefore no SMTP. `zoho_send.py` is the
 * documented path for sending as michael@crowelogic.com from a laptop, and it
 * cannot run here. The sending domain is southwestmushrooms.com because that is
 * what is verified on the Resend account; crowelogic.com is not, and verifying
 * it means DNS changes at Squarespace for a machine-to-owner alert nobody else
 * ever sees.
 *
 * The alert fires ONCE. A repeating "your data is ready" is worse than no alert
 * at all, because the second copy teaches you to ignore the first.
 */

/** Labeled launchpad rows before the comparison is worth reading. */
export const READABLE_SAMPLE = 100

/**
 * Alert recipient. Michael's own inbox: this is a machine-to-owner message and
 * never goes to a customer, so it does not touch the customer-email guardrails.
 */
const TO = "southwestfungi@gmail.com"
const FROM = "CroweTrade Engine <crowetrade@southwestmushrooms.com>"

export interface OriginStat {
  origin: string
  /** Labeled, non-voided decisions. Voided rows are corpus wreckage, not data. */
  labeled: number
  died: number
  enteredRet: number | null
  refusedRet: number | null
}

export interface CalibrationSnapshot {
  launchpad: OriginStat
  /** The promotional feed, which launchpad has to beat to be worth switching to. */
  baseline: OriginStat
  killed: boolean
  breakerOpen: boolean
  policyHash: string | null
}

/**
 * Normal CDF via Abramowitz and Stegun 7.1.26.
 *
 * JavaScript has no erf. The approximation is accurate to about 1.5e-7, which
 * is several orders of magnitude tighter than the sampling noise on a hundred
 * coin flips, so it is not the weak link in any claim made below.
 */
function normalCdf(z: number): number {
  const sign = z < 0 ? -1 : 1
  const x = Math.abs(z) / Math.SQRT2
  const t = 1 / (1 + 0.3275911 * x)
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-x * x)
  return 0.5 * (1 + sign * y)
}

export interface Separation {
  launchpadRate: number
  baselineRate: number
  z: number
  pValue: number
  significant: boolean
}

/**
 * Two-proportion z-test on the death rates.
 *
 * The whole launchpad thesis is "fresh bonding-curve mints die less often than
 * whatever is paying to be promoted." That is a difference of proportions, and
 * reporting the two percentages side by side without a test is how 10.8 versus
 * 52.4 got believed the first time. Returns null when either side is too thin
 * to test rather than returning a confident-looking number built on four rows.
 */
export function separation(a: OriginStat, b: OriginStat): Separation | null {
  if (a.labeled < 30 || b.labeled < 30) return null
  const p1 = a.died / a.labeled
  const p2 = b.died / b.labeled
  const pooled = (a.died + b.died) / (a.labeled + b.labeled)
  const se = Math.sqrt(pooled * (1 - pooled) * (1 / a.labeled + 1 / b.labeled))
  if (se === 0) return null
  const z = (p1 - p2) / se
  // Two-tailed: launchpad being dramatically WORSE is just as much a result as
  // it being better, and is the outcome that should stop the migration.
  const pValue = 2 * (1 - normalCdf(Math.abs(z)))
  return {
    launchpadRate: p1,
    baselineRate: p2,
    z,
    pValue,
    significant: pValue < 0.05,
  }
}

function pct(n: number | null): string {
  return n === null ? "no data" : `${n.toFixed(1)}%`
}

/**
 * A p-value is never zero. `toFixed(4)` renders 2e-8 as "0.0000", which reads
 * as certainty and is the sort of overstatement this whole alert exists to
 * avoid, so anything below the visible resolution switches to scientific.
 */
function pval(p: number): string {
  return p < 0.0001 ? p.toExponential(1) : p.toFixed(4)
}

/**
 * The verdict sentence.
 *
 * Written so the first line of the email answers the question that prompted it,
 * because an alert whose conclusion is buried under a table is a table, not an
 * alert.
 */
function verdict(sep: Separation | null): string {
  if (!sep) return "Not enough labeled rows on one side to test. Treat as inconclusive."
  const delta = (sep.baselineRate - sep.launchpadRate) * 100
  if (!sep.significant) {
    return (
      `No significant difference. Launchpad dies at ${(sep.launchpadRate * 100).toFixed(1)}% ` +
      `versus the promotional feed at ${(sep.baselineRate * 100).toFixed(1)}% ` +
      `(p = ${pval(sep.pValue)}). The earlier claim that launchpad is the healthier ` +
      `universe does NOT survive clean data. Switching discovery on this basis would be a guess.`
    )
  }
  if (delta > 0) {
    return (
      `Launchpad is significantly healthier: ${(sep.launchpadRate * 100).toFixed(1)}% death rate ` +
      `versus ${(sep.baselineRate * 100).toFixed(1)}% on the promotional feed, ` +
      `a ${delta.toFixed(1)} point gap (p = ${pval(sep.pValue)}). ` +
      `This is the first clean evidence for the launchpad thesis. It says entries are ` +
      `dying less often. It does NOT yet say the strategy makes money.`
    )
  }
  return (
    `Launchpad is significantly WORSE: ${(sep.launchpadRate * 100).toFixed(1)}% death rate ` +
    `versus ${(sep.baselineRate * 100).toFixed(1)}% on the promotional feed ` +
    `(p = ${pval(sep.pValue)}). The launchpad thesis is dead. Do not migrate discovery.`
  )
}

export function composeBody(snap: CalibrationSnapshot): { subject: string; text: string } {
  const sep = separation(snap.launchpad, snap.baseline)
  const headline = !sep
    ? "inconclusive"
    : !sep.significant
      ? "no edge found"
      : sep.launchpadRate < sep.baselineRate
        ? "launchpad wins"
        : "launchpad loses"

  const lines = [
    verdict(sep),
    "",
    `Launchpad: ${snap.launchpad.labeled} labeled, ${snap.launchpad.died} died.`,
    `  entered ${pct(snap.launchpad.enteredRet)}, refused ${pct(snap.launchpad.refusedRet)}`,
    `Promotional feed: ${snap.baseline.labeled} labeled, ${snap.baseline.died} died.`,
    `  entered ${pct(snap.baseline.enteredRet)}, refused ${pct(snap.baseline.refusedRet)}`,
    "",
    // Entered-versus-refused is the selection question, and it is separate from
    // the death-rate question. A universe can be healthier while our picks
    // inside it are still no better than the ones we threw away.
    "Selection check: if entered and refused returns are close, the gates are not",
    "picking winners, they are just picking. That was the finding on the",
    "promotional feed and it is the reason this comparison exists.",
    "",
    `Engine: ${snap.killed ? "KILLED" : "live"}, breaker ${snap.breakerOpen ? "OPEN" : "closed"}.`,
    `Policy: ${snap.policyHash ?? "unset"}`,
    "",
    "Still paper. No capital at risk. Nothing here authorizes funding an account.",
    "",
    "https://crowetrade-engine.yellow-block-3adc.workers.dev/api/positions",
  ]

  return {
    subject: `CroweTrade: launchpad re-validation ready (${headline})`,
    text: lines.join("\n"),
  }
}

/**
 * Send via Resend. Returns an error string rather than throwing, so a failed
 * alert degrades into a logged line instead of killing the trading tick that
 * called it.
 */
export async function send(
  apiKey: string,
  subject: string,
  text: string,
): Promise<{ ok: true } | { ok: false; error: string }> {
  try {
    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: FROM,
        to: [TO],
        reply_to: TO,
        subject,
        text,
      }),
    })
    if (!res.ok) {
      return { ok: false, error: `resend ${res.status}: ${(await res.text()).slice(0, 300)}` }
    }
    return { ok: true }
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) }
  }
}

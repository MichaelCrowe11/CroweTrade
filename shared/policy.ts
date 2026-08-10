/**
 * The policy envelope: the single object a user consents to.
 *
 * This is the consent framework Michael specced on 2026-08-08. One signed
 * object carries the legal waiver (by hash), the exact autonomy limits, and an
 * expiry. The user's wallet signature over this object is simultaneously the
 * legal consent record and the credential the signer service checks before any
 * transaction. Nothing executes outside a live, signed, unexpired envelope.
 *
 * Governance rules the envelope encodes, as agreed:
 *  - tighten instantly, loosen with delay (armAfter on any loosening change)
 *  - kill switch is instant and unconditional
 *  - a veto window follows every autonomous entry: within it, one action
 *    unwinds the position at market (a second trade, never a reversal;
 *    finality is real and the UI must never pretend otherwise)
 *  - every fill records the hash of the envelope that authorized it, so every
 *    trade has provable lineage: this fill, under this policy version, under
 *    this signed consent
 *
 * PAPER PHASE: signature and signer stay null and the engine trades imaginary
 * capital under the same envelope discipline, so the audit trail exists and is
 * demonstrable before real money does. Wallet signing lands with wallet
 * connect in the execution layer.
 */

import type { DiscoveryOrigin } from "./dexscreener.js"

export interface PolicyEnvelope {
  version: 1
  product: "crowetrade-paper" | "crowetrade-live"
  /** SHA-256 of the exact waiver text consented to (shared/waiver.md). */
  waiverSha256: string

  /** Hard caps, enforced at the signer for live and at the engine for paper. */
  perTradeCapSol: number
  dailyCapSol: number
  maxOpenPositions: number

  entry: {
    /** Minimum verdict allowed to open: "clear" only, or caution-and-better. */
    minVerdict: "clear" | "caution"
    maxTokenAgeMinutes: number
    /** Skip the launch window, where price gaps far past any stop. */
    minTokenAgeMinutes: number
    minLiquidityUsd: number
    /**
     * Refuse tokens already parabolic on the hour.
     *
     * Discovery surfaces tokens BECAUSE they are moving, so the naive strategy
     * systematically buys spikes. Both of the first honestly-priced losses were
     * entries into tokens already up triple digits, dead within three minutes.
     */
    maxChangeH1Pct: number
    /**
     * Cost hurdle: refuse entries whose buy-quote price impact exceeds this.
     *
     * Round trip costs roughly twice the one-way impact plus fees, and v1's
     * autopsy showed 61% of entries stopping out on exactly the thin pools
     * where impact ran 2-3% each way. A trade that starts 6% underwater on
     * costs needs the token to move 6% just to see zero. (Idea salvaged from
     * autonomous_trader's risk_agent: edge must clear summed costs, always.)
     */
    maxEntryImpactPct: number
    /**
     * Discovery sources this envelope may trade. An ALLOWLIST, not a set of
     * exclusions: a source added to the scanner later must be opted in
     * deliberately rather than inheriting permission to spend money.
     *
     * Narrowed to launchpad only on 2026-08-10, on REALIZED PnL rather than
     * forward returns. Of the lifetime -$385 across 145 quote-priced closes,
     * the promotional profile feed accounts for -$331 over 87 closes (-17.7%
     * average). Launchpad over 40 closes is -$16.67 (-3.6%), which is inside
     * the cost of trading. The two universes have different SHAPES, not just
     * different means: launchpad's 3 take-profits averaged +242.8% and its 9
     * time-stops +33.5%, so 30% of its trades paid for the other 70%. The
     * profile feed has no such tail, which is what "structurally
     * unprofitable" has meant here since the n=87 forward-return test.
     *
     * "held" is always permitted implicitly; it is not a discovery source but
     * the re-pricing of a position already open.
     */
    allowedOrigins: DiscoveryOrigin[]
    /**
     * Require this many of OUR OWN minute-ticks before entering, with price
     * higher than at the start of the window and liquidity not draining.
     * Confirmation from what we watched happen, not what a listing claims.
     */
    minObservedTicks: number
    /**
     * Minimum armed-model probability to enter; null = no model gate.
     *
     * The probability estimates P(30-min forward return clears the 6%
     * round-trip cost hurdle). An UNCOMPUTABLE probability blocks entry when
     * the gate is armed — unknown never passes, same as every safety gate.
     * The model sits BEHIND the gates and can only refuse, never override:
     * a hard veto stays a hard veto whatever the model believes.
     */
    /**
     * Refuse an entry whose price has already run this far since OUR OWN
     * first sight of the token.
     *
     * Added 2026-08-10 on measurement, and it is the most consequential entry
     * rule in the file. Across 73 entries, price moved +142.8% in the 23
     * minutes between first sight and entry, and 54 of 73 were bought HIGHER
     * than first sight. The calibration readout that appeared to show
     * selection working (+145.9% forward return on entered tokens) was
     * measuring almost exactly that pre-entry run, while the same trades
     * realized -14.6%. The engine was buying tops.
     *
     * maxChangeH1Pct cannot do this job: it reads the FEED's hourly change,
     * a third party's claim about the last hour. This reads the price we
     * recorded when we first saw the token against the price we are about to
     * pay, which is the only comparison that describes the actual trade.
     */
    maxDriftSinceFirstSightPct: number
    minModelProb: number | null
    /**
     * Identity of the exact frozen weights (shared/armed-model.ts). In the
     * envelope so that swapping models rolls the policy hash: a record earned
     * under one model must never quietly continue under another.
     */
    modelFingerprint: string | null
  }

  /**
   * Circuit breaker: consecutive stop-loss exits trip a pause on new entries.
   *
   * v1 took 22 stop-outs with nothing watching the sequence. Whatever the
   * entry logic is, a run of stops means the market regime and the strategy
   * disagree right now, and the cheapest response is to stand down briefly.
   * Exits keep managing regardless, as with the kill switch.
   */
  breaker: {
    consecutiveStopLimit: number
    cooldownMinutes: number
  }

  exit: {
    takeProfitPct: number
    stopLossPct: number
    /** Flat exit after this long regardless of price. Meme decay is real. */
    timeStopMinutes: number
    /** Human veto window after each autonomous entry. */
    vetoWindowMinutes: number
  }

  /** ISO time after which the envelope is dead and nothing trades. */
  expiresAt: string

  /** Wallet signature over the canonical hash. Null during the paper phase. */
  signature: string | null
  /** The signing wallet. Null during the paper phase. */
  signer: string | null
}

/**
 * Default paper policy, v1. Every number is a dial; the SHAPE (caps + gates +
 * exits + veto) is the contract. Loosening any of these in a live envelope
 * must re-arm with delay and require a fresh signature.
 */
export const PAPER_POLICY: PolicyEnvelope = {
  version: 1,
  product: "crowetrade-paper",
  waiverSha256: "unsigned-paper-phase",
  perTradeCapSol: 0.5,
  /**
   * Raised 10 -> 50 on 2026-08-08 to accelerate the validation sample toward
   * the 100-close funding criterion. This changes how MANY trades happen, not
   * the character of any trade — per-trade sizing, entry filters and exits are
   * untouched — so per-trade expectancy statistics stay comparable across the
   * hash boundary. A live envelope would never move this way without a fresh
   * signature; paper is where the dial is allowed to be cheap.
   */
  dailyCapSol: 50,
  maxOpenPositions: 8,
  entry: {
    minVerdict: "caution",
    maxTokenAgeMinutes: 90,
    /**
     * Lowered 15 -> 3 on 2026-08-09, and this is the most consequential dial
     * turn in the project so far.
     *
     * The 15-minute floor was tuned for the promotional feed, where tokens
     * arrive hours old. Launchpad tokens arrive SECONDS old and fall out of the
     * listing before they age in, so the floor did not filter them, it made
     * them unreachable: 346 decisions skipped as too-new, zero launchpad
     * entries ever.
     *
     * That matters because the two universes are not equally dangerous.
     * Measured over the same 30-minute horizon: launchpad death rate 0.108,
     * promotional feed 0.524. We proved one universe was dead and then kept
     * trading only that one, because the floor locked us out of the other.
     *
     * Three ticks of our own observation are still required before entry, so
     * "3 minutes" is the real floor regardless of what this says.
     */
    minTokenAgeMinutes: 3,
    minLiquidityUsd: 3_000,
    maxChangeH1Pct: 80,
    maxEntryImpactPct: 1.5,
    allowedOrigins: ["launchpad"],
    minObservedTicks: 3,
    /**
     * Armed 2026-08-09. The exit sweep closed the other door: the shipped
     * exit rule beat every alternative including no-stop, so entry selection
     * is the only lever left, and the unfiltered baseline is a measured
     * money-loser (52 closes, -$103.79 at arming time). 0.2 selects the
     * reliability bucket that observed a ~24% hit rate against a 5.8% base —
     * the first measured selection power this system has produced. Entries
     * will be far rarer on purpose; this cohort tests quality over volume
     * against the baseline cohort's record, the same head-to-head shape as
     * launchpad vs promotional.
     */
    /**
     * 30% is a deliberate first guess, not a fitted value, and it should be
     * revisited once the cohort has trades. The measured average run was
     * +142.8%, so this refuses the bulk of what was being bought while still
     * allowing a token to move while our own three confirming ticks accrue.
     * Too tight would refuse everything, since some drift is unavoidable
     * given the engine only enters after observing a token.
     */
    maxDriftSinceFirstSightPct: 30,
    minModelProb: 0.2,
    modelFingerprint: "m20260809-5743r-auc802",
  },
  breaker: {
    consecutiveStopLimit: 4,
    cooldownMinutes: 60,
  },
  exit: {
    /**
     * Take-profit widened from 60 to 120 on measured evidence.
     *
     * Stops do not hold on these assets: an observed -35% stop realized at
     * -43.6% because price gapped straight through it. With an effective loss
     * near 40% and a win rate near 35%, breakeven needs roughly 75%, so a 60%
     * target was mathematically losing before costs. 120 restores the
     * asymmetry a low win rate requires.
     */
    takeProfitPct: 120,
    stopLossPct: 35,
    /** These positions resolve in minutes; 45 was holding through the decay. */
    timeStopMinutes: 30,
    vetoWindowMinutes: 10,
  },
  expiresAt: "2027-01-01T00:00:00Z",
  signature: null,
  signer: null,
}

/** Stable stringify: keys sorted, so the hash is canonical across runtimes. */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`)
    return `{${entries.join(",")}}`
  }
  return JSON.stringify(value)
}

/**
 * The policy hash stamped on every fill and, for live envelopes, the exact
 * bytes the wallet signs. WebCrypto, so it runs identically in the renderer,
 * in Workers, and under node --test.
 */
export async function policyHash(p: PolicyEnvelope): Promise<string> {
  const unsigned = { ...p, signature: null }
  const bytes = new TextEncoder().encode(canonical(unsigned))
  const digest = await crypto.subtle.digest("SHA-256", bytes)
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("")
}

/**
 * The DUST envelope: the first policy allowed to spend real money.
 *
 * Sized so that being completely wrong is affordable. At roughly $84 SOL these
 * caps are about $1.70 per trade and $8.40 per day against a wallet holding
 * ~$20, which means the worst realistic outcome of a bug is a rounding error
 * on a dinner. That is the entire design goal: the number is chosen so the
 * lesson is cheap, not so the return is meaningful.
 *
 * It inherits every entry and exit rule from the paper policy, deliberately.
 * The paper record is only a rehearsal for this if the rules are the same
 * ones; changing strategy and execution in the same step would mean the first
 * live result could not be compared to anything.
 *
 * THE HASH CHANGES, WHICH STARTS A NEW COHORT. That is correct and wanted:
 * real fills must never be averaged into a record of imagined ones, and the
 * `execution` column keeps them separable even within the cohort.
 *
 * WHAT MUST BE EDITED BEFORE THIS IS USED, both deliberately left wrong so an
 * unedited copy fails closed:
 *
 *   expiresAt  — a near date. It is in the PAST as written, so an envelope
 *                nobody has reviewed refuses every trade.
 *   signer     — the trading wallet's address.
 *   signature  — that wallet's signature over this envelope's canonical hash,
 *                produced by `engine/scripts/sign-policy.mjs`.
 *
 * The signature is not regulatory theatre even for a sole operator. It is the
 * act that makes a fill traceable to a specific set of limits somebody agreed
 * to, rather than to a config file that happened to be deployed. Preflight
 * refuses an unsigned live envelope for that reason.
 */
export const LIVE_DUST_POLICY: PolicyEnvelope = {
  ...PAPER_POLICY,
  product: "crowetrade-live",
  waiverSha256: "personal-use-sole-operator",

  // ~$1.70 a trade, ~$8.40 a day, one position at a time. One position is not
  // a risk limit so much as a legibility one: with a single open trade, every
  // on-chain event during the test has exactly one possible explanation.
  perTradeCapSol: 0.02,
  dailyCapSol: 0.1,
  maxOpenPositions: 1,

  // MUST be updated to a near-future date before use. Past by default.
  expiresAt: "2000-01-01T00:00:00Z",
  signature: null,
  signer: null,
}

/**
 * The Ledger: one Durable Object holding the whole paper-trading state.
 *
 * The entire tick runs INSIDE the DO so all state mutation happens in one
 * place. An earlier version of this comment claimed a Durable Object is
 * "single-threaded by contract" and therefore serializes overlapping ticks.
 * That is FALSE and was a dangerous thing to believe: a DO's input gate does
 * not hold across `await`ed external I/O, and this tick awaits dozens of
 * fetches. A cron tick and a manual tick genuinely can interleave, both read
 * the same open positions and budget, and both enter. Serialization is
 * enforced explicitly by the tick lease below, not assumed.
 *
 * Every fill is stamped with the policy hash that authorized it. That lineage
 * is the audit trail: this fill, under this policy version. The positions and
 * fills tables are the funding artifact accruing in real time.
 */

import { DurableObject } from "cloudflare:workers"
import {
  fetchCandidates,
  fetchPairsForMints,
  type Candidate,
} from "../../shared/dexscreener.js"
import { fetchLaunchpadCandidates } from "../../shared/pumpfun.js"
import {
  fetchMintFacts,
  fetchTopHolderShare,
  configureRpc,
  type MintFacts,
} from "../../shared/solana.js"
import { evaluateGates, combineVerdict, type Verdict } from "../../shared/gates.js"
import { PAPER_POLICY, policyHash, type PolicyEnvelope } from "../../shared/policy.js"
import {
  decideEntries,
  decideExits,
  type OpenPosition,
  type Trajectory,
  type ModelRefusal,
} from "./strategy.js"
import { computeFeatures } from "../../shared/features.js"
import { fit, score, buildFeatureVector, type FeatureSnapshot } from "../../shared/model.js"
import { liveArmed } from "../../shared/preflight.js"
import { parseKeypair, base58, verifyPolicySignature } from "../../shared/signer.js"
import { executeSwap, walletBalanceSol } from "./execution/live.js"
import { ARMED_MODEL } from "../../shared/armed-model.js"
import { quoteBuy, quoteSell, LAMPORTS_PER_SOL } from "./execution/jupiter.js"
import { dryRunSwap } from "./execution/swap.js"
import { capture } from "./posthog.js"
import {
  composeBody,
  send as sendAlert,
  READABLE_SAMPLE,
  type OriginStat,
} from "./alert.js"

/** Slippage tolerance requested on every quote, entry and exit. */
const SLIPPAGE_BPS = 300

interface PositionRow {
  [key: string]: string | number | null
  id: string
  mint: string
  symbol: string
  entry_price: number
  size_sol: number
  size_usd: number
  token_amount: number
  opened_at: number
  closed_at: number | null
  exit_price: number | null
  exit_reason: string | null
  pnl_usd: number | null
  pnl_pct: number | null
  policy_hash: string
  verdict_entry: string
  veto_requested: number
}

export class Ledger extends DurableObject<Env> {
  private schemaReady = false

  private sql() {
    const sql = this.ctx.storage.sql
    if (!this.schemaReady) {
      sql.exec(`
        CREATE TABLE IF NOT EXISTS positions (
          id TEXT PRIMARY KEY,
          mint TEXT NOT NULL,
          symbol TEXT NOT NULL,
          entry_price REAL NOT NULL,
          size_sol REAL NOT NULL,
          size_usd REAL NOT NULL,
          token_amount REAL NOT NULL,
          opened_at INTEGER NOT NULL,
          closed_at INTEGER,
          exit_price REAL,
          exit_reason TEXT,
          pnl_usd REAL,
          pnl_pct REAL,
          policy_hash TEXT NOT NULL,
          verdict_entry TEXT NOT NULL,
          veto_requested INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS events (
          at INTEGER NOT NULL,
          kind TEXT NOT NULL,
          data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ticks (
          mint TEXT NOT NULL,
          at INTEGER NOT NULL,
          price REAL,
          liquidity_usd REAL,
          buys_24h INTEGER,
          sells_24h INTEGER,
          origin TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ticks_mint_at ON ticks (mint, at);
        CREATE TABLE IF NOT EXISTS creators (
          mint TEXT PRIMARY KEY,
          creator TEXT NOT NULL,
          first_seen INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_creators_creator ON creators (creator);
        CREATE TABLE IF NOT EXISTS decisions (
          mint TEXT PRIMARY KEY,
          at INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          price REAL,
          origin TEXT,
          verdict TEXT,
          features TEXT NOT NULL,
          eligible INTEGER NOT NULL,
          skip_reason TEXT,
          entered INTEGER NOT NULL DEFAULT 0,
          entry_impact_pct REAL,
          labeled INTEGER NOT NULL DEFAULT 0,
          forward_ret_pct REAL,
          died INTEGER,
          labeled_at INTEGER
        );
      `)
      // Rows written before 2026-08-08 were priced by an invented slippage
      // model later measured wrong by ~20x. They are TAGGED rather than
      // deleted: the mistake is part of the record, and the stats below simply
      // stop counting them so the headline number reflects real routes only.
      try {
        sql.exec("ALTER TABLE positions ADD COLUMN priced_by TEXT NOT NULL DEFAULT 'model'")
      } catch {
        // Column already present; nothing to migrate.
      }
      // Entry and exit pricing are tracked separately: a position can be
      // entered on a real quote and exited with no route at all, and calling
      // that whole round trip "venue-quoted" overstates the record's fidelity.
      try {
        sql.exec("ALTER TABLE positions ADD COLUMN exit_pricing TEXT")
      } catch {
        // Column already present.
      }
      try {
        sql.exec("ALTER TABLE decisions ADD COLUMN voided INTEGER NOT NULL DEFAULT 0")
      } catch {
        // Column already present.
      }
      // Quarantine the launchpad rows priced before the base-unit fix.
      //
      // total_supply arrives in base units, so decision-time prices were off by
      // 10^decimals and forward returns came out in the hundreds of millions of
      // percent. Deleting them would hide the mistake; leaving them labeled
      // would poison every query and every model fit. They are marked unlabeled
      // with a reason instead, so they stop counting and stay inspectable.
      try {
        sql.exec(
          // Cut on PRICE, not on the return. The first pass voided only absurd
          // returns and left rows that were equally wrong but quieter -- same
          // bad price, no computable return, still poisoning any re-label.
          // Buggy prices cluster near 1e-12 (market cap over BASE units) while
          // correct ones sit near 1e-6, so 1e-9 separates them cleanly.
          // Cut on TIME, not on price value.
          //
          // Threshold-chasing failed twice: a 1e-9 cut let rows at 4e-8
          // through, and every widening is a guess about how wrong a wrong
          // number was. The defensible statement is simpler: every launchpad
          // row written before the base-unit fix deployed is untrustworthy,
          // whatever it says. Post-fix rows accumulate clean from here.
          //
          // A PERMANENT flag, not labeled=0.
          //
          // The first version of this quarantine set labeled=0, which made the
          // rows eligible for labeling again -- so the labeler immediately
          // re-labeled them from the same bad price and re-corrupted the set.
          // The quarantine was fighting the labeler and losing. `voided` is
          // never cleared and the labeler skips it.
          `UPDATE decisions SET voided = 1, labeled = 0, forward_ret_pct = NULL, died = NULL
           WHERE origin = 'launchpad' AND voided = 0 AND at < 1786245220317`,
        )
      } catch {
        // Nothing to void.
      }
      // The armed model's probability at decision time, recorded on EVERY
      // decision (entered or refused), so live calibration stays checkable:
      // predicted probability vs what the label said, on data the model
      // could not have trained on.
      try {
        sql.exec("ALTER TABLE decisions ADD COLUMN model_prob REAL")
      } catch {
        // Column already present.
      }
      // Live execution provenance. `execution` defaults to 'paper' so every
      // row ever written before this migration is correctly labelled as
      // simulated: a record that cannot distinguish real fills from imagined
      // ones is worse than no record, and defaulting the other way would
      // retroactively claim 168 paper closes were real.
      try {
        sql.exec("ALTER TABLE positions ADD COLUMN execution TEXT NOT NULL DEFAULT 'paper'")
      } catch {
        // Column already present.
      }
      // On-chain signatures for the entry and the exit, so any live row can be
      // audited against the chain by anyone, including someone who does not
      // trust this engine's own accounting.
      try {
        sql.exec("ALTER TABLE positions ADD COLUMN entry_sig TEXT")
      } catch {
        // Column already present.
      }
      try {
        sql.exec("ALTER TABLE positions ADD COLUMN exit_sig TEXT")
      } catch {
        // Column already present.
      }
      this.schemaReady = true
    }
    return sql
  }

  private metaGet(key: string): string | null {
    const rows = this.sql().exec<{ value: string }>(
      "SELECT value FROM meta WHERE key = ?", key,
    ).toArray()
    return rows[0]?.value ?? null
  }

  private metaSet(key: string, value: string): void {
    this.sql().exec(
      "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      key, value,
    )
  }

  private event(kind: string, data: unknown): void {
    this.sql().exec("INSERT INTO events (at, kind, data) VALUES (?, ?, ?)",
      Date.now(), kind, JSON.stringify(data))
  }

  private openPositions(): OpenPosition[] {
    return this.sql().exec<PositionRow>(
      "SELECT * FROM positions WHERE closed_at IS NULL",
    ).toArray().map((r) => ({
      id: r.id,
      mint: r.mint,
      symbol: r.symbol,
      entryPriceUsd: r.entry_price,
      sizeSol: r.size_sol,
      sizeUsd: r.size_usd,
      tokenAmount: r.token_amount,
      openedAt: r.opened_at,
      policyHash: r.policy_hash,
      verdictAtEntry: r.verdict_entry as Verdict,
    }))
  }

  /**
   * Closes at a REAL sell quote when one exists.
   *
   * proceedsUsd comes from lamports Jupiter says the route actually returns for
   * our exact token amount, not from mark price times quantity. Mark price is
   * what the last trade printed; it is not what we can get out at, and on a
   * thin pool the gap between the two is the entire result.
   */
  /**
   * Is live execution enabled for THIS policy?
   *
   * Three independent conditions, all required: the environment flag is
   * exactly "1", a key is present, and the envelope itself is a live one.
   * Checked per call rather than cached, so removing the secret takes effect
   * on the next trade instead of the next deploy.
   */
  private liveEnabled(policy: PolicyEnvelope): boolean {
    return liveArmed(this.env as unknown as Record<string, unknown>)
      && policy.product === "crowetrade-live"
  }

  /**
   * The trading wallet's public address, derived from the configured keypair.
   *
   * Returns null on any malformation rather than throwing: a bad key must
   * degrade to "cannot trade live" rather than taking down the tick that also
   * manages existing positions.
   */
  private tradingOwner(): string | null {
    const raw = this.env.TRADING_KEYPAIR
    if (typeof raw !== "string" || raw.length === 0) return null
    try {
      return base58(parseKeypair(raw).publicKey)
    } catch {
      return null
    }
  }

  /** Was this position opened with real funds? Paper positions must never
   *  take the live exit path, even while live trading is armed. */
  private positionIsLive(id: string): boolean {
    return this.sql().exec<{ execution: string }>(
      "SELECT execution FROM positions WHERE id = ?", id,
    ).toArray()[0]?.execution === "live"
  }

  private async closePosition(
    p: OpenPosition,
    exitPriceUsd: number,
    reason: string,
    now: number,
    solUsd: number,
  ): Promise<void> {
    const decimals = Number(this.metaGet(`decimals:${p.mint}`) ?? "6")
    const baseUnits = BigInt(Math.floor(p.tokenAmount * 10 ** decimals))

    let proceeds: number
    let impact: number | null = null
    let route: string | null = null
    let pricing = "quote"

    const q = baseUnits > 0n ? await quoteSell(p.mint, baseUnits, SLIPPAGE_BPS) : null
    let exitSig: string | null = null
    const isLive = this.positionIsLive(p.id)

    if (q && solUsd > 0 && isLive && this.liveEnabled(PAPER_POLICY)) {
      // ── LIVE EXIT ────────────────────────────────────────────────────
      //
      // No entry guard here, deliberately. The kill switch, the daily cap and
      // the breaker stop NEW risk; none of them may prevent closing risk
      // already taken. A position we cannot sell is the worst state this
      // system has, worse than any single loss.
      const owner = this.tradingOwner()
      const key = this.env.TRADING_KEYPAIR as string
      const r = owner
        ? await executeSwap(q.raw, owner, key, p.mint, "exit", null)
        : null
      if (r?.ok && r.fill) {
        // Negative solDelta on a sell means SOL ARRIVED; proceeds are its
        // magnitude. Reading the quote here instead would re-introduce the
        // predicted-vs-realized gap this whole layer exists to close.
        proceeds = (Number(-r.fill.solDeltaLamports) / 1e9) * solUsd
        impact = q.priceImpactPct
        route = q.route
        exitSig = r.signature
        this.event("live_exit", {
          symbol: p.symbol, mint: p.mint, signature: exitSig, reason,
          solReceived: Number(-r.fill.solDeltaLamports) / 1e9,
          quotedSol: Number(q.outAmount) / LAMPORTS_PER_SOL,
        })
      } else {
        // A live position we failed to sell is NOT closed at a made-up price.
        // It stays open so the next tick retries, and the operator is told.
        this.event("live_exit_failed", {
          symbol: p.symbol, mint: p.mint, reason, error: r?.error ?? "no owner",
          signature: r?.signature ?? null, paidFee: r ? !r.refusedBeforeSend : false,
        })
        return
      }
    } else if (q && solUsd > 0) {
      proceeds = (Number(q.outAmount) / LAMPORTS_PER_SOL) * solUsd
      impact = q.priceImpactPct
      route = q.route
    } else {
      // No sell route means no buyer at any size: the position is unexitable,
      // which is a total loss, so proceeds are ZERO for every exit reason.
      //
      // The previous form marked non-safety exits to half the mark price and
      // still filed them as venue-quoted. That invented liquidity that did not
      // exist AND let a fabricated number into the headline record — the exact
      // dishonesty this engine exists to avoid. Such rows are now tagged
      // 'unroutable': they remain in the record because the loss is real, and
      // they stay distinguishable because the price was never quoted.
      proceeds = 0
      pricing = "unroutable"
    }

    const pnlUsd = proceeds - p.sizeUsd
    const pnlPct = (pnlUsd / p.sizeUsd) * 100
    const effective = p.tokenAmount > 0 ? proceeds / p.tokenAmount : 0

    // Circuit breaker bookkeeping: stops count up, take-profits reset, and a
    // run of stops at the limit trips a timed pause on NEW entries only.
    // Time-stops are neutral: they say "nothing happened", not "wrong again".
    // Loss-velocity trip (from the old repo's one production-quality risk
    // module): rate of loss, not level of loss. A drawdown limit sampled
    // per-minute observes the corpse; a velocity trip fires during the event.
    // Internal plumbing, not envelope policy, so the v2 hash stays stable.
    const VELOCITY_WINDOW_MS = 15 * 60_000
    const VELOCITY_MAX_LOSS_USD = 30
    const recentLoss = this.sql().exec<{ pnl: number | null }>(
      "SELECT SUM(pnl_usd) AS pnl FROM positions WHERE closed_at >= ? AND priced_by = 'quote'",
      now - VELOCITY_WINDOW_MS,
    ).one().pnl ?? 0
    if (recentLoss + pnlUsd < -VELOCITY_MAX_LOSS_USD) {
      const until = now + PAPER_POLICY.breaker.cooldownMinutes * 60_000
      this.metaSet("breaker_until", String(until))
      this.event("breaker", { tripped: true, kind: "loss-velocity", windowLossUsd: recentLoss + pnlUsd, until })
      capture(this.env, this.ctx, "breaker_tripped", {
        kind: "loss_velocity", window_loss_usd: recentLoss + pnlUsd,
        policy_hash: this.metaGet("policy_hash"),
      })
      this.queueBreakerAlert(now, until,
        `loss velocity — $${Math.abs(recentLoss + pnlUsd).toFixed(2)} lost inside ${VELOCITY_WINDOW_MS / 60_000} minutes (limit $${VELOCITY_MAX_LOSS_USD})`)
    }

    if (reason === "stop-loss" || reason === "safety-exit") {
      const consec = Number(this.metaGet("breaker_consec") ?? "0") + 1
      if (consec >= PAPER_POLICY.breaker.consecutiveStopLimit) {
        const until = now + PAPER_POLICY.breaker.cooldownMinutes * 60_000
        this.metaSet("breaker_until", String(until))
        this.metaSet("breaker_consec", "0")
        this.event("breaker", { tripped: true, until, afterConsecutiveStops: consec })
        capture(this.env, this.ctx, "breaker_tripped", {
          after_stops: consec, cooldown_minutes: PAPER_POLICY.breaker.cooldownMinutes,
          policy_hash: this.metaGet("policy_hash"),
        })
        this.queueBreakerAlert(now, until, `${consec} consecutive stop-loss exits`)
      } else {
        this.metaSet("breaker_consec", String(consec))
      }
    } else if (reason === "take-profit") {
      this.metaSet("breaker_consec", "0")
    }
    this.sql().exec(
      `UPDATE positions SET closed_at = ?, exit_price = ?, exit_reason = ?, pnl_usd = ?, pnl_pct = ?,
              exit_pricing = ?, exit_sig = ?
       WHERE id = ?`,
      now, effective, reason, pnlUsd, pnlPct, pricing, exitSig, p.id,
    )
    this.event("exit", { id: p.id, symbol: p.symbol, reason, pnlUsd, pnlPct, impact, route })
    capture(this.env, this.ctx, "paper_exit", {
      symbol: p.symbol, mint: p.mint, reason,
      pnl_usd: pnlUsd, pnl_pct: pnlPct, held_minutes: (now - p.openedAt) / 60_000,
      policy_hash: p.policyHash,
      // Real measured impact, so TCA can compare quoted cost to what the
      // strategy assumed. This is the number the old model got wrong by 20x.
      exit_price_impact_pct: impact, exit_route: route, routed: q !== null,
    })
  }

  /** Labeled, non-voided outcome stats for one discovery origin. */
  private originStat(origin: string): OriginStat {
    const r = this.sql().exec<{
      labeled: number | null; died: number | null
      entered_ret: number | null; refused_ret: number | null
    }>(
      `SELECT SUM(CASE WHEN labeled = 1 THEN 1 ELSE 0 END) AS labeled,
              SUM(CASE WHEN labeled = 1 AND died = 1 THEN 1 ELSE 0 END) AS died,
              AVG(CASE WHEN labeled = 1 AND entered = 1 THEN forward_ret_pct END) AS entered_ret,
              AVG(CASE WHEN labeled = 1 AND entered = 0 AND eligible = 1 THEN forward_ret_pct END) AS refused_ret
       FROM decisions WHERE voided = 0 AND origin = ?`,
      origin,
    ).one()
    return {
      origin,
      labeled: r.labeled ?? 0,
      died: r.died ?? 0,
      enteredRet: r.entered_ret,
      refusedRet: r.refused_ret,
    }
  }

  /**
   * Email Michael once, when the launchpad comparison first becomes readable.
   *
   * Called after every tick. The state machine exists because this method both
   * awaits network I/O and must never fire twice, and the DO input gate does
   * not hold across an await (the same trap documented at the top of this file,
   * which the tick lease exists to close). So the claim is written BEFORE the
   * send: two overlapping ticks cannot both see an unclaimed alert. A claim
   * older than ten minutes is treated as abandoned, which covers a Worker dying
   * mid-send, and a failed send clears the claim so the next tick retries.
   *
   * Failure is swallowed into a return value. An unreachable mail provider must
   * not take down the trading tick that called it.
   */
  async maybeAlert(): Promise<{ sent: boolean; reason: string }> {
    const state = this.metaGet("launchpad_alert") ?? ""
    if (state.startsWith("sent:")) return { sent: false, reason: "already sent" }
    if (state.startsWith("pending:")) {
      const since = Number(state.slice("pending:".length))
      if (Date.now() - since < 10 * 60_000) return { sent: false, reason: "send in flight" }
    }

    const launchpad = this.originStat("launchpad")
    if (launchpad.labeled < READABLE_SAMPLE) {
      return { sent: false, reason: `launchpad ${launchpad.labeled}/${READABLE_SAMPLE} labeled` }
    }

    const apiKey = this.env.RESEND_API_KEY
    if (!apiKey) return { sent: false, reason: "RESEND_API_KEY unset" }

    this.metaSet("launchpad_alert", `pending:${Date.now()}`)

    const { subject, text } = composeBody({
      launchpad,
      baseline: this.originStat("profile"),
      killed: this.metaGet("kill") === "1",
      breakerOpen: Number(this.metaGet("breaker_until") ?? 0) > Date.now(),
      policyHash: this.metaGet("policy_hash"),
    })

    const result = await sendAlert(apiKey, subject, text)
    if (!result.ok) {
      // Clear the claim so the next tick tries again. Leaving it pending would
      // trade a duplicate email for a silently lost one, which is the worse bug:
      // nobody notices the alert that never arrives.
      this.metaSet("launchpad_alert", "")
      this.event("alert_failed", { error: result.error })
      return { sent: false, reason: result.error }
    }
    this.metaSet("launchpad_alert", `sent:${Date.now()}`)
    this.event("alert_sent", { subject, labeled: launchpad.labeled })
    return { sent: true, reason: subject }
  }

  /**
   * Queue an operational alert for the post-tick flush.
   *
   * Queued, not sent, because the call sites live inside the trading tick and
   * an unreachable mail provider must never stall a close or an entry. The
   * flush runs from the router after the tick, the same seam maybeAlert uses.
   * Idempotent per key: episode semantics (what counts as "the same event
   * again") are the CALLER's job, encoded in the key it chooses.
   */
  private queueAlert(key: string, subject: string, text: string): void {
    if (this.metaGet(`opq:${key}`) !== null) return
    this.metaSet(`opq:${key}`, JSON.stringify({ subject, text, queuedAt: Date.now() }))
  }

  /**
   * One breaker episode emails once. Both trip sites re-fire while the
   * breaker is already open (loss-velocity re-trips on every close inside the
   * window, pushing `until` forward), so keying on `until` alone would send a
   * storm. The sent-marker holds the last alerted episode's expiry: a trip is
   * a NEW episode only after that expiry has passed.
   */
  private queueBreakerAlert(now: number, until: number, cause: string): void {
    if (Number(this.metaGet("opsent:breaker") ?? "0") >= now) return
    this.metaSet("opsent:breaker", String(until))
    this.queueAlert(
      `breaker:${until}`,
      "CroweTrade: circuit breaker tripped",
      [
        `The circuit breaker tripped and new entries are paused until ${new Date(until).toISOString()}.`,
        "",
        `Cause: ${cause}.`,
        "",
        "Exits keep managing open positions; entries resume automatically when",
        "the cooldown expires. No action is required — this email exists so a",
        "stand-down never happens silently.",
        "",
        "Still paper. No capital at risk.",
        "",
        "https://crowetrade-engine.yellow-block-3adc.workers.dev/api/positions",
      ].join("\n"),
    )
  }

  /**
   * Send queued operational alerts. Called from the router AFTER the tick, so
   * mail I/O never sits inside the trading path. A failed send keeps its queue
   * row and retries next flush — a duplicate email is a lesser bug than an
   * alert that silently never arrives. NOTE the honest limit: a STALLED engine
   * cannot email, because nothing is running; stall detection belongs to the
   * external hourly watch, not to the process being watched.
   */
  async flushAlerts(): Promise<{ sent: number; failed: number }> {
    const apiKey = this.env.RESEND_API_KEY
    if (!apiKey) return { sent: 0, failed: 0 }
    const rows = this.sql().exec<{ key: string; value: string }>(
      "SELECT key, value FROM meta WHERE key LIKE 'opq:%' LIMIT 5",
    ).toArray()
    let sent = 0
    let failed = 0
    for (const r of rows) {
      const p = JSON.parse(r.value) as {
        subject: string; text: string; queuedAt: number; pendingAt?: number
      }
      // Claim-before-send, same reasoning as maybeAlert: the input gate does
      // not hold across the await, and two overlapping flushes must not both
      // see an unclaimed row. An abandoned claim expires after ten minutes.
      if (p.pendingAt !== undefined && Date.now() - p.pendingAt < 10 * 60_000) continue
      this.metaSet(r.key, JSON.stringify({ ...p, pendingAt: Date.now() }))
      const result = await sendAlert(apiKey, p.subject, p.text)
      if (result.ok) {
        this.sql().exec("DELETE FROM meta WHERE key = ?", r.key)
        this.event("alert_sent", { kind: "operational", key: r.key, subject: p.subject })
        sent += 1
      } else {
        this.metaSet(r.key, JSON.stringify({ subject: p.subject, text: p.text, queuedAt: p.queuedAt }))
        this.event("alert_failed", { kind: "operational", key: r.key, error: result.error })
        failed += 1
      }
    }
    return { sent, failed }
  }

  /** The autonomous trading tick. Runs once per cron minute. */
  async tick(): Promise<{ entered: number; exited: number; scanned: number }> {
    const now = Date.now()
    const signal = new AbortController().signal
    const policy = PAPER_POLICY

    // Tick lease. Held for the duration, expires so a crashed tick cannot
    // wedge the engine forever. Without this, an overlapping tick double-spends
    // the daily cap and can open two positions in the same mint.
    const LEASE_MS = 3 * 60_000
    const leaseUntil = Number(this.metaGet("tick_lease") ?? "0")
    if (now < leaseUntil) {
      this.event("tick_skipped", { reason: "lease held", until: leaseUntil })
      return { entered: 0, exited: 0, scanned: 0 }
    }
    this.metaSet("tick_lease", String(now + LEASE_MS))

    // Point the shared RPC at Helius when the key is present. Without it the
    // holder call is rate-limited to uselessness and that gate stays blind.
    configureRpc(this.env.HELIUS_API_KEY)
    const hash = await policyHash(policy)
    if (this.metaGet("policy_hash") !== hash) {
      this.metaSet("policy_hash", hash)
      this.event("policy", { hash, policy })
    }

    const killed = this.metaGet("kill") === "1"

    // Scan the market. Discovery failing must not stop exit management, so the
    // two feeds fail independently.
    let candidates: Candidate[] = []
    let solUsd = Number(this.metaGet("sol_usd") ?? "0")
    try {
      // Pass the cached price so a rate-limited quote degrades to a slightly
      // stale SOL figure instead of a lost scan.
      // Refresh the SOL quote at most every five minutes. It was fetched every
      // tick purely to convert liquidity, which is a use that cannot tell the
      // difference, while consuming a request from the same rate limit
      // discovery needs. Passing 0 as the fallback forces a fetch when the
      // cache is cold or stale.
      const SOL_TTL_MS = 5 * 60_000
      const solAge = now - Number(this.metaGet("sol_usd_at") ?? "0")
      const wantFresh = solAge > SOL_TTL_MS || solUsd <= 0
      const scan = await fetchCandidates(signal, wantFresh ? 0 : solUsd)
      candidates = scan.candidates
      if (scan.solUsd > 0 && scan.solUsd !== solUsd) {
        solUsd = scan.solUsd
        this.metaSet("sol_usd", String(solUsd))
        this.metaSet("sol_usd_at", String(now))
      }
      this.metaSet("scanfail_consec", "0")
    } catch (e) {
      this.event("scan_error", { message: e instanceof Error ? e.message : String(e) })
      // Five straight failed scans is an outage, not a blip. Alert exactly at
      // the transition (== 5, not >=) so one episode emails once; the counter
      // resets on the next healthy scan, arming the alert for the next episode.
      const consec = Number(this.metaGet("scanfail_consec") ?? "0") + 1
      this.metaSet("scanfail_consec", String(consec))
      if (consec === 5) {
        this.queueAlert(
          `scanfault:${now}`,
          "CroweTrade: discovery scan failing",
          [
            `The discovery scan has failed ${consec} ticks in a row.`,
            "",
            `Last error: ${e instanceof Error ? e.message : String(e)}`,
            "",
            "The engine keeps ticking — exits still manage open positions and the",
            "launchpad source is fetched separately — but no promotional-feed",
            "candidates are arriving while this persists.",
            "",
            "https://crowetrade-engine.yellow-block-3adc.workers.dev/api/positions",
          ].join("\n"),
        )
      }
    }

    // Second discovery source, added after the promotional feed was MEASURED
    // unprofitable (entered -29.7% vs refused -30.3% at n=87: selection inside
    // that universe adds nothing). The launchpad lists every mint it created in
    // creation order, with no placement fee, so it is the whole launch universe
    // rather than a marketed slice. Both sources run tagged by origin so the
    // calibration loop decides between them on evidence, not on argument.
    if (solUsd > 0) {
      const launchpad = await fetchLaunchpadCandidates(solUsd, signal)
      const seen = new Set(candidates.map((c) => c.mint))
      for (const c of launchpad) {
        if (!seen.has(c.mint)) candidates.push(c)
      }
      // Remember every deployer we have ever seen mint a token. This is the
      // input the deployer-history gate has never had, and it only exists
      // because the launchpad names the creator.
      for (const c of launchpad) {
        if (c.creator) {
          this.sql().exec(
            "INSERT INTO creators (mint, creator, first_seen) VALUES (?, ?, ?) ON CONFLICT(mint) DO NOTHING",
            c.mint, c.creator, now,
          )
        }
      }
    }

    const open = this.openPositions()

    // Price and re-judge held tokens: union of fresh candidates and holdings.
    // Follow-up set: held positions AND every still-unlabeled decision.
    //
    // The second half is not optional bookkeeping, it is what makes the
    // calibration experiment valid. Held tokens are re-priced every tick
    // because we own them; refused tokens would otherwise stop being observed
    // the moment promotional discovery delists them, go stale, and get labeled
    // "died" by absence rather than by dying. That would kill the control
    // group on a technicality and make entered-vs-refused look decisive while
    // measuring nothing but which cohort we bothered to watch.
    const pendingLabel = this.sql().exec<{ mint: string }>(
      "SELECT mint FROM decisions WHERE labeled = 0 AND voided = 0",
    ).toArray().map((r) => r.mint)

    const inScan = new Set(candidates.map((c) => c.mint))
    const followUp = [...new Set([...open.map((p) => p.mint), ...pendingLabel])]
      .filter((m) => !inScan.has(m))

    let heldPairs: Candidate[] = []
    if (followUp.length > 0 && solUsd > 0) {
      heldPairs = await fetchPairsForMints(followUp, solUsd, signal)
        .catch(() => [] as Candidate[])
    }
    const everything = [...candidates, ...heldPairs]

    // Jupiter fallback pricing for tokens DexScreener cannot see.
    //
    // Fresh bonding-curve mints have no DexScreener pair for their first
    // minutes, so the pair-based follow-up returns nothing for them, their
    // ticks go stale, and the labeler calls them dead. That produced a 100%
    // death rate for launchpad-origin decisions on the first readout -- a
    // measurement artifact, not a market fact: a token with no pair still
    // quotes fine on Jupiter, which is the venue we would actually trade
    // through. Pricing the gap here keeps both cohorts observable, which is
    // the same failure the external audit caught in a different form.
    const priced = new Set(everything.map((c) => c.mint))
    const blind = [...new Set([...open.map((p) => p.mint), ...pendingLabel])]
      .filter((m) => !priced.has(m))
      .slice(0, 12) // bounded: one quote each, against the subrequest budget

    for (const mint of blind) {
      const decimals = Number(this.metaGet(`decimals:${mint}`) ?? "6")
      const q = await quoteBuy(mint, 0.1, SLIPPAGE_BPS)
      if (!q) continue // genuinely unroutable: no buyer, and THAT is a death
      const tokensOut = Number(q.outAmount) / 10 ** decimals
      if (tokensOut <= 0) continue
      const priceUsd = (0.1 * solUsd) / tokensOut
      this.sql().exec(
        "INSERT INTO ticks (mint, at, price, liquidity_usd, buys_24h, sells_24h, origin) VALUES (?, ?, ?, ?, ?, ?, ?)",
        mint, now, priceUsd, null, null, null, "jupiter-probe",
      )
    }

    // One RPC batch resolves authorities for the whole universe this tick.
    const facts = await fetchMintFacts(everything.map((c) => c.mint), signal).catch(
      () => new Map<string, MintFacts>(),
    )
    for (const c of everything) {
      const f = facts.get(c.mint)
      if (f) {
        c.snapshot.mintAuthority = f.mintAuthority
        c.snapshot.freezeAuthority = f.freezeAuthority
        // Decimals are needed later to convert a held float amount back into
        // base units for the sell quote, and the mint may be gone from
        // discovery by then, so persist it at first sight.
        this.metaSet(`decimals:${c.mint}`, String(f.decimals))
      }
    }

    // Record every priced observation. This is the beginning of the corpus:
    // OUR minute-by-minute view of each token's price, liquidity and flow,
    // which no promotional feed can pollute. Strategy v2's entry signal reads
    // trajectories from here — what WE watched happen — rather than trusting
    // a listing's snapshot. Pruned at 48h to respect DO storage.
    for (const c of everything) {
      this.sql().exec(
        "INSERT INTO ticks (mint, at, price, liquidity_usd, buys_24h, sells_24h, origin) VALUES (?, ?, ?, ?, ?, ?, ?)",
        c.mint, now, c.priceUsd, c.liquidityUsd, c.buys24h, c.sells24h, c.origin,
      )
    }
    this.sql().exec("DELETE FROM ticks WHERE at < ?", now - 48 * 3_600_000)

    // ── The calibration loop, half one: decision snapshots. ────────────────
    // One row per mint, taken the FIRST time we have enough of our own ticks
    // to compute features. Skipped tokens are recorded as deliberately as
    // entered ones: calibration lives in the counterfactuals — what happened
    // to the launches we refused — and a dataset of entries alone can never
    // separate the policy's judgment from the market's behavior.
    for (const c of candidates) {
      const exists = this.sql().exec<{ n: number }>(
        "SELECT COUNT(*) AS n FROM decisions WHERE mint = ?", c.mint,
      ).one().n
      if (exists > 0) continue

      const rows = this.sql().exec<{
        price: number | null; liquidity_usd: number | null
        buys_24h: number | null; sells_24h: number | null
      }>(
        "SELECT price, liquidity_usd, buys_24h, sells_24h FROM ticks WHERE mint = ? ORDER BY at DESC LIMIT 6",
        c.mint,
      ).toArray().reverse()
      if (rows.length < policy.entry.minObservedTicks) continue

      const feats = computeFeatures({
        prices: rows.map((r) => r.price ?? 0),
        liquidity: rows.map((r) => r.liquidity_usd ?? 0),
        buys24h: rows.map((r) => r.buys_24h ?? 0),
        sells24h: rows.map((r) => r.sells_24h ?? 0),
      })

      const verdict = combineVerdict(evaluateGates(c.snapshot))
      const ageMin = c.createdAt === null ? null : (now - c.createdAt) / 60_000
      const skipReason =
        c.origin !== "held" && !policy.entry.allowedOrigins.includes(c.origin)
          ? `origin-${c.origin}`
        : ageMin === null || ageMin < policy.entry.minTokenAgeMinutes ? "too-new"
        : ageMin > policy.entry.maxTokenAgeMinutes ? "too-old"
        : c.liquidityUsd === null || c.liquidityUsd < policy.entry.minLiquidityUsd ? "thin"
        : c.changeH1 === null || c.changeH1 > policy.entry.maxChangeH1Pct ? "parabolic"
        : verdict === "blocked" || verdict === "insufficient-data" ? `verdict-${verdict}`
        : null

      this.sql().exec(
        `INSERT INTO decisions (mint, at, symbol, price, origin, verdict, features, eligible, skip_reason, model_prob)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        c.mint, now, c.symbol, c.priceUsd, c.origin, verdict,
        JSON.stringify(feats), skipReason === null ? 1 : 0, skipReason,
        this.armedProbFor(c.mint, feats, c.origin, now),
      )
    }

    // Half two: outcomes. Thirty minutes after a decision snapshot, score it
    // from our own subsequent ticks. Every labeled row is one training example
    // for the calibrated edge model: features at decision time, then what the
    // market actually did. This is the dataset "crack the algorithm" needs.
    const toLabel = this.sql().exec<{ mint: string; price: number | null; features: string; eligible: number; entered: number }>(
      "SELECT mint, price, features, eligible, entered FROM decisions WHERE labeled = 0 AND voided = 0 AND at <= ? LIMIT 20",
      now - 30 * 60_000,
    ).toArray()
    for (const d of toLabel) {
      const latest = this.sql().exec<{ at: number; price: number | null; liquidity_usd: number | null }>(
        "SELECT at, price, liquidity_usd FROM ticks WHERE mint = ? ORDER BY at DESC LIMIT 1", d.mint,
      ).toArray()[0]
      const last = latest ?? null
      // Staleness now means "we followed it and the venue stopped pricing it",
      // because follow-up covers both cohorts. A delisted-but-alive token
      // still returns a pair quote; one that returns nothing has no market.
      const stale = last === null || now - last.at > 10 * 60_000
      const ret = d.price && d.price > 0 && last?.price
        ? ((last.price - d.price) / d.price) * 100
        : null
      // Dead = we stopped seeing it, its pool bled out, or it lost ~everything.
      // Disappearing from every feed IS the modal death and must count as one.
      // Thin-liquidity death only counts when liquidity was actually MEASURED.
      // A Jupiter probe tick carries a price but no liquidity figure, and
      // treating that null as "under $500" would mark every bonding-curve
      // token dead for the crime of being priced by quote instead of by pair.
      const liqKnown = last?.liquidity_usd !== null && last?.liquidity_usd !== undefined
      const died =
        stale ||
        (liqKnown && (last?.liquidity_usd ?? 0) < 500) ||
        (ret !== null && ret <= -90)
      this.sql().exec(
        "UPDATE decisions SET labeled = 1, forward_ret_pct = ?, died = ?, labeled_at = ? WHERE mint = ?",
        ret, died ? 1 : 0, now, d.mint,
      )
      capture(this.env, this.ctx, "outcome_labeled", {
        mint: d.mint, eligible: d.eligible === 1, entered: d.entered === 1,
        forward_ret_pct: ret, died, ...JSON.parse(d.features) as Record<string, unknown>,
      })
    }

    // Deployer history, computed from OUR OWN labeled outcomes.
    //
    // This gate has read "unknown" since the system was built, because no feed
    // sells deployer reputation. It becomes computable now only because the
    // launchpad names the creator and we have been labeling what happened to
    // every token we saw. Prior mints are counted from the creators table;
    // rugs are counted from decisions we labeled died. Costs no network call:
    // the corpus answers it.
    for (const c of everything) {
      const creator = c.creator ?? this.sql().exec<{ creator: string }>(
        "SELECT creator FROM creators WHERE mint = ?", c.mint,
      ).toArray()[0]?.creator
      if (!creator) continue
      const hist = this.sql().exec<{ prior: number; rugs: number }>(
        `SELECT COUNT(*) AS prior,
                SUM(CASE WHEN d.died = 1 THEN 1 ELSE 0 END) AS rugs
         FROM creators cr
         JOIN decisions d ON d.mint = cr.mint
         WHERE cr.creator = ? AND cr.mint != ? AND d.labeled = 1`,
        creator, c.mint,
      ).one()
      // Only assert history when we actually have some. Zero prior labeled
      // mints stays undefined, never a passing "0 rugs" that would flatter an
      // unknown deployer into looking proven.
      if (hist.prior > 0) {
        c.snapshot.deployerPriorMints = hist.prior
        c.snapshot.deployerPriorRugs = hist.rugs ?? 0
      }
    }

    // Holder concentration costs one call per token, so it is resolved only for
    // tokens that are actual entry candidates or already held — not for the
    // whole scan list, which would burn the daily credit budget on tokens the
    // policy would reject anyway.
    const heldMints = new Set(open.map((p) => p.mint))
    const worthChecking = everything.filter((c) => {
      if (heldMints.has(c.mint)) return true
      if (c.createdAt === null || c.liquidityUsd === null) return false
      const ageMin = (now - c.createdAt) / 60_000
      return (
        ageMin >= policy.entry.minTokenAgeMinutes &&
        ageMin <= policy.entry.maxTokenAgeMinutes &&
        c.liquidityUsd >= policy.entry.minLiquidityUsd &&
        c.changeH1 !== null &&
        c.changeH1 <= policy.entry.maxChangeH1Pct
      )
    })

    // Hard cap per tick. A Worker has a finite subrequest budget and the tick
    // already spends it on discovery, pricing, quotes and simulations; firing
    // one unbounded holder call per candidate on top of that exhausts it and
    // takes the whole tick down. Held positions are checked first because an
    // open position turning concentrated is more urgent than screening a new
    // one, and it also bounds the daily RPC credit spend.
    const HOLDER_CHECKS_PER_TICK = 10
    const ordered = [
      ...worthChecking.filter((c) => heldMints.has(c.mint)),
      ...worthChecking.filter((c) => !heldMints.has(c.mint)),
    ].slice(0, HOLDER_CHECKS_PER_TICK)

    await Promise.all(
      ordered.map(async (c) => {
        const f = facts.get(c.mint)
        if (!f) return
        const share = await fetchTopHolderShare(c.mint, f.supply, signal)
        if (share !== undefined) c.snapshot.topHolderShare = share
      }),
    )

    // Exits run even when killed: the kill switch stops NEW risk, never risk
    // management. Requested vetoes inside their window exit first.
    const prices = new Map<string, { priceUsd: number; verdict: Verdict }>()
    for (const c of everything) {
      if (c.priceUsd !== null && c.priceUsd > 0) {
        prices.set(c.mint, {
          priceUsd: c.priceUsd,
          verdict: combineVerdict(evaluateGates(c.snapshot)),
        })
      }
    }

    let exited = 0
    const vetoIds = new Set(
      this.sql().exec<{ id: string }>(
        "SELECT id FROM positions WHERE closed_at IS NULL AND veto_requested = 1",
      ).toArray().map((r) => r.id),
    )
    for (const p of open) {
      if (vetoIds.has(p.id)) {
        const cur = prices.get(p.mint)
        // A veto with no fresh price exits at entry price as the least-bad
        // honest option; the event records that the mark was stale.
        const px = cur?.priceUsd ?? p.entryPriceUsd
        await this.closePosition(p, px, "veto", now, solUsd)
        exited += 1
      }
    }
    const stillOpen = this.openPositions()
    for (const d of decideExits(stillOpen, prices, policy, now)) {
      await this.closePosition(d.position, d.exitPriceUsd, d.reason, now, solUsd)
      exited += 1
    }

    // Entries.
    let entered = 0
    const breakerUntil = Number(this.metaGet("breaker_until") ?? "0")
    const breakerOpen = now < breakerUntil
    // The envelope's own death date, ENFORCED, not just declared. The external
    // audit's first-listed unfixed finding: expiresAt was schema the tick never
    // read, meaning an expired consent would have kept trading. An expired
    // envelope stops NEW risk exactly like the kill switch — exits above this
    // line keep managing what is already open. An unparseable date counts as
    // expired: unknown never authorizes anything, entries least of all.
    const expiresMs = Date.parse(policy.expiresAt)
    const expired = Number.isNaN(expiresMs) || now >= expiresMs
    if (expired && this.metaGet("expired_noted") !== policy.expiresAt) {
      // Once per envelope, not once per tick: the record needs the fact, not
      // 1,440 copies of it a day. It also emails — an engine that stopped
      // trading because its consent lapsed must not look merely quiet.
      this.metaSet("expired_noted", policy.expiresAt)
      this.event("entry_skipped", { reason: "policy envelope expired", expiresAt: policy.expiresAt })
      this.queueAlert(
        `expired:${policy.expiresAt}`,
        "CroweTrade: policy envelope expired — entries stopped",
        [
          `The policy envelope expired at ${policy.expiresAt} and the engine has stopped entering.`,
          "Exits keep managing open positions. Deploy a fresh envelope to resume.",
          "",
          "https://crowetrade-engine.yellow-block-3adc.workers.dev/api/positions",
        ].join("\n"),
      )
    }
    if (!killed && !breakerOpen && !expired) {
      const day = new Date(now).toISOString().slice(0, 10)
      const spentToday = Number(this.metaGet(`spend:${day}`) ?? "0")
      let spentRunning = spentToday

      // Our own tape, per candidate: the last few minute-ticks recorded above,
      // oldest first, straight from what this engine itself observed.
      const trajectories = new Map<string, Trajectory>()
      for (const c of candidates) {
        const rows = this.sql().exec<{ price: number | null; liquidity_usd: number | null }>(
          "SELECT price, liquidity_usd FROM ticks WHERE mint = ? ORDER BY at DESC LIMIT ?",
          c.mint, policy.entry.minObservedTicks,
        ).toArray().reverse()
        trajectories.set(c.mint, {
          prices: rows.map((r) => r.price ?? 0),
          liquidity: rows.map((r) => r.liquidity_usd ?? 0),
        })
      }

      // Armed-model probabilities, from the same six-tick window the decision
      // snapshot uses. A candidate without enough of our own tape gets null,
      // and null does not pass an armed gate — unknown is never a pass.
      const modelProbs = new Map<string, number | null>()
      if (policy.entry.minModelProb !== null) {
        for (const c of candidates) {
          const rows = this.sql().exec<{
            price: number | null; liquidity_usd: number | null
            buys_24h: number | null; sells_24h: number | null
          }>(
            "SELECT price, liquidity_usd, buys_24h, sells_24h FROM ticks WHERE mint = ? ORDER BY at DESC LIMIT 6",
            c.mint,
          ).toArray().reverse()
          if (rows.length < policy.entry.minObservedTicks) {
            modelProbs.set(c.mint, null)
            continue
          }
          const feats = computeFeatures({
            prices: rows.map((r) => r.price ?? 0),
            liquidity: rows.map((r) => r.liquidity_usd ?? 0),
            buys24h: rows.map((r) => r.buys_24h ?? 0),
            sells24h: rows.map((r) => r.sells_24h ?? 0),
          })
          modelProbs.set(c.mint, this.armedProbFor(c.mint, feats, c.origin, now))
        }
      }

      const modelRefusals: ModelRefusal[] = []
      const entries = decideEntries(candidates, this.openPositions(), spentToday, solUsd, policy, now, trajectories, modelProbs, modelRefusals)
      // A few named examples plus a count: enough to see WHO the model is
      // refusing and how often, without an event per refusal flooding the log.
      for (const r of modelRefusals.slice(0, 5)) {
        this.event("entry_skipped", {
          symbol: r.symbol, mint: r.mint, reason: "model probability below minimum",
          prob: r.prob === null ? null : Number(r.prob.toFixed(3)),
        })
      }
      if (modelRefusals.length > 5) {
        this.event("entry_skipped", {
          reason: "model probability below minimum",
          additional: modelRefusals.length - 5,
        })
      }
      for (const e of entries) {
        const c = e.candidate
        if (c.priceUsd === null || c.liquidityUsd === null) continue

        // Price the entry off a REAL route. No route means no entry: a token we
        // cannot buy through Jupiter is one we could not have bought at all,
        // and inventing a fill would put a position in the book that never
        // could have existed.
        const decimals = Number(this.metaGet(`decimals:${c.mint}`) ?? "6")
        const q = await quoteBuy(c.mint, e.sizeSol, SLIPPAGE_BPS)
        if (!q) {
          this.event("entry_skipped", { symbol: c.symbol, mint: c.mint, reason: "no route" })
          continue
        }

        // Cost hurdle: one-way impact caps roughly half the round trip, and
        // v1's stop-outs concentrated in exactly the pools this refuses.
        const impactPct = q.priceImpactPct * 100
        if (impactPct > policy.entry.maxEntryImpactPct) {
          this.event("entry_skipped", {
            symbol: c.symbol, mint: c.mint, reason: "impact above cost hurdle",
            impactPct: Number(impactPct.toFixed(2)),
          })
          continue
        }

        // Build and simulate the real transaction before committing to the
        // position. A route that quotes but does not execute is a trade we
        // would have paid fees to fail at.
        const sim = await dryRunSwap(q.raw)
        if (!sim.ok) {
          this.event("entry_skipped", {
            symbol: c.symbol, mint: c.mint, reason: "simulation failed", error: sim.error,
          })
          capture(this.env, this.ctx, "entry_rejected", {
            symbol: c.symbol, mint: c.mint, reason: "simulation_failed", error: sim.error,
          })
          continue
        }

        // Priority fee is a real cost of entering and must be charged to the
        // position, or the record quietly understates what trading costs.
        const feeSol = (sim.priorityFeeLamports ?? 0) / LAMPORTS_PER_SOL
        let sizeUsd = (e.sizeSol + feeSol) * solUsd
        let tokenAmount = Number(q.outAmount) / 10 ** decimals
        let execution = "paper"
        let entrySig: string | null = null

        // ── LIVE ENTRY ──────────────────────────────────────────────────
        //
        // Reached only when both environment locks are open AND the envelope
        // is a live one. Everything above this point is identical for paper
        // and live, which is the property that makes the paper record
        // meaningful as a rehearsal.
        //
        // The numbers written to the book come from the CHAIN, not from the
        // quote: `q.outAmount` is what Jupiter predicted and `fill.tokenDelta`
        // is what arrived. Recording the prediction would reintroduce the
        // exact error the 20x slippage incident taught.
        if (this.liveEnabled(policy)) {
          const owner = this.tradingOwner()
          const key = this.env.TRADING_KEYPAIR as string
          const balance = await walletBalanceSol(owner ?? "")
          if (!owner || balance === null) {
            this.event("entry_skipped", { symbol: c.symbol, mint: c.mint, reason: "wallet unreadable" })
            continue
          }
          const result = await executeSwap(q.raw, owner, key, c.mint, "entry", {
            intent: {
              mint: c.mint, sizeSol: e.sizeSol, spentTodaySol: spentRunning,
              openPositions: this.openPositions().length, impactPct,
              simulationOk: sim.ok, walletBalanceSol: balance,
            },
            ctx: {
              policy, nowMs: now, killed, liveArmed: true,
              // Verified here rather than trusted: the guard is pure and
              // cannot do async crypto, so the caller owes it this answer.
              signatureVerified: policy.signature !== null && policy.signer !== null
                && await verifyPolicySignature(hash, policy.signer, policy.signature),
            },
          })
          if (!result.ok || !result.fill) {
            this.event("entry_skipped", {
              symbol: c.symbol, mint: c.mint, reason: "live entry refused",
              error: result.error, signature: result.signature, paidFee: !result.refusedBeforeSend,
            })
            continue
          }
          // Realized, from pre/post balances.
          const f = result.fill
          tokenAmount = Number(f.tokenDelta) / 10 ** f.decimals
          sizeUsd = (Number(f.solDeltaLamports) / 1e9) * solUsd
          execution = "live"
          entrySig = result.signature
          if (tokenAmount <= 0) {
            // Confirmed, but no tokens arrived. Something is wrong that this
            // code cannot reason about, so it records the event and does NOT
            // write a position it would then try to sell.
            this.event("entry_skipped", {
              symbol: c.symbol, mint: c.mint, reason: "live fill delivered no tokens",
              signature: entrySig,
            })
            continue
          }
          this.event("live_entry", {
            symbol: c.symbol, mint: c.mint, signature: entrySig,
            solSpent: Number(f.solDeltaLamports) / 1e9, tokens: tokenAmount,
            quotedTokens: Number(q.outAmount) / 10 ** decimals,
          })
        }

        if (tokenAmount <= 0) continue
        const entryPrice = sizeUsd / tokenAmount
        const id = crypto.randomUUID()
        this.sql().exec(
          `INSERT INTO positions
           (id, mint, symbol, entry_price, size_sol, size_usd, token_amount, opened_at, policy_hash, verdict_entry, priced_by, execution, entry_sig)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'quote', ?, ?)`,
          id, c.mint, c.symbol, entryPrice, e.sizeSol, sizeUsd, tokenAmount, now, hash, e.verdict,
          execution, entrySig,
        )
        this.sql().exec(
          "UPDATE decisions SET entered = 1, entry_impact_pct = ? WHERE mint = ? AND labeled = 0",
          impactPct, c.mint,
        )
        this.event("entry", {
          id, symbol: c.symbol, mint: c.mint, sizeSol: e.sizeSol, entryPrice,
          verdict: e.verdict, impact: q.priceImpactPct, route: q.route,
          modelProb: e.modelProb === null ? null : Number(e.modelProb.toFixed(3)),
        })
        capture(this.env, this.ctx, "paper_entry", {
          symbol: c.symbol, mint: c.mint, size_sol: e.sizeSol, entry_price: entryPrice,
          verdict: e.verdict, liquidity_usd: c.liquidityUsd, token_age_minutes: c.createdAt ? (now - c.createdAt) / 60_000 : null,
          policy_hash: hash,
          entry_price_impact_pct: q.priceImpactPct, entry_route: q.route,
          sim_units_consumed: sim.unitsConsumed, priority_fee_lamports: sim.priorityFeeLamports,
          model_prob: e.modelProb,
        })
        // Accumulate against a running total, not the pre-loop snapshot: the
        // previous form wrote spentToday + thisSize on every iteration, so a
        // tick that opened four positions recorded one position's spend and
        // the daily cap silently under-counted by 4x.
        spentRunning += e.sizeSol
        this.metaSet(`spend:${day}`, String(spentRunning))
        entered += 1
      }
    }

    capture(this.env, this.ctx, "engine_tick", {
      scanned: candidates.length, entered, exited, open: this.openPositions().length, killed,
    })
    // Release the lease so the next cron minute runs immediately rather than
    // waiting out the expiry.
    this.metaSet("tick_lease", "0")
    return { entered, exited, scanned: candidates.length }
  }

  /** Public read model for the terminal and the landing page. */
  summary(): unknown {
    const sql = this.sql()
    const open = sql.exec("SELECT * FROM positions WHERE closed_at IS NULL ORDER BY opened_at DESC").toArray()
    const closed = sql.exec(
      "SELECT * FROM positions WHERE closed_at IS NOT NULL ORDER BY closed_at DESC LIMIT 50",
    ).toArray()
    // Headline stats count REAL-ROUTE fills only. The model-priced rows stay
    // queryable and are reported separately rather than silently dropped.
    const totals = sql.exec<{ n: number; pnl: number | null; wins: number | null }>(
      `SELECT COUNT(*) AS n, SUM(pnl_usd) AS pnl,
              SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins
       FROM positions WHERE closed_at IS NOT NULL AND priced_by = 'quote'`,
    ).one()
    const legacy = sql.exec<{ n: number }>(
      "SELECT COUNT(*) AS n FROM positions WHERE closed_at IS NOT NULL AND priced_by = 'model'",
    ).one()

    // Per-policy cohorts. The lifetime aggregate mixes policy versions and is
    // therefore NOT the funding number: the 100-close criterion requires one
    // stable policy, and every policy change restarts that clock. Reporting
    // only the aggregate let a losing v1 and a different v2 average into a
    // single meaningless figure.
    const currentHash = this.metaGet("policy_hash")
    const cohorts = sql.exec<{ policy_hash: string; n: number; pnl: number | null; wins: number | null; unroutable: number | null }>(
      `SELECT policy_hash, COUNT(*) AS n, SUM(pnl_usd) AS pnl,
              SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN exit_pricing = 'unroutable' THEN 1 ELSE 0 END) AS unroutable
       FROM positions
       WHERE closed_at IS NOT NULL AND priced_by = 'quote'
       GROUP BY policy_hash ORDER BY n DESC`,
    ).toArray().map((r) => ({
      policyHash: r.policy_hash.slice(0, 8),
      current: r.policy_hash === currentHash,
      closed: r.n,
      pnlUsd: Number((r.pnl ?? 0).toFixed(2)),
      winRate: r.n > 0 ? Number(((r.wins ?? 0) / r.n).toFixed(3)) : null,
      unroutableExits: r.unroutable ?? 0,
    }))
    // Why the engine is or is not trading, without needing a log dive. A
    // silent engine and a capped engine look identical from the outside, and
    // that ambiguity cost a debugging cycle.
    const day = new Date().toISOString().slice(0, 10)
    const spentTodaySol = Number(this.metaGet(`spend:${day}`) ?? "0")
    const openCount = open.length
    const breakerUntil = Number(this.metaGet("breaker_until") ?? "0")
    const breakerOpen = Date.now() < breakerUntil
    const budget = {
      spentTodaySol,
      dailyCapSol: PAPER_POLICY.dailyCapSol,
      remainingSol: Math.max(0, PAPER_POLICY.dailyCapSol - spentTodaySol),
      openSlots: Math.max(0, PAPER_POLICY.maxOpenPositions - openCount),
      breaker: breakerOpen
        ? { open: true, until: new Date(breakerUntil).toISOString() }
        : { open: false, consecutiveStops: Number(this.metaGet("breaker_consec") ?? "0") },
      canEnter:
        this.metaGet("kill") !== "1" &&
        !breakerOpen &&
        openCount < PAPER_POLICY.maxOpenPositions &&
        PAPER_POLICY.dailyCapSol - spentTodaySol >= PAPER_POLICY.perTradeCapSol,
    }

    // The calibration dataset's own health readout: how many decisions taken,
    // how many labeled, and the early separation between what we entered and
    // what we refused. When labeled counts reach the hundreds, this block is
    // where the first evidence of (or against) edge will surface.
    const cal = sql.exec<{
      decisions: number; labeled: number; died: number | null
      entered_ret: number | null; skipped_ret: number | null
    }>(
      `SELECT COUNT(*) AS decisions,
              SUM(labeled) AS labeled,
              SUM(CASE WHEN labeled = 1 AND died = 1 THEN 1 ELSE 0 END) AS died,
              AVG(CASE WHEN labeled = 1 AND entered = 1 THEN forward_ret_pct END) AS entered_ret,
              AVG(CASE WHEN labeled = 1 AND entered = 0 AND eligible = 1 THEN forward_ret_pct END) AS skipped_ret
       FROM decisions WHERE voided = 0`,
    ).one()

    // Per-origin breakout. Adding a second discovery source is only an
    // experiment if the result can be read per source; a pooled average would
    // hide a good universe inside a bad one and vice versa. This is the readout
    // that says whether the launchpad move worked.
    // `decisions` counts LIVE rows only. It previously counted voided ones too,
    // which advertised 1583 launchpad decisions when roughly 1300 of them were
    // quarantined price-scale wreckage. A sample size that includes rows the
    // labeler refuses to touch is not a sample size, and reading it as one is
    // exactly how the original bad launchpad claim got believed.
    const byOrigin = sql.exec<{
      origin: string; n: number; voided: number; labeled: number; died: number | null
      entered_n: number; entered_ret: number | null; refused_ret: number | null
    }>(
      `SELECT COALESCE(origin, 'unknown') AS origin,
              SUM(CASE WHEN voided = 0 THEN 1 ELSE 0 END) AS n,
              SUM(voided) AS voided,
              SUM(CASE WHEN voided = 0 THEN labeled ELSE 0 END) AS labeled,
              SUM(CASE WHEN voided = 0 AND labeled = 1 AND died = 1 THEN 1 ELSE 0 END) AS died,
              SUM(CASE WHEN voided = 0 AND entered = 1 THEN 1 ELSE 0 END) AS entered_n,
              AVG(CASE WHEN voided = 0 AND labeled = 1 AND entered = 1 THEN forward_ret_pct END) AS entered_ret,
              AVG(CASE WHEN voided = 0 AND labeled = 1 AND entered = 0 AND eligible = 1 THEN forward_ret_pct END) AS refused_ret
       FROM decisions GROUP BY COALESCE(origin, 'unknown') ORDER BY n DESC`,
    ).toArray().map((r) => ({
      origin: r.origin,
      decisions: r.n,
      voided: r.voided ?? 0,
      labeled: r.labeled ?? 0,
      entered: r.entered_n,
      deathRate: (r.labeled ?? 0) > 0 ? Number((((r.died ?? 0) / (r.labeled ?? 1))).toFixed(3)) : null,
      avgForwardRetEnteredPct: r.entered_ret === null ? null : Number(r.entered_ret.toFixed(1)),
      avgForwardRetRefusedPct: r.refused_ret === null ? null : Number(r.refused_ret.toFixed(1)),
    }))

    // Why candidates are being turned away, counted. A gate that rejects
    // almost everything is either protecting the book or blinding it, and the
    // distribution is the only way to tell which.
    const skipReasons = sql.exec<{ reason: string; n: number }>(
      `SELECT COALESCE(skip_reason, 'eligible') AS reason, COUNT(*) AS n
       FROM decisions WHERE voided = 0 GROUP BY COALESCE(skip_reason, 'eligible') ORDER BY n DESC`,
    ).toArray().map((r) => ({ reason: r.reason, count: r.n }))

    return {
      mode: "paper",
      killed: this.metaGet("kill") === "1",
      policyHash: this.metaGet("policy_hash"),
      budget,
      byOrigin,
      skipReasons,
      calibration: {
        decisions: cal.decisions,
        labeled: cal.labeled ?? 0,
        // Distinguishes "no labels yet because none are due" from "labeling is
        // broken" — the same ambiguity that hid the simulate-403 for hours.
        oldestUnlabeledAgeMin: (() => {
          const r = sql.exec<{ at: number | null }>(
            "SELECT MIN(at) AS at FROM decisions WHERE labeled = 0 AND voided = 0",
          ).one().at
          return r === null ? null : Math.round((Date.now() - r) / 60_000)
        })(),
        dueForLabel: sql.exec<{ n: number }>(
          "SELECT COUNT(*) AS n FROM decisions WHERE labeled = 0 AND voided = 0 AND at <= ?",
          Date.now() - 30 * 60_000,
        ).one().n,
        deathRate: (cal.labeled ?? 0) > 0 ? (cal.died ?? 0) / (cal.labeled ?? 1) : null,
        avgForwardRetEnteredPct: cal.entered_ret,
        avgForwardRetEligibleSkippedPct: cal.skipped_ret,
      },
      // Alert plumbing, exposed for the same reason the skip reasons are: a
      // notification nobody receives and nobody can see failing is worse than
      // no notification, because it is silently trusted. The send path itself
      // is unverified (no test send was made), so `lastError` is the only thing
      // that will say so, and it needs somewhere to be read.
      alert: (() => {
        const state = this.metaGet("launchpad_alert") ?? ""
        const lp = byOrigin.find((o) => o.origin === "launchpad")
        const failure = sql.exec<{ data: string }>(
          "SELECT data FROM events WHERE kind = 'alert_failed' ORDER BY at DESC LIMIT 1",
        ).toArray()[0]
        return {
          state: state.startsWith("sent:")
            ? "sent"
            : state.startsWith("pending:")
              ? "sending"
              : "waiting",
          labeled: lp?.labeled ?? 0,
          needed: READABLE_SAMPLE,
          configured: Boolean(this.env.RESEND_API_KEY),
          lastError: failure ? (JSON.parse(failure.data) as { error?: string }).error : null,
        }
      })(),
      // Recent decisions, including rejections. The engine already recorded why
      // it declined each candidate; not exposing that made a silent engine
      // indistinguishable from a broken one.
      events: sql.exec("SELECT at, kind, data FROM events ORDER BY at DESC LIMIT 25").toArray(),
      open, closed,
      /** Per-policy cohorts. The funding criterion reads THIS, not lifetime. */
      cohorts,
      stats: {
        /** LIFETIME across all policy versions. Not the funding number. */
        closedCount: totals.n,
        totalPnlUsd: totals.pnl ?? 0,
        winRate: totals.n > 0 ? (totals.wins ?? 0) / totals.n : null,
        /** Excluded from the headline: priced by the retired slippage model. */
        excludedModelPriced: legacy.n,
      },
    }
  }

  /**
   * Counterfactual exit sweep over positions we actually took.
   *
   * For each closed real-quote position, replay OUR OWN recorded ticks between
   * entry and exit and ask what a given (takeProfit, stopLoss) pair would have
   * returned. This is not a backtest of entries — those are fixed, they really
   * happened — it is a backtest of the EXIT rule against observed prices, which
   * is the one counterfactual the tick history can answer honestly.
   *
   * Caveat carried in the output: replay assumes we could exit at the observed
   * mark, whereas real exits pay impact. Treat these as upper bounds and rank
   * rules against each other, never as achievable PnL.
   */
  /**
   * Armed-model probability for a mint from the SAME inputs training used:
   * computeFeatures over our own tape, measured liquidity from the nearest
   * tick in the three minutes before `at`, origin flag. One vector builder
   * serves training and this call — drift between the two is train/serve
   * skew, the failure mode that is silent until the live record diverges
   * from every backtest.
   */
  private armedProbFor(mint: string, feats: FeatureSnapshot, origin: string, at: number): number {
    const liq = this.sql().exec<{ liquidity_usd: number | null }>(
      `SELECT liquidity_usd FROM ticks
       WHERE mint = ? AND at <= ? AND at >= ? AND liquidity_usd IS NOT NULL
       ORDER BY at DESC LIMIT 1`,
      mint, at, at - 180_000,
    ).toArray()[0]?.liquidity_usd ?? null
    return score(ARMED_MODEL, buildFeatureVector(feats, liq, origin === "launchpad"))
  }

  exitSweep(): unknown {
    const sql = this.sql()
    const positions = sql.exec<{
      mint: string; entry_price: number; size_usd: number
      opened_at: number; origin: string | null
    }>(
      `SELECT p.mint, p.entry_price, p.size_usd, p.opened_at,
              (SELECT MIN(d.origin) FROM decisions d WHERE d.mint = p.mint) AS origin
       FROM positions p WHERE p.closed_at IS NOT NULL AND p.priced_by = 'quote'`,
    ).toArray()

    // Every rule replays the SAME fixed horizon from entry — the shipped
    // policy's 30-minute time stop — regardless of when the real position
    // closed. The earlier version replayed only [opened_at, closed_at], so a
    // position the live stop closed at minute 3 could never answer "what if we
    // had held", which is the exact question the realized record raises:
    // launchpad stops all lost while launchpad time-stops averaged +35.8%.
    // null tp = no target; null sl = no stop; both null = pure time exit.
    const HOLD_MS = 30 * 60_000
    const grid: { tp: number | null; sl: number | null }[] = [
      { tp: 120, sl: 35 }, // shipped policy, the baseline
      { tp: 120, sl: 50 },
      { tp: 120, sl: 70 },
      { tp: 120, sl: null },
      { tp: null, sl: 35 },
      { tp: null, sl: null },
    ]

    const results = grid.map((g) => {
      let pnl = 0, wins = 0, counted = 0, tpHits = 0, slHits = 0, expiries = 0
      const byOrigin = new Map<string, { counted: number; pnl: number }>()
      for (const p of positions) {
        const ticks = sql.exec<{ price: number | null }>(
          "SELECT price FROM ticks WHERE mint = ? AND at >= ? AND at <= ? ORDER BY at ASC",
          p.mint, p.opened_at, p.opened_at + HOLD_MS,
        ).toArray()
        if (ticks.length === 0 || p.entry_price <= 0) continue
        counted += 1

        const upper = g.tp === null ? null : p.entry_price * (1 + g.tp / 100)
        const lower = g.sl === null ? null : p.entry_price * (1 - g.sl / 100)
        let retPct: number | null = null
        for (const t of ticks) {
          const px = t.price
          if (px === null) continue
          // Stop checked first: within a one-minute bar we cannot see order,
          // and assuming the favorable fill is how backtests lie.
          if (lower !== null && px <= lower) { retPct = -(g.sl as number); slHits += 1; break }
          if (upper !== null && px >= upper) { retPct = g.tp as number; tpHits += 1; break }
        }
        if (retPct === null) {
          const last = ticks[ticks.length - 1]?.price ?? p.entry_price
          retPct = ((last - p.entry_price) / p.entry_price) * 100
          expiries += 1
        }
        const trade = p.size_usd * (retPct / 100)
        pnl += trade
        if (trade > 0) wins += 1
        const o = p.origin ?? "unknown"
        const agg = byOrigin.get(o) ?? { counted: 0, pnl: 0 }
        agg.counted += 1
        agg.pnl += trade
        byOrigin.set(o, agg)
      }
      return {
        rule: `${g.tp === null ? "NOTP" : `TP${g.tp}`}/${g.sl === null ? "NOSL" : `SL${g.sl}`}`,
        counted, tpHits, slHits, expiries,
        pnlUsd: Number(pnl.toFixed(2)),
        winRate: counted > 0 ? Number((wins / counted).toFixed(3)) : null,
        byOrigin: Object.fromEntries(
          [...byOrigin].map(([o, a]) => [o, { counted: a.counted, pnlUsd: Number(a.pnl.toFixed(2)) }]),
        ),
      }
    })

    return {
      note: "Exit-rule counterfactual on real entries and our own observed ticks, fixed 30-minute horizon. Ignores exit impact, so these are upper bounds — rank rules against each other, do not read as achievable PnL.",
      positions: positions.length,
      results: results.sort((a, b) => b.pnlUsd - a.pnlUsd),
    }
  }

  /**
   * Fits the edge model on labeled decisions and REPORTS. It does not arm it.
   *
   * Training is separated from deployment on purpose: a fit that looks good is
   * still a claim, and the sizing layer must not start consuming a probability
   * because a cron happened to produce one. Arming is a human decision made
   * against the reliability table below.
   *
   * The label is "did this clear the round-trip cost hurdle", not "did it go
   * up". A token that rose 3% while costing 6% to trade is a loss, and a model
   * trained on raw direction would learn to recommend exactly those.
   */
  trainModel(): unknown {
    const COST_HURDLE_PCT = 6
    const rows = this.sql().exec<{
      at: number; features: string; forward_ret_pct: number | null
      price: number | null; origin: string | null; liq_usd: number | null
    }>(
      // Liquidity was not captured on the feature snapshot, so it is joined
      // back from our own ticks: the nearest MEASURED depth in the three
      // minutes before the decision. Probe ticks carry null liquidity and do
      // not qualify — a bounded window keeps stale depth from masquerading as
      // decision-time depth. voided = 0 is defensive: today the quarantine
      // also clears `labeled`, but this query must not depend on that.
      `SELECT d.at, d.features, d.forward_ret_pct, d.price, d.origin,
              (SELECT t.liquidity_usd FROM ticks t
                WHERE t.mint = d.mint AND t.at <= d.at AND t.at >= d.at - 180000
                  AND t.liquidity_usd IS NOT NULL
                ORDER BY t.at DESC LIMIT 1) AS liq_usd
       FROM decisions d
       WHERE d.labeled = 1 AND d.forward_ret_pct IS NOT NULL AND d.voided = 0
       ORDER BY d.at ASC`,
    ).toArray()

    const training = rows.map((r) => {
      const f = JSON.parse(r.features) as Record<string, number | null>
      return {
        at: r.at,
        features: buildFeatureVector(f, r.liq_usd, r.origin === "launchpad"),
        label: ((r.forward_ret_pct ?? 0) > COST_HURDLE_PCT ? 1 : 0) as 0 | 1,
      }
    })

    const m = fit(training)
    const positives = training.filter((t) => t.label === 1).length
    return {
      note:
        "Fitted and reported only; NOT armed. Read `auc` (0.5 = coin flip) and the reliability table before trusting it. Label = forward return cleared a 6% round-trip cost hurdle.",
      labeledRows: training.length,
      positives,
      baseRate: training.length > 0 ? Number((positives / training.length).toFixed(3)) : null,
      fit: m,
      verdict:
        m === null
          ? "insufficient data to fit"
          : m.auc < 0.55
            ? "no usable signal yet: AUC is near chance"
            : m.auc < 0.65
              ? "weak signal; keep accumulating before trusting it"
              : "signal present; inspect reliability before arming",
    }
  }

  /**
   * Paid: survivability gates for an arbitrary mint, read from chain.
   *
   * This is the sellable primitive. An agent about to touch a token wants to
   * know whether the deployer can still print supply, whether the liquidity can
   * be pulled, and whether one wallet holds enough to crater it. Those are
   * chain reads, and unknown is reported as unknown rather than rounded to
   * safe, which is the difference between this and every "safe: true" boolean
   * an aggregator will sell you.
   */
  /**
   * Gates for many mints at once, for the terminal's scan list.
   *
   * Exists because the terminal was BLINDER THAN THE ENGINE. It recomputed
   * gates locally from the DexScreener bootstrap feed against the public RPC,
   * so LP lock, holder spread and deployer history read "unknown" on screen
   * while the engine could answer all three: it holds the Helius key, the
   * creators table, and the labeled corpus that turns a deployer into a rug
   * history. The screen was reporting less than the system knew, which also
   * made the audit's point that terminal and engine could disagree.
   *
   * Cost shape drives the signature. Mint authorities batch into ONE
   * getMultipleAccounts for the whole list, and deployer history is pure SQL
   * over data we already own, so both are effectively free per extra mint.
   * Top-holder share is one RPC PER MINT, so it is resolved only for the mint
   * the operator actually has selected, passed as `detail`. Every other mint
   * reports holder spread as unknown, which is not a shortcut: we genuinely
   * did not measure it, and saying so is the whole three-state discipline.
   */
  async gatesFor(mints: string[], detail?: string): Promise<unknown> {
    configureRpc(this.env.HELIUS_API_KEY)
    const signal = AbortSignal.timeout(8_000)
    const wanted = [...new Set(mints)].slice(0, 50)
    if (wanted.length === 0) return { gates: {} }

    const facts = await fetchMintFacts(wanted, signal).catch(() => new Map<string, MintFacts>())

    let detailShare: number | undefined
    if (detail && wanted.includes(detail)) {
      const f = facts.get(detail)
      if (f) detailShare = await fetchTopHolderShare(detail, f.supply, signal).catch(() => undefined)
    }

    // Deployer history for the whole batch in one pass. A creator with no
    // prior LABELED mints stays undefined rather than becoming a flattering
    // "0 rugs" -- no history is not a clean history.
    const history = new Map<string, { prior: number; rugs: number }>()
    for (const row of this.sql().exec<{ mint: string; prior: number; rugs: number }>(
      `SELECT cr.mint AS mint,
              (SELECT COUNT(*) FROM creators c2 JOIN decisions d2 ON d2.mint = c2.mint
                WHERE c2.creator = cr.creator AND c2.mint != cr.mint AND d2.labeled = 1) AS prior,
              (SELECT COUNT(*) FROM creators c3 JOIN decisions d3 ON d3.mint = c3.mint
                WHERE c3.creator = cr.creator AND c3.mint != cr.mint AND d3.labeled = 1 AND d3.died = 1) AS rugs
         FROM creators cr WHERE cr.mint IN (${wanted.map(() => "?").join(",")})`,
      ...wanted,
    ).toArray()) {
      if (row.prior > 0) history.set(row.mint, { prior: row.prior, rugs: row.rugs })
    }

    // Returns snapshot FIELDS, not evaluated gates, on purpose. The terminal
    // keeps running the same shared/gates.ts this engine runs, so the two can
    // only ever differ by their inputs -- and this call is precisely about
    // giving the terminal the engine's inputs. Shipping a verdict instead
    // would create a second place where "what the gates mean" is decided.
    // The verdict below is a convenience for non-terminal callers; the
    // terminal ignores it and evaluates for itself.
    const out: Record<string, unknown> = {}
    for (const mint of wanted) {
      const f = facts.get(mint)
      const h = history.get(mint)
      const snapshot = {
        mintAuthority: f?.mintAuthority,
        freezeAuthority: f?.freezeAuthority,
        topHolderShare: mint === detail ? detailShare : undefined,
        deployerPriorMints: h?.prior,
        deployerPriorRugs: h?.rugs,
      }
      out[mint] = {
        snapshot,
        verdict: combineVerdict(evaluateGates({
          mint, asOf: Date.now(), launchedAt: null,
          lpLockedBps: undefined, solReserveLamports: undefined, ...snapshot,
        })),
      }
    }
    return {
      gates: out,
      resolvedAuthorities: facts.size,
      holderCheckedFor: detailShare === undefined ? null : detail,
      note: "unknown means unmeasured, never safe. Holder spread is resolved only for the selected mint; one RPC per mint is too expensive for a whole scan list.",
    }
  }

  async safetyFor(mint: string): Promise<unknown> {
    configureRpc(this.env.HELIUS_API_KEY)
    const signal = new AbortController().signal

    const facts = await fetchMintFacts([mint], signal).catch(() => new Map<string, MintFacts>())
    const f = facts.get(mint)
    const share = f ? await fetchTopHolderShare(mint, f.supply, signal) : undefined

    // Deployer history from our own corpus, when we have seen this creator.
    const creator = this.sql().exec<{ creator: string }>(
      "SELECT creator FROM creators WHERE mint = ?", mint,
    ).toArray()[0]?.creator
    let priorMints: number | undefined
    let priorRugs: number | undefined
    if (creator) {
      const h = this.sql().exec<{ prior: number; rugs: number }>(
        `SELECT COUNT(*) AS prior, SUM(CASE WHEN d.died = 1 THEN 1 ELSE 0 END) AS rugs
         FROM creators cr JOIN decisions d ON d.mint = cr.mint
         WHERE cr.creator = ? AND cr.mint != ? AND d.labeled = 1`,
        creator, mint,
      ).one()
      if (h.prior > 0) { priorMints = h.prior; priorRugs = h.rugs ?? 0 }
    }

    const snapshot = {
      mint,
      asOf: Date.now(),
      launchedAt: null,
      mintAuthority: f?.mintAuthority,
      freezeAuthority: f?.freezeAuthority,
      lpLockedBps: undefined,
      topHolderShare: share,
      solReserveLamports: undefined,
      deployerPriorMints: priorMints,
      deployerPriorRugs: priorRugs,
    }
    const gates = evaluateGates(snapshot)
    return {
      mint,
      verdict: combineVerdict(gates),
      gates: gates.map((g) => ({ id: g.id, state: g.state, detail: g.detail, severity: g.severity })),
      creator: creator ?? null,
      note: "unknown means unmeasured, never safe. A verdict of 'caution' means at least one critical gate could not be resolved.",
    }
  }

  /** Paid: outcome statistics from the labeled corpus, split by origin. */
  corpusStats(): unknown {
    const sql = this.sql()
    const byOrigin = sql.exec<{
      origin: string; labeled: number; died: number | null; avg_ret: number | null
    }>(
      `SELECT COALESCE(origin,'unknown') AS origin,
              SUM(labeled) AS labeled,
              SUM(CASE WHEN labeled=1 AND died=1 THEN 1 ELSE 0 END) AS died,
              AVG(CASE WHEN labeled=1 THEN forward_ret_pct END) AS avg_ret
       FROM decisions GROUP BY COALESCE(origin,'unknown') HAVING SUM(labeled) > 0`,
    ).toArray().map((r) => ({
      origin: r.origin,
      labeled: r.labeled ?? 0,
      deathRate: (r.labeled ?? 0) > 0 ? Number(((r.died ?? 0) / (r.labeled ?? 1)).toFixed(3)) : null,
      avgForwardRet30mPct: r.avg_ret === null ? null : Number(r.avg_ret.toFixed(2)),
    }))
    const total = sql.exec<{ n: number }>("SELECT SUM(labeled) AS n FROM decisions").one().n ?? 0
    return {
      horizonMinutes: 30,
      totalLabeled: total,
      byOrigin,
      method:
        "Every eligible launch is snapshotted at decision time and scored 30 minutes later from our own observations. Refused launches are followed identically to entered ones, so the sample is not survivorship-biased.",
    }
  }

  /**
   * Read-only SQL over the corpus. The agent's research surface.
   *
   * This is the notebook, minus the kernel. The corpus is already relational --
   * decisions joined to ticks joined to creators is exactly the question
   * "which origin, at which age, with which flow, survived" -- so SQL is the
   * native language for it. A Python sandbox would add container security,
   * data egress and a whole runtime to answer the same questions less directly.
   *
   * SAFETY IS STRUCTURAL, not advisory:
   *  - a single statement only, so nothing can be smuggled after a semicolon
   *  - it must begin with SELECT or WITH; every mutating verb is rejected
   *    outright rather than filtered, because a blocklist is a race against
   *    whoever is more creative
   *  - the DO's sql API has no filesystem or network reach, so the worst case
   *    is a slow read of our own data
   *  - rows are capped, so a cartesian join returns a truncated answer instead
   *    of exhausting memory
   */
  researchQuery(sql: string): unknown {
    const trimmed = sql.trim().replace(/;\s*$/, "")

    if (trimmed.includes(";")) {
      return { error: "one statement only", detail: "semicolons are not permitted" }
    }
    if (!/^(select|with)\b/i.test(trimmed)) {
      return { error: "read only", detail: "queries must begin with SELECT or WITH" }
    }
    // Belt and braces behind the SELECT-only rule: CTEs can carry writes in
    // some dialects, and the cost of checking is one regex.
    if (/\b(insert|update|delete|drop|alter|create|replace|attach|pragma|vacuum)\b/i.test(trimmed)) {
      return { error: "read only", detail: "mutating statements are rejected" }
    }

    const MAX_ROWS = 500
    try {
      const rows = this.sql().exec(trimmed).toArray()
      return {
        rowCount: rows.length,
        truncated: rows.length > MAX_ROWS,
        rows: rows.slice(0, MAX_ROWS),
        schema: {
          decisions:
            "mint, at, symbol, price, origin, verdict, features (JSON), eligible, skip_reason, entered, entry_impact_pct, labeled, forward_ret_pct, died, labeled_at",
          ticks: "mint, at, price, liquidity_usd, buys_24h, sells_24h, origin",
          positions:
            "id, mint, symbol, entry_price, size_sol, size_usd, token_amount, opened_at, closed_at, exit_price, exit_reason, pnl_usd, pnl_pct, policy_hash, verdict_entry, priced_by, exit_pricing",
          creators: "mint, creator, first_seen",
        },
      }
    } catch (e) {
      // Return the database's own message: an agent iterating on a query needs
      // to know it misspelled a column, not that "the query failed".
      return { error: "query failed", detail: e instanceof Error ? e.message : String(e) }
    }
  }

  setKill(on: boolean): void {
    this.metaSet("kill", on ? "1" : "0")
    this.event("kill", { on })
    capture(this.env, this.ctx, "kill_switch", { on, policy_hash: this.metaGet("policy_hash") })
    // Every flip emails, because the kill switch is an authenticated action:
    // if this email arrives and Michael did not flip it, the admin token is
    // compromised and that is worth knowing within a minute, not at the next
    // manual check of the book.
    this.queueAlert(
      `kill:${Date.now()}`,
      `CroweTrade: kill switch ${on ? "ON" : "OFF"}`,
      [
        on
          ? "The kill switch is ON. New entries are stopped; exits keep managing open positions."
          : "The kill switch is OFF. The engine will resume entering on the next tick.",
        "",
        "If you did not do this, the admin token is compromised — rotate it now.",
        "",
        "https://crowetrade-engine.yellow-block-3adc.workers.dev/api/positions",
      ].join("\n"),
    )
  }

  /** Veto: allowed only inside the policy window; executes next tick. */
  requestVeto(id: string): { ok: boolean; reason?: string } {
    const rows = this.sql().exec<PositionRow>(
      "SELECT * FROM positions WHERE id = ? AND closed_at IS NULL", id,
    ).toArray()
    const p = rows[0]
    if (!p) return { ok: false, reason: "no such open position" }
    const ageMin = (Date.now() - p.opened_at) / 60_000
    if (ageMin > PAPER_POLICY.exit.vetoWindowMinutes) {
      return { ok: false, reason: `veto window (${PAPER_POLICY.exit.vetoWindowMinutes}m) has passed` }
    }
    this.sql().exec("UPDATE positions SET veto_requested = 1 WHERE id = ?", id)
    this.event("veto_requested", { id })
    return { ok: true }
  }
}

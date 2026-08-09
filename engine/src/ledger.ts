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
import { PAPER_POLICY, policyHash } from "../../shared/policy.js"
import {
  decideEntries,
  decideExits,
  type OpenPosition,
  type Trajectory,
} from "./strategy.js"
import { computeFeatures } from "../../shared/features.js"
import { fit } from "../../shared/model.js"
import { quoteBuy, quoteSell, LAMPORTS_PER_SOL } from "./execution/jupiter.js"
import { dryRunSwap } from "./execution/swap.js"
import { capture } from "./posthog.js"

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
    if (q && solUsd > 0) {
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
      capture(this.env, this.ctx, "breaker_tripped", { kind: "loss_velocity", window_loss_usd: recentLoss + pnlUsd })
    }

    if (reason === "stop-loss" || reason === "safety-exit") {
      const consec = Number(this.metaGet("breaker_consec") ?? "0") + 1
      if (consec >= PAPER_POLICY.breaker.consecutiveStopLimit) {
        const until = now + PAPER_POLICY.breaker.cooldownMinutes * 60_000
        this.metaSet("breaker_until", String(until))
        this.metaSet("breaker_consec", "0")
        this.event("breaker", { tripped: true, until, afterConsecutiveStops: consec })
        capture(this.env, this.ctx, "breaker_tripped", { after_stops: consec, cooldown_minutes: PAPER_POLICY.breaker.cooldownMinutes })
      } else {
        this.metaSet("breaker_consec", String(consec))
      }
    } else if (reason === "take-profit") {
      this.metaSet("breaker_consec", "0")
    }
    this.sql().exec(
      `UPDATE positions SET closed_at = ?, exit_price = ?, exit_reason = ?, pnl_usd = ?, pnl_pct = ?,
              exit_pricing = ?
       WHERE id = ?`,
      now, effective, reason, pnlUsd, pnlPct, pricing, p.id,
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
      const scan = await fetchCandidates(signal)
      candidates = scan.candidates
      solUsd = scan.solUsd
      this.metaSet("sol_usd", String(solUsd))
    } catch (e) {
      this.event("scan_error", { message: e instanceof Error ? e.message : String(e) })
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
        (c.origin === "boost" || c.origin === "both") && policy.entry.excludeBoosted ? "boosted"
        : ageMin === null || ageMin < policy.entry.minTokenAgeMinutes ? "too-new"
        : ageMin > policy.entry.maxTokenAgeMinutes ? "too-old"
        : c.liquidityUsd === null || c.liquidityUsd < policy.entry.minLiquidityUsd ? "thin"
        : c.changeH1 === null || c.changeH1 > policy.entry.maxChangeH1Pct ? "parabolic"
        : verdict === "blocked" || verdict === "insufficient-data" ? `verdict-${verdict}`
        : null

      this.sql().exec(
        `INSERT INTO decisions (mint, at, symbol, price, origin, verdict, features, eligible, skip_reason)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        c.mint, now, c.symbol, c.priceUsd, c.origin, verdict,
        JSON.stringify(feats), skipReason === null ? 1 : 0, skipReason,
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
    if (!killed && !breakerOpen) {
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

      const entries = decideEntries(candidates, this.openPositions(), spentToday, solUsd, policy, now, trajectories)
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
        const sizeUsd = (e.sizeSol + feeSol) * solUsd
        const tokenAmount = Number(q.outAmount) / 10 ** decimals
        if (tokenAmount <= 0) continue
        const entryPrice = sizeUsd / tokenAmount
        const id = crypto.randomUUID()
        this.sql().exec(
          `INSERT INTO positions
           (id, mint, symbol, entry_price, size_sol, size_usd, token_amount, opened_at, policy_hash, verdict_entry, priced_by)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'quote')`,
          id, c.mint, c.symbol, entryPrice, e.sizeSol, sizeUsd, tokenAmount, now, hash, e.verdict,
        )
        this.sql().exec(
          "UPDATE decisions SET entered = 1, entry_impact_pct = ? WHERE mint = ? AND labeled = 0",
          impactPct, c.mint,
        )
        this.event("entry", {
          id, symbol: c.symbol, mint: c.mint, sizeSol: e.sizeSol, entryPrice,
          verdict: e.verdict, impact: q.priceImpactPct, route: q.route,
        })
        capture(this.env, this.ctx, "paper_entry", {
          symbol: c.symbol, mint: c.mint, size_sol: e.sizeSol, entry_price: entryPrice,
          verdict: e.verdict, liquidity_usd: c.liquidityUsd, token_age_minutes: c.createdAt ? (now - c.createdAt) / 60_000 : null,
          policy_hash: hash,
          entry_price_impact_pct: q.priceImpactPct, entry_route: q.route,
          sim_units_consumed: sim.unitsConsumed, priority_fee_lamports: sim.priorityFeeLamports,
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
       FROM decisions`,
    ).one()

    // Per-origin breakout. Adding a second discovery source is only an
    // experiment if the result can be read per source; a pooled average would
    // hide a good universe inside a bad one and vice versa. This is the readout
    // that says whether the launchpad move worked.
    const byOrigin = sql.exec<{
      origin: string; n: number; labeled: number; died: number | null
      entered_n: number; entered_ret: number | null; refused_ret: number | null
    }>(
      `SELECT COALESCE(origin, 'unknown') AS origin,
              COUNT(*) AS n,
              SUM(labeled) AS labeled,
              SUM(CASE WHEN labeled = 1 AND died = 1 THEN 1 ELSE 0 END) AS died,
              SUM(CASE WHEN entered = 1 THEN 1 ELSE 0 END) AS entered_n,
              AVG(CASE WHEN labeled = 1 AND entered = 1 THEN forward_ret_pct END) AS entered_ret,
              AVG(CASE WHEN labeled = 1 AND entered = 0 AND eligible = 1 THEN forward_ret_pct END) AS refused_ret
       FROM decisions GROUP BY COALESCE(origin, 'unknown') ORDER BY n DESC`,
    ).toArray().map((r) => ({
      origin: r.origin,
      decisions: r.n,
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
       FROM decisions GROUP BY COALESCE(skip_reason, 'eligible') ORDER BY n DESC`,
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
  exitSweep(): unknown {
    const sql = this.sql()
    const positions = sql.exec<{
      mint: string; entry_price: number; size_usd: number
      opened_at: number; closed_at: number; token_amount: number
    }>(
      `SELECT mint, entry_price, size_usd, opened_at, closed_at, token_amount
       FROM positions WHERE closed_at IS NOT NULL AND priced_by = 'quote'`,
    ).toArray()

    const grid = [
      { tp: 40, sl: 25 }, { tp: 40, sl: 35 },
      { tp: 60, sl: 25 }, { tp: 60, sl: 35 },
      { tp: 80, sl: 35 }, { tp: 120, sl: 35 },
    ]

    const results = grid.map((g) => {
      let pnl = 0, wins = 0, counted = 0, tpHits = 0, slHits = 0, expiries = 0
      for (const p of positions) {
        const ticks = sql.exec<{ price: number | null }>(
          "SELECT price FROM ticks WHERE mint = ? AND at >= ? AND at <= ? ORDER BY at ASC",
          p.mint, p.opened_at, p.closed_at,
        ).toArray()
        if (ticks.length === 0 || p.entry_price <= 0) continue
        counted += 1

        const upper = p.entry_price * (1 + g.tp / 100)
        const lower = p.entry_price * (1 - g.sl / 100)
        let retPct: number | null = null
        for (const t of ticks) {
          const px = t.price
          if (px === null) continue
          // Stop checked first: within a one-minute bar we cannot see order,
          // and assuming the favorable fill is how backtests lie.
          if (px <= lower) { retPct = -g.sl; slHits += 1; break }
          if (px >= upper) { retPct = g.tp; tpHits += 1; break }
        }
        if (retPct === null) {
          const last = ticks[ticks.length - 1]?.price ?? p.entry_price
          retPct = ((last - p.entry_price) / p.entry_price) * 100
          expiries += 1
        }
        const trade = p.size_usd * (retPct / 100)
        pnl += trade
        if (trade > 0) wins += 1
      }
      return {
        rule: `TP${g.tp}/SL${g.sl}`,
        counted, tpHits, slHits, expiries,
        pnlUsd: Number(pnl.toFixed(2)),
        winRate: counted > 0 ? Number((wins / counted).toFixed(3)) : null,
      }
    })

    return {
      note: "Exit-rule counterfactual on real entries and our own observed ticks. Ignores exit impact, so these are upper bounds — rank rules against each other, do not read as achievable PnL.",
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
      price: number | null; origin: string | null
    }>(
      `SELECT d.at, d.features, d.forward_ret_pct, d.price, d.origin
       FROM decisions d
       WHERE d.labeled = 1 AND d.forward_ret_pct IS NOT NULL
       ORDER BY d.at ASC`,
    ).toArray()

    const training = rows.map((r) => {
      const f = JSON.parse(r.features) as Record<string, number | null>
      const liq = 0 // liquidity is not on the feature snapshot; see note below
      return {
        at: r.at,
        features: [
          f["netFlowShare"] ?? 0,
          f["flowAccel"] ?? 0,
          f["priceProgressPct"] ?? 0,
          f["liqTrendPct"] ?? 0,
          f["ticks"] ?? 0,
          liq,
        ],
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
    capture(this.env, this.ctx, "kill_switch", { on })
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

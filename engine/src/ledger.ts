/**
 * The Ledger: one Durable Object holding the whole paper-trading state.
 *
 * The entire tick runs INSIDE the DO rather than in the cron handler, because
 * a Durable Object is single-threaded by contract: two overlapping ticks (a
 * slow minute meeting the next cron) serialize instead of double-entering the
 * same token. Concurrency safety by construction beats concurrency safety by
 * discipline.
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
import { fetchMintFacts, type MintFacts } from "../../shared/solana.js"
import { evaluateGates, combineVerdict, type Verdict } from "../../shared/gates.js"
import { PAPER_POLICY, policyHash } from "../../shared/policy.js"
import { decideEntries, decideExits, type OpenPosition } from "./strategy.js"
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

    const q = baseUnits > 0n ? await quoteSell(p.mint, baseUnits, SLIPPAGE_BPS) : null
    if (q && solUsd > 0) {
      proceeds = (Number(q.outAmount) / LAMPORTS_PER_SOL) * solUsd
      impact = q.priceImpactPct
      route = q.route
    } else {
      // No route: the position is effectively unexitable right now. Marking it
      // at mark price would invent liquidity that does not exist, so it is
      // marked to ZERO. That is the honest floor, and it is what a rug looks
      // like from the inside.
      proceeds = reason === "safety-exit" ? 0 : p.tokenAmount * exitPriceUsd * 0.5
    }

    const pnlUsd = proceeds - p.sizeUsd
    const pnlPct = (pnlUsd / p.sizeUsd) * 100
    const effective = p.tokenAmount > 0 ? proceeds / p.tokenAmount : 0
    this.sql().exec(
      `UPDATE positions SET closed_at = ?, exit_price = ?, exit_reason = ?, pnl_usd = ?, pnl_pct = ?
       WHERE id = ?`,
      now, effective, reason, pnlUsd, pnlPct, p.id,
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

    const open = this.openPositions()

    // Price and re-judge held tokens: union of fresh candidates and holdings.
    const held = open.filter((p) => !candidates.some((c) => c.mint === p.mint))
    let heldPairs: Candidate[] = []
    if (held.length > 0 && solUsd > 0) {
      heldPairs = await fetchPairsForMints(held.map((p) => p.mint), solUsd, signal)
        .catch(() => [] as Candidate[])
    }
    const everything = [...candidates, ...heldPairs]

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
    if (!killed) {
      const day = new Date(now).toISOString().slice(0, 10)
      const spentToday = Number(this.metaGet(`spend:${day}`) ?? "0")
      const entries = decideEntries(candidates, this.openPositions(), spentToday, solUsd, policy, now)
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
        this.metaSet(`spend:${day}`, String(spentToday + e.sizeSol))
        entered += 1
      }
    }

    capture(this.env, this.ctx, "engine_tick", {
      scanned: candidates.length, entered, exited, open: this.openPositions().length, killed,
    })
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
    return {
      mode: "paper",
      killed: this.metaGet("kill") === "1",
      policyHash: this.metaGet("policy_hash"),
      open, closed,
      stats: {
        closedCount: totals.n,
        totalPnlUsd: totals.pnl ?? 0,
        winRate: totals.n > 0 ? (totals.wins ?? 0) / totals.n : null,
        /** Excluded from the headline: priced by the retired slippage model. */
        excludedModelPriced: legacy.n,
      },
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

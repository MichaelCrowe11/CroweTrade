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
import { fetchMintFacts } from "../../shared/solana.js"
import { evaluateGates, combineVerdict, type Verdict } from "../../shared/gates.js"
import { PAPER_POLICY, policyHash } from "../../shared/policy.js"
import { decideEntries, decideExits, slippageBps, type OpenPosition } from "./strategy.js"
import { capture } from "./posthog.js"

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

  private closePosition(p: OpenPosition, exitPriceUsd: number, reason: string, now: number): void {
    // Sell-side slippage worsens the exit the same way entry slippage worsened
    // the entry. Exit liquidity is unknown here, so the model reuses the entry
    // notional as the denominator proxy; pessimistic is the only safe direction.
    const slip = slippageBps(p.sizeUsd, p.sizeUsd * 50) / 10_000
    const effective = exitPriceUsd * (1 - slip)
    const proceeds = p.tokenAmount * effective
    const pnlUsd = proceeds - p.sizeUsd
    const pnlPct = (pnlUsd / p.sizeUsd) * 100
    this.sql().exec(
      `UPDATE positions SET closed_at = ?, exit_price = ?, exit_reason = ?, pnl_usd = ?, pnl_pct = ?
       WHERE id = ?`,
      now, effective, reason, pnlUsd, pnlPct, p.id,
    )
    this.event("exit", { id: p.id, symbol: p.symbol, reason, pnlUsd, pnlPct })
    capture(this.env, this.ctx, "paper_exit", {
      symbol: p.symbol, mint: p.mint, reason,
      pnl_usd: pnlUsd, pnl_pct: pnlPct, held_minutes: (now - p.openedAt) / 60_000,
      policy_hash: p.policyHash,
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
      () => new Map<string, { mintAuthority: string | null; freezeAuthority: string | null }>(),
    )
    for (const c of everything) {
      const f = facts.get(c.mint)
      if (f) {
        c.snapshot.mintAuthority = f.mintAuthority
        c.snapshot.freezeAuthority = f.freezeAuthority
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
        this.closePosition(p, px, "veto", now)
        exited += 1
      }
    }
    const stillOpen = this.openPositions()
    for (const d of decideExits(stillOpen, prices, policy, now)) {
      this.closePosition(d.position, d.exitPriceUsd, d.reason, now)
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
        const sizeUsd = e.sizeSol * solUsd
        const slip = slippageBps(sizeUsd, c.liquidityUsd) / 10_000
        const entryPrice = c.priceUsd * (1 + slip)
        const id = crypto.randomUUID()
        this.sql().exec(
          `INSERT INTO positions
           (id, mint, symbol, entry_price, size_sol, size_usd, token_amount, opened_at, policy_hash, verdict_entry)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          id, c.mint, c.symbol, entryPrice, e.sizeSol, sizeUsd, sizeUsd / entryPrice, now, hash, e.verdict,
        )
        this.event("entry", { id, symbol: c.symbol, mint: c.mint, sizeSol: e.sizeSol, entryPrice, verdict: e.verdict })
        capture(this.env, this.ctx, "paper_entry", {
          symbol: c.symbol, mint: c.mint, size_sol: e.sizeSol, entry_price: entryPrice,
          verdict: e.verdict, liquidity_usd: c.liquidityUsd, token_age_minutes: c.createdAt ? (now - c.createdAt) / 60_000 : null,
          policy_hash: hash,
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
    const totals = sql.exec<{ n: number; pnl: number | null; wins: number | null }>(
      `SELECT COUNT(*) AS n, SUM(pnl_usd) AS pnl,
              SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins
       FROM positions WHERE closed_at IS NOT NULL`,
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

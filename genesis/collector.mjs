// CroweTrade Genesis collector. Records every pump.fun creation and the first
// fifteen minutes of every trade on it, at transaction resolution, into a local
// SQLite file. Two sockets: PumpPortal's free creation stream (metadata, dev
// buy) and the public Solana RPC log stream on the pump.fun program (trades,
// decoded from the anchor TradeEvent). Nothing here trades. It watches.
import { DatabaseSync } from "node:sqlite"
import { mkdirSync, writeFileSync } from "node:fs"
import { homedir } from "node:os"

const DIR = `${homedir()}/crowetrade-genesis`
mkdirSync(DIR, { recursive: true })
const db = new DatabaseSync(`${DIR}/genesis.db`)
db.exec(`
  PRAGMA journal_mode = WAL;
  PRAGMA synchronous = NORMAL;
  CREATE TABLE IF NOT EXISTS tokens (
    mint TEXT PRIMARY KEY, created_at INTEGER NOT NULL, sig TEXT, creator TEXT, name TEXT, symbol TEXT, uri TEXT,
    dev_sol REAL, dev_tokens REAL, vsol0 REAL, vtok0 REAL, mcap_sol0 REAL, source TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_tokens_created ON tokens (created_at);
  CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY, mint TEXT NOT NULL, ts INTEGER NOT NULL, block_ts INTEGER, sig TEXT, is_buy INTEGER NOT NULL,
    sol REAL NOT NULL, tokens REAL NOT NULL, user TEXT, vsol REAL NOT NULL, vtok REAL NOT NULL, real_sol REAL
  );
  CREATE INDEX IF NOT EXISTS idx_trades_mint_ts ON trades (mint, ts);
  CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades (ts);
  CREATE TABLE IF NOT EXISTS heartbeat (key TEXT PRIMARY KEY, value TEXT NOT NULL);
`)
const insToken = db.prepare(`INSERT OR IGNORE INTO tokens (mint, created_at, sig, creator, name, symbol, uri, dev_sol, dev_tokens, vsol0, vtok0, mcap_sol0, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
const insTrade = db.prepare(`INSERT INTO trades (mint, ts, block_ts, sig, is_buy, sol, tokens, user, vsol, vtok, real_sol) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
const setHb = db.prepare(`INSERT OR REPLACE INTO heartbeat (key, value) VALUES (?, ?)`)

const TRACK_MS = 15 * 60_000
const RETENTION_MS = 7 * 24 * 3_600_000
const PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
const TRADE_DISC = Buffer.from([189, 219, 127, 211, 78, 230, 97, 238])
const RPC_WS = (process.env.GENESIS_RPC_WS || "wss://api.mainnet-beta.solana.com,wss://solana-rpc.publicnode.com").split(",")
const tracked = new Map() // mint -> created_at
const stats = { creates: 0, trades: 0, logMsgs: 0, dropped: 0, reconnects: 0, startedAt: Date.now() }

function b58(buf) {
  const A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  let n = BigInt("0x" + buf.toString("hex")); let s = ""
  while (n > 0n) { s = A[Number(n % 58n)] + s; n /= 58n }
  for (const b of buf) { if (b === 0) s = "1" + s; else break }
  return s
}
function decodeTrade(b) {
  if (b.length < 8 + 32 + 8 + 8 + 1 + 32 + 8 + 8 + 8 + 8) return null
  let o = 8
  const mint = b58(b.subarray(o, o + 32)); o += 32
  const solAmount = Number(b.readBigUInt64LE(o)) / 1e9; o += 8
  const tokenAmount = Number(b.readBigUInt64LE(o)) / 1e6; o += 8
  const isBuy = b[o] === 1; o += 1
  const user = b58(b.subarray(o, o + 32)); o += 32
  const blockTs = Number(b.readBigInt64LE(o)); o += 8
  const vSol = Number(b.readBigUInt64LE(o)) / 1e9; o += 8
  const vTok = Number(b.readBigUInt64LE(o)) / 1e6; o += 8
  const realSol = Number(b.readBigUInt64LE(o)) / 1e9
  return { mint, solAmount, tokenAmount, isBuy, user, blockTs, vSol, vTok, realSol }
}
function log(msg) { process.stdout.write(`${new Date().toISOString()} ${msg}\n`) }

// ── Creation stream (PumpPortal, free tier) ─────────────────────────────────
function connectCreates() {
  const ws = new WebSocket("wss://pumpportal.fun/api/data")
  ws.onopen = () => { ws.send(JSON.stringify({ method: "subscribeNewToken" })); log("creates: connected") }
  ws.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data) } catch { return }
    if (m.txType !== "create" || !m.mint) return
    const now = Date.now()
    insToken.run(m.mint, now, m.signature ?? null, m.traderPublicKey ?? null, m.name ?? null, m.symbol ?? null, m.uri ?? null,
      m.solAmount ?? null, m.initialBuy ?? null, m.vSolInBondingCurve ?? null, m.vTokensInBondingCurve ?? null, m.marketCapSol ?? null, "pumpportal")
    tracked.set(m.mint, now)
    stats.creates += 1
  }
  ws.onclose = () => { stats.reconnects += 1; log("creates: closed, reconnecting in 3s"); setTimeout(connectCreates, 3000) }
  ws.onerror = (e) => { log(`creates: error ${e.message ?? e}`) }
}

// ── Trade stream (Solana RPC logs on the pump.fun program) ──────────────────
let rpcIdx = 0
function connectTrades() {
  const url = RPC_WS[rpcIdx % RPC_WS.length]
  const ws = new WebSocket(url)
  let alive = true
  ws.onopen = () => {
    ws.send(JSON.stringify({ jsonrpc: "2.0", id: 1, method: "logsSubscribe", params: [{ mentions: [PUMP] }, { commitment: "processed" }] }))
    log(`trades: connected ${url}`)
  }
  ws.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data) } catch { return }
    if (m.error) { log(`trades: rpc error ${JSON.stringify(m.error)}`); rpcIdx += 1; ws.close(); return }
    const v = m.params?.result?.value; if (!v) return
    stats.logMsgs += 1
    const now = Date.now()
    for (const line of v.logs || []) {
      if (!line.startsWith("Program data: ")) continue
      const b = Buffer.from(line.slice(14), "base64")
      if (!b.subarray(0, 8).equals(TRADE_DISC)) continue
      const t = decodeTrade(b); if (!t) continue
      let created = tracked.get(t.mint)
      if (created === undefined) {
        // A fresh curve seen first on chain (PumpPortal missed or lagged it):
        // a bonding curve with under 1 SOL of real reserves is minutes old.
        if (t.realSol < 1 && t.vSol < 32) {
          insToken.run(t.mint, now, v.signature ?? null, t.isBuy ? t.user : null, null, null, null, t.isBuy ? t.solAmount : null, t.isBuy ? t.tokenAmount : null, t.vSol, t.vTok, null, "chain")
          tracked.set(t.mint, now); created = now; stats.creates += 1
        } else { stats.dropped += 1; continue }
      }
      if (now - created > TRACK_MS) { tracked.delete(t.mint); continue }
      insTrade.run(t.mint, now, t.blockTs, v.signature ?? null, t.isBuy ? 1 : 0, t.solAmount, t.tokenAmount, t.user, t.vSol, t.vTok, t.realSol)
      stats.trades += 1
    }
  }
  ws.onclose = () => { if (!alive) return; alive = false; stats.reconnects += 1; rpcIdx += 1; log("trades: closed, reconnecting in 3s"); setTimeout(connectTrades, 3000) }
  ws.onerror = (e) => { log(`trades: error ${e.message ?? e}`) }
}

// ── Housekeeping ─────────────────────────────────────────────────────────────
setInterval(() => {
  const now = Date.now()
  for (const [mint, at] of tracked) if (now - at > TRACK_MS) tracked.delete(mint)
  setHb.run("at", String(now)); setHb.run("stats", JSON.stringify({ ...stats, tracked: tracked.size }))
  writeFileSync(`${DIR}/heartbeat.json`, JSON.stringify({ at: now, ...stats, tracked: tracked.size }))
  log(`hb creates=${stats.creates} trades=${stats.trades} tracked=${tracked.size} logMsgs=${stats.logMsgs} reconnects=${stats.reconnects}`)
}, 60_000)
setInterval(() => {
  db.prepare("DELETE FROM trades WHERE ts < ?").run(Date.now() - RETENTION_MS)
  db.prepare("DELETE FROM tokens WHERE created_at < ?").run(Date.now() - RETENTION_MS)
}, 6 * 3_600_000)

connectCreates()
connectTrades()
log("genesis collector up")

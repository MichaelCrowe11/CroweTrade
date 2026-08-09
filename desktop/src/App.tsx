import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { fetchCandidates, type Candidate } from "./feed/dexscreener.js"
import { fetchMintFacts } from "./feed/solana.js"
import { evaluateGates, combineVerdict, type Verdict } from "./safety/gates.js"
import { Annunciator } from "./components/Annunciator.js"
import { PriceChart } from "./components/PriceChart.js"
import { age, usd, shortMint } from "./components/format.js"
import { Spark } from "./components/Spark.js"
import { Rail } from "./shell/Rail.js"
import { Workspace } from "./shell/Workspace.js"
import { AnalystPanel } from "./shell/AnalystPanel.js"
import type { Panel } from "./shell/panels.js"

const REFRESH_MS = 20_000
const ENGINE = "https://crowetrade-engine.yellow-block-3adc.workers.dev"

interface EnginePosition {
  id: string
  symbol: string
  mint: string
  entry_price: number
  size_sol: number
  opened_at: number
  verdict_entry: string
}

interface EngineEvent {
  at: number
  kind: string
  data: string
}

interface EngineSummary {
  killed: boolean
  open: EnginePosition[]
  stats: { closedCount: number; totalPnlUsd: number; winRate: number | null }
  budget?: { spentTodaySol: number; dailyCapSol: number; canEnter: boolean }
  events?: EngineEvent[]
}

/** One line per engine decision: what, who, and why when it declined. */
function describeEvent(e: EngineEvent): { label: string; detail: string; kind: string } {
  try {
    const d = JSON.parse(e.data) as Record<string, unknown>
    const sym = typeof d["symbol"] === "string" ? d["symbol"] : ""
    switch (e.kind) {
      case "entry":
        return { kind: "entry", label: `ENTER ${sym}`, detail: `${String(d["verdict"])}` }
      case "exit": {
        const pnl = typeof d["pnlUsd"] === "number" ? d["pnlUsd"] : 0
        return {
          kind: pnl >= 0 ? "exit-win" : "exit-loss",
          label: `EXIT ${sym}`,
          detail: `${String(d["reason"])} ${pnl >= 0 ? "+" : "-"}$${Math.abs(pnl).toFixed(2)}`,
        }
      }
      case "entry_skipped":
        return { kind: "skip", label: `SKIP ${sym}`, detail: String(d["reason"] ?? "") }
      case "kill":
        return { kind: "skip", label: "KILL", detail: d["on"] ? "engaged" : "released" }
      default:
        return { kind: "skip", label: e.kind.toUpperCase(), detail: "" }
    }
  } catch {
    return { kind: "skip", label: e.kind, detail: "" }
  }
}

/** Verdict, or null when the combination policy has not been written yet. */
type VerdictState = { kind: "ok"; verdict: Verdict } | { kind: "unset" }

/**
 * combineVerdict is deliberately unimplemented, so calling it throws.
 *
 * Catching here rather than letting it crash is the point: the panel still
 * renders every individual gate, which is the factual layer, and only the
 * judgment layer on top is missing. The operator sees exactly what the system
 * knows and exactly what it has not been told how to decide.
 */
function safeVerdict(gates: ReturnType<typeof evaluateGates>): VerdictState {
  try {
    return { kind: "ok", verdict: combineVerdict(gates) }
  } catch {
    return { kind: "unset" }
  }
}

const VERDICT_NOTE: Record<Verdict, string> = {
  clear: "no critical gate failed",
  caution: "tradeable at reduced size",
  blocked: "a critical gate failed",
  "insufficient-data": "too little is known to judge",
}

export default function App() {
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [solUsd, setSolUsd] = useState<number | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [now, setNow] = useState(() => Date.now())
  const [engine, setEngine] = useState<EngineSummary | null>(null)
  // Rolling price trace per mint, built from successive scans. The engine keeps
  // its own tick history, but the terminal polls on its own schedule and this
  // costs no extra request.
  const [trace, setTrace] = useState<Map<string, number[]>>(new Map())

  // One controller per mount, aborted on unmount, so an in-flight request from
  // a closing window cannot resolve into a setState on a dead component.
  const abortRef = useRef<AbortController | null>(null)

  const load = useCallback(async () => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const scan = await fetchCandidates(controller.signal)

      // Chain reads are additive: show the aggregator rows immediately, then
      // resolve the authority gates from chain as they land. Blocking the list
      // on a rate-limited public RPC would make the app feel broken when it is
      // merely uninformed.
      setCandidates(scan.candidates)
      setSolUsd(scan.solUsd)
      setTrace((prev) => {
        const next = new Map(prev)
        for (const c of scan.candidates) {
          if (c.priceUsd === null) continue
          // 40 points is about 13 minutes at the current refresh, which is the
          // window where these tokens actually resolve.
          next.set(c.mint, [...(next.get(c.mint) ?? []), c.priceUsd].slice(-40))
        }
        return next
      })
      setError(null)
      setSelected((cur) => cur ?? scan.candidates[0]?.mint ?? null)

      const facts = await fetchMintFacts(
        scan.candidates.map((c) => c.mint),
        controller.signal,
      )
      if (controller.signal.aborted || facts.size === 0) return

      setCandidates((rows) =>
        rows.map((row) => {
          const f = facts.get(row.mint)
          if (!f) return row
          return {
            ...row,
            snapshot: {
              ...row.snapshot,
              mintAuthority: f.mintAuthority,
              freezeAuthority: f.freezeAuthority,
            },
          }
        }),
      )
    } catch (e) {
      if (controller.signal.aborted) return
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (!controller.signal.aborted) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
    const feed = setInterval(() => void load(), REFRESH_MS)
    // The cloud engine's paper book, read-only. Its absence must cost nothing:
    // the terminal observes markets fine while the engine is unreachable.
    const pullEngine = () => {
      fetch(`${ENGINE}/api/positions`)
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => setEngine(d as EngineSummary | null))
        .catch(() => setEngine(null))
    }
    pullEngine()
    const engineTimer = setInterval(pullEngine, 30_000)
    // Ages are relative to wall clock, so they need their own tick or every row
    // would read the same age until the next feed poll landed.
    const clock = setInterval(() => setNow(Date.now()), 1_000)
    return () => {
      clearInterval(feed)
      clearInterval(clock)
      clearInterval(engineTimer)
      abortRef.current?.abort()
    }
  }, [load])

  const active = useMemo(
    () => candidates.find((c) => c.mint === selected) ?? null,
    [candidates, selected],
  )

  const gates = useMemo(() => (active ? evaluateGates(active.snapshot) : []), [active])
  const verdict = useMemo(() => (active ? safeVerdict(gates) : null), [active, gates])

  const flagFor = useCallback((c: Candidate): Verdict => {
    const v = safeVerdict(evaluateGates(c.snapshot))
    return v.kind === "ok" ? v.verdict : "insufficient-data"
  }, [])

  // Panel renderers. Each is a self-contained readout so the workspace can
  // place it anywhere without any of them knowing where they sit.

  const renderScan = () => (
    <>
      {error && <p className="empty">{error}</p>}
      {!error && candidates.length === 0 && <p className="empty">No candidates.</p>}
      {candidates.map((c) => {
        const chUp = (c.changeH1 ?? 0) >= 0
        return (
          <button
            key={c.mint}
            type="button"
            className="scan__row"
            aria-selected={c.mint === selected}
            onClick={() => setSelected(c.mint)}
          >
            <span className="scan__symbol">{c.symbol}</span>
            {c.changeH1 !== null && (
              <span className={`scan__change mono ${chUp ? "scan__change--up" : "scan__change--down"}`}>
                {chUp ? "+" : ""}
                {c.changeH1.toFixed(1)}%
              </span>
            )}
            <span className={`scan__flag scan__flag--${flagFor(c)}`} aria-hidden="true" />
            <Spark points={trace.get(c.mint) ?? []} up={chUp} />
            <span className="scan__age mono">
              {age(c.createdAt, now)} / {usd(c.liquidityUsd)}
              {c.origin === "launchpad" && <span className="scan__origin"> LP</span>}
              {(c.origin === "boost" || c.origin === "both") && (
                <span className="scan__origin scan__origin--paid"> AD</span>
              )}
            </span>
          </button>
        )
      })}
    </>
  )

  const renderPrimary = () => {
    if (!active) return <p className="empty">Select a candidate.</p>
    return (
      <div className="primary">
        <div className="primary__identity">
          <div className="primary__titlerow">
            <h1 className="primary__symbol">{active.symbol}</h1>
            <span className="primary__venue mono">{active.dex}</span>
          </div>
          <span className="primary__name">
            {active.name}
            <span className="primary__mint mono"> / {shortMint(active.mint)}</span>
          </span>
        </div>
        <PriceChart pool={active.pool} mint={active.mint} />
        <Stat k="Price" v={usd(active.priceUsd)} />
        <Stat k="Liquidity" v={usd(active.liquidityUsd)} />
        <Pressure buys={active.buys24h} sells={active.sells24h} />
      </div>
    )
  }

  const renderGates = () => {
    if (!active) return <p className="empty">Select a candidate.</p>
    return (
      <div className="primary">
        {verdict?.kind === "ok" && (
          <div className={`verdict verdict--${verdict.verdict}`}>
            <span className="verdict__word">{verdict.verdict.replace("-", " ")}</span>
            <span className="verdict__note">{VERDICT_NOTE[verdict.verdict]}</span>
          </div>
        )}
        {verdict?.kind === "unset" && (
          <div className="verdict verdict--unset">
            <span className="verdict__word">policy unset</span>
            <span className="verdict__note">gates below are live; the combination rule is not written yet</span>
          </div>
        )}
        <Annunciator gates={gates} />
      </div>
    )
  }

  const renderBook = () => (
    <div className="exec">
      <div className="exec__row">
        <span className="exec__key">ENGINE</span>
        <span className={`exec__value ${engine ? "exec__value--observe" : ""}`}>
          {engine ? (engine.killed ? "KILLED" : "TRADING") : "unreachable"}
        </span>
      </div>
      {engine && (
        <>
          <div className="exec__row">
            <span className="exec__key">OPEN</span>
            <span className="exec__value">{engine.open.length}</span>
          </div>
          <div className="exec__row">
            <span className="exec__key">CLOSED</span>
            <span className="exec__value">{engine.stats.closedCount}</span>
          </div>
          <div className="exec__row">
            <span className="exec__key">WIN RATE</span>
            <span className="exec__value">
              {engine.stats.winRate === null ? "--" : `${Math.round(engine.stats.winRate * 100)}%`}
            </span>
          </div>
          <div className="exec__row">
            <span className="exec__key">SIM PNL</span>
            <span className={`exec__value ${engine.stats.totalPnlUsd >= 0 ? "exec__value--up" : "exec__value--down"}`}>
              {`${engine.stats.totalPnlUsd >= 0 ? "+" : "-"}$${Math.abs(engine.stats.totalPnlUsd).toFixed(2)}`}
            </span>
          </div>
          {engine.budget && (
            <div className="exec__row">
              <span className="exec__key">DAY SPEND</span>
              <span className="exec__value">
                {engine.budget.spentTodaySol.toFixed(2)} / {engine.budget.dailyCapSol} SOL
              </span>
            </div>
          )}
        </>
      )}
      {engine?.events && (
        <div className="decisions">
          {engine.events.slice(0, 14).map((e) => {
            const v = describeEvent(e)
            return (
              <div key={`${e.at}-${v.label}`} className={`decision decision--${v.kind}`}>
                <span className="decision__label mono">{v.label}</span>
                <span className="decision__detail">{v.detail}</span>
                <span className="decision__age mono">{age(e.at, now)}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )

  /**
   * Research panel. Opens the selected mint on a block explorer in the system
   * browser rather than embedding a webview: an Electron webview loading
   * arbitrary remote pages next to a trading surface is an attack surface, and
   * the operator already has a real browser with their sessions in it.
   */
  const renderBrowser = () => (
    <div className="research">
      {!active && <p className="empty">Select a candidate.</p>}
      {active && (
        <>
          <p className="research__hint">Look up {active.symbol} on chain.</p>
          <div className="research__links">
            {[
              ["Solscan", `https://solscan.io/token/${active.mint}`],
              ["DexScreener", `https://dexscreener.com/solana/${active.mint}`],
              ["Pump.fun", `https://pump.fun/coin/${active.mint}`],
              ["Birdeye", `https://birdeye.so/token/${active.mint}?chain=solana`],
            ].map(([label, href]) => (
              <a key={label} className="research__link" href={href} target="_blank" rel="noreferrer">
                {label}
              </a>
            ))}
          </div>
          <p className="research__mint mono">{active.mint}</p>
        </>
      )}
    </div>
  )

  return (
    <div className="app">
      <header className="header">
        <div className="header__brand">
          <span className="header__wordmark">
            Crowe<span className="header__wordmark-accent">Trade</span>
          </span>
          <span className="header__product">Operator Terminal</span>
        </div>

        <div className="pulse mono" aria-label="Market pulse">
          <span className="pulse__item">
            <span className="pulse__key">SOL</span>
            {solUsd !== null ? `$${solUsd.toFixed(2)}` : "--"}
          </span>
          <span className="pulse__sep" />
          <span className="pulse__item">
            <span className="pulse__key">TRACKED</span>
            {candidates.length}
          </span>
          <span className="pulse__sep" />
          <span className="pulse__item">
            <span className="pulse__key">NEWEST</span>
            {candidates[0] ? age(candidates[0].createdAt, now) : "--"}
          </span>
        </div>

        <span className={`eyebrow${error ? " eyebrow--idle" : ""}`}>
          <span className="eyebrow__dot" />
          {error ? "feed down" : loading ? "connecting" : "bootstrap feed"}
        </span>
      </header>

      <div className="body">
        <Rail />
        <Workspace
          render={(panel: Panel) => {
            switch (panel.type) {
              case "scan":
                return renderScan()
              case "chart":
                return renderPrimary()
              case "gates":
                return renderGates()
              case "book":
                return renderBook()
              case "analyst":
                return <AnalystPanel mint={active?.mint ?? null} />
              case "browser":
                return renderBrowser()
            }
          }}
        />
      </div>
    </div>
  )
}

function Stat({ k, v }: { k: string; v: string }) {
  const unknown = v === "unknown"
  return (
    <div className="stat">
      <span className="stat__key">{k}</span>
      <span className={`stat__value mono${unknown ? " stat__value--muted" : ""}`}>{v}</span>
    </div>
  )
}

/**
 * Buy pressure as a proportion, not two bare counts. 725 buys / 507 sells
 * takes arithmetic to read; a bar at 59% does not.
 */
function Pressure({ buys, sells }: { buys: number | null; sells: number | null }) {
  if (buys === null || sells === null || buys + sells === 0) {
    return <Stat k="Flow 24h" v="unknown" />
  }
  const share = buys / (buys + sells)
  return (
    <div className="stat stat--tall">
      <span className="stat__key">Flow 24h</span>
      <div className="pressure">
        <div className="pressure__track" role="img"
          aria-label={`${buys.toLocaleString()} buys, ${sells.toLocaleString()} sells`}>
          <div className="pressure__buys" style={{ width: `${(share * 100).toFixed(1)}%` }} />
        </div>
        <span className="pressure__caption mono">
          {buys.toLocaleString()} B / {sells.toLocaleString()} S
        </span>
      </div>
    </div>
  )
}

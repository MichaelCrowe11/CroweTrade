import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { fetchCandidates, type Candidate } from "./feed/dexscreener.js"
import { fetchMintFacts } from "./feed/solana.js"
import { evaluateGates, combineVerdict, type Verdict } from "./safety/gates.js"
import { Annunciator } from "./components/Annunciator.js"
import { PriceChart } from "./components/PriceChart.js"
import { age, usd, shortMint } from "./components/format.js"
import { Spark } from "./components/Spark.js"
import { Rail } from "./shell/Rail.js"
import { Workspace, CloseIcon } from "./shell/Workspace.js"
import { AnalystPanel } from "./shell/AnalystPanel.js"
import { BrowserPanel } from "./shell/BrowserPanel.js"
import { usePanels, type Panel } from "./shell/panels.js"
import { DURATIONS, EASINGS, MAGNITUDES } from "./shell/motion.js"
import { describeEvent, type EngineEvent } from "./engine/events.js"
import { standingOf, countdown, pct, gapPt } from "./engine/standing.js"

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

/**
 * The read model of GET /api/positions. Everything beyond the original five
 * fields is optional so an older engine, or a summary trimmed mid-deploy,
 * degrades to the smaller readout instead of a crash. The additions exist so
 * the terminal can explain a SILENT engine: a book that stopped trading must
 * name its reason (breaker, cap, slots, kill), because this project has lost
 * hours twice to declines that were recorded but never surfaced.
 */
interface EngineSummary {
  mode?: string
  killed: boolean
  policyHash?: string
  open: EnginePosition[]
  stats: {
    closedCount: number
    totalPnlUsd: number
    winRate: number | null
    excludedModelPriced?: number
  }
  budget?: {
    spentTodaySol: number
    dailyCapSol: number
    remainingSol?: number
    openSlots?: number
    canEnter: boolean
    breaker?: { open: boolean; until: string | null }
  }
  byOrigin?: {
    origin: string
    decisions: number
    voided: number
    labeled: number
    entered: number
    deathRate: number | null
    avgForwardRetEnteredPct: number | null
    avgForwardRetRefusedPct: number | null
  }[]
  skipReasons?: { reason: string; count: number }[]
  calibration?: {
    decisions: number
    labeled: number
    oldestUnlabeledAgeMin: number | null
    dueForLabel: number
    deathRate: number | null
    avgForwardRetEnteredPct: number | null
    avgForwardRetEligibleSkippedPct: number | null
  }
  alert?: {
    state: string
    labeled: number
    needed: number
    configured: boolean
    lastError: string | null
  }
  cohorts?: {
    policyHash: string
    current: boolean
    closed: number
    pnlUsd: number
    winRate: number | null
    unroutableExits: number
  }[]
  events?: EngineEvent[]
}

/** Death rate as a plain unsigned percent; null is "--", never a zero. */
function deathPct(rate: number | null | undefined): string {
  if (rate === null || rate === undefined || !Number.isFinite(rate)) return "--"
  return `${(rate * 100).toFixed(0)}%`
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
  const analystOpen = usePanels((s) => s.analystOpen)
  const closeAnalyst = usePanels((s) => s.closeAnalyst)
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
    // Dev-only: the self-shot rail can seed an engine fixture so states like
    // an open breaker are photographable without waiting for the live engine
    // to be in them. A present fixture replaces polling for the session, and
    // is CONSUMED on read: a leftover fixture surviving into the next real
    // launch would pin a phantom breaker on screen forever, which is the
    // exact quiet-vs-broken confusion this readout exists to prevent.
    let engineTimer: ReturnType<typeof setInterval> | undefined
    let fixture: EngineSummary | null = null
    try {
      const raw = localStorage.getItem("crowetrade-engine-fixture")
      if (raw) {
        localStorage.removeItem("crowetrade-engine-fixture")
        fixture = JSON.parse(raw) as EngineSummary
      }
    } catch {
      fixture = null
    }
    if (fixture) {
      setEngine(fixture)
    } else {
      pullEngine()
      engineTimer = setInterval(pullEngine, 30_000)
    }
    // Ages are relative to wall clock, so they need their own tick or every row
    // would read the same age until the next feed poll landed.
    const clock = setInterval(() => setNow(Date.now()), 1_000)
    return () => {
      clearInterval(feed)
      clearInterval(clock)
      if (engineTimer) clearInterval(engineTimer)
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

  /**
   * The book: engine standing first, because a quiet engine and a broken one
   * must never look the same. When the engine declines to enter, the word at
   * the top names why: the kill switch, the breaker (with a live countdown),
   * the day cap, or full slots. An unexplained block renders as PAUSED in the
   * alarm hue rather than as a healthy TRADING, on the principle that unknown
   * never reads as pass.
   */
  const renderBook = () => {
    const standing = engine ? standingOf(engine) : null
    const current = engine?.cohorts?.find((c) => c.current) ?? null
    return (
      <div className="exec">
        {!engine && (
          <div className="standing standing--unreachable">
            <span className="standing__word">UNREACHABLE</span>
            <span className="standing__note">
              the terminal observes markets on its own; the paper book returns when the engine does
            </span>
          </div>
        )}

        {engine && standing && (
          <div className={`standing standing--${standing.state}`}>
            <span className="standing__word">
              {standing.state === "killed" && "KILLED"}
              {standing.state === "breaker" && "HOLDING"}
              {standing.state === "cap" && "DAY CAP"}
              {standing.state === "slots" && "SLOTS FULL"}
              {standing.state === "paused" && "PAUSED"}
              {standing.state === "trading" && "TRADING"}
            </span>
            <span className="standing__note">
              {standing.state === "killed" && "kill switch engaged; exits continue, entries do not"}
              {standing.state === "breaker" &&
                (standing.untilMs !== null ? (
                  <>
                    circuit breaker open; entries resume in{" "}
                    <span className="mono standing__clock">{countdown(standing.untilMs, now)}</span>
                  </>
                ) : (
                  "circuit breaker open; no deadline reported"
                ))}
              {standing.state === "cap" &&
                `daily budget spent: ${engine.budget?.spentTodaySol.toFixed(2)} of ${engine.budget?.dailyCapSol} SOL`}
              {standing.state === "slots" && "every position slot is in use"}
              {standing.state === "paused" && "entries blocked; the engine did not name a cause"}
              {standing.state === "trading" &&
                (engine.budget
                  ? `${engine.budget.remainingSol?.toFixed(1) ?? "?"} SOL and ${engine.budget.openSlots ?? "?"} slots free`
                  : "entries permitted")}
            </span>
          </div>
        )}

        {engine && (
          <>
            <div className="exec__row">
              <span className="exec__key">MODE</span>
              <span className="exec__value exec__value--observe">
                {(engine.mode ?? "unknown").toUpperCase()}
                {engine.policyHash ? ` / ${engine.policyHash.slice(0, 8)}` : ""}
              </span>
            </div>
            <div className="exec__row">
              <span className="exec__key">OPEN</span>
              <span className="exec__value">
                {engine.open.length}
                {engine.budget?.openSlots !== undefined ? ` / ${engine.budget.openSlots} slots free` : ""}
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

            {/* The funding number is THIS policy's record. Lifetime spans every
                policy version this engine ever ran, including ones whose bugs
                are already fixed, so it lives in a footnote, never up here. */}
            {current ? (
              <>
                <div className="exec__section mono">THIS POLICY</div>
                <div className="exec__row">
                  <span className="exec__key">CLOSED</span>
                  <span className="exec__value">{current.closed}</span>
                </div>
                <div className="exec__row">
                  <span className="exec__key">PNL</span>
                  <span
                    className={`exec__value ${current.pnlUsd >= 0 ? "exec__value--up" : "exec__value--down"}`}
                  >
                    {`${current.pnlUsd >= 0 ? "+" : "-"}$${Math.abs(current.pnlUsd).toFixed(2)}`}
                  </span>
                </div>
                <div className="exec__row">
                  <span className="exec__key">WIN RATE</span>
                  <span className="exec__value">
                    {current.winRate === null ? "--" : `${Math.round(current.winRate * 100)}%`}
                  </span>
                </div>
                {current.unroutableExits > 0 && (
                  <div className="exec__row">
                    <span className="exec__key">UNROUTABLE</span>
                    <span className="exec__value exec__value--down">{current.unroutableExits}</span>
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="exec__row">
                  <span className="exec__key">CLOSED</span>
                  <span className="exec__value">{engine.stats.closedCount}</span>
                </div>
                <div className="exec__row">
                  <span className="exec__key">WIN RATE</span>
                  <span className="exec__value">
                    {engine.stats.winRate === null
                      ? "--"
                      : `${Math.round(engine.stats.winRate * 100)}%`}
                  </span>
                </div>
                <div className="exec__row">
                  <span className="exec__key">SIM PNL</span>
                  <span
                    className={`exec__value ${engine.stats.totalPnlUsd >= 0 ? "exec__value--up" : "exec__value--down"}`}
                  >
                    {`${engine.stats.totalPnlUsd >= 0 ? "+" : "-"}$${Math.abs(engine.stats.totalPnlUsd).toFixed(2)}`}
                  </span>
                </div>
              </>
            )}

            {engine.alert && (
              <div className="exec__row">
                <span className="exec__key">ALERT</span>
                {!engine.alert.configured ? (
                  <span className="exec__value exec__value--down">NOT CONFIGURED</span>
                ) : engine.alert.lastError ? (
                  <span className="exec__value exec__value--down">
                    FAILED: {engine.alert.lastError}
                  </span>
                ) : (
                  <span className="exec__value">
                    {engine.alert.state === "sent"
                      ? `sent, ${engine.alert.labeled} labeled`
                      : `${engine.alert.state} ${engine.alert.labeled}/${engine.alert.needed} labeled`}
                  </span>
                )}
              </div>
            )}

            {current && (
              <p className="exec__note">
                Lifetime, {engine.cohorts?.length ?? 0} policies: {engine.stats.closedCount}{" "}
                closes, {engine.stats.totalPnlUsd >= 0 ? "+" : "-"}$
                {Math.abs(engine.stats.totalPnlUsd).toFixed(2)},{" "}
                {engine.stats.winRate === null ? "--" : `${Math.round(engine.stats.winRate * 100)}%`}{" "}
                win
                {engine.stats.excludedModelPriced
                  ? `; ${engine.stats.excludedModelPriced} model-priced closes excluded`
                  : ""}
                . Judge the current policy, not this line.
              </p>
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
  }

  /**
   * Calibration: the honest record of whether selection works. The primary
   * readout is entered-versus-refused forward returns, side by side with the
   * gap between them, because two numbers that should diverge and do not is
   * the most important thing this product currently knows about itself.
   * Quarantined (voided) rows are shown as excluded and never enter a sample
   * size; a missing return renders as "--", never as zero.
   */
  const renderCalibration = () => {
    if (!engine) {
      return <p className="empty">Engine unreachable. Calibration reads come from the live engine.</p>
    }
    const cal = engine.calibration
    const origins = engine.byOrigin ?? []
    const skips = engine.skipReasons ?? []
    const maxSkip = Math.max(1, ...skips.map((s) => s.count))
    const totalSkip = skips.reduce((n, s) => n + s.count, 0)
    // Forward returns carry DIRECTION, so they read on the green/red axis;
    // a missing number is muted, never colored and never zero.
    const tone = (v: number | null | undefined) =>
      v === null || v === undefined ? "calib__num--muted" : v >= 0 ? "calib__num--up" : "calib__num--down"
    const gapValue =
      cal &&
      cal.avgForwardRetEnteredPct !== null &&
      cal.avgForwardRetEligibleSkippedPct !== null
        ? cal.avgForwardRetEnteredPct - cal.avgForwardRetEligibleSkippedPct
        : null
    return (
      <div className="calib">
        {cal && (
          <div className="calib__test">
            <span className="calib__eyebrow mono">SELECTION TEST</span>
            <div className="calib__pair">
              <div className="calib__side">
                <span className={`calib__num mono ${tone(cal.avgForwardRetEnteredPct)}`}>
                  {pct(cal.avgForwardRetEnteredPct)}
                </span>
                <span className="calib__key mono">ENTERED</span>
              </div>
              <div className="calib__side calib__side--gap">
                <span className={`calib__num calib__num--gap mono ${tone(gapValue)}`}>
                  {gapPt(cal.avgForwardRetEnteredPct, cal.avgForwardRetEligibleSkippedPct)}
                </span>
                <span className="calib__key mono">GAP</span>
              </div>
              <div className="calib__side">
                <span className={`calib__num mono ${tone(cal.avgForwardRetEligibleSkippedPct)}`}>
                  {pct(cal.avgForwardRetEligibleSkippedPct)}
                </span>
                <span className="calib__key mono">REFUSED</span>
              </div>
            </div>
            <p className="calib__note">
              30 minute forward return, {cal.labeled} labeled decisions. If selection works,
              entered must beat refused. A gap near zero means the gates are not separating
              winners from the universe they choose from.
            </p>
          </div>
        )}

        {origins.length > 0 && (
          <div className="calib__block">
            <span className="calib__eyebrow mono">BY ORIGIN</span>
            {origins.map((o) => (
              <div className="origin" key={o.origin}>
                <div className="origin__head">
                  <span className="origin__name mono">{o.origin.toUpperCase()}</span>
                  <span className="origin__sample">
                    {o.labeled} labeled of {o.decisions}
                    {o.voided > 0 ? `, ${o.voided.toLocaleString()} quarantined excluded` : ""}
                  </span>
                </div>
                <div className="origin__stats mono">
                  <span className="origin__stat">
                    <span className="origin__k">DEATH</span> {deathPct(o.deathRate)}
                  </span>
                  <span className="origin__stat">
                    <span className="origin__k">ENTERED</span> {o.entered}
                  </span>
                  <span className="origin__stat">
                    <span className="origin__k">ENT RET</span> {pct(o.avgForwardRetEnteredPct)}
                  </span>
                  <span className="origin__stat">
                    <span className="origin__k">REF RET</span> {pct(o.avgForwardRetRefusedPct)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {skips.length > 0 && (
          <div className="calib__block">
            <span className="calib__eyebrow mono">DISCOVERY SKIPS</span>
            {skips.map((s) => (
              <div className="skip" key={s.reason}>
                <span className="skip__reason">{s.reason}</span>
                <div className="skip__track">
                  <div
                    className={`skip__fill ${s.reason === "eligible" ? "skip__fill--eligible" : ""}`}
                    style={{ width: `${((s.count / maxSkip) * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="skip__count mono">{s.count}</span>
              </div>
            ))}
            <p className="calib__note">
              {totalSkip.toLocaleString()} decisions in the current window. "eligible" is what
              survived every gate; everything else names the gate that refused it.
            </p>
          </div>
        )}

        {cal && (
          <p className="calib__ops mono">
            labeled {cal.labeled}/{cal.decisions} · due {cal.dueForLabel} · oldest unlabeled{" "}
            {cal.oldestUnlabeledAgeMin ?? "--"}m · death {deathPct(cal.deathRate)}
          </p>
        )}
      </div>
    )
  }

  /**
   * Research panel: a real embedded browser, not a link list. The page lives
   * in a main-process WebContentsView (sandboxed, context-isolated, node-free,
   * own session partition, popups to the system browser) because an iframe is
   * not a browser: the sites an operator needs refuse embedding. This renderer
   * only draws the chrome. The earlier link-list rationale (webview as attack
   * surface) is answered by the isolation above, not forgotten.
   */
  const renderBrowser = (panel: Panel) => (
    <BrowserPanel
      panelId={panel.id}
      initialUrl={
        typeof panel.payload?.["url"] === "string"
          ? (panel.payload["url"] as string)
          : active
            ? `https://solscan.io/token/${active.mint}`
            : "https://solscan.io"
      }
    />
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
        {/* The Analyst pops out from the left edge beside the rail, Cortex's
            conversation-panel pattern. It DOCKS rather than overlays: browser
            panels are native views composited above this page, so anything
            that floated over the workspace would render underneath them. */}
        <AnimatePresence initial={false}>
          {analystOpen && (
            <motion.aside
              className="drawer"
              aria-label="Analyst drawer"
              initial={{ x: -MAGNITUDES.slide, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -MAGNITUDES.slide, opacity: 0 }}
              transition={{ duration: DURATIONS.smooth, ease: EASINGS.snap }}
            >
              <header className="drawer__head">
                <span className="ws__title">Analyst</span>
                <button
                  type="button"
                  className="ws__act drawer__close"
                  onClick={closeAnalyst}
                  aria-label="Close the Analyst drawer"
                >
                  <CloseIcon />
                </button>
              </header>
              <AnalystPanel mint={active?.mint ?? null} />
            </motion.aside>
          )}
        </AnimatePresence>
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
              case "calibration":
                return renderCalibration()
              case "browser":
                return renderBrowser(panel)
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

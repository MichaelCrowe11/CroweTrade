import { useEffect, useState } from "react"

/**
 * The primary readout: 1-minute price action, hand-rolled SVG.
 *
 * No chart library on purpose. A library brings its own typography, its own
 * easing, its own tooltip chrome, and the design brief is a Tektronix scope,
 * not a product tour. Everything here is drawn with the same tokens as the
 * rest of the surface, and the whole component costs zero dependencies.
 *
 * Candles arrive over the IPC bridge because the OHLCV host sends no CORS
 * headers; see electron/main.ts.
 */

declare global {
  interface Window {
    crowetrade?: { candles?: (pool: string) => Promise<number[][]> }
  }
}

const W = 720
const H = 240
const PAD_R = 74 // room for the price axis
const PAD_Y = 14
const VOL_H = 34 // volume strip along the bottom, inside H

type State =
  | { kind: "loading" }
  | { kind: "empty" }
  | { kind: "ok"; candles: number[][] }

function fmtPrice(v: number): string {
  if (v >= 1) return v.toFixed(2)
  return v.toPrecision(3)
}

function fmtClock(ts: number): string {
  const d = new Date(ts * 1000)
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`
}

export function PriceChart({ pool, mint }: { pool: string | null; mint: string }) {
  const [state, setState] = useState<State>({ kind: "loading" })

  useEffect(() => {
    let dead = false
    setState({ kind: "loading" })
    const bridge = window.crowetrade?.candles
    if (!pool || !bridge) {
      setState({ kind: "empty" })
      return
    }
    const pull = () => {
      bridge(pool)
        .then((rows) => {
          if (dead) return
          setState(rows.length >= 2 ? { kind: "ok", candles: rows } : { kind: "empty" })
        })
        .catch(() => {
          if (!dead) setState({ kind: "empty" })
        })
    }
    pull()
    const t = setInterval(pull, 30_000)
    return () => {
      dead = true
      clearInterval(t)
    }
    // mint in the deps so switching tokens that share a null pool still resets.
  }, [pool, mint])

  if (state.kind !== "ok") {
    return (
      <div className="chart chart--empty">
        <span className="chart__status">
          {state.kind === "loading" ? "reading candles" : "no candle history yet"}
        </span>
      </div>
    )
  }

  const candles = state.candles
  const closes = candles.map((c) => c[4] ?? 0)
  const vols = candles.map((c) => c[5] ?? 0)
  const first = closes[0] ?? 0
  const last = closes[closes.length - 1] ?? 0
  const lo = Math.min(...closes)
  const hi = Math.max(...closes)
  const span = hi - lo || hi || 1
  const maxVol = Math.max(...vols, 1)
  const up = last >= first
  const changePct = first > 0 ? ((last - first) / first) * 100 : 0

  const plotW = W - PAD_R
  const plotH = H - PAD_Y * 2 - VOL_H
  const x = (i: number) => (i / (candles.length - 1)) * plotW
  const y = (v: number) => PAD_Y + (1 - (v - lo) / span) * plotH

  const line = closes.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ")
  const area = `${line} L${plotW},${PAD_Y + plotH} L0,${PAD_Y + plotH} Z`

  // Four horizontal grid rules with their price labels.
  const rules = [0, 1 / 3, 2 / 3, 1].map((f) => ({ yy: PAD_Y + f * plotH, v: hi - f * span }))

  const t0 = candles[0]?.[0] ?? 0
  const t1 = candles[candles.length - 1]?.[0] ?? 0

  return (
    <div className="chart">
      <div className="chart__head">
        <span className="chart__price mono">${fmtPrice(last)}</span>
        <span className={`chart__change mono ${up ? "chart__change--up" : "chart__change--down"}`}>
          {up ? "+" : ""}
          {changePct.toFixed(1)}% / {candles.length}m
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="chart__svg" role="img"
        aria-label={`Price ${fmtPrice(last)} dollars, ${up ? "up" : "down"} ${Math.abs(changePct).toFixed(1)} percent over ${candles.length} minutes`}>
        <defs>
          <linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--clm-gold)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--clm-gold)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {rules.map((r) => (
          <g key={r.yy}>
            <line x1="0" y1={r.yy} x2={plotW} y2={r.yy} className="chart__grid" />
            <text x={plotW + 8} y={r.yy + 3} className="chart__axis">
              {fmtPrice(r.v)}
            </text>
          </g>
        ))}

        {/* Volume strip: activity texture under the price, deliberately faint. */}
        {vols.map((v, i) => {
          const bh = (v / maxVol) * (VOL_H - 6)
          return bh > 0.5 ? (
            <rect
              key={i}
              x={x(i) - plotW / candles.length / 2 + 0.5}
              y={H - PAD_Y - bh}
              width={Math.max(plotW / candles.length - 1, 0.75)}
              height={bh}
              className="chart__vol"
            />
          ) : null
        })}

        <path d={area} fill="url(#chartFill)" />
        <path d={line} className="chart__line" />

        {/* Last-price marker: dashed rule across, dot on the tip. */}
        <line x1="0" y1={y(last)} x2={plotW} y2={y(last)} className="chart__last" />
        <circle cx={x(closes.length - 1)} cy={y(last)} r="3" className="chart__tip" />

        <text x="0" y={H - 2} className="chart__axis">{fmtClock(t0)}</text>
        <text x={plotW} y={H - 2} textAnchor="end" className="chart__axis">{fmtClock(t1)}</text>
      </svg>
    </div>
  )
}

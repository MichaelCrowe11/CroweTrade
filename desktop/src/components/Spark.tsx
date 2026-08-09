/**
 * Row sparkline: the shape of a token's recent hour, at a glance.
 *
 * A scan list of names and percentages tells you what happened; it does not
 * tell you what is happening. Two tokens both at +40% can be one climbing and
 * one collapsing off a spike, and on this market that difference is the entire
 * decision. The trace is the smallest thing that carries it.
 *
 * Deliberately unlabelled and unaxised. It is a texture the eye reads while
 * scanning, not a chart to be measured; the real chart is the primary readout.
 */

export function Spark({ points, up }: { points: number[]; up: boolean }) {
  if (points.length < 2) {
    return <span className="spark spark--empty" aria-hidden="true" />
  }

  const w = 64
  const h = 18
  const lo = Math.min(...points)
  const hi = Math.max(...points)
  const span = hi - lo || hi || 1

  const d = points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * w
      // Inset by 1px top and bottom so the stroke is never clipped at extremes.
      const y = 1 + (1 - (p - lo) / span) * (h - 2)
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")

  return (
    <svg
      className={`spark ${up ? "spark--up" : "spark--down"}`}
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <path d={`${d} L${w},${h} L0,${h} Z`} className="spark__fill" />
      <path d={d} className="spark__line" />
    </svg>
  )
}

/**
 * The two-marker inline subset the Analyst actually emits: **strong** and
 * `code`. Not a markdown engine; anything else stays literal, and an UNCLOSED
 * marker stays literal too, which is what makes this safe to run over a
 * half-streamed answer: "**TP60" reads as typed text until its closer lands,
 * then resolves to bold on the next paint.
 *
 * Dependency-free so node --test can exercise it directly.
 */

export interface InlineSegment {
  kind: "text" | "strong" | "code" | "num" | "strong-num"
  text: string
}

/**
 * The instrument rule, applied to prose: financial figures inside an answer
 * are their own segments so the renderer can set them in tabular mono.
 * Matches signed currency, percents, amounts with a SOL unit, and bare
 * counts; leaves numbers embedded in hyphenated words ("30-minute") as text,
 * and never re-segments code spans.
 */
const NUM_RE = /[-+]?\$?\d[\d,]*(?:\.\d+)?(?:%|\s?SOL\b)?/g

function splitNumbers(seg: InlineSegment): InlineSegment[] {
  const numKind = seg.kind === "strong" ? "strong-num" : "num"
  const out: InlineSegment[] = []
  let cursor = 0
  for (const m of seg.text.matchAll(NUM_RE)) {
    const start = m.index
    const end = start + m[0].length
    const before = seg.text[start - 1]
    const after = seg.text[end]
    // A digit glued to a letter or hyphen on either side is part of a word
    // ("30-minute", "sha256"), not a figure.
    if ((before && /[\w-]/.test(before)) || (after && /[\w-]/.test(after))) continue
    if (start > cursor) out.push({ kind: seg.kind, text: seg.text.slice(cursor, start) })
    out.push({ kind: numKind, text: m[0] })
    cursor = end
  }
  if (cursor < seg.text.length) out.push({ kind: seg.kind, text: seg.text.slice(cursor) })
  return out.length > 0 ? out : [seg]
}

export function segmentRich(text: string): InlineSegment[] {
  return segmentInline(text).flatMap((seg) =>
    seg.kind === "code" ? [seg] : splitNumbers(seg),
  )
}

const MARKERS: { open: string; kind: "strong" | "code" }[] = [
  { open: "**", kind: "strong" },
  { open: "`", kind: "code" },
]

export function segmentInline(text: string): InlineSegment[] {
  const out: InlineSegment[] = []
  let plain = ""
  let i = 0

  const flush = () => {
    if (plain) {
      out.push({ kind: "text", text: plain })
      plain = ""
    }
  }

  while (i < text.length) {
    const marker = MARKERS.find((m) => text.startsWith(m.open, i))
    if (marker) {
      const close = text.indexOf(marker.open, i + marker.open.length)
      const inner = close === -1 ? "" : text.slice(i + marker.open.length, close)
      if (close !== -1 && inner.length > 0) {
        flush()
        out.push({ kind: marker.kind, text: inner })
        i = close + marker.open.length
        continue
      }
    }
    plain += text[i]
    i++
  }
  flush()
  return out
}

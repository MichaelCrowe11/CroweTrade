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
  kind: "text" | "strong" | "code"
  text: string
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

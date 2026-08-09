/**
 * Minimal SSE parser for the Foundry responses stream.
 *
 * Dependency-free so node --test can run it directly (the strip-types runner
 * cannot resolve cross-module .js specifiers), and deliberately tolerant:
 * comments, event: lines, [DONE], blank data and unparseable payloads are
 * dropped rather than thrown, because a hiccup in a keep-alive line must not
 * kill an answer that is mid-sentence.
 *
 * Events are blank-line separated; each data: line is expected to carry one
 * complete JSON object with a type field (the shape verified against the live
 * endpoint). Multi-line data: continuation is not in that vocabulary, so each
 * data: line parses alone.
 */

export interface SseJsonEvent {
  type: string
  [key: string]: unknown
}

export function createSseParser(): { push(chunk: string): SseJsonEvent[] } {
  let buf = ""
  return {
    push(chunk) {
      // Normalizing per push is idempotent for already-clean text and heals a
      // CRLF pair split across chunk boundaries (the lone \r waits in buf).
      buf = (buf + chunk).replace(/\r\n/g, "\n")
      const events: SseJsonEvent[] = []
      let idx: number
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        for (const line of raw.split("\n")) {
          if (!line.startsWith("data:")) continue
          const payload = line.slice(5).trim()
          if (!payload || payload === "[DONE]") continue
          try {
            const parsed = JSON.parse(payload) as { type?: unknown }
            if (typeof parsed["type"] === "string") events.push(parsed as SseJsonEvent)
          } catch {
            // Noise on the wire is not an error in the answer.
          }
        }
      }
      return events
    },
  }
}

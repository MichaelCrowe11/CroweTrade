/**
 * One line per engine decision: what, who, and why when it declined.
 *
 * Moved out of App.tsx so it is testable (dependency-free per the strip-types
 * constraint) and because the event vocabulary now covers the states where the
 * engine deliberately does nothing: a breaker trip, a failed alert, a scan
 * error. Those are precisely the lines an operator needs when the book goes
 * quiet, so falling through to a bare uppercased kind is not acceptable for
 * them.
 *
 * View kinds: entry, exit-win, exit-loss, skip, breaker, fail, info.
 */

export interface EngineEvent {
  at: number
  kind: string
  data: string
}

export interface EventView {
  label: string
  detail: string
  kind: string
}

export function describeEvent(e: EngineEvent): EventView {
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
      case "breaker": {
        if (!d["tripped"]) return { kind: "breaker", label: "BREAKER", detail: "released" }
        const until = typeof d["until"] === "number" ? d["until"] : null
        const hold = until !== null ? `; holding ${Math.max(0, Math.round((until - e.at) / 60_000))}m` : ""
        const cause =
          d["kind"] === "loss-velocity"
            ? "loss velocity trip"
            : typeof d["afterConsecutiveStops"] === "number"
              ? `${d["afterConsecutiveStops"]} consecutive stops`
              : "tripped"
        return { kind: "breaker", label: "BREAKER", detail: `${cause}${hold}` }
      }
      case "alert_sent":
        return { kind: "info", label: "ALERT", detail: String(d["subject"] ?? "sent") }
      case "alert_failed":
        return { kind: "fail", label: "ALERT FAILED", detail: String(d["error"] ?? "") }
      case "scan_error":
        return { kind: "fail", label: "SCAN ERROR", detail: String(d["message"] ?? "") }
      default:
        return { kind: "skip", label: e.kind.toUpperCase(), detail: "" }
    }
  } catch {
    return { kind: "skip", label: e.kind, detail: "" }
  }
}

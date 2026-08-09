/**
 * Why the engine is or is not entering, as one derived word.
 *
 * Dependency-free on purpose: node --test with type stripping cannot resolve
 * the `.js`-specifier imports the app modules use between themselves, so
 * test-critical logic lives in modules that import nothing.
 *
 * The ordering is the honesty policy. Kill outranks the breaker because it is
 * the operator's own hand; the breaker outranks budget states because it is
 * the safety system asserting itself; and a blocked entry with NO visible
 * cause must surface as "paused", never as "trading" -- an unexplained quiet
 * engine is exactly the state that has cost this project hours twice, and it
 * must not be rendered as a healthy one.
 */

export interface StandingInput {
  killed: boolean
  budget?: {
    spentTodaySol?: number
    dailyCapSol?: number
    remainingSol?: number
    openSlots?: number
    canEnter?: boolean
    breaker?: { open: boolean; until: string | null }
  }
}

export type Standing =
  | { state: "killed" }
  | { state: "breaker"; untilMs: number | null }
  | { state: "cap" }
  | { state: "slots" }
  | { state: "paused" }
  | { state: "trading" }

export function standingOf(input: StandingInput): Standing {
  if (input.killed) return { state: "killed" }
  const b = input.budget
  if (b?.breaker?.open) {
    const parsed = b.breaker.until === null ? NaN : Date.parse(b.breaker.until ?? "")
    return { state: "breaker", untilMs: Number.isFinite(parsed) ? parsed : null }
  }
  if (b?.canEnter === false) {
    if (typeof b.remainingSol === "number" && b.remainingSol <= 0) return { state: "cap" }
    if (typeof b.openSlots === "number" && b.openSlots <= 0) return { state: "slots" }
    return { state: "paused" }
  }
  return { state: "trading" }
}

/** "12:34" to a deadline; clamps at "0:00"; empty when there is no deadline. */
export function countdown(untilMs: number | null, nowMs: number): string {
  if (untilMs === null) return ""
  const left = Math.max(0, Math.floor((untilMs - nowMs) / 1000))
  const m = Math.floor(left / 60)
  const s = left % 60
  return `${m}:${String(s).padStart(2, "0")}`
}

/**
 * The selection edge in points: entered forward return minus refused forward
 * return. Positive means the gates picked better than what they rejected.
 * Either side missing means the comparison does not exist yet, not that it is
 * zero.
 */
export function gapPt(
  entered: number | null | undefined,
  refused: number | null | undefined,
): string {
  if (
    entered === null || entered === undefined || !Number.isFinite(entered) ||
    refused === null || refused === undefined || !Number.isFinite(refused)
  ) {
    return "--"
  }
  const gap = entered - refused
  return `${gap > 0 ? "+" : ""}${gap.toFixed(1)}pt`
}

/**
 * Signed percentage, one decimal. null stays "--": a missing forward return
 * (no entries yet) is not a zero return, and rendering it as one would be the
 * same lie as a blank gate reading as a pass.
 */
export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--"
  const rounded = value.toFixed(1)
  if (rounded === "0.0" || rounded === "-0.0") return "0.0%"
  return `${value > 0 ? "+" : ""}${rounded}%`
}

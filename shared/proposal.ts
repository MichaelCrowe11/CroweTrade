/**
 * Policy proposals: the artifact an agent hands a human.
 *
 * The seam this exists to create. Agents are good at reading a corpus and
 * noticing things; they are unreliable at knowing whether what they noticed is
 * already handled. On 2026-08-10 the Analyst recommended dropping the profile
 * discovery feed — correct reasoning, well caveated, and already shipped hours
 * earlier. It had no way to see that, because it was reading a lifetime metric
 * that described policies no longer running.
 *
 * So a proposal is validated against the RUNNING policy before a human ever
 * reads it. A change already in effect comes back as a no-op with the current
 * value attached, which is a two-second dismissal instead of a research
 * project. That single check is most of this file's value.
 *
 * What this deliberately does NOT do: apply anything. It produces a reviewable
 * object. The engine's behaviour changes when a signed envelope is deployed,
 * never because an agent was persuasive. Agents propose, humans sign, machines
 * execute.
 *
 * No runtime imports, so it is testable under strip-types.
 */

/** Fields an agent may propose changing. An allowlist, not a free path: an
 *  agent must not be able to propose editing `product`, `signature` or
 *  `signer`, because those are the consent record itself. */
export const PROPOSABLE = [
  "perTradeCapSol",
  "dailyCapSol",
  "maxOpenPositions",
  "entry.minVerdict",
  "entry.maxTokenAgeMinutes",
  "entry.minTokenAgeMinutes",
  "entry.minLiquidityUsd",
  "entry.maxChangeH1Pct",
  "entry.maxEntryImpactPct",
  "entry.maxDriftSinceFirstSightPct",
  "entry.minObservedTicks",
  "entry.minModelProb",
  "entry.allowedOrigins",
  "entry.maxDeployerPriorMints",
  "entry.maxDeployerPriorRugs",
  "breaker.consecutiveStopLimit",
  "breaker.cooldownMinutes",
  "exit.takeProfitPct",
  "exit.stopLossPct",
  "exit.timeStopMinutes",
] as const

export type ProposablePath = (typeof PROPOSABLE)[number]

/** Paths where `null` is a legitimate value meaning "gate not applied".
 *  Listed explicitly rather than inferred: accepting null for `dailyCapSol`
 *  would read as "no daily cap", which is the opposite of a safe default. */
export const NULLABLE = [
  "entry.minModelProb", "exit.takeProfitPct",
  "entry.maxDeployerPriorMints", "entry.maxDeployerPriorRugs",
] as const

export interface ProposedChange {
  path: string
  /** The value the agent wants. */
  to: unknown
}

export interface ValidatedChange {
  path: string
  from: unknown
  to: unknown
  /** True when `to` already equals `from`: nothing to do. */
  noop: boolean
  /**
   * Does this REDUCE the engine's freedom to act? Tighten-vs-loosen is the
   * governance axis the envelope already documents: tighten instantly, loosen
   * with delay. Null when the direction is not meaningful for this field.
   */
  tightens: boolean | null
}

export interface ProposalResult {
  ok: boolean
  changes: ValidatedChange[]
  /** Reasons the proposal cannot be accepted as written. */
  errors: string[]
  /** True when EVERY change is a no-op: the agent proposed the status quo. */
  entirelyNoop: boolean
}

function readPath(obj: unknown, path: string): unknown {
  let cur: unknown = obj
  for (const part of path.split(".")) {
    if (typeof cur !== "object" || cur === null) return undefined
    cur = (cur as Record<string, unknown>)[part]
  }
  return cur
}

/** Deep-ish equality, enough for the scalar and string-array values a policy
 *  actually holds. Order matters for arrays because allowedOrigins order is
 *  not meaningful but a reordering is still not a change worth deploying. */
function sameValue(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && [...a].sort().every((v, i) => v === [...b].sort()[i])
  }
  return a === b
}

/**
 * Which direction reduces the engine's freedom, per field.
 *
 * Stated explicitly rather than inferred, because the sign is not guessable:
 * RAISING minModelProb tightens (fewer entries qualify) while RAISING
 * dailyCapSol loosens (more money may move). Getting this backwards would
 * label a loosening as safe-to-apply-instantly, which is the one mistake the
 * governance rule exists to prevent.
 */
const LOWER_IS_TIGHTER: Record<string, boolean> = {
  perTradeCapSol: true,
  dailyCapSol: true,
  maxOpenPositions: true,
  "entry.maxTokenAgeMinutes": true,
  "entry.maxChangeH1Pct": true,
  "entry.maxEntryImpactPct": true,
  "entry.maxDriftSinceFirstSightPct": true,
  "entry.maxDeployerPriorMints": true,
  "entry.maxDeployerPriorRugs": true,
  "exit.stopLossPct": true,
  "exit.timeStopMinutes": true,
  "breaker.consecutiveStopLimit": true,
  // Raising these RESTRICTS: a higher floor admits fewer trades.
  "entry.minLiquidityUsd": false,
  "entry.minTokenAgeMinutes": false,
  "entry.minObservedTicks": false,
  "entry.minModelProb": false,
  "breaker.cooldownMinutes": false,
}

function tightensBy(path: string, from: unknown, to: unknown): boolean | null {
  if (path === "entry.allowedOrigins" && Array.isArray(from) && Array.isArray(to)) {
    // Fewer permitted sources is tighter; adding one is a loosening.
    if (to.length < from.length) return true
    if (to.length > from.length) return false
    return null
  }
  if (path === "entry.minVerdict") {
    // "clear" is stricter than "caution".
    if (from === "caution" && to === "clear") return true
    if (from === "clear" && to === "caution") return false
    return null
  }
  const lowerTighter = LOWER_IS_TIGHTER[path]
  if (lowerTighter === undefined) return null
  // A nullable gate, where null means the gate is not applied at all.
  // Arming one always TIGHTENS and disarming always LOOSENS, whichever way
  // the number itself runs, because the change is the existence of a refusal
  // rather than its level. Without this, disarming a gate also disarms the
  // governance label on re-arming it: `tightensBy` fell through the
  // number check and returned null, and an operator reviewing the proposal
  // to turn the gate back on would be shown "unclassified" for the one
  // change the tighten-instant rule most obviously covers.
  if (from === null && typeof to === "number") return true
  if (typeof from === "number" && to === null) return false
  if (typeof from !== "number" || typeof to !== "number") return null
  return lowerTighter ? to < from : to > from
}

/**
 * Validate a set of proposed changes against the policy actually running.
 *
 * Returns a structured verdict rather than throwing, because the caller is an
 * agent whose output a human will read: a refusal with reasons is information,
 * an exception is a dead end.
 */
export function validateProposal(
  current: unknown,
  changes: ProposedChange[],
): ProposalResult {
  const errors: string[] = []
  const validated: ValidatedChange[] = []

  if (changes.length === 0) {
    return { ok: false, changes: [], errors: ["proposal contains no changes"], entirelyNoop: false }
  }

  const seen = new Set<string>()
  for (const c of changes) {
    if (!(PROPOSABLE as readonly string[]).includes(c.path)) {
      errors.push(`${c.path} is not a proposable field`)
      continue
    }
    if (seen.has(c.path)) {
      errors.push(`${c.path} appears more than once`)
      continue
    }
    seen.add(c.path)

    const from = readPath(current, c.path)
    if (from === undefined) {
      errors.push(`${c.path} does not exist on the current policy`)
      continue
    }
    // Type must match. An agent proposing a string for a numeric cap is a
    // malformed proposal, not a policy question.
    //
    // Except for gates that are genuinely nullable, where null is a VALUE
    // meaning "not applied" rather than a missing one. The check used to
    // accept null -> number but refuse number -> null, so the model gate could
    // be proposed armed and never proposed disarmed. Found 2026-08-11 while
    // disarming it by hand: the Analyst could not have proposed the change
    // the operator was making, and would have reported a type error rather
    // than a policy disagreement.
    const sameType = Array.isArray(from)
      ? Array.isArray(c.to)
      : typeof from === typeof c.to
        || (from === null && c.to !== undefined)
        || (c.to === null && (NULLABLE as readonly string[]).includes(c.path))
    if (!sameType) {
      errors.push(`${c.path} expects ${Array.isArray(from) ? "an array" : typeof from}`)
      continue
    }
    if (typeof c.to === "number" && (!Number.isFinite(c.to) || c.to < 0)) {
      errors.push(`${c.path} must be a finite non-negative number`)
      continue
    }

    validated.push({
      path: c.path,
      from,
      to: c.to,
      noop: sameValue(from, c.to),
      tightens: tightensBy(c.path, from, c.to),
    })
  }

  const entirelyNoop = validated.length > 0 && validated.every((v) => v.noop)
  if (entirelyNoop) {
    errors.push(
      "every proposed change is already in effect: the running policy already does this",
    )
  }
  return { ok: errors.length === 0, changes: validated, errors, entirelyNoop }
}

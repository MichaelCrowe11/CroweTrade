/**
 * Workflow spec validation, dependency-free and tested.
 *
 * A workflow is a model-authored program the operator keeps: named, linear,
 * explicit steps. NO templating between steps on purpose: Cortex's canvas
 * shipped two of its three launch bugs in ref interpolation, and its command
 * mode's open review finding is interpolated model output flowing into a
 * shell. Steps here are independent and literal; chaining happens inside a
 * step (pipes) where the operator can read it.
 *
 * Validation refuses rather than repairs, and says why, because the caller
 * is a model that can fix its input when told what was wrong.
 */

export type WfStep =
  | { kind: "command"; command: string }
  | { kind: "python"; code: string }
  | { kind: "panels"; rows: string[][] }
  | { kind: "notebook"; path: string }

export interface WorkflowSpec {
  name: string
  description: string
  steps: WfStep[]
}

export type WfValidation = { ok: true; workflow: WorkflowSpec } | { ok: false; reason: string }

const PANEL_TYPES = new Set(["scan", "chart", "gates", "book", "calibration", "browser"])
const MAX_STEPS = 12
const MAX_TEXT = 4000

export function validateWorkflow(input: unknown): WfValidation {
  const raw = input as { name?: unknown; description?: unknown; steps?: unknown } | null
  if (!raw || typeof raw !== "object") return { ok: false, reason: "workflow must be an object" }

  const name = typeof raw.name === "string" ? raw.name.trim().slice(0, 48) : ""
  if (!name) return { ok: false, reason: "name is required" }
  const description =
    typeof raw.description === "string" ? raw.description.trim().slice(0, 200) : ""

  if (!Array.isArray(raw.steps) || raw.steps.length === 0) {
    return { ok: false, reason: "steps must be a non-empty array" }
  }
  if (raw.steps.length > MAX_STEPS) {
    return { ok: false, reason: `at most ${MAX_STEPS} steps` }
  }

  const steps: WfStep[] = []
  for (const s of raw.steps as { kind?: unknown; command?: unknown; code?: unknown; rows?: unknown }[]) {
    if (s?.kind === "command" && typeof s.command === "string" && s.command.trim()) {
      if (s.command.length > MAX_TEXT) return { ok: false, reason: "a command step is too long" }
      if (/(^|\s)sudo(\s|$)/.test(s.command)) {
        return { ok: false, reason: "sudo is not available to workflows" }
      }
      steps.push({ kind: "command", command: s.command })
    } else if (s?.kind === "python" && typeof s.code === "string" && s.code.trim()) {
      if (s.code.length > MAX_TEXT) return { ok: false, reason: "a python step is too long" }
      steps.push({ kind: "python", code: s.code })
    } else if (s?.kind === "notebook") {
      // A bare .ipynb name only: the executor resolves it under the app's own
      // notebooks directory, so a saved workflow can never be talked into
      // executing a notebook from anywhere else on disk.
      const p = (s as { path?: unknown }).path
      if (typeof p !== "string" || !/^[A-Za-z0-9][A-Za-z0-9 _.-]*\.ipynb$/.test(p) || p.includes("..")) {
        return { ok: false, reason: "notebook path must be a bare name ending in .ipynb" }
      }
      steps.push({ kind: "notebook", path: p })
    } else if (s?.kind === "panels" && Array.isArray(s.rows)) {
      const rows = (s.rows as unknown[][])
        .map((row) => (Array.isArray(row) ? row.filter((t) => PANEL_TYPES.has(String(t))) : []))
        .filter((row) => row.length > 0)
        .map((row) => row.map(String))
      if (rows.length === 0) return { ok: false, reason: "a panels step named no known panel types" }
      steps.push({ kind: "panels", rows })
    } else {
      return { ok: false, reason: `unknown or empty step kind: ${String(s?.kind)}` }
    }
  }

  return { ok: true, workflow: { name, description, steps } }
}

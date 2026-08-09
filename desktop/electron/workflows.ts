import * as fs from "node:fs"
import * as path from "node:path"
import { app } from "electron"
import { validateWorkflow, type WorkflowSpec } from "./wfspec"
import { runCommand, runPython, type Emit } from "./exec"

/**
 * The workflow shelf: model-authored programs the operator keeps.
 *
 * File-backed in userData as plain JSON so the collection survives updates
 * and can be read or edited by hand. Saving is upsert-by-name (models
 * iterate); running replays the steps through the SAME visible execution
 * primitives the Orchestrator uses live, and a failed step stops the run
 * rather than plowing on, because a workflow that half-ran and said nothing
 * is the silent-engine problem all over again.
 */

export interface SavedWorkflow extends WorkflowSpec {
  id: string
  author: string
  createdAt: number
  updatedAt: number
  lastRunAt?: number
  lastResult?: string
}

function shelfPath(): string {
  return path.join(app.getPath("userData"), "workflows.json")
}

export function listWorkflows(): SavedWorkflow[] {
  try {
    const raw = JSON.parse(fs.readFileSync(shelfPath(), "utf8")) as unknown
    return Array.isArray(raw) ? (raw as SavedWorkflow[]) : []
  } catch {
    return []
  }
}

function persist(list: SavedWorkflow[]): void {
  fs.writeFileSync(shelfPath(), JSON.stringify(list, null, 2))
}

export function saveWorkflow(
  input: unknown,
  author: string,
): { ok: true; workflow: SavedWorkflow } | { ok: false; reason: string } {
  const v = validateWorkflow(input)
  if (!v.ok) return v
  const id = v.workflow.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  if (!id) return { ok: false, reason: "name reduces to an empty id" }
  const list = listWorkflows()
  const existing = list.find((w) => w.id === id)
  const now = Date.now()
  const saved: SavedWorkflow = existing
    ? { ...existing, ...v.workflow, updatedAt: now }
    : { ...v.workflow, id, author, createdAt: now, updatedAt: now }
  persist([...list.filter((w) => w.id !== id), saved])
  return { ok: true, workflow: saved }
}

export function deleteWorkflow(id: string): void {
  persist(listWorkflows().filter((w) => w.id !== id))
}

function recordRun(id: string, result: string): void {
  persist(
    listWorkflows().map((w) =>
      w.id === id ? { ...w, lastRunAt: Date.now(), lastResult: result } : w,
    ),
  )
}

/** Find by id or by name, so models can say run_workflow("engine health"). */
export function resolveWorkflow(ref: string): SavedWorkflow | undefined {
  const list = listWorkflows()
  const slug = ref
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return list.find((w) => w.id === ref) ?? list.find((w) => w.id === slug)
}

export async function runWorkflow(ref: string, emit: Emit): Promise<string> {
  const wf = resolveWorkflow(ref)
  if (!wf) return `workflow not found: ${ref}`
  emit({ kind: "term", text: `== workflow ${wf.name}: ${wf.steps.length} step(s) ==\n` })
  let result = "ok"
  for (let i = 0; i < wf.steps.length; i++) {
    const step = wf.steps[i]
    if (!step) continue
    emit({ kind: "wf-step", id: wf.id, step: i + 1, of: wf.steps.length })
    if (step.kind === "command") {
      const r = await runCommand(step.command, undefined, emit)
      if (!r.startsWith("exit 0")) {
        result = `failed at step ${i + 1}`
        break
      }
    } else if (step.kind === "python") {
      const r = await runPython(step.code, emit)
      if (!r.startsWith("exit 0")) {
        result = `failed at step ${i + 1}`
        break
      }
    } else {
      emit({ kind: "panels", action: "arrange", rows: step.rows })
    }
  }
  recordRun(wf.id, result)
  emit({ kind: "wf-done", id: wf.id, result })
  emit({ kind: "wf-changed" })
  return result
}

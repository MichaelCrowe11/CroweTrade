import * as fs from "node:fs"
import * as path from "node:path"
import { app } from "electron"
import { buildNotebook } from "./nbdoc"
import { runProgram, type Emit } from "./exec"

/**
 * The notebook runtime: model-authored research that leaves an artifact.
 *
 * Notebooks live under the app's own data directory and execute with the
 * research venv's kernel (provisioned by scripts/setup, checked here, and
 * reported honestly when absent rather than silently falling back to a bare
 * python that lacks the libraries). Execution writes outputs back into the
 * .ipynb, so the file IS the research record; the run also streams into the
 * visible terminal lane like everything else the machine does.
 */

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9 _.-]*\.ipynb$/

function notebooksDir(): string {
  const dir = path.join(app.getPath("userData"), "notebooks")
  fs.mkdirSync(dir, { recursive: true })
  return dir
}

function kernelPython(): string | null {
  const py = path.join(app.getPath("userData"), "research-venv/bin/python")
  return fs.existsSync(py) ? py : null
}

function runnerScript(): string {
  // Packaged: the runner ships as an extraResource beside the asar, because
  // python cannot read a file that lives inside one. isPackaged is legitimate
  // here; it selects a resource location, not a dev server.
  return app.isPackaged
    ? path.join(process.resourcesPath, "scripts/nb_runner.py")
    : path.join(__dirname, "../scripts/nb_runner.py")
}

function resolveName(name: string): string | null {
  const withExt = name.endsWith(".ipynb") ? name : `${name}.ipynb`
  if (!NAME_RE.test(withExt) || withExt.includes("..")) return null
  return withExt
}

export function saveNotebook(
  name: unknown,
  cells: unknown,
): { ok: true; file: string } | { ok: false; reason: string } {
  if (typeof name !== "string" || !name.trim()) return { ok: false, reason: "name is required" }
  const file = resolveName(name.trim())
  if (!file) return { ok: false, reason: "name must be a bare notebook name, no paths" }
  if (!Array.isArray(cells) || !cells.every((c) => typeof c === "string")) {
    return { ok: false, reason: "cells must be an array of python source strings" }
  }
  let doc: string
  try {
    doc = buildNotebook(cells as string[])
  } catch (e) {
    return { ok: false, reason: e instanceof Error ? e.message : String(e) }
  }
  fs.writeFileSync(path.join(notebooksDir(), file), doc)
  return { ok: true, file }
}

export function listNotebooks(): { file: string; modifiedAt: number }[] {
  return fs
    .readdirSync(notebooksDir())
    .filter((f) => f.endsWith(".ipynb"))
    .map((f) => ({
      file: f,
      modifiedAt: fs.statSync(path.join(notebooksDir(), f)).mtimeMs,
    }))
}

export async function runNotebook(name: string, emit: Emit): Promise<string> {
  const file = resolveName(name)
  if (!file) return "refused: notebook name must be a bare name ending in .ipynb"
  const full = path.join(notebooksDir(), file)
  if (!fs.existsSync(full)) return `notebook not found: ${file}`
  const py = kernelPython()
  if (!py) {
    const msg =
      "research kernel not provisioned; run scripts/setup-research.sh and try again"
    emit({ kind: "term", text: `${msg}\n` })
    return msg
  }
  return runProgram(py, [runnerScript(), full], emit, `[notebook] ${file}`)
}

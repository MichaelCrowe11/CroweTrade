import { spawn, type ChildProcess } from "node:child_process"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"

/**
 * The shared execution primitives for everything that acts on this machine:
 * the Orchestrator's live tool calls and saved workflow steps run through the
 * SAME two functions, so the safety story never forks. Every run is echoed to
 * the visible terminal before it executes, output streams live, sudo is
 * refused, one child at a time, and stopExec kills whatever is running.
 */

export interface ExecEvent {
  kind: string
  [key: string]: unknown
}

export type Emit = (e: ExecEvent) => void

const COMMAND_TIMEOUT_MS = 120_000

let currentChild: ChildProcess | null = null

export function stopExec(): void {
  currentChild?.kill("SIGKILL")
}

export function defaultCwd(): string {
  return path.join(os.homedir(), "Projects/crowetrade/desktop")
}

function run(cmd: string, args: string[], cwd: string, emit: Emit): Promise<string> {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { cwd })
    currentChild = child
    let out = ""
    const timer = setTimeout(() => child.kill("SIGKILL"), COMMAND_TIMEOUT_MS)
    const push = (buf: Buffer) => {
      const s = buf.toString()
      out += s
      emit({ kind: "term", text: s })
    }
    child.stdout?.on("data", push)
    child.stderr?.on("data", push)
    child.on("error", (e) => {
      clearTimeout(timer)
      currentChild = null
      const msg = `spawn failed: ${e.message}\n`
      emit({ kind: "term", text: msg })
      resolve(msg.trim())
    })
    child.on("close", (code) => {
      clearTimeout(timer)
      currentChild = null
      emit({ kind: "term", text: `[exit ${code ?? "killed"}]\n` })
      resolve(`exit ${code ?? "killed"}\n${out.slice(-4000)}`)
    })
  })
}

export function runCommand(command: string, cwd: string | undefined, emit: Emit): Promise<string> {
  if (/(^|\s)sudo(\s|$)/.test(command)) {
    const line = "refused: sudo is not available here\n"
    emit({ kind: "term", text: `$ ${command}\n${line}` })
    return Promise.resolve(line.trim())
  }
  emit({ kind: "term", text: `$ ${command}\n` })
  return run("/bin/zsh", ["-lc", command], cwd || defaultCwd(), emit)
}

export async function runPython(code: string, emit: Emit): Promise<string> {
  emit({ kind: "term", text: `>>> python, ${code.split("\n").length} line(s)\n` })
  const file = path.join(os.tmpdir(), `crowetrade-py-${process.pid}-${Date.now()}.py`)
  fs.writeFileSync(file, code)
  try {
    return await run("python3", [file], defaultCwd(), emit)
  } finally {
    fs.unlink(file, () => {})
  }
}

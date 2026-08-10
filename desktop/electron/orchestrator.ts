import { llm, streamCompletion } from "./engine"
import { runCommand, runPython, stopExec, type Emit as ExecEmit } from "./exec"
import { saveWorkflow, listWorkflows, runWorkflow } from "./workflows"
import { saveNotebook, listNotebooks, runNotebook } from "./notebook"

/**
 * The Orchestrator: an agent harness that runs the terminal, visibly.
 *
 * It reaches its model through the engine's inference passthrough, exactly as
 * the Analyst does, so this app holds no model credential. Its CONTRACT is the
 * opposite though: where the Analyst is read-only by construction, the
 * Orchestrator ACTS. It runs shell commands on this machine and rearranges
 * the workspace. The safety model is visibility
 * plus a hand on the cord: every command it runs is echoed to the terminal
 * pane BEFORE it executes, all output streams live, and stop() kills the
 * loop and whatever child process is running. sudo is refused outright.
 *
 * Tool results flow back through the chat-completions function-call protocol:
 * stream a round, assemble tool_calls from their streamed fragments, execute
 * them in order, append each result as a tool message, repeat until the model
 * answers in prose or the round budget runs out.
 */

const MAX_ROUNDS = 12
const COMMAND_TIMEOUT_MS = 120_000
const PANEL_TYPES = ["scan", "chart", "gates", "book", "calibration", "browser"] as const

const INSTRUCTIONS = `You are the CroweTrade Orchestrator, the operator-side agent of the CroweTrade
terminal. You run on the operator's own machine with their own shell, and every
command you run is displayed live in the terminal pane beside your transcript.

You can: run shell commands (run_command), and manage the workspace
(open_panel, close_panel, arrange_layout, reset_layout). Panel types:
scan, chart, gates, book, calibration, browser.

Rules, non-negotiable:
- Narrate briefly what you are doing and why as you work. Plain prose, no
  emojis, no em dashes.
- Never invent numbers about the engine or the market. If asked about the
  book, read the live engine (curl the public read API at
  https://crowetrade-engine.yellow-block-3adc.workers.dev/api/positions) or
  open the relevant panel, and quote what came back.
- You have NO trading authority. Kill, veto, and policy changes need a bearer
  token you must not read, and you never send transactions. If asked to
  trade, decline and say why.
- Do not read or print secrets (.env files, keys, tokens) unless the operator
  explicitly names the file and asks.
- Prefer small, inspectable commands over long compound ones. Never sudo.
- When the operator asks for a workspace ("show me the book and the browser"),
  use arrange_layout with a sensible row split rather than opening panels one
  at a time.
- run_python executes a python3 script for analysis and research; print what
  matters, the operator watches the terminal.
- WORKFLOWS are saved, named, linear programs the operator keeps on the
  Workflows panel. Steps are literal and independent: kind "command"
  {command}, kind "python" {code}, kind "panels" {rows}. There is NO
  templating between steps; chain inside a step with pipes if needed. When
  the operator asks to keep, save, or reuse a procedure, save_workflow it,
  then confirm what you saved. run_workflow replays one by name, visibly.
- NOTEBOOKS are how research is kept: save_notebook writes code cells to a
  named .ipynb, run_notebook executes it with the research kernel (pandas
  available) and writes the outputs back into the file, so the notebook IS
  the research record. Prefer a notebook over run_python whenever the
  analysis is worth keeping or repeating; a workflow step
  {"kind":"notebook","path":"name.ipynb"} replays it.`

interface FnTool {
  type: "function"
  name: string
  description: string
  parameters: Record<string, unknown>
}

const TOOLS: FnTool[] = [
  {
    type: "function",
    name: "run_command",
    description:
      "Run a shell command on the operator's machine (zsh -lc). Output streams to the visible terminal; you receive the exit code and the tail of the output. 120s timeout. No sudo.",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "The command line to run." },
        cwd: {
          type: "string",
          description: "Working directory. Defaults to the CroweTrade desktop repo.",
        },
      },
      required: ["command"],
    },
  },
  {
    type: "function",
    name: "arrange_layout",
    description:
      "Replace the whole workspace layout. rows is an array of rows, each an array of panel types; panels sharing a row sit side by side, rows stack. Existing panels of a type are reused with their state.",
    parameters: {
      type: "object",
      properties: {
        rows: {
          type: "array",
          items: { type: "array", items: { type: "string", enum: [...PANEL_TYPES] } },
        },
      },
      required: ["rows"],
    },
  },
  {
    type: "function",
    name: "open_panel",
    description: "Open one panel (or focus it if it is single-instance and already open).",
    parameters: {
      type: "object",
      properties: { type: { type: "string", enum: [...PANEL_TYPES] } },
      required: ["type"],
    },
  },
  {
    type: "function",
    name: "close_panel",
    description: "Close the first open panel of the given type.",
    parameters: {
      type: "object",
      properties: { type: { type: "string", enum: [...PANEL_TYPES] } },
      required: ["type"],
    },
  },
  {
    type: "function",
    name: "reset_layout",
    description: "Return the workspace to the default scan | chart | gates layout.",
    parameters: { type: "object", properties: {} },
  },
  {
    type: "function",
    name: "run_python",
    description:
      "Run a python3 script on the operator's machine for analysis or research. Output streams to the visible terminal; you receive the exit code and output tail. 120s timeout.",
    parameters: {
      type: "object",
      properties: { code: { type: "string", description: "The python source to run." } },
      required: ["code"],
    },
  },
  {
    type: "function",
    name: "save_workflow",
    description:
      "Save a named workflow to the operator's Workflows panel. Steps are literal and run in order; a failing step stops the run. No templating between steps. Saving an existing name updates it.",
    parameters: {
      type: "object",
      properties: {
        name: { type: "string" },
        description: { type: "string" },
        steps: {
          type: "array",
          description:
            'Each step is one of: {"kind":"command","command":"..."}, {"kind":"python","code":"..."}, {"kind":"panels","rows":[["book","calibration"]]}.',
          items: { type: "object" },
        },
      },
      required: ["name", "steps"],
    },
  },
  {
    type: "function",
    name: "list_workflows",
    description: "List the saved workflows: names, descriptions, step counts, last results.",
    parameters: { type: "object", properties: {} },
  },
  {
    type: "function",
    name: "run_workflow",
    description: "Run a saved workflow by name. Its steps execute visibly in the terminal.",
    parameters: {
      type: "object",
      properties: { name: { type: "string" } },
      required: ["name"],
    },
  },
  {
    type: "function",
    name: "save_notebook",
    description:
      "Save python code cells as a named Jupyter notebook in the research library. Overwrites an existing name.",
    parameters: {
      type: "object",
      properties: {
        name: { type: "string", description: "Bare notebook name, e.g. liquidity-scan.ipynb" },
        cells: { type: "array", items: { type: "string" }, description: "Python source per cell." },
      },
      required: ["name", "cells"],
    },
  },
  {
    type: "function",
    name: "run_notebook",
    description:
      "Execute a saved notebook with the research kernel. Cell outputs stream to the terminal and are written back into the .ipynb.",
    parameters: {
      type: "object",
      properties: { name: { type: "string" } },
      required: ["name"],
    },
  },
  {
    type: "function",
    name: "list_notebooks",
    description: "List the saved research notebooks.",
    parameters: { type: "object", properties: {} },
  },
]

export interface OrchEvent {
  kind: string
  [key: string]: unknown
}

type Emit = ExecEmit

let stopped = false

export function stopOrchestrator(): void {
  stopped = true
  stopExec()
}

interface FnCall {
  call_id: string
  name: string
  arguments: string
}

/**
 * One streamed round through the engine's inference passthrough.
 *
 * The protocol changed with the move off Azure: the Responses API carried
 * conversation state server-side via previous_response_id, and chat
 * completions does not. So the loop below now owns the full message history
 * and resends it each round. That is more bytes on the wire and strictly
 * simpler to reason about, because the exact context the model sees lives in
 * one array this file controls rather than in a remote session.
 */
async function streamRound(
  messages: unknown[],
  emit: Emit,
): Promise<{ text: string; calls: FnCall[] }> {
  // The two APIs nest the tool schema differently: Responses put name and
  // parameters at the top level, chat completions wraps them in `function`.
  // Converting here keeps TOOLS above readable as one flat declaration.
  const tools = TOOLS.map((t) => ({
    type: "function" as const,
    function: { name: t.name, description: t.description, parameters: t.parameters },
  }))
  const res = await llm({ messages, tools, stream: true })
  const { text, toolCalls } = await streamCompletion(res, {
    onText: (t) => emit({ kind: "assistant_delta", text: t }),
    // Reasoning stays out of the transcript; the operator wants the plan and
    // the commands, not the model's working-out.
  })
  return {
    text,
    calls: toolCalls.map((c, i) => ({
      call_id: c.id ?? `call_${i}`,
      name: c.name,
      arguments: c.args || "{}",
    })),
  }
}

export async function runOrchestrator(goal: string, emit: Emit): Promise<{ text: string }> {
  stopped = false

  const messages: unknown[] = [
    { role: "system", content: INSTRUCTIONS },
    { role: "user", content: goal },
  ]
  let lastText = ""

  for (let round = 0; round < MAX_ROUNDS; round++) {
    if (stopped) return { text: lastText || "stopped" }
    emit({ kind: "round", n: round + 1 })
    const { text, calls } = await streamRound(messages, emit)
    if (text) lastText = text
    if (calls.length === 0 || stopped) {
      emit({ kind: "done" })
      return { text: lastText }
    }

    messages.push({
      role: "assistant",
      content: text,
      tool_calls: calls.map((c) => ({
        id: c.call_id,
        type: "function",
        function: { name: c.name, arguments: c.arguments },
      })),
    })

    for (const call of calls) {
      if (stopped) break
      let args: Record<string, unknown> = {}
      try {
        args = JSON.parse(call.arguments) as Record<string, unknown>
      } catch {
        args = {}
      }
      emit({ kind: "tool_call", name: call.name, args })

      let result = "done"
      if (call.name === "run_command" && typeof args["command"] === "string") {
        result = await runCommand(
          args["command"],
          typeof args["cwd"] === "string" ? args["cwd"] : undefined,
          emit,
        )
      } else if (call.name === "run_python" && typeof args["code"] === "string") {
        result = await runPython(args["code"], emit)
      } else if (call.name === "save_workflow") {
        const saved = saveWorkflow(args, "orchestrator")
        result = saved.ok
          ? `saved workflow '${saved.workflow.name}' with ${saved.workflow.steps.length} step(s)`
          : `refused: ${saved.reason}`
        emit({ kind: "wf-changed" })
      } else if (call.name === "list_workflows") {
        result = JSON.stringify(
          listWorkflows().map((w) => ({
            name: w.name,
            description: w.description,
            steps: w.steps.length,
            lastResult: w.lastResult ?? "never run",
          })),
        )
      } else if (call.name === "run_workflow" && typeof args["name"] === "string") {
        result = await runWorkflow(args["name"], emit)
      } else if (call.name === "save_notebook") {
        const saved = saveNotebook(args["name"], args["cells"])
        result = saved.ok ? `saved notebook ${saved.file}` : `refused: ${saved.reason}`
      } else if (call.name === "run_notebook" && typeof args["name"] === "string") {
        result = await runNotebook(args["name"], emit)
      } else if (call.name === "list_notebooks") {
        result = JSON.stringify(listNotebooks())
      } else if (call.name === "arrange_layout") {
        emit({ kind: "panels", action: "arrange", rows: args["rows"] })
      } else if (call.name === "open_panel") {
        emit({ kind: "panels", action: "open", panelType: args["type"] })
      } else if (call.name === "close_panel") {
        emit({ kind: "panels", action: "close", panelType: args["type"] })
      } else if (call.name === "reset_layout") {
        emit({ kind: "panels", action: "reset" })
      } else {
        result = `unknown tool ${call.name}`
      }
      messages.push({ role: "tool", tool_call_id: call.call_id, content: result })
    }
  }

  emit({ kind: "done" })
  return { text: lastText || "round budget exhausted" }
}

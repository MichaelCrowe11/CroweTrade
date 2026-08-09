import { execFileSync } from "node:child_process"
import { createSseParser } from "./sse"
import { runCommand, runPython, stopExec, type Emit as ExecEmit } from "./exec"
import { saveWorkflow, listWorkflows, runWorkflow } from "./workflows"

/**
 * The Orchestrator: an agent harness that runs the terminal, visibly.
 *
 * It holds the same credential rail as the Analyst (the operator's own az
 * login; nothing stored) but a different contract: where the Analyst is
 * read-only by construction, the Orchestrator ACTS. It runs shell commands on
 * this machine and rearranges the workspace. The safety model is visibility
 * plus a hand on the cord: every command it runs is echoed to the terminal
 * pane BEFORE it executes, all output streams live, and stop() kills the
 * loop and whatever child process is running. sudo is refused outright.
 *
 * Tool results flow back through the responses API's function-call protocol:
 * stream a round, collect completed function_call items, execute them in
 * order, submit function_call_output items against previous_response_id,
 * repeat until the model answers in prose or the round budget runs out.
 */

const FOUNDRY =
  "https://crowelm-prod-eastus2.services.ai.azure.com/api/projects/crowelm-foundry"
const MODEL = "gpt-5.6-sol"
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
  then confirm what you saved. run_workflow replays one by name, visibly.`

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

/** One streamed round. Returns the response id, its text, and any tool calls. */
async function streamRound(
  token: string,
  input: unknown,
  previousId: string | null,
  emit: Emit,
): Promise<{ id: string; text: string; calls: FnCall[] }> {
  const res = await fetch(`${FOUNDRY}/openai/v1/responses`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      instructions: INSTRUCTIONS,
      tools: TOOLS,
      input,
      stream: true,
      ...(previousId ? { previous_response_id: previousId } : {}),
    }),
  })
  if (!res.ok || !res.body) {
    throw new Error(`orchestrator ${res.status}: ${(await res.text()).slice(0, 200)}`)
  }

  const parser = createSseParser()
  const decoder = new TextDecoder()
  let id = ""
  let text = ""
  const calls: FnCall[] = []

  for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
    for (const evt of parser.push(decoder.decode(chunk, { stream: true }))) {
      if (evt.type === "response.output_text.delta") {
        const d = evt["delta"]
        if (typeof d === "string") {
          text += d
          emit({ kind: "assistant_delta", text: d })
        }
      } else if (evt.type === "response.output_item.done") {
        const item = evt["item"] as
          | { type?: string; call_id?: string; name?: string; arguments?: string }
          | undefined
        if (item?.type === "function_call" && item.call_id && item.name) {
          calls.push({ call_id: item.call_id, name: item.name, arguments: item.arguments ?? "{}" })
        }
      } else if (evt.type === "response.completed") {
        const r = evt["response"] as { id?: string } | undefined
        if (r?.id) id = r.id
      } else if (evt.type === "response.failed" || evt.type === "error") {
        throw new Error(`stream failed: ${JSON.stringify(evt).slice(0, 200)}`)
      }
    }
  }
  return { id, text, calls }
}

export async function runOrchestrator(goal: string, emit: Emit): Promise<{ text: string }> {
  stopped = false
  const token = execFileSync(
    "az",
    ["account", "get-access-token", "--resource", "https://ai.azure.com", "--query", "accessToken", "-o", "tsv"],
    { encoding: "utf8" },
  ).trim()

  let input: unknown = goal
  let previousId: string | null = null
  let lastText = ""

  for (let round = 0; round < MAX_ROUNDS; round++) {
    if (stopped) return { text: lastText || "stopped" }
    emit({ kind: "round", n: round + 1 })
    const { id, text, calls } = await streamRound(token, input, previousId, emit)
    if (text) lastText = text
    if (calls.length === 0 || stopped) {
      emit({ kind: "done" })
      return { text: lastText }
    }

    const outputs: { type: "function_call_output"; call_id: string; output: string }[] = []
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
      outputs.push({ type: "function_call_output", call_id: call.call_id, output: result })
    }

    input = outputs
    previousId = id
  }

  emit({ kind: "done" })
  return { text: lastText || "round budget exhausted" }
}

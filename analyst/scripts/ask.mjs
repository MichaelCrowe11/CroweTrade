#!/usr/bin/env node
/**
 * Ask the published CroweTrade Analyst a question and print its answer.
 *
 * Exists because driving the Foundry threads/runs API through chained shell
 * one-liners is quoting hell: the assistant payload contains a multi-line
 * markdown system prompt, and one bad capture turns every later curl into a
 * parse error that looks like an API failure but is not.
 *
 * Usage: AZ_TOKEN=$(az account get-access-token --resource https://ai.azure.com \
 *          --query accessToken -o tsv) node scripts/ask.mjs "your question"
 */

const EP = "https://crowelm-prod-eastus2.services.ai.azure.com/api/projects/crowelm-foundry"
const AID = process.env.CT_AGENT_ID ?? "asst_TNm2vmK8klo6LWtFRkm9vzmG"
const API = "api-version=v1"
const token = process.env.AZ_TOKEN
if (!token) {
  console.error("AZ_TOKEN missing")
  process.exit(1)
}

const H = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }

async function call(path, init = {}) {
  const res = await fetch(`${EP}${path}${path.includes("?") ? "&" : "?"}${API}`, { headers: H, ...init })
  const text = await res.text()
  try {
    return JSON.parse(text)
  } catch {
    throw new Error(`non-JSON from ${path}: ${res.status} ${text.slice(0, 200)}`)
  }
}

const question = process.argv.slice(2).join(" ") || "How are we doing? Give me the honest read."

const thread = await call("/threads", { method: "POST", body: "{}" })
if (!thread.id) throw new Error(`no thread: ${JSON.stringify(thread).slice(0, 200)}`)

await call(`/threads/${thread.id}/messages`, {
  method: "POST",
  body: JSON.stringify({ role: "user", content: question }),
})

let run = await call(`/threads/${thread.id}/runs`, {
  method: "POST",
  body: JSON.stringify({ assistant_id: AID }),
})
if (!run.id) throw new Error(`no run: ${JSON.stringify(run).slice(0, 300)}`)

const deadline = Date.now() + 240_000
while (!["completed", "failed", "expired", "cancelled", "requires_action"].includes(run.status)) {
  if (Date.now() > deadline) throw new Error("timed out waiting for run")
  await new Promise((r) => setTimeout(r, 4000))
  run = await call(`/threads/${thread.id}/runs/${run.id}`)
}

console.log(`run: ${run.status}`)
if (run.status !== "completed") {
  console.log("error:", JSON.stringify(run.last_error))
  // Surface which tools it tried, since a tool failure and a model failure look
  // identical from the run status alone.
  const steps = await call(`/threads/${thread.id}/runs/${run.id}/steps`)
  for (const s of steps.data ?? []) {
    console.log(` step ${s.type} ${s.status}`, JSON.stringify(s.last_error ?? ""))
  }
  process.exit(1)
}

const steps = await call(`/threads/${thread.id}/runs/${run.id}/steps`)
const tools = (steps.data ?? []).flatMap((s) =>
  (s.step_details?.tool_calls ?? []).map((t) => t.function?.name ?? t.type),
)
console.log(`tools called: ${tools.length ? tools.join(", ") : "NONE"}`)

const msgs = await call(`/threads/${thread.id}/messages`)
const answer = (msgs.data ?? []).find((m) => m.role === "assistant")
console.log("\n--- answer ---\n")
console.log((answer?.content ?? []).map((c) => c.text?.value).join("\n"))

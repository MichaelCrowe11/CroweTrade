#!/usr/bin/env node
/**
 * CroweTrade Analyst setup wizard.
 *
 * Checks prerequisites, discovers which models can actually serve the analyst
 * on this Foundry project, registers the agent, and verifies it end to end
 * against the live engine.
 *
 * WHY THIS EXISTS: the same setup was attempted by hand and blocked for hours
 * on a misleading error. The legacy Assistants API (/assistants) force-stamps
 * `temperature` and `top_p` onto every agent; the gpt-5.x family rejects both;
 * and with a tool attached that rejection surfaces as a generic
 * "server_error: Sorry, something went wrong" with EMPTY run steps. The fix is
 * to use the modern route (/openai/v1/responses), which has no sampling fields
 * and works with gpt-5.6-sol. This wizard encodes that so nobody rediscovers it.
 *
 *   node scripts/setup.mjs          # check, register, verify
 *   node scripts/setup.mjs --check  # prerequisites only, changes nothing
 */

import { execSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, "..")
const EP = "https://crowelm-prod-eastus2.services.ai.azure.com/api/projects/crowelm-foundry"
const ENGINE = "https://crowetrade-engine.yellow-block-3adc.workers.dev"
const AGENT = "crowetrade-analyst"
const checkOnly = process.argv.includes("--check")

const ok = (m) => console.log(`  ok    ${m}`)
const bad = (m) => console.log(`  FAIL  ${m}`)
const step = (m) => console.log(`\n${m}`)

function die(msg, fix) {
  bad(msg)
  if (fix) console.log(`\n  Fix: ${fix}`)
  process.exit(1)
}

step("1. Prerequisites")

let token
try {
  token = execSync(
    "az account get-access-token --resource https://ai.azure.com --query accessToken -o tsv",
    { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] },
  ).trim()
  ok("Azure token acquired")
} catch {
  die("no Azure token", "run `az login`, then re-run this wizard")
}

const H = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }

const health = await fetch(`${ENGINE}/api/health`).then((r) => r.json()).catch(() => null)
if (!health?.ok) die("engine unreachable", `check ${ENGINE}/api/health`)
ok(`engine reachable, mode=${health.mode}`)
if (health.mode !== "paper") {
  console.log("  NOTE  engine is NOT in paper mode; the analyst will say so prominently")
}

step("2. Which models can serve the analyst here")

// Probe candidates in preference order. gpt-4o is listed last on purpose: it
// works but its deployment capacity is small and it rate-limits under load.
const candidates = ["gpt-5.6-sol", "gpt-5.5", "gpt-4o"]
let model = null
for (const m of candidates) {
  const r = await fetch(`${EP}/openai/v1/responses`, {
    method: "POST",
    headers: H,
    body: JSON.stringify({ model: m, input: "ok" }),
  })
  if (r.ok) {
    ok(`${m} responds`)
    model ??= m
  } else {
    const e = await r.text()
    bad(`${m}: ${e.slice(0, 90)}`)
  }
}
if (!model) die("no model can serve responses", "check deployments in the Foundry portal")
console.log(`\n  Selected: ${model}`)

if (checkOnly) {
  console.log("\n--check: prerequisites verified, nothing changed.")
  process.exit(0)
}

step("3. Register the agent")

const spec = JSON.parse(readFileSync(join(ROOT, "config/engine-openapi.json"), "utf8"))
const instructions = readFileSync(join(ROOT, "agent/instructions.md"), "utf8")
const tools = [{
  type: "openapi",
  openapi: {
    name: "crowetrade_engine_read",
    description: "Read-only access to the live CroweTrade engine.",
    auth: { type: "anonymous" },
    spec,
  },
}]

const reg = await fetch(`${EP}/agents?api-version=v1`, {
  method: "POST",
  headers: H,
  body: JSON.stringify({
    name: AGENT,
    description: "Read-only conversational analysis surface for the CroweTrade engine.",
    definition: { kind: "prompt", model, instructions, tools },
  }),
})
if (reg.ok) ok(`agent "${AGENT}" registered on ${model}`)
else bad(`registration returned ${reg.status} (it may already exist; continuing)`)

step("4. Verify end to end against the live engine")

const res = await fetch(`${EP}/openai/v1/responses`, {
  method: "POST",
  headers: H,
  body: JSON.stringify({
    model,
    instructions,
    tools,
    input: "In one sentence: how many closed trades does the CURRENT policy cohort have?",
  }),
})
if (!res.ok) die(`verification failed: ${(await res.text()).slice(0, 200)}`)
const body = await res.json()
const called = (body.output ?? []).filter((o) => o.type !== "message" && o.type !== "reasoning")
if (called.length === 0) {
  bad("the model answered WITHOUT calling the engine — treat that answer as unfounded")
} else {
  ok(`engine consulted: ${called.map((c) => c.name ?? c.type).join(", ")}`)
}
const answer = (body.output ?? [])
  .filter((o) => o.type === "message")
  .flatMap((o) => (o.content ?? []).map((c) => c.text))
  .join("\n")
console.log(`\n  Answer: ${answer.slice(0, 300)}`)

step("Ready. How to use it")
console.log(`
  Ask a question from the terminal:

    export AZ_TOKEN=$(az account get-access-token --resource https://ai.azure.com --query accessToken -o tsv)
    node scripts/ask.mjs "why did it skip PEPE?"
    node scripts/ask.mjs "how are we doing?"
    node scripts/ask.mjs "which exit rule looks best, and what is the caveat?"

  It is READ-ONLY by construction: it holds three GET operations and no
  credentials. Kill, veto and policy changes need a bearer token it does not
  have, so a conversation cannot move the book.

  Model in use: ${model}
  Agent registered as: ${AGENT}
`)

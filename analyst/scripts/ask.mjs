#!/usr/bin/env node
/**
 * Ask the CroweTrade Analyst a question.
 *
 * Uses /openai/v1/responses with the agent's instructions and tools sent
 * inline. That route is the one that works; the legacy Assistants API
 * (/assistants) force-stamps `temperature` and `top_p`, the gpt-5.x family
 * rejects both, and with a tool attached the rejection surfaces as a generic
 * "server_error" with empty run steps -- which cost hours of chasing a phantom
 * tool bug. Run scripts/setup.mjs first; it verifies all of this.
 *
 * Usage:
 *   export AZ_TOKEN=$(az account get-access-token --resource https://ai.azure.com --query accessToken -o tsv)
 *   node scripts/ask.mjs "why did it skip PEPE?"
 */

import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..")
const EP = "https://crowelm-prod-eastus2.services.ai.azure.com/api/projects/crowelm-foundry"
const MODEL = process.env.CT_MODEL ?? "gpt-5.6-sol"

const token = process.env.AZ_TOKEN
if (!token) {
  console.error("AZ_TOKEN missing. Run:")
  console.error("  export AZ_TOKEN=$(az account get-access-token --resource https://ai.azure.com --query accessToken -o tsv)")
  process.exit(1)
}

const question = process.argv.slice(2).join(" ") || "How are we doing? Give me the honest read."

const res = await fetch(`${EP}/openai/v1/responses`, {
  method: "POST",
  headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
  body: JSON.stringify({
    model: MODEL,
    instructions: readFileSync(join(ROOT, "agent/instructions.md"), "utf8"),
    tools: [{
      type: "openapi",
      openapi: {
        name: "crowetrade_engine_read",
        description: "Read-only access to the live CroweTrade engine.",
        auth: { type: "anonymous" },
        spec: JSON.parse(readFileSync(join(ROOT, "config/engine-openapi.json"), "utf8")),
      },
    }],
    input: question,
  }),
})

if (!res.ok) {
  console.error(`HTTP ${res.status}`)
  console.error((await res.text()).slice(0, 600))
  process.exit(1)
}

const body = await res.json()

// An answer with no tool call is the model talking from its prompt rather than
// from the ledger. Surface that rather than letting it pass as grounded.
const calls = (body.output ?? [])
  .filter((o) => o.type !== "message" && o.type !== "reasoning")
  .map((o) => o.name ?? o.type)
console.log(calls.length ? `[engine consulted: ${calls.join(", ")}]\n` : "[WARNING: answered without consulting the engine]\n")

console.log(
  (body.output ?? [])
    .filter((o) => o.type === "message")
    .flatMap((o) => (o.content ?? []).map((c) => c.text))
    .join("\n"),
)

import { test } from "node:test"
import assert from "node:assert/strict"
import { validateWorkflow } from "./wfspec.ts"

const good = {
  name: "engine-health",
  description: "Curl the engine and report the breaker.",
  steps: [
    { kind: "command", command: "curl -s https://example.com/api | jq .budget.breaker" },
    { kind: "python", code: "print('ok')" },
    { kind: "panels", rows: [["book", "calibration"]] },
  ],
}

test("a well-formed workflow validates and round-trips", () => {
  const r = validateWorkflow(good)
  assert.ok(r.ok)
  assert.equal(r.ok && r.workflow.name, "engine-health")
  assert.equal(r.ok && r.workflow.steps.length, 3)
})

test("sudo anywhere in a command refuses the whole workflow, with a reason", () => {
  const r = validateWorkflow({
    ...good,
    steps: [{ kind: "command", command: "sudo rm -rf /tmp/x" }],
  })
  assert.ok(!r.ok)
  assert.match(!r.ok ? r.reason : "", /sudo/)
})

test("unknown step kinds and empty step lists are refused", () => {
  const bad = validateWorkflow({ ...good, steps: [{ kind: "teleport" }] })
  assert.ok(!bad.ok)
  const empty = validateWorkflow({ ...good, steps: [] })
  assert.ok(!empty.ok)
})

test("panels rows are filtered to known types and an all-invalid step refuses", () => {
  const r = validateWorkflow({
    ...good,
    steps: [{ kind: "panels", rows: [["book", "nonsense"], ["nope"]] }],
  })
  assert.ok(r.ok)
  assert.deepEqual(r.ok && r.workflow.steps[0], { kind: "panels", rows: [["book"]] })
  const allBad = validateWorkflow({ ...good, steps: [{ kind: "panels", rows: [["nope"]] }] })
  assert.ok(!allBad.ok)
})

test("names are trimmed and length-capped, oversize steps refuse", () => {
  const r = validateWorkflow({ ...good, name: "  engine health  " })
  assert.equal(r.ok && r.workflow.name, "engine health")
  const long = validateWorkflow({
    ...good,
    steps: [{ kind: "command", command: "x".repeat(5000) }],
  })
  assert.ok(!long.ok)
  const noName = validateWorkflow({ ...good, name: "   " })
  assert.ok(!noName.ok)
})

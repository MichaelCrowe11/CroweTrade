import { test } from "node:test"
import assert from "node:assert/strict"
import { planArrangement, type ArrangeInput } from "./arrange.ts"

const existing: ArrangeInput[] = [
  { id: "scan-1", type: "scan", focused: true, row: 0 },
  { id: "browser-7", type: "browser", focused: false, row: 0, payload: { url: "https://solscan.io" } },
]

test("existing panels are reused with their ids and payloads, rows reassigned", () => {
  const out = planArrangement(existing, [["browser"], ["scan"]])
  const browser = out.find((p) => p.type === "browser")
  const scan = out.find((p) => p.type === "scan")
  assert.equal(browser?.id, "browser-7")
  assert.deepEqual(browser?.payload, { url: "https://solscan.io" })
  assert.equal(browser?.row, 0)
  assert.equal(scan?.id, "scan-1")
  assert.equal(scan?.row, 1)
})

test("types not present get fresh panels with ids of their own", () => {
  const out = planArrangement(existing, [["scan", "chart"]])
  const chart = out.find((p) => p.type === "chart")
  assert.ok(chart)
  assert.match(chart?.id ?? "", /^chart-/)
  assert.equal(chart?.row, 0)
})

test("panels absent from the plan are dropped", () => {
  const out = planArrangement(existing, [["scan"]])
  assert.equal(out.length, 1)
  assert.equal(out[0]?.type, "scan")
})

test("exactly one panel ends up focused even when the focused one was dropped", () => {
  const out = planArrangement(
    [{ id: "gates-1", type: "gates", focused: true, row: 0 }, ...existing],
    [["browser", "chart"]],
  )
  assert.equal(out.filter((p) => p.focused).length, 1)
})

test("an empty plan yields an empty list for the caller to reject", () => {
  assert.deepEqual(planArrangement(existing, []), [])
})

test("duplicate types in one plan reuse existing panels first, then create", () => {
  const out = planArrangement(existing, [["browser", "browser"]])
  assert.equal(out.length, 2)
  assert.equal(out[0]?.id, "browser-7")
  assert.notEqual(out[1]?.id, "browser-7")
})

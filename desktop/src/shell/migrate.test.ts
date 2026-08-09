import { test } from "node:test"
import assert from "node:assert/strict"
import { migratePanelsV1, type StoredPanel } from "./migrate.ts"

/**
 * v0 persisted workspaces could contain "analyst" panels. In v1 the Analyst is
 * a drawer, not a workspace panel, so rehydrating an old workspace must drop
 * them or the workspace renders a headed panel with no body.
 */

const FALLBACK: StoredPanel[] = [{ id: "chart-default", type: "chart", focused: true, row: 0 }]

test("analyst panels are dropped, others kept in order", () => {
  const out = migratePanelsV1(
    [
      { id: "a", type: "scan", focused: false, row: 0 },
      { id: "b", type: "analyst", focused: false, row: 0 },
      { id: "c", type: "gates", focused: true, row: 1 },
    ],
    FALLBACK,
  )
  assert.deepEqual(
    out.map((p) => p.type),
    ["scan", "gates"],
  )
})

test("focus moves to the first survivor when the analyst held it", () => {
  const out = migratePanelsV1(
    [
      { id: "a", type: "analyst", focused: true, row: 0 },
      { id: "b", type: "scan", focused: false, row: 0 },
      { id: "c", type: "book", focused: false, row: 1 },
    ],
    FALLBACK,
  )
  assert.ok(out.every((p) => p.type !== "analyst"))
  assert.equal(out.filter((p) => p.focused).length, 1)
  assert.equal(out[0]?.focused, true)
  assert.equal(out[0]?.id, "b")
})

test("focus is untouched when the focused panel survives", () => {
  const out = migratePanelsV1(
    [
      { id: "a", type: "analyst", focused: false, row: 0 },
      { id: "b", type: "scan", focused: false, row: 0 },
      { id: "c", type: "book", focused: true, row: 1 },
    ],
    FALLBACK,
  )
  assert.ok(out.every((p) => p.type !== "analyst"))
  assert.equal(out.find((p) => p.id === "c")?.focused, true)
  assert.equal(out.find((p) => p.id === "b")?.focused, false)
})

test("a workspace that was only the analyst falls back to the default layout", () => {
  const out = migratePanelsV1([{ id: "a", type: "analyst", focused: true, row: 0 }], FALLBACK)
  assert.deepEqual(out, FALLBACK)
})

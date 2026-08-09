import { test } from "node:test"
import assert from "node:assert/strict"
import { buildNotebook } from "./nbdoc.ts"

test("cells become a valid nbformat 4 document with a python3 kernelspec", () => {
  const doc = JSON.parse(buildNotebook(["print('a')", "x = 1\nx"])) as {
    nbformat: number
    cells: { cell_type: string; source: string; outputs: unknown[] }[]
    metadata: { kernelspec?: { name?: string } }
  }
  assert.equal(doc.nbformat, 4)
  assert.equal(doc.cells.length, 2)
  assert.equal(doc.cells[0]?.cell_type, "code")
  assert.equal(doc.cells[0]?.source, "print('a')")
  assert.deepEqual(doc.cells[0]?.outputs, [])
  assert.equal(doc.metadata.kernelspec?.name, "python3")
})

test("blank cells are dropped and an all-blank notebook is refused", () => {
  const doc = JSON.parse(buildNotebook(["", "print('x')", "   "])) as { cells: unknown[] }
  assert.equal(doc.cells.length, 1)
  assert.throws(() => buildNotebook(["", "  "]), /at least one/)
})

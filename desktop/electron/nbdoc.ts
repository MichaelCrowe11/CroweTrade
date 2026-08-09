/**
 * Notebook document builder, dependency-free and tested: model-authored code
 * cells become a minimal valid nbformat 4 document that nbclient can execute
 * with the research kernel. Blank cells are dropped; a notebook with nothing
 * to run is refused rather than saved as an empty artifact.
 */

export function buildNotebook(cells: string[]): string {
  const code = cells.map((c) => c ?? "").filter((c) => c.trim().length > 0)
  if (code.length === 0) throw new Error("a notebook needs at least one non-empty cell")
  return JSON.stringify(
    {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {
        kernelspec: { name: "python3", display_name: "Python 3", language: "python" },
        language_info: { name: "python" },
      },
      cells: code.map((source) => ({
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source,
      })),
    },
    null,
    1,
  )
}

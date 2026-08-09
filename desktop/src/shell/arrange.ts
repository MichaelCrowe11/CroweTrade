/**
 * Layout planning for the orchestrator's arrange tool.
 *
 * Dependency-free per the strip-types testing constraint. Given the current
 * panels and a target layout (rows of panel types), produce the new panel
 * list: existing panels of a requested type are REUSED, ids and payloads
 * intact, so rearranging never reloads a browser page or loses panel state;
 * missing types get fresh panels; panels not in the plan are dropped; and
 * exactly one panel ends up focused, because a workspace where nothing is
 * focused strands every keyboard affordance.
 */

export interface ArrangeInput {
  id: string
  type: string
  focused: boolean
  row?: number
  payload?: Record<string, unknown>
}

let arrangeSeq = 0

export function planArrangement(existing: ArrangeInput[], rows: string[][]): ArrangeInput[] {
  const pool = new Map<string, ArrangeInput[]>()
  for (const p of existing) {
    pool.set(p.type, [...(pool.get(p.type) ?? []), p])
  }

  const out: ArrangeInput[] = []
  rows.forEach((rowTypes, rowIdx) => {
    for (const type of rowTypes) {
      const reusable = pool.get(type)?.shift()
      if (reusable) {
        out.push({ ...reusable, row: rowIdx, focused: false })
      } else {
        arrangeSeq++
        out.push({ id: `${type}-arr${arrangeSeq}`, type, focused: false, row: rowIdx })
      }
    }
  })

  if (out.length > 0) {
    const previouslyFocused = existing.find((p) => p.focused)
    const keep = previouslyFocused && out.find((p) => p.id === previouslyFocused.id)
    const target = keep ?? out[0]
    if (target) target.focused = true
  }
  return out
}

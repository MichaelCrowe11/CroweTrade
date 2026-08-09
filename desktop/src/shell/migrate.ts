/**
 * Persisted-workspace migrations.
 *
 * Dependency-free on purpose: node --test with type stripping cannot resolve
 * the `.js`-specifier imports the app modules use between themselves, so
 * test-critical logic lives in modules that import nothing. panels.ts wires
 * this into zustand's persist(migrate).
 */

export interface StoredPanel {
  id: string
  type: string
  focused: boolean
  row?: number
  payload?: Record<string, unknown>
}

/**
 * v0 -> v1: the Analyst left the workspace and became a drawer, so persisted
 * "analyst" panels must be dropped or the workspace rehydrates a headed panel
 * with no body. Focus follows the same rule the store uses on close: if the
 * dropped panel held it, the first survivor takes it.
 */
export function migratePanelsV1(panels: StoredPanel[], fallback: StoredPanel[]): StoredPanel[] {
  const remaining = panels.filter((p) => p.type !== "analyst")
  if (remaining.length === 0) return fallback
  if (!remaining.some((p) => p.focused) && remaining[0]) {
    remaining[0] = { ...remaining[0], focused: true }
  }
  return remaining
}

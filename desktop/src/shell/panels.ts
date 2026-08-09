/**
 * Panel store, ported from Crowe Cortex (src/csep/panels.ts).
 *
 * Cortex already solved the operator-workspace problem for this estate: a left
 * rail that pops panels into a row-based grid, where panels sharing a row split
 * horizontally and separate rows stack. It is the house pattern, it is proven
 * in a shipping app, and it is portable — zustand plus a motion token file, no
 * global store coupling and no backend.
 *
 * This is the Cortex model with a trading vocabulary: instead of terminal and
 * farm panels, CroweTrade opens a scan, a chart, safety gates, the paper book,
 * an analyst, and a browser for looking a token up on chain.
 */

import { create } from "zustand"
import { persist } from "zustand/middleware"

export type PanelType = "scan" | "chart" | "gates" | "book" | "analyst" | "browser"

export interface Panel {
  id: string
  type: PanelType
  focused: boolean
  /** Panel-type extras. The browser uses it for an initial URL. */
  payload?: Record<string, unknown>
  /** Vertical band. Same row = side by side; different rows = stacked. */
  row?: number
}

export const PANEL_LABELS: Record<PanelType, string> = {
  scan: "Scan",
  chart: "Chart",
  gates: "Gates",
  book: "Book",
  analyst: "Analyst",
  browser: "Browser",
}

/**
 * Single-instance types focus the existing panel instead of duplicating.
 *
 * The browser is deliberately multi-instance: comparing two mints side by side
 * on chain is the whole reason to have one. Everything else is a single view of
 * a single truth, and two of them would just disagree with each other.
 */
const SINGLE_INSTANCE: ReadonlySet<PanelType> = new Set<PanelType>([
  "scan",
  "chart",
  "gates",
  "book",
  "analyst",
])

interface PanelsState {
  panels: Panel[]
  addPanel: (type: PanelType, focus?: boolean, payload?: Record<string, unknown>) => void
  closePanel: (id: string) => void
  focusPanel: (id: string) => void
  splitFocusedDown: () => void
  closeAll: () => void
  reset: () => void
}

/** Smallest unused row, so closing a row and splitting again reuses the gap. */
function nextFreeRow(panels: Panel[]): number {
  const used = new Set(panels.map((p) => p.row ?? 0))
  let r = 0
  while (used.has(r)) r++
  return r
}

function makeId(type: PanelType): string {
  return `${type}-${Date.now()}-${Math.floor(Math.random() * 1000)}`
}

/** The layout an operator gets on first run: watch, read, judge. */
const DEFAULT_PANELS: Panel[] = [
  { id: "scan-default", type: "scan", focused: false, row: 0 },
  { id: "chart-default", type: "chart", focused: true, row: 0 },
  { id: "gates-default", type: "gates", focused: false, row: 0 },
]

export const usePanels = create<PanelsState>()(
  persist(
    (set, get) => ({
      panels: DEFAULT_PANELS,

      addPanel: (type, focus = true, payload) => {
        const panels = get().panels
        if (SINGLE_INSTANCE.has(type)) {
          const existing = panels.find((p) => p.type === type)
          if (existing) {
            if (focus) get().focusPanel(existing.id)
            return
          }
        }
        const id = makeId(type)
        // New panels join the focused panel's row rather than starting their
        // own: an operator adding the analyst wants it beside what they are
        // looking at, not stacked underneath it.
        const row = panels.find((p) => p.focused)?.row ?? 0
        set({
          panels: [
            ...panels.map((p) => ({ ...p, focused: focus ? false : p.focused })),
            { id, type, focused: focus, payload, row },
          ],
        })
      },

      closePanel: (id) => {
        const remaining = get().panels.filter((p) => p.id !== id)
        // Never leave an empty workspace: closing the last panel returns the
        // default layout rather than a void the operator has to rebuild.
        if (remaining.length === 0) {
          set({ panels: DEFAULT_PANELS })
          return
        }
        if (!remaining.some((p) => p.focused) && remaining[0]) {
          remaining[0] = { ...remaining[0], focused: true }
        }
        set({ panels: remaining })
      },

      focusPanel: (id) =>
        set({ panels: get().panels.map((p) => ({ ...p, focused: p.id === id })) }),

      splitFocusedDown: () => {
        const panels = get().panels
        const focused = panels.find((p) => p.focused)
        if (!focused) return
        const id = makeId(focused.type)
        set({
          panels: [
            ...panels.map((p) => ({ ...p, focused: false })),
            { ...focused, id, focused: true, row: nextFreeRow(panels) },
          ],
        })
      },

      closeAll: () => set({ panels: DEFAULT_PANELS }),
      reset: () => set({ panels: DEFAULT_PANELS }),
    }),
    { name: "crowetrade-panels" },
  ),
)

/** Panels grouped into rows, in row order. The workspace renders this. */
export function rowsOf(panels: Panel[]): Panel[][] {
  const byRow = new Map<number, Panel[]>()
  for (const p of panels) {
    const r = p.row ?? 0
    byRow.set(r, [...(byRow.get(r) ?? []), p])
  }
  return [...byRow.entries()].sort(([a], [b]) => a - b).map(([, v]) => v)
}

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
 * and a browser for looking a token up on chain. The Analyst is NOT a
 * workspace panel: it is a drawer docked at the left edge (state below),
 * following Cortex's conversation panel, because a conversation runs alongside
 * whatever is being read rather than competing with it for row space.
 */

import { create } from "zustand"
import { persist } from "zustand/middleware"
import { migratePanelsV1, type StoredPanel } from "./migrate.js"
import { planArrangement } from "./arrange.js"

export type PanelType =
  | "scan"
  | "chart"
  | "gates"
  | "book"
  | "calibration"
  | "workflows"
  | "browser"

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
  calibration: "Calibration",
  workflows: "Workflows",
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
  "calibration",
  "workflows",
])

interface PanelsState {
  panels: Panel[]
  /** The Analyst drawer, docked left of the workspace. Not a panel. */
  analystOpen: boolean
  /** The Orchestrator console, same dock. One side surface at a time,
   * Cortex's pattern: the rail switches the surface, switching collapses
   * the other. */
  orchestratorOpen: boolean
  addPanel: (type: PanelType, focus?: boolean, payload?: Record<string, unknown>) => void
  closePanel: (id: string) => void
  focusPanel: (id: string) => void
  splitFocusedDown: () => void
  /** Replace the whole layout: rows of types, existing panels reused. */
  arrange: (rows: PanelType[][]) => void
  toggleAnalyst: () => void
  closeAnalyst: () => void
  toggleOrchestrator: () => void
  closeOrchestrator: () => void
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
      analystOpen: false,
      orchestratorOpen: false,

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
        // own: an operator adding the book wants it beside what they are
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

      arrange: (rows) => {
        const valid = rows
          .map((row) => row.filter((t) => t in PANEL_LABELS))
          .filter((row) => row.length > 0)
        const planned = planArrangement(get().panels, valid) as Panel[]
        // An empty plan is a request for nothing; the workspace never goes
        // empty, so it falls back to the default layout instead.
        set({ panels: planned.length > 0 ? planned : DEFAULT_PANELS })
      },

      toggleAnalyst: () =>
        set({ analystOpen: !get().analystOpen, orchestratorOpen: false }),
      closeAnalyst: () => set({ analystOpen: false }),
      toggleOrchestrator: () =>
        set({ orchestratorOpen: !get().orchestratorOpen, analystOpen: false }),
      closeOrchestrator: () => set({ orchestratorOpen: false }),

      closeAll: () => set({ panels: DEFAULT_PANELS }),
      reset: () => set({ panels: DEFAULT_PANELS, analystOpen: false, orchestratorOpen: false }),
    }),
    {
      name: "crowetrade-panels",
      version: 1,
      // v0 workspaces could hold "analyst" panels; migrate.ts drops them and
      // hands focus over exactly the way closePanel would.
      migrate: (persisted, version) => {
        const state = persisted as { panels?: StoredPanel[]; analystOpen?: boolean }
        if (version < 1) {
          return {
            ...state,
            panels: migratePanelsV1(state.panels ?? [], DEFAULT_PANELS) as Panel[],
            // The operator had the analyst open as a panel; keep it available
            // in its new home rather than silently vanishing it.
            analystOpen: (state.panels ?? []).some((p) => p.type === "analyst"),
          }
        }
        return state
      },
    },
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

import { Fragment, useEffect, type ReactNode } from "react"
import { motion } from "motion/react"
import { Group, Panel as SizePanel, Separator } from "react-resizable-panels"
import { usePanels, rowsOf, PANEL_LABELS, type Panel } from "./panels.js"
import { DURATIONS, EASINGS, MAGNITUDES } from "./motion.js"

/**
 * The workspace: rows of panels, ported from Cortex's Workspace.
 *
 * Panels sharing a row split horizontally; rows stack. That is the whole
 * layout model, and it is deliberately not a drag-to-arrange tile engine: this
 * is an instrument, so the operator opens and closes readouts rather than
 * carpentering a workspace. Wave's tile layout was evaluated and rejected for
 * exactly that reason, and because it cannot be lifted without its store.
 *
 * Sizing is react-resizable-panels, the same Group/Panel/Separator shape
 * Cortex ships. One deliberate difference: Cortex renders a different tree for
 * the single-row case; here the outer vertical Group is permanent, because a
 * changed tree shape remounts every panel, and remounting a browser panel
 * destroys its native WebContentsView mid-read. Rows are also keyed by their
 * STORED row number, not their render index, for the same reason.
 *
 * Panel entry keeps the house rise-in, but exit animation was given up when
 * the library took ownership of the tree: AnimatePresence needs to keep an
 * exiting child rendered, and a Group re-laying out its survivors does not.
 */

/* Sensible opening proportions so a fresh row does not slice itself into
 * equal thirds: lists and lamp grids start narrower, readouts wider. Types
 * without an entry share the remainder equally. */
/* Width is attention. The scan list is the tape the operator reads
 * continuously and it was the NARROWEST panel at 26% while the chart took
 * whatever was left, so a single price got more width than the whole market.
 * Gates and book both carry dense labelled readouts that were wrapping. */
const DEFAULT_SIZES: Partial<Record<Panel["type"], string>> = {
  scan: "30%",
  chart: "26%",
  gates: "32%",
  book: "36%",
  calibration: "42%",
}

function SplitIcon() {
  // A panel with a rule through its middle: what splitting down produces.
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="5" width="16" height="14" rx="1" />
        <path d="M4 12h16" />
      </g>
    </svg>
  )
}

export function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
      <path
        d="M6 6l12 12M18 6L6 18"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function Workspace({ render }: { render: (panel: Panel) => ReactNode }) {
  const panels = usePanels((s) => s.panels)
  const closePanel = usePanels((s) => s.closePanel)
  const focusPanel = usePanels((s) => s.focusPanel)
  const splitFocusedDown = usePanels((s) => s.splitFocusedDown)
  const rows = rowsOf(panels)

  // ⌘D splits the focused panel into a new row below, same binding as Cortex.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey && !e.shiftKey && !e.altKey && (e.key === "d" || e.key === "D")) {
        e.preventDefault()
        splitFocusedDown()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [splitFocusedDown])

  const renderPanel = (panel: Panel) => (
    <SizePanel
      id={panel.id}
      minSize="15%"
      defaultSize={DEFAULT_SIZES[panel.type]}
      className="ws__panel"
    >
      <motion.div
        className={`ws__panelbox ${panel.focused ? "ws__panelbox--focused" : ""}`}
        // Clicking anywhere in a panel focuses it, so the next panel opened
        // lands beside the one being read.
        onMouseDown={() => focusPanel(panel.id)}
        initial={{ opacity: 0, y: MAGNITUDES.rise }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: DURATIONS.smooth, ease: EASINGS.snap }}
      >
        <header className="ws__head">
          <span className="ws__title">{PANEL_LABELS[panel.type]}</span>
          <span className="ws__acts">
            <button
              type="button"
              className="ws__act"
              onClick={(e) => {
                e.stopPropagation()
                focusPanel(panel.id)
                splitFocusedDown()
              }}
              title="Split down (⌘D)"
              aria-label={`Split ${PANEL_LABELS[panel.type]} into a new row`}
            >
              <SplitIcon />
            </button>
            <button
              type="button"
              className="ws__act ws__act--close"
              onClick={(e) => {
                e.stopPropagation()
                closePanel(panel.id)
              }}
              aria-label={`Close ${PANEL_LABELS[panel.type]}`}
            >
              <CloseIcon />
            </button>
          </span>
        </header>
        <div className="ws__body">{render(panel)}</div>
      </motion.div>
    </SizePanel>
  )

  return (
    <div className="ws">
      <Group orientation="vertical" className="ws__grid">
        {rows.map((row, rowIdx) => {
          const rowKey = `row-${row[0]?.row ?? rowIdx}`
          return (
            <Fragment key={rowKey}>
              <SizePanel id={rowKey} minSize="15%" className="ws__rowpanel">
                <Group orientation="horizontal" className="ws__grid">
                  {row.map((panel, i) => (
                    <Fragment key={panel.id}>
                      {renderPanel(panel)}
                      {i < row.length - 1 && <Separator className="ws__handle" />}
                    </Fragment>
                  ))}
                </Group>
              </SizePanel>
              {rowIdx < rows.length - 1 && (
                <Separator className="ws__handle ws__handle--row" />
              )}
            </Fragment>
          )
        })}
      </Group>
    </div>
  )
}

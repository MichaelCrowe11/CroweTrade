import type { ReactNode } from "react"
import { AnimatePresence, motion } from "motion/react"
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
 */

export function Workspace({ render }: { render: (panel: Panel) => ReactNode }) {
  const panels = usePanels((s) => s.panels)
  const closePanel = usePanels((s) => s.closePanel)
  const focusPanel = usePanels((s) => s.focusPanel)
  const rows = rowsOf(panels)

  return (
    <div className="ws">
      {rows.map((row, i) => (
        <div className="ws__row" key={i}>
          <AnimatePresence initial={false}>
            {row.map((panel) => (
              <motion.section
                key={panel.id}
                className={`ws__panel ${panel.focused ? "ws__panel--focused" : ""}`}
                // Clicking anywhere in a panel focuses it, so the next panel
                // opened lands beside the one being read.
                onMouseDown={() => focusPanel(panel.id)}
                initial={{ opacity: 0, y: MAGNITUDES.rise }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: MAGNITUDES.rise }}
                transition={{ duration: DURATIONS.smooth, ease: EASINGS.snap }}
                layout
              >
                <header className="ws__head">
                  <span className="ws__title">{PANEL_LABELS[panel.type]}</span>
                  <button
                    type="button"
                    className="ws__close"
                    onClick={(e) => {
                      e.stopPropagation()
                      closePanel(panel.id)
                    }}
                    aria-label={`Close ${PANEL_LABELS[panel.type]}`}
                  >
                    <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
                      <path
                        d="M6 6l12 12M18 6L6 18"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                      />
                    </svg>
                  </button>
                </header>
                <div className="ws__body">{render(panel)}</div>
              </motion.section>
            ))}
          </AnimatePresence>
        </div>
      ))}
    </div>
  )
}

import { motion } from "motion/react"
import { usePanels, PANEL_LABELS, type PanelType } from "./panels.js"
import { DURATIONS, EASINGS, MAGNITUDES } from "./motion.js"

/**
 * The left rail, ported in spirit from Cortex's LeftRail.
 *
 * Icons are drawn rather than imported: an icon font or sprite sheet arrives
 * with its own stroke weight and optical sizing, and the house language here is
 * a 1.6px round-cap stroke on a 24px box, which is what the desktop renderer
 * uses. Six shapes is less code than a dependency.
 *
 * The Analyst button is the odd one out: it toggles the drawer docked beside
 * this rail rather than opening a workspace panel, following Cortex's
 * conversation panel. Same button language, different destination.
 */

type RailKey = PanelType | "analyst"

const ICONS: Record<RailKey, JSX.Element> = {
  // Scan: a list.
  scan: (
    <>
      <path d="M4 7h16M4 12h16M4 17h10" />
    </>
  ),
  // Chart: a rising trace.
  chart: (
    <>
      <path d="M4 17l5-6 4 3 7-8" />
      <path d="M4 20h16" opacity="0.4" />
    </>
  ),
  // Gates: an annunciator lamp grid.
  gates: (
    <>
      <rect x="4" y="5" width="7" height="6" rx="1" />
      <rect x="13" y="5" width="7" height="6" rx="1" />
      <rect x="4" y="13" width="7" height="6" rx="1" />
      <rect x="13" y="13" width="7" height="6" rx="1" />
    </>
  ),
  // Book: a ledger.
  book: (
    <>
      <path d="M5 4h11a2 2 0 012 2v14H7a2 2 0 01-2-2z" />
      <path d="M9 9h7M9 13h5" opacity="0.5" />
    </>
  ),
  // Analyst: the assistant mark, a conversation.
  analyst: (
    <>
      <path d="M20 12a7 7 0 01-7 7H8l-4 3v-5a7 7 0 017-9h2a7 7 0 017 4z" />
    </>
  ),
  // Browser: a window with a chrome bar.
  browser: (
    <>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="M3 9h18" />
    </>
  ),
}

const LABELS: Record<RailKey, string> = { ...PANEL_LABELS, analyst: "Analyst" }

const ORDER: RailKey[] = ["scan", "chart", "gates", "book", "analyst", "browser"]

export function Rail() {
  const panels = usePanels((s) => s.panels)
  const analystOpen = usePanels((s) => s.analystOpen)
  const addPanel = usePanels((s) => s.addPanel)
  const toggleAnalyst = usePanels((s) => s.toggleAnalyst)
  const open = new Set<RailKey>(panels.map((p) => p.type))
  if (analystOpen) open.add("analyst")

  return (
    <nav className="rail" aria-label="Panels">
      {ORDER.map((key, i) => (
        <motion.button
          key={key}
          type="button"
          className={`rail__btn ${open.has(key) ? "rail__btn--open" : ""}`}
          onClick={() => (key === "analyst" ? toggleAnalyst() : addPanel(key))}
          title={LABELS[key]}
          aria-label={key === "analyst" ? "Toggle the Analyst drawer" : `Open ${LABELS[key]}`}
          aria-pressed={key === "analyst" ? analystOpen : undefined}
          initial={{ opacity: 0, x: -MAGNITUDES.slide }}
          animate={{ opacity: 1, x: 0 }}
          transition={{
            duration: DURATIONS.smooth,
            ease: EASINGS.snap,
            // Staggered so the rail assembles on launch rather than appearing.
            delay: i * 0.04,
          }}
        >
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <g
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {ICONS[key]}
            </g>
          </svg>
          <span className="rail__label">{LABELS[key]}</span>
        </motion.button>
      ))}
    </nav>
  )
}

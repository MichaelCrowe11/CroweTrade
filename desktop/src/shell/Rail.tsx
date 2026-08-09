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

type RailKey = PanelType | "analyst" | "orchestrator"

/* Every icon carries ONE gold accent: the element that is the point of the
 * surface (the lit lamp, the trace tip, the live cursor). The set stays
 * stroke-drawn in the house 1.6 weight; the accent is what makes it ours. */
const ICONS: Record<RailKey, JSX.Element> = {
  // Scan: a list, the selected row marked.
  scan: (
    <>
      <path d="M7 7h13M7 12h13M7 17h7" />
      <circle cx="4" cy="7" r="1.3" fill="var(--clm-gold)" stroke="none" />
    </>
  ),
  // Chart: a rising trace with a lit tip.
  chart: (
    <>
      <path d="M4 17l5-6 4 3 7-8" />
      <path d="M4 20h16" opacity="0.4" />
      <circle cx="20" cy="6" r="1.6" fill="var(--clm-gold)" stroke="none" />
    </>
  ),
  // Gates: an annunciator lamp grid, one lamp lit.
  gates: (
    <>
      <rect x="4" y="5" width="7" height="6" rx="1" />
      <rect x="13" y="5" width="7" height="6" rx="1" />
      <rect x="4" y="13" width="7" height="6" rx="1" />
      <rect x="13" y="13" width="7" height="6" rx="1" />
      <circle cx="7.5" cy="8" r="1.4" fill="var(--clm-gold)" stroke="none" />
    </>
  ),
  // Book: a ledger with its ribbon.
  book: (
    <>
      <path d="M5 4h11a2 2 0 012 2v14H7a2 2 0 01-2-2z" />
      <path d="M9 9h7M9 13h5" opacity="0.5" />
      <path d="M13.5 4v4.6l1.5-1.1 1.5 1.1V4" fill="var(--clm-gold)" stroke="none" />
    </>
  ),
  // Calibration: two measured bars; the marks carry the accent.
  calibration: (
    <>
      <path d="M4 8h11M4 16h11" />
      <path d="M19 5.5v5M15 13.5v5" stroke="var(--clm-gold)" />
    </>
  ),
  // Workflows: a saved program, input node wired to a lit output.
  workflows: (
    <>
      <rect x="4" y="4" width="6" height="5" rx="1" />
      <path d="M7 9v5a2 2 0 002 2h5" />
      <circle cx="17" cy="16" r="2.4" fill="var(--clm-gold)" stroke="none" />
    </>
  ),
  // Analyst: the assistant mark, a conversation with a live word in it.
  analyst: (
    <>
      <path d="M20 12a7 7 0 01-7 7H8l-4 3v-5a7 7 0 017-9h2a7 7 0 017 4z" />
      <circle cx="11.5" cy="12.5" r="1.4" fill="var(--clm-gold)" stroke="none" />
    </>
  ),
  // Orchestrator: a prompt with a live cursor.
  orchestrator: (
    <>
      <path d="M4 7l5 5-5 5" />
      <rect x="12" y="15.7" width="8" height="2.6" rx="1.3" fill="var(--clm-gold)" stroke="none" />
    </>
  ),
  // Browser: a window, the address dot live.
  browser: (
    <>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="M3 9h18" />
      <circle cx="6" cy="7" r="1.1" fill="var(--clm-gold)" stroke="none" />
    </>
  ),
}

const LABELS: Record<RailKey, string> = {
  ...PANEL_LABELS,
  analyst: "Analyst",
  orchestrator: "Orchestrator",
}

const ORDER: RailKey[] = [
  "orchestrator",
  "scan",
  "chart",
  "gates",
  "book",
  "calibration",
  "workflows",
  "analyst",
  "browser",
]

/** The two side surfaces share the dock; their buttons toggle, panels open. */
const DOCK: ReadonlySet<RailKey> = new Set<RailKey>(["analyst", "orchestrator"])

export function Rail() {
  const panels = usePanels((s) => s.panels)
  const analystOpen = usePanels((s) => s.analystOpen)
  const orchestratorOpen = usePanels((s) => s.orchestratorOpen)
  const addPanel = usePanels((s) => s.addPanel)
  const toggleAnalyst = usePanels((s) => s.toggleAnalyst)
  const toggleOrchestrator = usePanels((s) => s.toggleOrchestrator)
  const open = new Set<RailKey>(panels.map((p) => p.type))
  if (analystOpen) open.add("analyst")
  if (orchestratorOpen) open.add("orchestrator")

  return (
    <nav className="rail" aria-label="Panels">
      {ORDER.map((key, i) => (
        <motion.button
          key={key}
          type="button"
          className={`rail__btn ${open.has(key) ? "rail__btn--open" : ""}`}
          onClick={() =>
            key === "analyst"
              ? toggleAnalyst()
              : key === "orchestrator"
                ? toggleOrchestrator()
                : addPanel(key)
          }
          title={LABELS[key]}
          aria-label={DOCK.has(key) ? `Toggle ${LABELS[key]}` : `Open ${LABELS[key]}`}
          aria-pressed={
            key === "analyst" ? analystOpen : key === "orchestrator" ? orchestratorOpen : undefined
          }
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

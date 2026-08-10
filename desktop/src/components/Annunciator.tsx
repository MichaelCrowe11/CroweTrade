import type { GateResult } from "../safety/gates.js"

/**
 * The annunciator panel.
 *
 * Modelled on an aircraft annunciator rather than a data table, because the
 * underlying facts genuinely are a set of discrete named conditions and that is
 * what an annunciator is for. A table asks you to read six numbers and form a
 * judgment; a panel of lamps asks you to notice which one is lit.
 *
 * Three states, never two. Every competing tool collapses "we do not know" into
 * a blank cell or a zero, and a zero reads as a fact.
 */
export function Annunciator({ gates }: { gates: GateResult[] }) {
  return (
    <div className="annunciator" role="group" aria-label="Survivability gates">
      {gates.map((g) => (
        <div
          key={g.id}
          className={`lamp lamp--${g.state} lamp--${g.severity}`}
          // The visual state is colour plus glow, neither of which a screen
          // reader conveys, so the state is also spelled out in the label.
          aria-label={`${g.label}: ${g.state}, ${g.detail}`}
        >
          <span className="lamp__led" aria-hidden="true" />
          <span className="lamp__label">{g.label}</span>
          <span className="lamp__detail">{g.detail}</span>
          {/* The state named in words. Colour and shape already carry it, but
              neither answers "is that good?" for a reader who does not know
              that a revoked mint authority is the outcome you want. */}
          <span className="lamp__verdict" aria-hidden="true">
            {g.state === "pass" ? "OK" : g.state === "fail" ? "RISK" : "UNKNOWN"}
          </span>
        </div>
      ))}
    </div>
  )
}

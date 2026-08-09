import { useEffect, useState } from "react"
import { usePanels } from "./panels.js"

/**
 * The workflow shelf: model-authored programs the operator keeps.
 *
 * Cards, not a node graph, on purpose for v1: a CroweTrade workflow is
 * linear and literal (no templating between steps, per the Cortex canvas's
 * scar tissue), so the honest visualization is the step list itself. Running
 * one replays it through the visible terminal; the Orchestrator console
 * opens so the machine is watched, never trusted blind.
 */

interface WfCard {
  id: string
  name: string
  description: string
  author: string
  steps: { kind: string }[]
  updatedAt: number
  lastRunAt?: number
  lastResult?: string
}

const STEP_GLYPH: Record<string, string> = { command: "$", python: ">>>", panels: "[]" }

export function WorkflowsPanel() {
  const [items, setItems] = useState<WfCard[]>([])
  const [running, setRunning] = useState<Record<string, string>>({})
  const [confirming, setConfirming] = useState<string | null>(null)

  const refresh = () => {
    void window.crowetrade?.workflows?.list().then((l) => setItems(l as WfCard[]))
  }

  useEffect(() => {
    refresh()
    const off = window.crowetrade?.orchestrator?.onEvent?.((e) => {
      if (e["kind"] === "wf-changed") refresh()
      if (e["kind"] === "wf-step") {
        setRunning((r) => ({ ...r, [String(e["id"])]: `step ${e["step"]} of ${e["of"]}` }))
      }
      if (e["kind"] === "wf-done") {
        setRunning((r) => {
          const next = { ...r }
          delete next[String(e["id"])]
          return next
        })
      }
    })
    return () => off?.()
  }, [])

  const run = (id: string) => {
    // The machine must be watchable while it works.
    usePanels.setState({ orchestratorOpen: true, analystOpen: false })
    setRunning((r) => ({ ...r, [id]: "starting" }))
    void window.crowetrade?.workflows?.run(id)
  }

  if (items.length === 0) {
    return (
      <p className="empty">
        No workflows yet. Ask the Orchestrator to write a procedure and save it, and it lands
        here for reuse.
      </p>
    )
  }

  return (
    <div className="wfl">
      {items
        .slice()
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .map((w) => (
          <article key={w.id} className="wfl__card">
            <header className="wfl__head">
              <span className="wfl__name">{w.name}</span>
              <span className="wfl__meta mono">
                {w.author} · {w.steps.length} step{w.steps.length === 1 ? "" : "s"}
              </span>
            </header>
            {w.description && <p className="wfl__desc">{w.description}</p>}
            <div className="wfl__steps mono">
              {w.steps.map((s, i) => (
                <span key={i} className={`wfl__step wfl__step--${s.kind}`}>
                  {STEP_GLYPH[s.kind] ?? "?"} {s.kind}
                </span>
              ))}
            </div>
            <footer className="wfl__foot">
              <span className="wfl__status mono">
                {running[w.id]
                  ? running[w.id]
                  : w.lastResult
                    ? `last run: ${w.lastResult}`
                    : "never run"}
              </span>
              <span className="wfl__actions">
                <button
                  type="button"
                  className="wfl__run"
                  disabled={Boolean(running[w.id])}
                  onClick={() => run(w.id)}
                >
                  Run
                </button>
                {confirming === w.id ? (
                  <button
                    type="button"
                    className="wfl__delete wfl__delete--armed"
                    onClick={() => {
                      setConfirming(null)
                      void window.crowetrade?.workflows?.delete(w.id)
                    }}
                  >
                    Sure?
                  </button>
                ) : (
                  <button
                    type="button"
                    className="wfl__delete"
                    onClick={() => setConfirming(w.id)}
                  >
                    Delete
                  </button>
                )}
              </span>
            </footer>
          </article>
        ))}
    </div>
  )
}

import { useEffect, useRef, useState } from "react"
import { motion } from "motion/react"
import { usePanels, type PanelType, PANEL_LABELS } from "./panels.js"
import { AIAvatarSwirl, type SwirlState } from "./AIAvatarSwirl.js"
import { segmentInline } from "./markdown.js"
import { DURATIONS, EASINGS, MAGNITUDES } from "./motion.js"

/**
 * The Orchestrator console: the agent that runs the terminal, from a sheet
 * that pops out above the workspace (Cortex's pop-out surface idiom, docked
 * rather than floated because native browser views composite above the page).
 *
 * Two lanes: the conversation on the left, the machine on the right. Every
 * command the agent runs prints in the terminal lane BEFORE it executes and
 * its output streams live; the STOP control kills the loop and the running
 * process. Panel actions arrive as events and are applied to the same panels
 * store the rail uses, so the agent and the operator share one workspace
 * truth.
 */

interface OrchTurn {
  role: "you" | "orchestrator"
  text: string
  pending?: boolean
}

function AnswerText({ text }: { text: string }) {
  return (
    <>
      {segmentInline(text).map((seg, i) =>
        seg.kind === "strong" ? (
          <strong key={i}>{seg.text}</strong>
        ) : seg.kind === "code" ? (
          <code key={i} className="mono turn__code">
            {seg.text}
          </code>
        ) : (
          <span key={i}>{seg.text}</span>
        ),
      )}
    </>
  )
}

export function Orchestrator({ onCollapse }: { onCollapse: () => void }) {
  const [turns, setTurns] = useState<OrchTurn[]>([])
  const [draft, setDraft] = useState("")
  const [busy, setBusy] = useState(false)
  const [live, setLive] = useState("")
  const [term, setTerm] = useState("")
  const termRef = useRef<HTMLPreElement>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const off = window.crowetrade?.orchestrator?.onEvent?.((e) => {
      const kind = e["kind"]
      if (kind === "term" && typeof e["text"] === "string") {
        setTerm((t) => (t + e["text"]).slice(-30_000))
      } else if (kind === "assistant_delta" && typeof e["text"] === "string") {
        setLive((t) => t + (e["text"] as string))
      } else if (kind === "round") {
        // A new model round starts a fresh paragraph of narration.
        setLive((t) => (t ? `${t}\n\n` : t))
      } else if (kind === "tool_call") {
        const name = String(e["name"] ?? "tool")
        if (name !== "run_command") {
          setTerm((t) => `${t}[${name} ${JSON.stringify(e["args"] ?? {})}]\n`.slice(-30_000))
        }
      } else if (kind === "panels") {
        const store = usePanels.getState()
        const action = e["action"]
        if (action === "arrange" && Array.isArray(e["rows"])) {
          store.arrange(e["rows"] as PanelType[][])
        } else if (action === "open" && typeof e["panelType"] === "string" && e["panelType"] in PANEL_LABELS) {
          store.addPanel(e["panelType"] as PanelType)
        } else if (action === "close" && typeof e["panelType"] === "string") {
          const hit = store.panels.find((p) => p.type === e["panelType"])
          if (hit) store.closePanel(hit.id)
        } else if (action === "reset") {
          store.reset()
        }
      } else if (kind === "error" && typeof e["message"] === "string") {
        setTerm((t) => `${t}\n[error] ${e["message"]}\n`.slice(-30_000))
      }
    })
    return () => off?.()
  }, [])

  useEffect(() => {
    termRef.current?.scrollTo({ top: termRef.current.scrollHeight })
  }, [term])

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" })
  }, [turns, live])

  const swirlState: SwirlState = !busy ? "idle" : live === "" ? "thinking" : "responding"

  async function run(goal: string) {
    if (!goal.trim() || busy) return
    const api = window.crowetrade?.orchestrator
    if (!api) return
    setDraft("")
    setBusy(true)
    setLive("")
    setTurns((t) => [...t, { role: "you", text: goal }, { role: "orchestrator", text: "", pending: true }])
    try {
      const res = await api.ask(goal)
      setTurns((t) => [...t.slice(0, -1), { role: "orchestrator", text: res.text }])
    } catch (e) {
      setTurns((t) => [
        ...t.slice(0, -1),
        { role: "orchestrator", text: e instanceof Error ? e.message : String(e) },
      ])
    } finally {
      setBusy(false)
      setLive("")
    }
  }

  return (
    <motion.section
      className="orch"
      aria-label="Orchestrator console"
      initial={{ opacity: 0, x: -MAGNITUDES.slide }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -MAGNITUDES.slide }}
      transition={{ duration: DURATIONS.smooth, ease: EASINGS.snap }}
    >
      <header className="orch__head">
        <span className="ws__title">Orchestrator</span>
        <button
          type="button"
          className="ws__act"
          onClick={onCollapse}
          aria-label="Collapse the Orchestrator"
          title="Collapse"
        >
          <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
            <path
              d="M15 6l-6 6 6 6"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </header>
      <div className="orch__conv">
        <div className="orch__transcript">
          {turns.length === 0 && (
            <div className="orch__empty">
              <AIAvatarSwirl state="idle" size={44} storm="active" />
              <div className="orch__intro">
                <span className="orch__name">Orchestrator</span>
                <span className="orch__hint">
                  Runs commands in the visible terminal and arranges the workspace. Try: check
                  the engine and lay out the panels I need to judge it.
                </span>
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            <div key={i} className={`turn turn--${t.role === "you" ? "you" : "analyst"}`}>
              <span className="turn__who mono">
                {t.role === "orchestrator" && (
                  <AIAvatarSwirl state={t.pending ? swirlState : "idle"} size={18} storm="active" />
                )}
                {t.role === "you" ? "YOU" : "ORCHESTRATOR"}
              </span>
              <p className="turn__text">
                <AnswerText text={t.pending ? live : t.text} />
                {t.pending && (
                  <motion.span
                    className="turn__caret"
                    animate={{ opacity: [1, 0.2, 1] }}
                    transition={{ duration: 0.9, repeat: Infinity, ease: "easeInOut" }}
                  >
                    |
                  </motion.span>
                )}
              </p>
            </div>
          ))}
          <div ref={endRef} />
        </div>
        <form
          className="orch__composer"
          onSubmit={(e) => {
            e.preventDefault()
            void run(draft)
          }}
        >
          <input
            className="analyst__input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={busy ? "working" : "Give the orchestrator a goal"}
            disabled={busy}
            aria-label="Give the orchestrator a goal"
          />
          {busy ? (
            <button
              type="button"
              className="orch__stop"
              onClick={() => void window.crowetrade?.orchestrator?.stop()}
            >
              Stop
            </button>
          ) : (
            <button type="submit" className="analyst__send" disabled={!draft.trim()}>
              Run
            </button>
          )}
        </form>
      </div>

      <pre ref={termRef} className="orch__term mono" aria-label="Orchestrator terminal">
        {term || "terminal idle; commands the orchestrator runs appear here before they execute"}
      </pre>
    </motion.section>
  )
}

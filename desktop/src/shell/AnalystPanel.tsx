import { useState, useRef, useEffect } from "react"
import { motion } from "motion/react"
import { DURATIONS, EASINGS, MAGNITUDES } from "./motion.js"
import { AIAvatarSwirl, type SwirlState } from "./AIAvatarSwirl.js"
import { AnswerBody } from "./AnswerText.js"

/**
 * The Analyst: ask the system about itself, in the same window as the book.
 *
 * Composition follows Cortex (ConversationCanvas + ChatInput) and the four-part
 * decomposition Wave's AI panel arrived at: transcript, status, input, and a
 * grounding indicator. That last one is specific to this product and is not
 * decoration -- an answer that did not consult the engine is the model talking
 * from its prompt, and on a surface where people read numbers to decide things,
 * an ungrounded answer must look different from a grounded one.
 *
 * The identity is the AIAvatarSwirl, and per the house rule its motion is tied
 * to LIVE streaming: deltas paint as they arrive from the main process, engine
 * reads surface the moment they happen (grounding visible in real time, not
 * disclosed after the fact), and the swirl's storm state follows what the
 * Analyst is actually doing. The reveal cadence is Cortex's AnswerStream.
 *
 * Read-only by construction. The Analyst holds three GET operations and no
 * credentials; kill, veto and policy changes need a bearer token it does not
 * have, so nothing said here can move the book.
 */

interface Turn {
  role: "you" | "analyst"
  text: string
  /** Which engine endpoints the answer consulted. Empty means ungrounded. */
  consulted?: string[]
  pending?: boolean
}

const SUGGESTIONS = [
  "How are we doing?",
  "Why did it skip the newest tokens?",
  "Which exit rule looks best, and what is the caveat?",
]

/* Cortex AnswerStream cadence: reveal toward the streamed buffer at a fixed
 * tick so bursty network chunks still read as steady writing. */
const REVEAL_INTERVAL_MS = 12
const CHARS_PER_TICK = 2

function useTypewriter(target: string): string {
  const [shown, setShown] = useState("")
  useEffect(() => {
    if (target === "") {
      setShown("")
      return
    }
    if (shown.length >= target.length) return
    const t = window.setInterval(() => {
      setShown((d) => {
        if (d.length >= target.length) {
          window.clearInterval(t)
          return d
        }
        return target.slice(0, d.length + CHARS_PER_TICK)
      })
    }, REVEAL_INTERVAL_MS)
    return () => window.clearInterval(t)
  }, [target, shown.length])
  return shown
}

/** "crowetrade_engine_read_getPositions" reads as "getPositions" on screen. */
function toolLabel(name: string): string {
  return name.replace(/^crowetrade_engine_read_/, "")
}

export function AnalystPanel({ mint }: { mint?: string | null }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState("")
  const [busy, setBusy] = useState(false)
  const [live, setLive] = useState("")
  const [liveTools, setLiveTools] = useState<string[]>([])
  const endRef = useRef<HTMLDivElement>(null)

  const shown = useTypewriter(live)

  useEffect(() => {
    const offDelta = window.crowetrade?.onAskDelta?.((d) => setLive((t) => t + d))
    const offTool = window.crowetrade?.onAskTool?.((n) => setLiveTools((t) => [...t, n]))
    return () => {
      offDelta?.()
      offTool?.()
    }
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [turns, shown])

  const swirlState: SwirlState = !busy ? "idle" : live === "" ? "thinking" : "responding"

  async function ask(question: string) {
    if (!question.trim() || busy) return
    setDraft("")
    setBusy(true)
    setLive("")
    setLiveTools([])
    setTurns((t) => [...t, { role: "you", text: question }, { role: "analyst", text: "", pending: true }])

    try {
      // The bridge is exposed by the main process, which holds the Azure token.
      // Putting the credential in the renderer would make it reachable from
      // page context, and this window loads no remote content but should not
      // rely on that staying true.
      const bridge = window.crowetrade?.ask
      if (!bridge) throw new Error("analyst bridge unavailable; run scripts/setup.mjs")
      const res = await bridge(question)
      setTurns((t) => [
        ...t.slice(0, -1),
        { role: "analyst", text: res.text, consulted: res.consulted },
      ])
    } catch (e) {
      setTurns((t) => [
        ...t.slice(0, -1),
        { role: "analyst", text: e instanceof Error ? e.message : String(e), consulted: [] },
      ])
    } finally {
      setBusy(false)
      setLive("")
      setLiveTools([])
    }
  }

  return (
    <div className="analyst">
      <div className="analyst__transcript">
        {turns.length === 0 && (
          <div className="analyst__empty">
            <div className="analyst__hero">
              <AIAvatarSwirl state="idle" size={64} />
              <span className="analyst__name">CroweTrade Analyst</span>
              <span className="analyst__tagline">
                reads the live engine, answers grounded or says it is not
              </span>
            </div>
            <div className="analyst__suggest">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className="analyst__chip" onClick={() => ask(s)}>
                  {s}
                </button>
              ))}
              {mint && (
                <button
                  type="button"
                  className="analyst__chip"
                  onClick={() => ask(`What do you know about the mint ${mint}? Is it safe to touch?`)}
                >
                  Ask about the selected token
                </button>
              )}
            </div>
          </div>
        )}

        {turns.map((t, i) => (
          <motion.div
            key={i}
            className={`turn turn--${t.role}`}
            initial={{ opacity: 0, y: MAGNITUDES.rise }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: DURATIONS.quick, ease: EASINGS.snap }}
          >
            <span className={`turn__who mono${t.pending ? " turn__who--working" : ""}`}>
              {t.role === "analyst" && (
                <AIAvatarSwirl state={t.pending ? swirlState : "idle"} size={20} storm="active" />
              )}
              <span className="turn__whoname">{t.role === "you" ? "YOU" : "ANALYST"}</span>
            </span>

            {t.pending ? (
              <>
                {liveTools.length > 0 && (
                  <span className="turn__grounded mono">
                    {liveTools.map((n, j) => (
                      <span key={j} className="turn__read">
                        reading {toolLabel(n)}
                      </span>
                    ))}
                  </span>
                )}
                {shown === "" ? (
                  <span className="turn__thinking">
                    reading the ledger
                    <span className="turn__dots" aria-hidden="true" />
                  </span>
                ) : (
                  <p className="turn__text">
                    <AnswerBody text={shown} />
                    <motion.span
                      className="turn__caret"
                      animate={{ opacity: [1, 0.2, 1] }}
                      transition={{ duration: 0.9, repeat: Infinity, ease: "easeInOut" }}
                    >
                      |
                    </motion.span>
                  </p>
                )}
              </>
            ) : (
              <>
                {t.consulted !== undefined && (
                  <span
                    className={`turn__grounded mono ${t.consulted.length ? "" : "turn__grounded--none"}`}
                  >
                    {t.consulted.length
                      ? `engine consulted: ${t.consulted.map(toolLabel).join(", ")}`
                      : "answered without consulting the engine"}
                  </span>
                )}
                <p className="turn__text">
                  <AnswerBody text={t.text} />
                </p>
              </>
            )}
          </motion.div>
        ))}
        <div ref={endRef} />
      </div>

      <form
        className="analyst__composer"
        onSubmit={(e) => {
          e.preventDefault()
          void ask(draft)
        }}
      >
        <input
          className="analyst__input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={busy ? "reading the ledger" : "Ask the analyst"}
          disabled={busy}
          aria-label="Ask the analyst"
        />
        <button type="submit" className="analyst__send" disabled={busy || !draft.trim()}>
          Ask
        </button>
      </form>
    </div>
  )
}

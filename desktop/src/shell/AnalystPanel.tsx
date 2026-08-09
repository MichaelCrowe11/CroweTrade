import { useState, useRef, useEffect } from "react"
import { motion } from "motion/react"
import { DURATIONS, EASINGS, MAGNITUDES } from "./motion.js"

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

export function AnalystPanel({ mint }: { mint?: string | null }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState("")
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [turns])

  async function ask(question: string) {
    if (!question.trim() || busy) return
    setDraft("")
    setBusy(true)
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
    }
  }

  return (
    <div className="analyst">
      <div className="analyst__transcript">
        {turns.length === 0 && (
          <div className="analyst__empty">
            <p className="analyst__hint">
              Ask about the book, a refusal, or the record. Answers are read from
              the live engine, not recalled.
            </p>
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
            <span className="turn__who mono">{t.role === "you" ? "YOU" : "ANALYST"}</span>
            {t.pending ? (
              <span className="turn__thinking">
                reading the ledger
                <span className="turn__dots" aria-hidden="true" />
              </span>
            ) : (
              <>
                {t.consulted !== undefined && (
                  <span
                    className={`turn__grounded mono ${t.consulted.length ? "" : "turn__grounded--none"}`}
                  >
                    {t.consulted.length
                      ? `engine consulted: ${t.consulted.join(", ")}`
                      : "answered without consulting the engine"}
                  </span>
                )}
                <p className="turn__text">{t.text}</p>
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

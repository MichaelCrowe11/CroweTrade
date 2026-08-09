import { segmentRich } from "./markdown.js"

/**
 * How the assistants typeset an answer.
 *
 * Two house rules meet here. The instrument rule: every financial figure in
 * the prose sets in tabular mono, so -$55.62 reads as a readout, not a word.
 * The editorial rule: the opening paragraph is the lede, set in the display
 * face, because both assistants are instructed to lead with the verdict and
 * the type should say so before the words do.
 */

export function AnswerText({ text }: { text: string }) {
  return (
    <>
      {segmentRich(text).map((seg, i) =>
        seg.kind === "strong" ? (
          <strong key={i}>{seg.text}</strong>
        ) : seg.kind === "strong-num" ? (
          <strong key={i} className="mono turn__num">
            {seg.text}
          </strong>
        ) : seg.kind === "code" ? (
          <code key={i} className="mono turn__code">
            {seg.text}
          </code>
        ) : seg.kind === "num" ? (
          <span key={i} className="mono turn__num">
            {seg.text}
          </span>
        ) : (
          <span key={i}>{seg.text}</span>
        ),
      )}
    </>
  )
}

export function AnswerBody({ text }: { text: string }) {
  const brk = text.indexOf("\n\n")
  const lede = brk === -1 ? text : text.slice(0, brk)
  const rest = brk === -1 ? "" : text.slice(brk + 2)
  return (
    <>
      <span className="turn__lede">
        <AnswerText text={lede} />
      </span>
      {rest && <AnswerText text={rest} />}
    </>
  )
}

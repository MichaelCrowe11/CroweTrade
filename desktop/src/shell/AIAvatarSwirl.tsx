import { useEffect, useRef, useState } from "react"

/**
 * AIAvatarSwirl, the assistant identity mark, ported from the Crowe Logic AI
 * chat surface (components/chat/ai-avatar-swirl.tsx) where it is a KEPT
 * feature: the token storm tied to live streaming output is part of the
 * product's identity, and a static mark on an operator chat surface is
 * explicitly the wrong choice. This port is a refinement, not a replacement:
 *
 * - The center is a DRAWN spiral rather than the web app's raster avatar,
 *   because this terminal's icon language is stroked vector (see Rail.tsx)
 *   and a trading surface ships nobody's photograph.
 * - The token pools speak this product's vocabulary: quotes, gates, mints,
 *   verdicts, cohorts. The swirl thinks in the words the engine thinks in.
 * - Every color derives from the --clm ladder. Gold only: the swirl is brand
 *   and safety, never direction.
 *
 * PERFORMANCE: the storm animates by writing transform and opacity straight
 * onto the particle elements inside the rAF loop, never through React state.
 * The original re-rendered the component per frame per instance, which is a
 * real cost on an always-open terminal; the physics are unchanged, only the
 * write path moved. Chromium stops firing rAF in hidden documents, so a
 * backgrounded window pays nothing. Reduced motion collapses everything to a
 * calm static ring.
 */

export type SwirlState = "idle" | "thinking" | "responding"

interface Particle {
  token: string
  color: string
  speed: number
  angle: number
  radius: number
}

const IDLE_TOKENS = ["◦", "◉", "⊙", "◌", "tick", "scan"]
const THINKING_TOKENS = [
  "quote",
  "impact",
  "gate",
  "mint",
  "liq",
  "route",
  "slot",
  "hold",
  "curve",
  "pnl",
  "{}",
  "()",
  "sol",
  "veto",
]
const RESPONDING_TOKENS = [
  "verdict",
  "entry",
  "exit",
  "book",
  "cohort",
  "labeled",
  "refusal",
  "=>",
  "return",
  "answer",
]

/* Warm variation inside the brand, derived from tokens so the storm moves
 * when the palette does. Six stops keep it reading as motion, not a fill. */
const STORM_COLORS = [
  "var(--clm-gold)",
  "var(--clm-gold-soft)",
  "var(--clm-gold-deep)",
  "color-mix(in srgb, var(--clm-gold-soft) 72%, white)",
  "color-mix(in srgb, var(--clm-gold) 65%, var(--clm-cream))",
  "color-mix(in srgb, var(--clm-gold-deep) 80%, var(--clm-gold))",
]

/* The token storm is decorative; users who opt out get a calm static ring. */
function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    if (!window.matchMedia) return
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)")
    const apply = () => setReduced(mq.matches)
    apply()
    mq.addEventListener?.("change", apply)
    return () => mq.removeEventListener?.("change", apply)
  }, [])
  return reduced
}

/** The mark itself: a spiral coiling inward, stroke-driven, gold gradient. */
function SpiralMark({ size }: { size: number }) {
  return (
    <svg
      viewBox="0 0 48 48"
      width={size}
      height={size}
      fill="none"
      strokeWidth="2"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="ct-swirl-grad" x1="0" y1="0" x2="48" y2="48">
          <stop offset="0%" stopColor="var(--clm-gold-soft)" />
          <stop offset="100%" stopColor="var(--clm-gold-deep)" />
        </linearGradient>
      </defs>
      <path
        d="M24 6 C 35 6, 42 14, 42 24 C 42 33, 35 40, 26 40 C 18 40, 12 34, 12 26 C 12 19, 17 14, 24 14 C 30 14, 34 18, 34 24 C 34 28, 31 31, 27 31"
        stroke="url(#ct-swirl-grad)"
      />
      <circle cx="27" cy="31" r="1.6" fill="var(--clm-gold)" stroke="none" />
    </svg>
  )
}

export function AIAvatarSwirl({
  state,
  size = 40,
  storm = "always",
}: {
  state: SwirlState
  size?: number
  /**
   * "active" storms only while thinking or responding. For the small inline
   * turn avatar, an idle storm spills tokens over transcript text; motion
   * tied to work is the identity, idle drift at that size is just noise.
   */
  storm?: "always" | "active"
}) {
  const reduced = useReducedMotion()
  const stormy = storm === "always" || state !== "idle"

  /* Display list: React renders the spans once per (state, size) change;
   * physics and per-frame writes live entirely in refs. */
  const [display, setDisplay] = useState<{ token: string; color: string }[]>([])
  const physicsRef = useRef<Particle[]>([])
  const spanRefs = useRef<(HTMLSpanElement | null)[]>([])
  const markRef = useRef<HTMLDivElement>(null)
  const markOpacityRef = useRef(1)

  const auraSpin = state === "thinking" ? 5 : state === "responding" ? 8 : 14
  const ringSpin = state === "thinking" ? 8 : state === "responding" ? 12 : 20
  const auraOpacity = state === "thinking" ? 0.7 : state === "responding" ? 0.55 : 0.32

  /* Breathe: a few writes per couple of seconds, straight to the mark. */
  useEffect(() => {
    const setMark = (v: number) => {
      markOpacityRef.current = v
      if (markRef.current) markRef.current.style.opacity = String(v)
    }
    if (reduced || state === "idle") {
      setMark(1)
      return
    }
    const breathe = setInterval(
      () => {
        setMark(0.72)
        setTimeout(() => setMark(1), 260)
        setTimeout(() => setMark(0.86), 560)
        setTimeout(() => setMark(1), 900)
      },
      state === "thinking" ? 2000 : 2800,
    )
    return () => {
      clearInterval(breathe)
      setMark(1)
    }
  }, [state, reduced])

  /* Seed the storm: build physics into refs, render spans once. */
  useEffect(() => {
    if (!stormy) {
      physicsRef.current = []
      setDisplay([])
      return
    }
    const count = state === "thinking" ? 14 : state === "responding" ? 10 : 7
    const pool =
      state === "thinking" ? THINKING_TOKENS : state === "responding" ? RESPONDING_TOKENS : IDLE_TOKENS

    physicsRef.current = Array.from({ length: count }, (_, i) => {
      const angle = (i / count) * Math.PI * 2
      return {
        token: pool[(i + Math.floor(Math.random() * pool.length)) % pool.length] ?? "◦",
        color: STORM_COLORS[i % STORM_COLORS.length] ?? "var(--clm-gold)",
        speed:
          state === "thinking"
            ? 1 + Math.random() * 1.5
            : state === "responding"
              ? 0.45 + Math.random() * 0.85
              : 0.25 + Math.random() * 0.4,
        angle,
        radius: size * 0.72 + Math.random() * (size * 0.34),
      }
    })
    spanRefs.current = []
    setDisplay(physicsRef.current.map((p) => ({ token: p.token, color: p.color })))

    /* Reduced motion: place the ring once, statically, and stop. */
    if (reduced) {
      queueMicrotask(() => {
        physicsRef.current.forEach((p, i) => {
          const el = spanRefs.current[i]
          if (!el) return
          el.style.transform = `translate(${Math.cos(p.angle) * p.radius}px, ${Math.sin(p.angle) * p.radius}px)`
          el.style.opacity = "0.5"
        })
      })
    }
  }, [size, state, reduced, stormy])

  /* The loop: physics unchanged from the ported original, write path direct. */
  useEffect(() => {
    if (reduced || !stormy) return
    let frame = 0
    let time = 0

    const animate = () => {
      time += state === "thinking" ? 0.03 : state === "responding" ? 0.023 : 0.015
      const mark = markOpacityRef.current

      physicsRef.current.forEach((p, i) => {
        const el = spanRefs.current[i]
        if (!el) return
        p.angle += p.speed * 0.015
        let radius = p.radius
        let opacity: number
        let scale: number
        let x: number
        let y: number

        if (state === "thinking") {
          const drift =
            Math.sin(time * 2 + i) * (size * 0.2) + Math.cos(time * 1.45 + i * 0.3) * (size * 0.14)
          radius = p.radius + Math.sin(time * 2.35 + i) * (size * 0.18)
          opacity = 0.58 + Math.sin(time * 3 + i) * 0.35
          scale = 0.8 + Math.sin(time * 2.2 + i) * 0.28
          x = Math.cos(p.angle) * (radius + drift)
          y = Math.sin(p.angle) * (radius + drift)
        } else if (state === "responding") {
          const pull = 0.72 + Math.sin(time * 1.55 + i * 0.45) * 0.16
          radius = p.radius * pull
          opacity = 0.55 + Math.sin(time * 1.9 + i) * 0.22
          scale = 0.82 + Math.sin(time * 2.1 + i) * 0.18
          x = Math.cos(p.angle) * (radius * 0.8)
          y = Math.sin(p.angle) * (radius * 0.62)
        } else {
          const drift = Math.sin(time * 1.4 + i) * (size * 0.1)
          opacity = 0.42 + Math.sin(time + i) * 0.24
          scale = 0.92 + Math.sin(time * 1.35 + i) * 0.12
          x = Math.cos(p.angle) * (p.radius + drift)
          y = Math.sin(p.angle) * (p.radius + drift)
        }

        el.style.transform = `translate(${x}px, ${y}px) scale(${scale})`
        el.style.opacity = String(Math.max(0, opacity * mark))
      })
      frame = requestAnimationFrame(animate)
    }

    animate()
    return () => cancelAnimationFrame(frame)
  }, [size, state, reduced, stormy])

  return (
    <div className="swirl" style={{ width: size, height: size }} aria-hidden="true">
      {/* Rotating gold energy field: soft aura plus a crisp counter-rotating
          ring for depth. Spins faster while the Analyst works. */}
      <div
        className="swirl__aura"
        style={{
          opacity: auraOpacity,
          animation: reduced ? "none" : `swirl-spin ${auraSpin}s linear infinite`,
        }}
      />
      <div
        className="swirl__ring"
        style={{
          opacity: state === "idle" ? 0.5 : 0.85,
          animation: reduced ? "none" : `swirl-spin ${ringSpin}s linear infinite reverse`,
        }}
      />
      <div className="swirl__storm">
        {display.map((p, i) => (
          <span
            key={i}
            ref={(el) => {
              spanRefs.current[i] = el
            }}
            className="swirl__tok mono"
            style={{
              color: p.color,
              opacity: 0,
              fontSize: p.token.length > 3 ? "8px" : "9px",
            }}
          >
            {p.token}
          </span>
        ))}
      </div>
      <div
        ref={markRef}
        className="swirl__mark"
        style={{ transform: state === "thinking" ? "scale(1.03)" : "scale(1)" }}
      >
        <SpiralMark size={size} />
      </div>
    </div>
  )
}

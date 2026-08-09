import { useEffect, useState } from "react"

/**
 * AIAvatarSwirl, the assistant identity mark, ported from the Crowe Logic AI
 * chat surface (components/chat/ai-avatar-swirl.tsx) where it is a KEPT
 * feature: the token storm tied to live streaming output is part of the
 * product's identity, and a static mark on an operator chat surface is
 * explicitly the wrong choice. This port is a refinement, not a replacement:
 *
 * - The center is a DRAWN spiral rather than the web app's raster avatar,
 *   because this terminal's icon language is stroked vector (see Rail.tsx)
 *   and a trading surface ships no photograph of anyone.
 * - The token pools speak this product's vocabulary: quotes, gates, mints,
 *   verdicts, cohorts. The swirl thinks in the words the engine thinks in.
 * - Every color derives from the --clm ladder. Gold only: the swirl is brand
 *   and safety, never direction.
 *
 * Three states: idle (ambient drift), thinking (the Analyst is reading the
 * ledger, storm tightens and quickens), responding (tokens settle as text
 * streams out). Reduced motion collapses everything to a calm static ring.
 */

export type SwirlState = "idle" | "thinking" | "responding"

interface StormParticle {
  id: number
  token: string
  x: number
  y: number
  color: string
  speed: number
  angle: number
  radius: number
  opacity: number
  scale: number
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
  const [particles, setParticles] = useState<StormParticle[]>([])
  const [markOpacity, setMarkOpacity] = useState(1)
  const reduced = useReducedMotion()
  const stormy = storm === "always" || state !== "idle"

  const auraSpin = state === "thinking" ? 5 : state === "responding" ? 8 : 14
  const ringSpin = state === "thinking" ? 8 : state === "responding" ? 12 : 20
  const auraOpacity = state === "thinking" ? 0.7 : state === "responding" ? 0.55 : 0.32

  useEffect(() => {
    if (reduced) {
      setMarkOpacity(1)
      return
    }
    if (state === "thinking" || state === "responding") {
      const breathe = setInterval(
        () => {
          setMarkOpacity(0.72)
          setTimeout(() => setMarkOpacity(1), 260)
          setTimeout(() => setMarkOpacity(0.86), 560)
          setTimeout(() => setMarkOpacity(1), 900)
        },
        state === "thinking" ? 2000 : 2800,
      )
      return () => clearInterval(breathe)
    }
    setMarkOpacity(1)
    return
  }, [state, reduced])

  useEffect(() => {
    if (!stormy) {
      setParticles([])
      return
    }
    const count = state === "thinking" ? 14 : state === "responding" ? 10 : 7
    const pool =
      state === "thinking" ? THINKING_TOKENS : state === "responding" ? RESPONDING_TOKENS : IDLE_TOKENS

    setParticles(
      Array.from({ length: count }, (_, i) => {
        const angle = (i / count) * Math.PI * 2
        const radius = size * 0.72 + Math.random() * (size * 0.34)
        return {
          id: i,
          token: pool[(i + Math.floor(Math.random() * pool.length)) % pool.length] ?? "◦",
          x: reduced ? Math.cos(angle) * radius : 0,
          y: reduced ? Math.sin(angle) * radius : 0,
          color: STORM_COLORS[i % STORM_COLORS.length] ?? "var(--clm-gold)",
          speed:
            state === "thinking"
              ? 1 + Math.random() * 1.5
              : state === "responding"
                ? 0.45 + Math.random() * 0.85
                : 0.25 + Math.random() * 0.4,
          angle,
          radius,
          opacity: reduced ? 0.5 : 1,
          scale: 1,
        }
      }),
    )
  }, [size, state, reduced, stormy])

  useEffect(() => {
    if (reduced || !stormy) return
    let frame = 0
    let time = 0

    const animate = () => {
      time += state === "thinking" ? 0.03 : state === "responding" ? 0.023 : 0.015
      setParticles((prev) =>
        prev.map((p) => {
          const angle = p.angle + p.speed * 0.015
          let radius = p.radius
          let opacity = p.opacity
          let scale = 1
          let x = 0
          let y = 0

          if (state === "thinking") {
            const drift =
              Math.sin(time * 2 + p.id) * (size * 0.2) + Math.cos(time * 1.45 + p.id * 0.3) * (size * 0.14)
            radius = p.radius + Math.sin(time * 2.35 + p.id) * (size * 0.18)
            opacity = 0.58 + Math.sin(time * 3 + p.id) * 0.35
            scale = 0.8 + Math.sin(time * 2.2 + p.id) * 0.28
            x = Math.cos(angle) * (radius + drift)
            y = Math.sin(angle) * (radius + drift)
          } else if (state === "responding") {
            const pull = 0.72 + Math.sin(time * 1.55 + p.id * 0.45) * 0.16
            radius = p.radius * pull
            opacity = 0.55 + Math.sin(time * 1.9 + p.id) * 0.22
            scale = 0.82 + Math.sin(time * 2.1 + p.id) * 0.18
            x = Math.cos(angle) * (radius * 0.8)
            y = Math.sin(angle) * (radius * 0.62)
          } else {
            const drift = Math.sin(time * 1.4 + p.id) * (size * 0.1)
            opacity = 0.42 + Math.sin(time + p.id) * 0.24
            scale = 0.92 + Math.sin(time * 1.35 + p.id) * 0.12
            x = Math.cos(angle) * (p.radius + drift)
            y = Math.sin(angle) * (p.radius + drift)
          }

          return { ...p, angle, radius, opacity, scale, x, y }
        }),
      )
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
        {particles.map((p) => (
          <span
            key={p.id}
            className="swirl__tok mono"
            style={{
              transform: `translate(${p.x}px, ${p.y}px) scale(${p.scale})`,
              color: p.color,
              opacity: p.opacity * markOpacity,
              fontSize: p.token.length > 3 ? "8px" : "9px",
            }}
          >
            {p.token}
          </span>
        ))}
      </div>
      <div
        className="swirl__mark"
        style={{
          transform: state === "thinking" ? "scale(1.03)" : "scale(1)",
          opacity: markOpacity,
        }}
      >
        <SpiralMark size={size} />
      </div>
    </div>
  )
}

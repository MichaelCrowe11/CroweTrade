/**
 * Motion tokens, ported from Crowe Cortex (src/motion/tokens.ts).
 *
 * Every animation pulls from here so timing stays coherent. Scattered ad-hoc
 * durations are how an interface starts feeling assembled rather than designed.
 */

export const DURATIONS = {
  instant: 0.08,
  quick: 0.18,
  smooth: 0.32,
  expand: 0.45,
  breathe: 1.8,
} as const

export const EASINGS = {
  /** Crisp out-back for elements entering the canvas. Matches --clm-ease. */
  snap: [0.16, 1, 0.3, 1] as const,
  /** Symmetric, for ambient loops that should not draw the eye. */
  ambient: [0.4, 0, 0.6, 1] as const,
} as const

export const MAGNITUDES = {
  /** Vertical fade-in distance, px. */
  rise: 8,
  /** Horizontal slide-in distance, px. */
  slide: 16,
} as const

/**
 * The armed edge model: weights FROZEN at arming time.
 *
 * This is a constant, not a training pipeline, on purpose. The mandatory
 * discipline from the calibration work: a model may only trade on decisions
 * labeled BEFORE its own deploy time, or it trains on markets it influenced
 * and the calibration claim silently dies. Freezing the fit as source makes
 * the boundary physical — /api/train keeps refitting and REPORTING as new
 * labels accrue, and none of it touches what trades until a human replaces
 * this constant and bumps the policy's modelFingerprint.
 *
 * Provenance: fitted 2026-08-09 over 5,743 clean labeled decisions (voided
 * excluded), temporal 75/25 split, clamped feature vector (see
 * buildFeatureVector). Test AUC 0.802 on 1,436 held-out rows, base rate 5.8%.
 * Reliability at the gate's operating point: predicted 0.264 vs observed
 * 0.238 (n=42) in the 0.2-0.4 bucket — a ~4x lift over base. Signal is
 * dominated by measured liquidity depth; isLaunchpad carries -0.13, so the
 * fit is not merely re-encoding which feed a token came from.
 *
 * The label this probability estimates: P(30-minute forward return clears a
 * 6% round-trip cost hurdle). It is NOT P(profit) — exits and realized
 * slippage sit on top.
 */
import type { ModelFit } from "./model.js"

/** Identity of these exact weights; the policy envelope carries it so a new
 *  model rolls the policy hash and starts a new cohort. */
export const ARMED_MODEL_FINGERPRINT = "m20260809-5743r-auc802"

export const ARMED_MODEL: ModelFit = {
  weights: [-0.0385, 0.0269, 0.0352, 0.0791, 0.0072, 0.4533, -0.374, -0.1252],
  bias: -3.1824,
  mean: [0.0063, 0.0027, -3.3933, -19.3336, 3.0306, 0.8271, 0.9011, 0.9524],
  std: [0.093, 0.1536, 69.9775, 102.5518, 0.3017, 1.0831, 0.2985, 0.2129],
  trainN: 4307,
  testN: 1436,
  baseRate: 0.058,
  accuracy: 0.942,
  auc: 0.802,
  reliability: [
    { bucket: "0.0-0.2", predicted: 0.05, observed: 0.052, n: 1392 },
    { bucket: "0.2-0.4", predicted: 0.264, observed: 0.238, n: 42 },
    { bucket: "0.4-0.6", predicted: 0.418, observed: 0.5, n: 2 },
  ],
  featureNames: [
    "netFlowShare",
    "flowAccel",
    "priceProgressPct",
    "liqTrendPct",
    "ticksObserved",
    "logLiquidityUsd",
    "liqKnown",
    "isLaunchpad",
  ],
}

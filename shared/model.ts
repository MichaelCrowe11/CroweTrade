/**
 * The edge model: logistic regression over decision-time features.
 *
 * Deliberately the smallest thing that can work. The question is a calibrated
 * probability that a launch is worth touching, and a linear model over six
 * features answers it with coefficients a human can read and argue with. A
 * larger model would be harder to trust and, on a few hundred samples, would
 * mostly memorise noise.
 *
 * Three disciplines are non-negotiable here, because each one is a documented
 * way this exact problem produces beautiful worthless results:
 *
 * 1. TEMPORAL split, never random. Randomly holding out rows lets the model
 *    learn from the future of the same market session it is scored on. Every
 *    split in this file is by decision time.
 * 2. Standardise using TRAIN statistics only, then apply them to test. Fitting
 *    the scaler on all data leaks test distribution into training.
 * 3. Calibration is the target, not accuracy. A model that says 70% must be
 *    right about 70% of the time, because the number feeds position sizing.
 *    Accuracy alone can look fine while the probabilities are meaningless.
 */

/** Feature vector, ordered. Order is part of the contract with the weights. */
export const FEATURE_NAMES = [
  "netFlowShare",
  "flowAccel",
  "priceProgressPct",
  "liqTrendPct",
  "ageMinutes",
  "logLiquidityUsd",
] as const

export interface TrainingRow {
  at: number
  features: number[]
  /** 1 when the forward return cleared the round-trip cost hurdle. */
  label: 0 | 1
}

export interface ModelFit {
  weights: number[]
  bias: number
  mean: number[]
  std: number[]
  trainN: number
  testN: number
  /** Share of test rows labeled 1. The rate a constant predictor would get. */
  baseRate: number
  /** Test accuracy. Reported, but NOT the thing that matters. */
  accuracy: number
  /** Ranking quality, 0.5 = coin flip. The honest headline. */
  auc: number
  /** Calibration bins: predicted probability vs observed frequency. */
  reliability: { bucket: string; predicted: number; observed: number; n: number }[]
  featureNames: string[]
}

function standardise(rows: number[][]): { mean: number[]; std: number[] } {
  const d = rows[0]?.length ?? 0
  const mean = new Array(d).fill(0)
  const std = new Array(d).fill(1)
  if (rows.length === 0) return { mean, std }
  for (let j = 0; j < d; j++) {
    let s = 0
    for (const r of rows) s += r[j] ?? 0
    mean[j] = s / rows.length
    let v = 0
    for (const r of rows) v += ((r[j] ?? 0) - mean[j]) ** 2
    // Guard a zero-variance feature: dividing by it produces Infinity and
    // poisons every downstream weight.
    std[j] = Math.sqrt(v / rows.length) || 1
  }
  return { mean, std }
}

const apply = (r: number[], mean: number[], std: number[]) =>
  r.map((x, j) => (x - (mean[j] ?? 0)) / (std[j] ?? 1))

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

/** Area under the ROC curve, computed by rank. Ties share their average rank. */
export function auc(scores: number[], labels: number[]): number {
  const pos = labels.filter((l) => l === 1).length
  const neg = labels.length - pos
  if (pos === 0 || neg === 0) return 0.5 // undefined; report a coin flip
  const idx = scores.map((s, i) => ({ s, l: labels[i] ?? 0 })).sort((a, b) => a.s - b.s)
  let rank = 1
  let sumPosRanks = 0
  let i = 0
  while (i < idx.length) {
    let j = i
    while (j + 1 < idx.length && idx[j + 1]?.s === idx[i]?.s) j++
    const avgRank = (rank + (rank + (j - i))) / 2
    for (let k = i; k <= j; k++) if (idx[k]?.l === 1) sumPosRanks += avgRank
    rank += j - i + 1
    i = j + 1
  }
  return (sumPosRanks - (pos * (pos + 1)) / 2) / (pos * neg)
}

/**
 * Fits the model. Returns null when there is not enough data to say anything,
 * which is a legitimate and frequent answer early on.
 */
export function fit(rows: TrainingRow[], minRows = 80): ModelFit | null {
  if (rows.length < minRows) return null

  // Temporal split: oldest 75% trains, newest 25% scores.
  const sorted = [...rows].sort((a, b) => a.at - b.at)
  const cut = Math.floor(sorted.length * 0.75)
  const train = sorted.slice(0, cut)
  const test = sorted.slice(cut)
  if (train.length < 30 || test.length < 10) return null

  const { mean, std } = standardise(train.map((r) => r.features))
  const X = train.map((r) => apply(r.features, mean, std))
  const y = train.map((r) => r.label)

  const d = X[0]?.length ?? 0
  let w = new Array(d).fill(0)
  let b = 0
  const lr = 0.1
  const l2 = 0.01 // ridge penalty: small samples overfit without it

  for (let epoch = 0; epoch < 600; epoch++) {
    const gw = new Array(d).fill(0)
    let gb = 0
    for (let i = 0; i < X.length; i++) {
      const xi = X[i] ?? []
      let z = b
      for (let j = 0; j < d; j++) z += (w[j] ?? 0) * (xi[j] ?? 0)
      const err = sigmoid(z) - (y[i] ?? 0)
      for (let j = 0; j < d; j++) gw[j] += err * (xi[j] ?? 0)
      gb += err
    }
    for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / X.length + l2 * (w[j] ?? 0))
    b -= lr * (gb / X.length)
  }

  const predict = (f: number[]) => {
    const x = apply(f, mean, std)
    let z = b
    for (let j = 0; j < d; j++) z += (w[j] ?? 0) * (x[j] ?? 0)
    return sigmoid(z)
  }

  const scores = test.map((r) => predict(r.features))
  const labels = test.map((r) => r.label)
  const correct = scores.filter((s, i) => (s >= 0.5 ? 1 : 0) === labels[i]).length

  // Reliability: does a predicted 60% actually happen 60% of the time?
  const buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
  const reliability: ModelFit["reliability"] = []
  for (let k = 0; k < buckets.length - 1; k++) {
    const lo = buckets[k] ?? 0
    const hi = buckets[k + 1] ?? 1
    const inBin = scores.map((s, i) => ({ s, l: labels[i] ?? 0 })).filter((o) => o.s >= lo && o.s < hi)
    if (inBin.length === 0) continue
    reliability.push({
      bucket: `${lo.toFixed(1)}-${hi.toFixed(1)}`,
      predicted: Number((inBin.reduce((a, o) => a + o.s, 0) / inBin.length).toFixed(3)),
      observed: Number((inBin.reduce((a, o) => a + o.l, 0) / inBin.length).toFixed(3)),
      n: inBin.length,
    })
  }

  return {
    weights: w.map((x) => Number(x.toFixed(4))),
    bias: Number(b.toFixed(4)),
    mean: mean.map((x) => Number(x.toFixed(4))),
    std: std.map((x) => Number(x.toFixed(4))),
    trainN: train.length,
    testN: test.length,
    baseRate: Number((labels.reduce((a: number, l) => a + l, 0) / labels.length).toFixed(3)),
    accuracy: Number((correct / test.length).toFixed(3)),
    auc: Number(auc(scores, labels).toFixed(3)),
    reliability,
    featureNames: [...FEATURE_NAMES],
  }
}

/** Scores one feature vector with a stored fit. */
export function score(m: ModelFit, features: number[]): number {
  const x = features.map((v, j) => (v - (m.mean[j] ?? 0)) / (m.std[j] ?? 1))
  let z = m.bias
  for (let j = 0; j < x.length; j++) z += (m.weights[j] ?? 0) * (x[j] ?? 0)
  return sigmoid(z)
}

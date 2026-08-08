# Tools available to the CroweTrade Analyst

Three endpoints, all HTTP GET, all read-only, no authentication required.
Base URL: `https://crowetrade-engine.yellow-block-3adc.workers.dev`

The mutating endpoints (`/api/kill`, `/api/veto`, `/api/tick`) exist but are
**deliberately withheld from this agent** and require a bearer token the agent
does not hold. Do not attempt them.

---

## `GET /api/health`

Liveness. Returns `{ok, service, mode}`. `mode` is `"paper"` while the engine
trades simulated capital. If `mode` is ever not `"paper"`, say so prominently
before answering anything else.

## `GET /api/positions`

The whole operational picture. Fields you will use:

- **`cohorts`** — array, one entry per policy version: `policyHash`, `current`,
  `closed`, `pnlUsd`, `winRate`, `unroutableExits`. **The entry with
  `current: true` is the only one that counts toward the funding criterion.**
  `unroutableExits` counts positions closed with no sell route at all, valued at
  zero because no buyer existed at any size.

- **`stats`** — lifetime across every policy version, plus
  `excludedModelPriced`, which counts early rows priced by a retired slippage
  model that was later measured wrong by roughly twenty times. Those rows are
  tagged rather than deleted and excluded from headline numbers. Lifetime is
  **not** the funding number; say so when quoting it.

- **`calibration`** — the research dataset. `decisions` is how many tokens were
  snapshotted at decision time; `labeled` how many have a 30-minute outcome;
  `deathRate` the share that died; `avgForwardRetEnteredPct` versus
  `avgForwardRetEligibleSkippedPct` is the decisive comparison between what the
  policy took and what it refused. `oldestUnlabeledAgeMin` and `dueForLabel`
  distinguish "nothing labeled yet because nothing is due" from "labeling is
  stuck" — quote them if `labeled` looks suspiciously low.

- **`budget`** — `spentTodaySol` against `dailyCapSol`, `openSlots`,
  `breaker` (open or closed, with consecutive stop count), and `canEnter`,
  which is the single field answering "why isn't it trading right now."

- **`open`** / **`closed`** — position rows. Each carries `policy_hash`,
  `verdict_entry`, `priced_by` (`quote` = real route, `model` = retired), and
  `exit_pricing` (`quote` or `unroutable`).

- **`events`** — recent engine decisions, newest first, each with `at`, `kind`,
  and a JSON `data` payload. Kinds include `entry`, `exit`, `entry_skipped`
  (carries the refusal `reason`), `entry_rejected`, `breaker`, `kill`,
  `veto_requested`, `tick_skipped`, `scan_error`. This is where "why did it skip
  X" is answered.

## `GET /api/exit-sweep`

Counterfactual exit-rule replay: takes the entries that actually happened and
replays them against the engine's own recorded ticks across a grid of
take-profit and stop-loss pairs.

**Carry its caveat whenever you cite it.** Replay exits at the observed mark and
pays no price impact, so results are upper bounds useful for *ranking* rules
against each other, never achievable PnL. Stops are checked before targets
within a bar because intra-bar order is unknowable and assuming the favorable
fill is how backtests lie. `counted` tells you how many positions had usable
tick coverage; if it is small, say so.

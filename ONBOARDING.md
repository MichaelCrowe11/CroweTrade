# CroweTrade: working on this repo

Read this before your first change. Most of what follows is not style advice,
it is a list of ways this codebase has already wasted a day of someone's time.

## Start a Codespace, not a local checkout

Click Code, then Codespaces, then create. You get Linux, Node 24, the test
runner and the Worker types, with no toolchain to install. This matters most on
Windows: the build scripts, the signing rail and the screenshot rail are all
macOS-only, and none of them are needed to do useful work here.

If you insist on a local clone, `.gitattributes` keeps line endings sane, but
you are on your own for the rest.

## What in this repo is alive

The repository has a large Python trading stack in `src/`, plus `fly*.toml`,
`k8s/`, `helm/`, `docker/` and a pile of DEPLOYMENT*.md files. **All of that is
superseded.** It is a 2025 institutional-CEX system that does not apply to the
current product, and parts of it are actively wrong (its take-cost analysis
returns hardcoded constants, its risk guard has a no-op kill switch, its
point-in-time test asserts against a lambda defined inside the test itself).

What is alive is three directories:

- **`engine/`** is a Cloudflare Worker. A cron fires every minute and drives a
  trading tick inside a Durable Object called `Ledger`, which holds a SQLite
  database with every position, tick, decision and label. This is the system.
- **`shared/`** is the domain logic: safety gates, policy envelopes, feature
  extraction, the model, the signer, the preflight guard. Both the engine and
  the terminal import it, which is what stops the verdict on screen and the
  verdict that trades from ever disagreeing.
- **`desktop/`** is the operator terminal, an Electron app. You do not need to
  run it to be useful.

There is exactly **one** Durable Object instance, `idFromName("global")`. All
state is there. That is why two people can work on this at once without
building anything: the shared workspace is server-side and already exists.

## The rules that are not obvious

**Never merge to `main`.** `.github/workflows/deploy.yml` still fires Fly and
k8s deploys on push to main and would deploy the superseded Python stack over
the live system. Gating it requires a token scope Michael has not granted yet.
Work on branches, open pull requests, and let Michael merge.

**Test-critical logic goes in `shared/`, and `shared/` stays import-free.** The
test runner is `node --test --experimental-strip-types`, which cannot resolve
the `.js` import specifiers that engine files use. So anything you want covered
by a test must live in a `shared/` module with no imports of its own, and its
test goes in `desktop/src/safety/` importing it by `.ts` path. `shared/auth.ts`
and its test are the smallest example of the pattern.

**The test glob must be quoted.** `npm test` in `desktop/` works. A bare
directory argument silently fails to match.

**Stage by explicit path. Never `git add -A` or `git commit -a`.** Multiple
sessions and people share this tree. A broad add once swept a half-finished
test file into someone else's commit without its module, leaving a commit where
the suite failed on fresh checkout.

**Every scripted edit must assert its anchor before replacing.** Python
`str.replace` no-ops silently on a miss. An edit script that printed "success"
unconditionally cost an hour of debugging a deploy problem that did not exist.
Assert the target is present first, and re-read the file after.

**A deployed fix may not be running yet.** A Durable Object keeps executing the
old script version until the instance cycles, and the every-minute cron keeps
it permanently warm so it never does. If a fix appears not to have landed,
that is the first suspect, not the second. The escape hatch: deploy with
`"crons": []`, wait about 90 seconds for the instance to go idle and evict,
then redeploy with the cron restored.

## How to read the numbers without fooling yourself

This project has produced confident wrong answers three separate times, each
from a metric measured at a different moment than the event it described. Take
these as house rules.

**Segment by policy hash before judging anything.** Every fill is stamped with
the hash of the policy envelope that produced it. Lifetime totals mix
incompatible strategies and mean nothing. The `cohorts` array on the summary is
the honest view, and only the one flagged `current` counts toward the funding
criterion.

**Never quote `avgForwardRetEnteredPct` as evidence that selection works.** It
is measured from the decision snapshot, which is taken at first sight, while
`entered` flips later when the position actually opens. On the same tokens it
reads +145.9% while realised performance was -14.6% per trade. The gap is
almost exactly the price run between first sight and entry: we buy things that
have already moved.

**Rows marked `voided` never join a sample.** They are quarantined corrupt data
and the column is never cleared. An earlier attempt to void rows by resetting
their `labeled` flag made them eligible for labelling again, and the labeller
promptly re-corrupted them.

**Unknown is not a pass.** The safety gates have three states, not two, and the
third one is drawn as a hollow ring rather than a dim dot because a panel of
dim dots reads as "switched off". If you add a gate, make sure it can say it
does not know, and make sure that never counts as permission.

**Sort any counterfactual by return and look at the top rows before trusting
the mean.** Two contaminated rows once moved an average by 90 points.

## Access

You have a **research token**. It reaches:

- `POST /api/research` with `{"sql": "..."}` for read-only SQL over the whole
  corpus. SELECT and WITH only, one statement, 500 rows per page.
- `POST /api/analyst` for the grounded Analyst, `POST /api/gates`,
  `GET /api/proposals`.
- Everything already public: `GET /api/positions`, `/api/health`, `/api/train`,
  `/api/exit-sweep`.

It does not reach `kill`, `veto`, `tick` or `/api/llm`, and it is a separate
secret from the operator's, so it can be revoked without disturbing anything
else. See `shared/auth.ts` for why the asymmetry is written the way it is.

Send it as `Authorization: Bearer <token>`. Do not put it in a file that git
can see. `engine/.dev.vars` is gitignored and is the right place locally.

## Live trading is built and inert

Everything needed to trade real money exists: a preflight guard, an Ed25519
signer verified against mainnet, balance-based reconciliation, and exactly one
module that can broadcast. **None of it is armed**, and three independent
conditions must all hold before it can be: an environment flag set to exactly
`"1"`, a keypair present, and a policy envelope whose product is
`crowetrade-live`. Absence of any one leaves the engine behaving exactly as it
does now.

Do not arm anything. Do not add a send path outside
`engine/src/execution/live.ts`.

## Known and expected

`tsc` on `engine/` reports three errors about `TRADING_KEYPAIR` not existing on
`Env`. That is correct and deliberate: the secret is absent, so the generated
types do not include it. Do not "fix" it by adding the secret.

## Where the work actually is

As of 2026-08-11 the engine has entered **zero** positions since 17:58 UTC on
08-10, while reporting perfect health. Three entry-parameter blockers have been
found and two fixed. The diagnostic that matters is `entryFunnel` on
`/api/positions`: it counts survivors at each stage and its buckets sum to the
scanned count, so the largest bucket is trustworthy. Current answer is
`thinLiquidity`, with `tooOld` behind it.

The open question underneath that is whether a viable age-and-liquidity window
exists on the pump.fun launchpad at all. Two queries about it disagreed wildly
and both were wrong, for a reason worth internalising: ordering by most recent
tick samples the newest mints, whose "best liquidity so far" is just their
first minute. There is no trustworthy read on this yet. It is a good first
piece of work.

The standing caveat on everything: the machine is real, the strategy is not
proven. The record is deeply negative and the honest goal is 100 closes under
one stable policy, not a profitable afternoon.

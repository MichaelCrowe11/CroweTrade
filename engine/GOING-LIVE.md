# Going live: the $20 dust test

This is the checklist that takes CroweTrade from paper to a first real
transaction. It is written for twenty dollars, not for a funded strategy.

**Why twenty dollars is the right first number.** It is not a bet. The paper
record is negative and nothing here changes that. What twenty dollars buys is
a set of engineering facts that cannot be established on paper: that the caps
hold when the money is real, that a fill can be confirmed against the chain and
reconciled into the book, that the kill switch stops a live path and not just a
simulated one, and that the difference between a quoted fill and a realized one
is measured rather than assumed. Every one of those is a thing that has to work
before any larger number is safe, and every one is cheaper to learn wrong at
twenty dollars than at two thousand.

**What it does not buy.** Evidence that the strategy works. That still requires
100+ closes positive under one stable policy, and the current record is 168
closes at -$429 across six cohorts. Do not read a successful dust test as a
green light to size up.

---

## What is already built and tested

- `shared/preflight.ts` — the guard that runs immediately before any real
  transaction. Pure, deny-by-default, 17 tests. It re-checks the kill switch,
  live arming, envelope type, expiry, wallet signature, per-trade cap, daily
  cap, position slots, wallet balance including an exit reserve, simulation
  result, and price impact. A refusal names the first thing wrong.
- `engine/src/execution/live.ts` — the ONLY module in the codebase that can
  broadcast, verified: `sendTransaction` appears in exactly one other file,
  inside a comment saying it does not exist there. Entries and exits share this
  path, because an engine that can enter live and cannot exit live is worse
  than one that does neither. The one asymmetry is policy, not mechanism:
  exits skip the entry guard, since the kill switch, daily cap and breaker
  exist to stop NEW risk and must never trap a position that is already open.
- `engine/src/execution/swap.ts` — unchanged. Builds and simulates, never
  sends. Still the entry gate.

- `shared/signer.ts` — Ed25519 signing via WebCrypto, which workerd supports
  natively, so no crypto dependency ships. 10 tests at the byte level. The
  guarantee it enforces: signing changes EXACTLY the 64 signature bytes and
  passes the message through untouched, so "we signed what we simulated" is
  checkable rather than assumed. A transaction declaring more signers than we
  hold keys for is REFUSED rather than partially signed.
  **VERIFIED AGAINST LIVE MAINNET 2026-08-10:** a real Jupiter swap
  transaction, signed by this module, submitted to `simulateTransaction` with
  `sigVerify: true`, returned `AccountNotFound` — not a signature error. The
  network accepted the signature and failed only on the unfunded wallet, which
  is proof the crypto is correct, obtained without spending anything.

- `shared/reconcile.ts` — what ACTUALLY happened, read from the chain's pre-
  and post-balances rather than from the quote. 10 tests. The cases it is built
  around are the ones where a naive reader records a confident wrong number:
  missing transaction meta returns null rather than a zero fill (a zero fill
  would tell the book the trade did nothing), a mint held across two token
  accounts is summed rather than sampled, balances belonging to another owner
  are never credited to us, and amounts stay bigint so a 2^53+1 base-unit
  balance survives. `tca()` compares quoted against realized — the real number
  the old Python stack only claimed to compute.

- **WIRED INTO THE TICK (2026-08-10).** Entries and exits both route through
  `live.ts` when — and only when — `liveEnabled()` is true, which requires all
  three of: `LIVE_TRADING` exactly `"1"`, a `TRADING_KEYPAIR` present, and the
  envelope's product being `crowetrade-live`. With any one absent the engine
  behaves exactly as it did yesterday, which is what makes the paper record a
  meaningful rehearsal rather than a different program.

  Three details worth knowing before you arm:
  - Positions carry `execution` ('paper' | 'live') and the on-chain
    `entry_sig` / `exit_sig`, so any live row is auditable against the chain by
    someone who does not trust this engine's accounting. `execution` defaults
    to 'paper', so the 168 historical closes stay correctly labelled simulated.
  - A **paper** position never takes the live exit path, even while live is
    armed. Checked per position, not globally.
  - A live exit that FAILS does not close the position at an invented price.
    It stays open, the next tick retries, and `live_exit_failed` is recorded.

## What is NOT built yet

- A second-machine test of the whole path.
- Counsel sign-off on custody and the waiver. Still the real gate.

---

## Michael's steps, in order

The order is the safety property. Each step is reversible until the last one.

### 1. Counsel first, before any key exists

Custody and the waiver draft (`shared/waiver.md`) need a lawyer's eyes. This
gates real money regardless of how good the record gets, and it has the longest
lead time of anything on this list. Starting it costs nothing.

### 2. Create a dedicated throwaway wallet

Not your main wallet. A fresh keypair whose entire balance is the dust:

```
solana-keygen new --no-bip39-passphrase --outfile ~/.crowetrade-dust.json
chmod 600 ~/.crowetrade-dust.json
solana address -k ~/.crowetrade-dust.json
```

The reason it is throwaway: total loss of this wallet must be an acceptable
outcome, because the point of the exercise is to find out what happens.

### 3. Fund it with ~$20 of SOL

Roughly 0.2 SOL at current prices. Send it to the address printed above.
Verify:

```
solana balance -k ~/.crowetrade-dust.json
```

### 4. Write a LIVE envelope and sign it

The current envelope is `crowetrade-paper` and preflight refuses to spend real
funds under it. A live envelope needs `product: "crowetrade-live"`, caps sized
for dust rather than for paper, a near expiry, and the wallet's signature over
its canonical hash.

Suggested dust caps — deliberately far below the paper values:

```
perTradeCapSol: 0.02      // ~$2 per trade
dailyCapSol:    0.10      // ~$10 per day
maxOpenPositions: 1
expiresAt: <48 hours out>
```

A near expiry is a feature. If something goes wrong and nobody notices, the
envelope dies on its own.

### 5. Set the Worker secrets

```
cd engine
npx wrangler secret put TRADING_KEYPAIR   # paste the keypair JSON
npx wrangler secret put LIVE_TRADING      # the single character: 1
```

Both are required. Either one absent leaves the path inert, and `LIVE_TRADING`
must be exactly `1` — `true`, `yes` and ` 1` all leave it disarmed on purpose.

### 6. Confirm the kill switch works BEFORE arming

```
TOKEN=$(cat engine/.admin-token)
curl -X POST $ENGINE/api/kill -H "Authorization: Bearer $TOKEN" -d '{"on":true}'
```

Confirm the book shows KILLED, then turn it off again. A kill switch you have
not tested is a kill switch you are hoping about.

### 7. Watch the first trade land

The first live entry should be watched in real time, not discovered later.
Confirm on chain with the signature the engine records, and check that the
position written to the book matches what actually happened on chain rather
than what was quoted.

### 8. Stop and read the result

One confirmed round trip is the whole deliverable. Compare quoted price against
realized fill, and check whether the daily cap accounting matches what the
wallet actually spent. If those two numbers disagree, stop and fix it before
anything larger.

---

## Reverting

`npx wrangler secret delete LIVE_TRADING` disarms everything immediately and
takes effect on the next trade, not the next deploy. The kill switch is faster
still and stops entries without stopping exits.

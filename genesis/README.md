# Genesis feed

The other vantage point. The engine sees the market once a minute through a
listing poll; by then the first seconds of a launch, where the price actually
moves, are over. Genesis watches those seconds directly.

- `collector.mjs` runs on the Pro under launchd (`launchd/`). Two sockets:
  PumpPortal's free creation stream (metadata, dev buy) and the public Solana
  RPC log stream on the pump.fun program, decoded from the anchor TradeEvent.
  Every creation and its first fifteen minutes of trades, at transaction
  resolution with curve reserves, into `~/crowetrade-genesis/genesis.db`
  (WAL, 7-day retention). It never trades.
- `genesis_report.py` replays a 0.1 SOL paper entry at creation + k seconds
  with the curve's own arithmetic (constant product on virtual reserves, 1%
  fee each way), exits by hold or take-profit on the reserves observed at
  that moment, stratified by what was knowable at entry (dev buy, buyers so
  far, sells so far, creator launch count). Runs daily at 06:50 Phoenix and
  POSTs the summary to the engine (`/api/genesis`, admin), which prints it in
  the 07:00 digest.

Measured 2026-08-31 at install: the public RPC stream delivers ~400 log
messages a second, ~45 decoded trades a second, free.

# CroweTrade

CroweTrade was a paper-trading engine that generated signals, sized positions, checked risk limits, and routed simulated orders for Crowe Logic's own strategy research.

## Status

`archived`. Development stopped in September 2025. The last commit on `main` is dated 2025-09-13. A `revival/paper-engine` branch received commits until 2026-08-31 and was never merged. The product was retired. Nothing is deployed: `crowetrade.com` did not resolve when checked on 2026-09-10. The code is kept for reference.

## Install and first run

Not maintained. No supported install path.

For reference, on 2026-09-10 with Python 3.13 in a fresh virtualenv, `pip install -e '.[dev]'` succeeded and `python -m pytest --no-cov --continue-on-collection-errors` reported 94 passed, 42 failed, 19 errors, with 4 test modules that fail to import.

## What runs today

Nothing is maintained.

- The Python package under `src/crowetrade` installs from `pyproject.toml`.
- Four test modules fail at import: a `NameError` in `src/crowetrade/messaging/__init__.py`, a `NameError` in `tests/unit/test_backtesting_basic.py`, and a missing `pandas` dependency that `pyproject.toml` does not declare.
- The Fly.io, Docker, Kubernetes and Helm files target `crowetrade.com`, which did not resolve on 2026-09-10.

## Limits

- This is not financial advice. It is a code archive.
- It was never validated with real money or live pricing. The venue factory in `src/crowetrade/services/execution_service/venue_adapter.py` returns a no-trade adapter when `PAPER_MODE=true` and raises `NotImplementedError` for live production mode without a venue adapter. The committed deployment config `.env.production` sets `PAPER_MODE=true`. Tests run against mock adapters only.
- The Coinbase adapter in `src/crowetrade/services/execution_service/coinbase_adapter.py` contains order code that no test imports or exercises. Do not use it to trade.
- Strategy, risk, and backtest results in this repository are simulated. No test or committed config runs against a live market.

## License and contact

`pyproject.toml` declares MIT. The repository has no LICENSE file.

Contact: michael@crowelogic.com

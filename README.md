# CroweTrade

This repository contains the architecture and a minimal Python skeleton for the Crowe Logic Parallel Financial Agent Ecosystem.

Quick start

- Install (development): `pip install -e .[dev]`
- Run tests: `pytest -q`
- Lint/format: `ruff check . && black --check .`

Key paths

- `src/crowetrade/` — core contracts and live agents
- `specs/` — JSON Schemas and examples for core events; policy examples
- `.github/instructions/Architecture.instructions.md` — full blueprint
- `.github/workflows/ci.yml` — CI pipeline

Roadmap

- Sprint 0: scaffolding, contracts, schemas, CI (this PR)
- Sprint 1: feature store, optimizer, risk overlays, backtest stub
- Sprint 2: execution, TCA, governance hooks, canary mode

Deployment

- Fly.io (preferred)
	- Export a Fly token as `FLYIO_TOKEN`.
	- Deploy both services with health checks:
		- `tools/deploy.sh fly`

- Local (Docker Compose fallback)
	- Bring up both services locally with health checks:
		- `tools/deploy.sh local`

Health endpoints

- Execution: `https://<host>:<port>/health` (local: `http://localhost:18080/health`)
- Portfolio: `https://<host>:<port>/health` (local: `http://localhost:18081/health`)

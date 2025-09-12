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

## Persistent Feature Store & Policy Hot-Reload

This codebase now includes:

1. Disk-backed Feature Store (`DiskFeatureStore`)
	- Append-only JSONL per key with auto-incrementing version and SHA-256 content hash.
	- Retrieval of historical snapshots: `history(key, limit)` and `get_version(key, n)`.
2. Policy Hot-Reload (`PolicyRegistry`)
	- Automatic mtime detection; policies reloaded when underlying YAML changes.
	- Stable `policy_hash` exposed and attached to every emitted `Signal` for auditability.

Example usage:

```python
from crowetrade.feature_store.store import DiskFeatureStore
from crowetrade.core.policy import PolicyRegistry

fs = DiskFeatureStore("./feature_history")
fs.put("AAPL", {"mom": 0.12, "vol": 0.34})
print(fs.get("AAPL").version)
print(len(fs.history("AAPL")))

registry = PolicyRegistry("specs/policies")
pol = registry.load("cs_mom_v1")
print(registry.policy_hash("cs_mom_v1"))
```

Backward compatibility: `InMemoryFeatureStore.put(key, values, version=...)` still accepted.


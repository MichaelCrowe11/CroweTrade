# CroweLang - CroweTrade Integration Specification (Embedded Copy)

This is the in-repo mirror of the CroweLang integration spec to ensure build-time availability and provenance.

(Original maintained in crowe-lang repo.)

## Compiler Targets
- Primary: Python (CroweTrade live agents)
- Secondary: React/TypeScript (UI prototyping), future: WASM

## Minimal Contract
Generated Python strategies must expose:
- `class <StrategyName>Strategy` with methods:
  - `generate_signals(features: dict) -> list[dict]`
  - `risk_limits() -> dict`
  - `execution_policy() -> dict`

## Drift Detection
A CI check will compare hashed AST of `.crowe` sources vs generated `.py` artifacts.

## Risk Mapping
CroweLang `risk` block → Policy YAML fields: gross/net, dd_intraday, leverage.

## Next Steps
See `ROADMAP.md` for staged adoption.

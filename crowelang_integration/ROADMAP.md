# CroweLang → CroweTrade Adoption Roadmap

## Phase 0 – Bootstrap (Current)
- Copy integration spec & exemplar strategy.
- Provide generator shim that can be replaced by direct npm/ts-node invocation later.
- Validate generated Python compiles and passes lint.

## Phase 1 – Compiler Bridge
- Add script to run `crowe-compiler` (node) to emit Python into `src/crowetrade/strategies/generated`.
- CI step: detect drift between `.crowe` sources and generated `.py` artifacts.
- Add unit tests asserting strategy class shape & basic signal function behavior.

## Phase 2 – Runtime Adapters
- Implement adapter translating CroweLang store/actions to CroweTrade FeatureVectors & Signals.
- Map CroweLang `risk` declarations to PolicyRegistry entries.

## Phase 3 – Live Experimentation
- Shadow deploy one CroweLang-generated strategy; collect metrics vs reference Python strategy.
- Canary gating: capital cap + risk guard overlays.

## Phase 4 – Native Integration
- Replace bespoke Python strategy modules with CroweLang sources + generated code in build.
- Add incremental hot-reload for research lane.

## Phase 5 – Advanced Features
- Multi-target generation (Python + wasm for sandbox simulation).
- Deterministic replay & provenance stamping.
- Static analyzers for risk rule completeness.

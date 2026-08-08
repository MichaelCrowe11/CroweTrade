# CroweTrade Analyst

The conversational surface of CroweTrade. Ask it what the system holds, why it
refused a token, how the record actually reads, and what the evidence supports.

Built as an Azure AI Foundry agent so it travels: once published it appears
wherever your Foundry agents do, phone included, rather than only at the desk.

## Why it needs no knowledge index

Every other Crowe Foundry agent grounds on a vector index. This one deliberately
does not. Its ground truth is the engine's **live** read API, because a snapshot
of yesterday's book would answer today's questions confidently and wrongly, and
confident wrongness is the exact failure the whole system exists to prevent.

## Read-only is a boundary, not a setting

The agent holds three GET endpoints and no credentials. The kill switch, veto
and policy changes require a bearer token it does not have. A model that can be
talked into flipping the kill switch is a vulnerability, not a feature, so the
conversational surface reads and explains while the operator's own authenticated
controls act.

Fetched data is treated as data. A token named "IGNORE PREVIOUS INSTRUCTIONS" is
a token, not a request.

## Layout

```
agent/instructions.md   system prompt: identity, boundaries, honesty rules
agent/tools.md          what each endpoint returns and how to read it
agent/config.yaml       model params + OpenAPI tool binding
config/engine-openapi.yaml   the three GET operations, typed
.foundry/agent-metadata.yaml project + hub + eval wiring
.foundry/datasets/           blocking honesty eval suite
```

## The eval suite is blocking on purpose

`crowetrade-analyst-honesty-v1.jsonl` tests the things that are the product:
leading with the current cohort rather than flattering lifetime aggregates,
never calling simulated results profit, refusing to act on the book, resisting
prompt injection through token names, quoting recorded refusal reasons instead
of inventing plausible ones, and carrying the exit sweep's upper-bound caveat.

A regression there is not a quality dip. It is a system that can flatter its own
losing record, which is the one failure that would make this project worthless.

## Publishing

The workspace is complete and its tools are verified against the live engine.
Publishing to the Foundry project requires the portal (this Azure deployment's
data-plane API version rejects agent-create over CLI, the same trap documented
for `crowe-product-description-writer`):

1. Foundry portal, project `crowelm-foundry` on hub `crowelm-prod-eastus2`.
2. New agent, name `crowetrade-analyst`.
3. Paste `agent/instructions.md` as instructions.
4. Actions, add OpenAPI tool, upload `config/engine-openapi.yaml`, auth: none.
5. Temperature 0.2.
6. Run the honesty suite before enabling anywhere.

## Verified live

All three operations answered correctly against the running engine on
2026-08-08: health reports paper mode, positions exposes cohorts, calibration,
budget, events, open and closed, and the exit sweep returns its caveat with
ranked rules.

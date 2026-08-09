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

## Setup

One command. It checks prerequisites, discovers which models can serve the
analyst, registers the agent, and verifies it end to end against the live
engine:

```
node scripts/setup.mjs          # check, register, verify
node scripts/setup.mjs --check  # prerequisites only, changes nothing
```

## Using it

```
export AZ_TOKEN=$(az account get-access-token --resource https://ai.azure.com --query accessToken -o tsv)
node scripts/ask.mjs "how are we doing?"
node scripts/ask.mjs "why did it skip PEPE?"
node scripts/ask.mjs "which exit rule looks best, and what is the caveat?"
```

Every answer is prefixed with which engine endpoints it consulted. An answer
that reports `answered without consulting the engine` is the model talking from
its prompt rather than the ledger, and should not be trusted.

## The blocker that was NOT a platform limit

This was blocked for hours on a misleading error, and the cause is worth
recording because nothing about it is discoverable from the message.

The legacy Assistants API (`/assistants`) **force-stamps `temperature` and
`top_p` onto every agent**, defaulting them when omitted and ignoring `null` on
update. The gpt-5.x family **rejects both parameters outright**. With a tool
attached, that specific rejection surfaces as a generic
`server_error: Sorry, something went wrong` with **empty run steps**, which
looks exactly like a broken OpenAPI tool. Stripping `tools` to `[]` reveals the
real message instantly; that is the diagnostic.

The fix is the modern route, `/openai/v1/responses`, which has no sampling
fields at all and works with gpt-5.6-sol. Anthropic models remain unavailable
on the agent path here (`invalid_deployment`); they are chat-completions only.

## Verified live

All three operations answered correctly against the running engine on
2026-08-08: health reports paper mode, positions exposes cohorts, calibration,
budget, events, open and closed, and the exit sweep returns its caveat with
ranked rules.

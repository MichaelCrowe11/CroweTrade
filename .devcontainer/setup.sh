#!/usr/bin/env bash
#
# First-create setup for the CroweTrade devcontainer.
#
# Idempotent: safe to run again by hand after a rebuild. Every step reports
# what it could not do rather than continuing quietly, because a half-built
# environment that looks finished costs more than one that stops and says so.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
fail=0

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m    SKIPPED: %s\033[0m\n' "$1"; fail=1; }

step "Terminal dependencies (desktop/)"
# Electron itself is downloaded here and is useless in a container with no
# display. That is expected. The renderer, the type checker and the test
# runner all come from the same install, and those are what matter here.
if (cd desktop && npm install --no-audit --no-fund); then
  echo "    ok"
else
  warn "npm install failed in desktop/"
fi

step "Engine local variables (engine/.dev.vars)"
if [ -f engine/.dev.vars ]; then
  echo "    already present, left alone"
elif cp engine/.dev.vars.example engine/.dev.vars; then
  echo "    created from .dev.vars.example, all values empty"
else
  warn "could not create engine/.dev.vars"
fi

step "Worker types (engine/worker-configuration.d.ts)"
# Gitignored, so a fresh clone has none and the engine will not typecheck
# until this runs. Generated FROM .dev.vars, which is why the step above has
# to come first.
if (cd engine && npx --yes wrangler@4 types); then
  echo "    ok"
else
  warn "wrangler types failed; engine typecheck will report missing Env keys"
fi

step "Test suite"
if (cd desktop && npm test >/tmp/crowetrade-test.log 2>&1); then
  tail -8 /tmp/crowetrade-test.log
else
  warn "tests did not pass on a clean checkout; see /tmp/crowetrade-test.log"
fi

printf '\n'
if [ "$fail" -eq 0 ]; then
  printf '\033[32mReady.\033[0m Read ONBOARDING.md before your first change.\n'
else
  printf '\033[33mReady with gaps.\033[0m Re-run: bash .devcontainer/setup.sh\n'
fi
printf 'Repo root: %s\n' "$ROOT"

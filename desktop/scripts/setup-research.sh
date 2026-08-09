#!/usr/bin/env bash
# Provision the CroweTrade research kernel: an isolated venv under the app's
# own data directory, holding the notebook execution stack. Idempotent.
set -euo pipefail
VENV="$HOME/Library/Application Support/CroweTrade/research-venv"
uv venv "$VENV" --python 3.13
uv pip install --python "$VENV/bin/python" ipykernel nbclient nbformat pandas
"$VENV/bin/python" -c "import nbclient, nbformat, pandas; print('research kernel ready')"

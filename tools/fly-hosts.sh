#!/usr/bin/env bash
set -euo pipefail

if ! command -v flyctl >/dev/null 2>&1; then
  echo "Installing flyctl..."
  curl -fsSL https://fly.io/install.sh | sh -s -- -b /usr/local/bin
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Installing jq..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y >/dev/null 2>&1 || true
    sudo apt-get install -y jq >/dev/null 2>&1 || true
  fi
fi

export FLYCTL_ACCESS_TOKEN="${FLYIO_TOKEN:-${FLY_API_TOKEN:-}}"

get_host() {
  local app="$1"
  local json host
  json=$(flyctl status --app "$app" --json)
  host=$(echo "$json" | jq -r '(.hostname // .Hostname // .App.Status.Hostname // empty)')
  echo "$host"
}

exec_app="crowetrade-execution"
port_app="crowetrade-portfolio"

exec_host=$(get_host "$exec_app" || true)
port_host=$(get_host "$port_app" || true)

echo "Execution: ${exec_host:-<none>}"
echo "Portfolio: ${port_host:-<none>}"
if [ -n "${exec_host:-}" ]; then echo "Exec health: https://${exec_host}/health"; fi
if [ -n "${port_host:-}" ]; then echo "Port health: https://${port_host}/health"; fi

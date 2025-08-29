#!/bin/bash
set -euo pipefail

# Simple Fly.io deploy script
# Usage: FLYIO_TOKEN=your_token ./deploy-fly-simple.sh

if [ -z "${FLYIO_TOKEN:-}" ]; then
    echo "❌ FLYIO_TOKEN not set"
    echo ""
    echo "Get your token from: https://fly.io/dashboard"
    echo "Then run: FLYIO_TOKEN=your_token ./deploy-fly-simple.sh"
    exit 1
fi

export FLYCTL_ACCESS_TOKEN="$FLYIO_TOKEN"
export PATH="/home/codespace/.fly/bin:$PATH"

echo "🚀 Deploying to Fly.io..."

# Ensure apps exist
echo "📋 Ensuring apps exist..."
flyctl apps show crowetrade-execution >/dev/null 2>&1 || flyctl apps create crowetrade-execution
flyctl apps show crowetrade-portfolio >/dev/null 2>&1 || flyctl apps create crowetrade-portfolio

# Deploy execution service
echo "🔨 Deploying execution service..."
flyctl deploy -c fly.execution.toml --remote-only --now

# Deploy portfolio service  
echo "📊 Deploying portfolio service..."
flyctl deploy -c fly.portfolio.toml --remote-only --now

# Get hostnames and test health
echo "🏥 Checking health..."
exec_host=$(flyctl status -a crowetrade-execution --json | jq -r '.hostname // empty')
port_host=$(flyctl status -a crowetrade-portfolio --json | jq -r '.hostname // empty')

echo ""
echo "✅ Deployment complete!"
echo "Execution: ${exec_host:-<pending>}"
echo "Portfolio: ${port_host:-<pending>}"

if [ -n "${exec_host:-}" ]; then
    echo "Exec health: https://${exec_host}/health"
    curl -fsS "https://${exec_host}/health" && echo " ✓"
fi

if [ -n "${port_host:-}" ]; then
    echo "Port health: https://${port_host}/health" 
    curl -fsS "https://${port_host}/health" && echo " ✓"
fi

echo ""
echo "🎉 Both services deployed and healthy!"

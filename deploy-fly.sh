#!/bin/bash

# CroweTrade Fly.io Deployment Script
# This script deploys the CroweTrade platform to your existing Fly.io apps

set -e

echo "🚀 Starting CroweTrade deployment to Fly.io..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first."
    exit 1
fi

# Check authentication
if ! flyctl auth whoami &> /dev/null; then
    echo "❌ Not authenticated with Fly.io. Please run 'flyctl auth login'"
    exit 1
fi

echo "✅ Fly.io authentication verified"

# Deploy Frontend to crowetrade-web
echo "🌐 Deploying frontend to crowetrade-web..."
flyctl deploy --config fly.frontend.toml --app crowetrade-web

# Deploy API to crowetrade-api  
echo "🔧 Deploying API to crowetrade-api..."
flyctl deploy --config fly.api.toml --app crowetrade-api

# Deploy Execution Service to crowetrade-execution
echo "⚡ Deploying execution service to crowetrade-execution..."
flyctl deploy --config fly.execution.toml --app crowetrade-execution

# Deploy Portfolio Service to crowetrade-portfolio
echo "📊 Deploying portfolio service to crowetrade-portfolio..."
flyctl deploy --config fly.portfolio.toml --app crowetrade-portfolio

echo "🎉 Deployment complete! Your apps should be available at:"
echo "Frontend: https://crowetrade-web.fly.dev"
echo "API: https://crowetrade-api.fly.dev"
echo "Execution: https://crowetrade-execution.fly.dev" 
echo "Portfolio: https://crowetrade-portfolio.fly.dev"

echo "🔗 Setting up custom domains..."
flyctl certs create crowetrade.com --app crowetrade-web
flyctl certs create api.crowetrade.com --app crowetrade-api

echo "✨ Deployment and domain setup complete!"

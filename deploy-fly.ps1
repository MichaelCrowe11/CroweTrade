# CroweTrade Fly.io Deployment Script for Windows
# This script deploys the CroweTrade platform to your existing Fly.io apps

Write-Host "🚀 Starting CroweTrade deployment to Fly.io..." -ForegroundColor Green

# Check if flyctl is installed
try {
    $null = flyctl auth whoami
    Write-Host "✅ Fly.io authentication verified" -ForegroundColor Green
}
catch {
    Write-Host "❌ Not authenticated with Fly.io. Please run 'flyctl auth login'" -ForegroundColor Red
    exit 1
}

# Deploy Frontend to crowetrade-web
Write-Host "🌐 Deploying frontend to crowetrade-web..." -ForegroundColor Cyan
flyctl deploy --config fly.frontend.toml --app crowetrade-web

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Frontend deployment failed" -ForegroundColor Red
    exit 1
}

# Deploy API to crowetrade-api  
Write-Host "🔧 Deploying API to crowetrade-api..." -ForegroundColor Cyan
flyctl deploy --config fly.api.toml --app crowetrade-api

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ API deployment failed" -ForegroundColor Red
    exit 1
}

# Deploy Execution Service to crowetrade-execution
Write-Host "⚡ Deploying execution service to crowetrade-execution..." -ForegroundColor Cyan
flyctl deploy --config fly.execution.toml --app crowetrade-execution

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Execution service deployment failed" -ForegroundColor Red
    exit 1
}

# Deploy Portfolio Service to crowetrade-portfolio
Write-Host "📊 Deploying portfolio service to crowetrade-portfolio..." -ForegroundColor Cyan
flyctl deploy --config fly.portfolio.toml --app crowetrade-portfolio

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Portfolio service deployment failed" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 Deployment complete! Your apps should be available at:" -ForegroundColor Green
Write-Host "Frontend: https://crowetrade-web.fly.dev" -ForegroundColor Yellow
Write-Host "API: https://crowetrade-api.fly.dev" -ForegroundColor Yellow
Write-Host "Execution: https://crowetrade-execution.fly.dev" -ForegroundColor Yellow
Write-Host "Portfolio: https://crowetrade-portfolio.fly.dev" -ForegroundColor Yellow

Write-Host "🔗 Setting up custom domains..." -ForegroundColor Cyan
flyctl certs create crowetrade.com --app crowetrade-web
flyctl certs create api.crowetrade.com --app crowetrade-api

Write-Host "✨ Deployment and domain setup complete!" -ForegroundColor Green

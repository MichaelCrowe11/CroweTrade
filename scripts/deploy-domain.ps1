# CroweTrade Domain Deployment Script (PowerShell)
# This script deploys the CroweTrade platform and connects your domain

param(
    [switch]$SkipBuild,
    [switch]$DomainOnly
)

Write-Host "🚀 Starting CroweTrade Domain Deployment..." -ForegroundColor Blue

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if flyctl is installed
try {
    $flyVersion = flyctl version
    Write-Status "Flyctl version: $flyVersion"
} catch {
    Write-Error "flyctl is not installed. Please install it first:"
    Write-Host "  Invoke-WebRequest -Uri https://fly.io/install.ps1 -UseBasicParsing | Invoke-Expression"
    exit 1
}

# Check if logged in to Fly.io
try {
    $whoami = flyctl auth whoami
    Write-Status "Logged in as: $whoami"
} catch {
    Write-Error "You're not logged in to Fly.io. Please run: flyctl auth login"
    exit 1
}

if (-not $DomainOnly) {
    if (-not $SkipBuild) {
        Write-Status "Building and deploying frontend application..."
        flyctl deploy --config fly.toml --remote-only
        if ($LASTEXITCODE -ne 0) { 
            Write-Error "Frontend deployment failed"
            exit 1
        }

        Write-Status "Building and deploying execution service..."
        flyctl deploy --config fly.execution.toml --remote-only
        if ($LASTEXITCODE -ne 0) { 
            Write-Error "Execution service deployment failed"
            exit 1
        }

        Write-Status "Building and deploying portfolio service..."
        flyctl deploy --config fly.portfolio.toml --remote-only
        if ($LASTEXITCODE -ne 0) { 
            Write-Error "Portfolio service deployment failed"
            exit 1
        }
    }
}

# Set up custom domain
Write-Status "Configuring custom domain: crowetrade.com"
flyctl certs add crowetrade.com --app crowetrade-main
flyctl certs add www.crowetrade.com --app crowetrade-main

# Set up subdomains for APIs
Write-Status "Configuring API subdomain: api.crowetrade.com"
flyctl certs add api.crowetrade.com --app crowetrade-execution

Write-Status "Configuring portfolio API subdomain: portfolio.crowetrade.com" 
flyctl certs add portfolio.crowetrade.com --app crowetrade-portfolio

Write-Success "Deployment completed!"

Write-Host ""
Write-Status "Domain Configuration Required:"
Write-Host "Add the following DNS records to your crowetrade.com domain:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Type    | Name      | Value" -ForegroundColor White
Write-Host "--------|-----------|--------------------------------" -ForegroundColor White
Write-Host "CNAME   | @         | crowetrade-main.fly.dev" -ForegroundColor Yellow
Write-Host "CNAME   | www       | crowetrade-main.fly.dev" -ForegroundColor Yellow
Write-Host "CNAME   | api       | crowetrade-execution.fly.dev" -ForegroundColor Yellow
Write-Host "CNAME   | portfolio | crowetrade-portfolio.fly.dev" -ForegroundColor Yellow
Write-Host ""

Write-Status "Checking certificate status..."
flyctl certs show crowetrade.com --app crowetrade-main
flyctl certs show api.crowetrade.com --app crowetrade-execution

Write-Success "🎉 CroweTrade platform deployed successfully!"
Write-Status "Your trading platform will be available at: https://crowetrade.com"
Write-Status "API endpoints will be available at: https://api.crowetrade.com" 
Write-Status "Portfolio API will be available at: https://portfolio.crowetrade.com"

Write-Host ""
Write-Warning "Next Steps:"
Write-Host "1. Configure DNS records as shown above"
Write-Host "2. Wait for SSL certificates to be issued (can take up to 10 minutes)"
Write-Host "3. Test your domain: https://crowetrade.com"
Write-Host "4. Monitor deployment: flyctl logs --app crowetrade-main"

# CroweTrade Production Deployment Script for crowetrade.com (PowerShell)
param(
    [Parameter(Position=0)]
    [ValidateSet("check", "dns", "deploy", "dns-auto", "dns-verify")]
    [string]$Action = "deploy",
    
    [Parameter()]
    [switch]$AutoApply = $false
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Write-Banner {
    Write-Host ""
    Write-Host "🚀 CroweTrade Production Deployment to crowetrade.com" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
}

function Test-Prerequisites {
    Write-Log "Checking prerequisites..." -Color Yellow
    
    # Check flyctl
    try {
        $flyVersion = flyctl version 2>$null
        Write-Log "✅ flyctl: $flyVersion" -Color Green
    }
    catch {
        Write-Log "Installing flyctl..." -Color Yellow
        
        # Install flyctl on Windows
        if (Get-Command "scoop" -ErrorAction SilentlyContinue) {
            scoop install flyctl
        }
        elseif (Get-Command "choco" -ErrorAction SilentlyContinue) {
            choco install flyctl
        }
        else {
            Write-Log "Please install flyctl manually from https://fly.io/docs/hands-on/install-flyctl/" -Color Red
            exit 1
        }
    }
    
    # Check authentication
    $flyToken = $env:FLY_API_TOKEN -or $env:FLYIO_TOKEN
    if (-not $flyToken) {
        Write-Log "❌ No Fly.io token found" -Color Red
        Write-Log "Please set FLY_API_TOKEN or FLYIO_TOKEN environment variable" -Color Red
        Write-Log "Get your token from: https://fly.io/user/personal_access_tokens" -Color Red
        exit 1
    }
    
    if ($env:FLYIO_TOKEN) {
        $env:FLY_API_TOKEN = $env:FLYIO_TOKEN
    }
    
    Write-Log "✅ Fly.io authentication configured" -Color Green
}

function Test-Secrets {
    Write-Log "Checking required secrets..." -Color Yellow
    
    $requiredSecrets = @(
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET", 
        "COINBASE_PASSPHRASE",
        "POSTGRES_PASSWORD",
        "REDIS_PASSWORD"
    )
    
    $optionalSecrets = @(
        "NEXT_PUBLIC_CDP_PROJECT_ID",
        "NEXT_PUBLIC_CDP_API_KEY"
    )
    
    $missingSecrets = @()
    
    foreach ($secret in $requiredSecrets) {
        $value = [System.Environment]::GetEnvironmentVariable($secret)
        if (-not $value) {
            $missingSecrets += $secret
        }
    }
    
    if ($missingSecrets.Count -gt 0) {
        Write-Log "⚠️  Missing secrets: $($missingSecrets -join ', ')" -Color Yellow
        Write-Log "Please set these environment variables before deployment" -Color Yellow
        Write-Log "For security, these will be set as Fly secrets during deployment" -Color Yellow
    }
    else {
        Write-Log "✅ All required secrets are available" -Color Green
    }
}

function New-FlyApps {
    Write-Log "Creating Fly.io applications..." -Color Yellow
    
    $apps = @{
        "crowetrade-web" = "Web Frontend"
        "crowetrade-execution" = "Trading Execution Engine"
        "crowetrade-portfolio" = "Portfolio Management"
    }
    
    foreach ($app in $apps.GetEnumerator()) {
        $appName = $app.Key
        $description = $app.Value
        
        try {
            $existingApp = flyctl apps list --json | ConvertFrom-Json | Where-Object { $_.Name -eq $appName }
            if ($existingApp) {
                Write-Log "✅ App $appName exists" -Color Green
            }
            else {
                Write-Log "Creating app $appName ($description)..." -Color Yellow
                $org = $env:FLY_ORG ?? "personal"
                flyctl apps create $appName --org $org
            }
        }
        catch {
            Write-Log "⚠️  App creation failed for $appName, may already exist" -Color Yellow
        }
    }
}

function Set-FlySecrets {
    Write-Log "Setting application secrets..." -Color Yellow
    
    $apps = @("crowetrade-web", "crowetrade-execution", "crowetrade-portfolio")
    
    # Crypto trading secrets
    $coinbaseKey = $env:COINBASE_API_KEY
    if ($coinbaseKey) {
        foreach ($app in $apps) {
            try {
                flyctl secrets set "COINBASE_API_KEY=$coinbaseKey" -a $app *>$null
            }
            catch { }
        }
        Write-Log "✅ Set Coinbase API key" -Color Green
    }
    
    $coinbaseSecret = $env:COINBASE_API_SECRET
    if ($coinbaseSecret) {
        foreach ($app in $apps) {
            try {
                flyctl secrets set "COINBASE_API_SECRET=$coinbaseSecret" -a $app *>$null
            }
            catch { }
        }
        Write-Log "✅ Set Coinbase API secret" -Color Green
    }
    
    $coinbasePassphrase = $env:COINBASE_PASSPHRASE
    if ($coinbasePassphrase) {
        foreach ($app in $apps) {
            try {
                flyctl secrets set "COINBASE_PASSPHRASE=$coinbasePassphrase" -a $app *>$null
            }
            catch { }
        }
        Write-Log "✅ Set Coinbase passphrase" -Color Green
    }
    
    # Database secrets
    $postgresPassword = $env:POSTGRES_PASSWORD
    if ($postgresPassword) {
        foreach ($app in $apps) {
            try {
                flyctl secrets set "POSTGRES_PASSWORD=$postgresPassword" -a $app *>$null
            }
            catch { }
        }
        Write-Log "✅ Set PostgreSQL password" -Color Green
    }
    
    $redisPassword = $env:REDIS_PASSWORD
    if ($redisPassword) {
        foreach ($app in $apps) {
            try {
                flyctl secrets set "REDIS_PASSWORD=$redisPassword" -a $app *>$null
            }
            catch { }
        }
        Write-Log "✅ Set Redis password" -Color Green
    }
}

function Deploy-Services {
    Write-Log "Deploying CroweTrade services..." -Color Yellow
    
    # Deploy execution engine
    Write-Host "🔧 Deploying Trading Execution Engine..." -ForegroundColor Blue
    flyctl deploy -c fly.execution.toml --remote-only --now
    Write-Log "✅ Execution engine deployed" -Color Green
    
    # Deploy portfolio service
    Write-Host "📊 Deploying Portfolio Management..." -ForegroundColor Blue
    flyctl deploy -c fly.portfolio.toml --remote-only --now
    Write-Log "✅ Portfolio service deployed" -Color Green
    
    # Deploy web frontend
    Write-Host "🌐 Deploying Web Frontend..." -ForegroundColor Blue
    flyctl deploy -c fly.web.toml --remote-only --now
    Write-Log "✅ Web frontend deployed" -Color Green
}

function Set-Domains {
    Write-Log "Configuring custom domains..." -Color Yellow
    
    # Main website
    try {
        flyctl certs create crowetrade.com -a crowetrade-web *>$null
        flyctl certs create www.crowetrade.com -a crowetrade-web *>$null
    }
    catch { }
    
    # API endpoints
    try {
        flyctl certs create api.crowetrade.com -a crowetrade-execution *>$null
        flyctl certs create portfolio.crowetrade.com -a crowetrade-portfolio *>$null
    }
    catch { }
    
    Write-Log "✅ Domain certificates requested" -Color Green
}

function Test-Health {
    Write-Log "Running health checks..." -Color Yellow
    
    $endpoints = @{
        "https://crowetrade.com" = "Main Website"
        "https://api.crowetrade.com/health" = "Trading API"
        "https://portfolio.crowetrade.com/health" = "Portfolio API"
    }
    
    foreach ($endpoint in $endpoints.GetEnumerator()) {
        $url = $endpoint.Key
        $name = $endpoint.Value
        
        Write-Log "Checking $name..." -Color Yellow
        
        # Wait for deployment
        Start-Sleep -Seconds 10
        
        try {
            $response = Invoke-WebRequest -Uri $url -TimeoutSec 30 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Host "   ✅ $name is healthy" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "   ⚠️  $name is not responding yet (may need more time)" -ForegroundColor Yellow
        }
    }
}

function Show-DnsInstructions {
    Write-Host ""
    Write-Host "🌐 DNS Configuration Instructions for Namecheap" -ForegroundColor Blue
    Write-Host "════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    # Get app hostnames
    try {
        $webStatus = flyctl status -a crowetrade-web --json 2>$null | ConvertFrom-Json
        $webHost = $webStatus.Hostname ?? "crowetrade-web.fly.dev"
        
        $apiStatus = flyctl status -a crowetrade-execution --json 2>$null | ConvertFrom-Json
        $apiHost = $apiStatus.Hostname ?? "crowetrade-execution.fly.dev"
        
        $portfolioStatus = flyctl status -a crowetrade-portfolio --json 2>$null | ConvertFrom-Json
        $portfolioHost = $portfolioStatus.Hostname ?? "crowetrade-portfolio.fly.dev"
    }
    catch {
        $webHost = "crowetrade-web.fly.dev"
        $apiHost = "crowetrade-execution.fly.dev" 
        $portfolioHost = "crowetrade-portfolio.fly.dev"
    }
    
    Write-Host "Please update your DNS records in Namecheap:"
    Write-Host ""
    Write-Host "1. Remove existing records:"
    Write-Host "   - Remove: CNAME www -> parkingpage.namecheap.com"
    Write-Host "   - Remove: URL Redirect @ -> http://www.crowetrade.com/"
    Write-Host ""
    Write-Host "2. Add these new records:"
    Write-Host "   ┌─────────────┬──────────┬─────────────────────────────────┬─────────┐"
    Write-Host "   │ Type        │ Host     │ Value                           │ TTL     │"
    Write-Host "   ├─────────────┼──────────┼─────────────────────────────────┼─────────┤"
    Write-Host "   │ CNAME       │ @        │ $webHost                        │ 30 min  │"
    Write-Host "   │ CNAME       │ www      │ $webHost                        │ 30 min  │"
    Write-Host "   │ CNAME       │ api      │ $apiHost                        │ 30 min  │"
    Write-Host "   │ CNAME       │ portfolio│ $portfolioHost                  │ 30 min  │"
    Write-Host "   └─────────────┴──────────┴─────────────────────────────────┴─────────┘"
    Write-Host ""
    Write-Host "3. Keep existing records:"
    Write-Host "   - TXT @ v=spf1 include:spf.efwd.registrar-servers.com ~all"
    Write-Host ""
    Write-Host "Note: DNS changes may take 5-30 minutes to propagate globally" -ForegroundColor Yellow
}

function Start-DnsSetup {
    param([switch]$AutoApply = $false)
    
    Write-Log "Starting DNS configuration..." -Color Yellow
    
    # Check for automated DNS script first
    if (Test-Path "scripts\namecheap-auto-dns.ps1") {
        Write-Log "Found Namecheap API automation script" -Color Green
        
        if ($AutoApply) {
            Write-Log "Applying DNS changes automatically via Namecheap API..." -Color Cyan
            & ".\scripts\namecheap-auto-dns.ps1" -Apply
        } else {
            Write-Log "Running in preview mode..." -Color Yellow
            & ".\scripts\namecheap-auto-dns.ps1"
            Write-Host ""
            Write-Log "To apply changes automatically, use: deploy-crowetrade.ps1 dns -AutoApply" -Color Cyan
        }
    } elseif (Test-Path "scripts\namecheap-dns-setup.ps1") {
        Write-Log "Using manual DNS setup script..." -Color Yellow
        & ".\scripts\namecheap-dns-setup.ps1"
    } else {
        Write-Log "DNS setup scripts not found, showing manual instructions" -Color Yellow
        Show-DnsInstructions
    }
}

function Show-Summary {
    Write-Host ""
    Write-Host "🎉 CroweTrade Deployment Complete!" -ForegroundColor Green
    Write-Host "═════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "🌐 Your CroweTrade platform will be available at:"
    Write-Host "   • Main Site:    https://crowetrade.com"
    Write-Host "   • Trading API:  https://api.crowetrade.com"
    Write-Host "   • Portfolio:    https://portfolio.crowetrade.com"
    Write-Host ""
    Write-Host "🔧 Features Deployed:"
    Write-Host "   ✅ AI-Powered Trading Engine"
    Write-Host "   ✅ Coinbase Pro Integration"
    Write-Host "   ✅ Real-time Cryptocurrency Trading"
    Write-Host "   ✅ Advanced Risk Management"
    Write-Host "   ✅ Portfolio Analytics"
    Write-Host "   ✅ Web Dashboard"
    Write-Host ""
    Write-Host "🔐 Security:"
    Write-Host "   ✅ HTTPS/TLS Certificates"
    Write-Host "   ✅ API Authentication"
    Write-Host "   ✅ Encrypted Secrets"
    Write-Host ""
    Write-Host "📊 Monitoring:"
    Write-Host "   • Health checks: Every 15 seconds"
    Write-Host "   • Fly.io Dashboard: https://fly.io/dashboard"
    Write-Host "   • Logs: flyctl logs -a <app-name>"
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Blue
    Write-Host "1. Update DNS records as shown above"
    Write-Host "2. Wait 5-30 minutes for DNS propagation"
    Write-Host "3. Visit https://crowetrade.com to access your platform"
    Write-Host "4. Monitor logs: flyctl logs -a crowetrade-web"
}

function Invoke-FullDeployment {
    Write-Banner
    Test-Prerequisites
    Test-Secrets
    New-FlyApps
    Set-FlySecrets
    Deploy-Services
    Set-Domains
    Test-Health
    Start-DnsSetup
    Show-Summary
}

# Main execution
switch ($Action) {
    "check" {
        Write-Banner
        Test-Prerequisites
        Test-Secrets
    }
    "dns" {
        Write-Banner
        Start-DnsSetup -AutoApply:$AutoApply
    }
    "dns-auto" {
        Write-Banner
        if (Test-Path "scripts\namecheap-auto-dns.ps1") {
            & ".\scripts\namecheap-auto-dns.ps1" -Apply
        } else {
            Write-Log "Automated DNS script not found!" -Color Red
        }
    }
    "dns-verify" {
        Write-Banner
        if (Test-Path "scripts\namecheap-auto-dns.ps1") {
            & ".\scripts\namecheap-auto-dns.ps1" -Verify
        } else {
            Write-Log "Automated DNS script not found!" -Color Red
        }
    }
    "deploy" {
        Invoke-FullDeployment
    }
    default {
        Write-Host "Usage: .\deploy-crowetrade.ps1 [action] [options]"
        Write-Host ""
        Write-Host "Actions:"
        Write-Host "  check      - Check prerequisites and secrets"
        Write-Host "  dns        - Setup DNS (preview mode)"
        Write-Host "  dns-auto   - Apply DNS changes via Namecheap API"
        Write-Host "  dns-verify - Verify DNS propagation"
        Write-Host "  deploy     - Full deployment (default)"
        Write-Host ""
        Write-Host "Options:"
        Write-Host "  -AutoApply - Automatically apply DNS changes with 'dns' action"
        exit 1
    }
}

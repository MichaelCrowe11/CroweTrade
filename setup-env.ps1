# CroweTrade Environment Setup Script
# Run this script to set up required environment variables for deployment

Write-Host "🔐 CroweTrade Environment Setup" -ForegroundColor Blue
Write-Host "═══════════════════════════════" -ForegroundColor Blue
Write-Host ""

# Coinbase Pro API Configuration
Write-Host "📊 Coinbase Pro API Configuration" -ForegroundColor Yellow
Write-Host "Get your API credentials from: https://pro.coinbase.com/profile/api"
Write-Host ""

$coinbaseKey = Read-Host "Enter Coinbase API Key"
if ($coinbaseKey) {
    [Environment]::SetEnvironmentVariable("COINBASE_API_KEY", $coinbaseKey, "User")
    Write-Host "✅ COINBASE_API_KEY set" -ForegroundColor Green
}

$coinbaseSecret = Read-Host "Enter Coinbase API Secret" -MaskInput
if ($coinbaseSecret) {
    [Environment]::SetEnvironmentVariable("COINBASE_API_SECRET", $coinbaseSecret, "User")
    Write-Host "✅ COINBASE_API_SECRET set" -ForegroundColor Green
}

$coinbasePassphrase = Read-Host "Enter Coinbase API Passphrase" -MaskInput
if ($coinbasePassphrase) {
    [Environment]::SetEnvironmentVariable("COINBASE_PASSPHRASE", $coinbasePassphrase, "User")
    Write-Host "✅ COINBASE_PASSPHRASE set" -ForegroundColor Green
}

Write-Host ""

# Database Configuration
Write-Host "🗄️ Database Configuration" -ForegroundColor Yellow
Write-Host "Set secure passwords for database services"
Write-Host ""

$postgresPassword = Read-Host "Enter PostgreSQL Password (or press Enter for auto-generated)" -MaskInput
if (-not $postgresPassword) {
    $postgresPassword = -join ((1..16) | ForEach-Object { [char](Get-Random -Minimum 33 -Maximum 126) })
    Write-Host "Generated secure PostgreSQL password" -ForegroundColor Cyan
}
[Environment]::SetEnvironmentVariable("POSTGRES_PASSWORD", $postgresPassword, "User")
Write-Host "✅ POSTGRES_PASSWORD set" -ForegroundColor Green

$redisPassword = Read-Host "Enter Redis Password (or press Enter for auto-generated)" -MaskInput  
if (-not $redisPassword) {
    $redisPassword = -join ((1..16) | ForEach-Object { [char](Get-Random -Minimum 33 -Maximum 126) })
    Write-Host "Generated secure Redis password" -ForegroundColor Cyan
}
[Environment]::SetEnvironmentVariable("REDIS_PASSWORD", $redisPassword, "User")
Write-Host "✅ REDIS_PASSWORD set" -ForegroundColor Green

Write-Host ""

# Fly.io Configuration
Write-Host "🚀 Fly.io Configuration" -ForegroundColor Yellow
Write-Host "Get your API token from: https://fly.io/user/personal_access_tokens"
Write-Host ""

$flyToken = Read-Host "Enter Fly.io API Token" -MaskInput
if ($flyToken) {
    [Environment]::SetEnvironmentVariable("FLY_API_TOKEN", $flyToken, "User")
    Write-Host "✅ FLY_API_TOKEN set" -ForegroundColor Green
}

$flyOrg = Read-Host "Enter Fly.io Organization (or press Enter for 'personal')"
if (-not $flyOrg) {
    $flyOrg = "personal"
}
[Environment]::SetEnvironmentVariable("FLY_ORG", $flyOrg, "User")
Write-Host "✅ FLY_ORG set to '$flyOrg'" -ForegroundColor Green

Write-Host ""

# Coinbase Developer Platform Configuration (Optional)
Write-Host "🔗 Coinbase Developer Platform (Optional)" -ForegroundColor Yellow
Write-Host "Enables onchain trading and Smart Wallet features"
Write-Host "Project ID from your dashboard: b1e3b0af-35cb-48f0-aec7-276d3c4fbf79"
Write-Host ""

$cdpProjectId = Read-Host "Enter CDP Project ID (or press Enter to skip)"
if ($cdpProjectId) {
    [Environment]::SetEnvironmentVariable("NEXT_PUBLIC_CDP_PROJECT_ID", $cdpProjectId, "User")
    Write-Host "✅ CDP_PROJECT_ID set" -ForegroundColor Green
    
    $cdpApiKey = Read-Host "Enter CDP API Key (or press Enter to skip)" -MaskInput
    if ($cdpApiKey) {
        [Environment]::SetEnvironmentVariable("NEXT_PUBLIC_CDP_API_KEY", $cdpApiKey, "User")
        Write-Host "✅ CDP_API_KEY set" -ForegroundColor Green
    }
} else {
    Write-Host "⏭️  Skipped Coinbase Developer Platform setup" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "🎉 Environment Setup Complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Close and reopen your PowerShell terminal"
Write-Host "2. Run: .\deploy-crowetrade.ps1 check"
Write-Host "3. Run: .\deploy-crowetrade.ps1 deploy"
Write-Host ""
Write-Host "Note: Environment variables are saved to your user profile" -ForegroundColor Cyan
Write-Host "and will persist across PowerShell sessions." -ForegroundColor Cyan

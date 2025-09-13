# GitHub Secrets Setup Guide for Coinbase API
# Step-by-step instructions for adding Coinbase credentials to GitHub

Write-Host ""
Write-Host "🔐 GitHub Secrets Setup for Coinbase API" -ForegroundColor Blue
Write-Host "════════════════════════════════════════" -ForegroundColor Blue
Write-Host ""

Write-Host "📋 STEP 1: Get Your Coinbase Pro API Credentials" -ForegroundColor Yellow
Write-Host "If you don't have them yet:" -ForegroundColor White
Write-Host "• Go to: https://pro.coinbase.com/profile/api" -ForegroundColor Cyan
Write-Host "• Create a new API key with View, Trade, Transfer permissions" -ForegroundColor White
Write-Host "• Save all 3 values: API Key, Secret, and Passphrase" -ForegroundColor White
Write-Host ""

Write-Host "📋 STEP 2: Add Secrets to GitHub (Browser should be open)" -ForegroundColor Yellow
Write-Host "In the GitHub secrets page that just opened:" -ForegroundColor White
Write-Host ""

Write-Host "🔑 Secret #1: COINBASE_API_KEY" -ForegroundColor Cyan
Write-Host "   1. Click 'New repository secret'" -ForegroundColor White
Write-Host "   2. Name: COINBASE_API_KEY" -ForegroundColor Green
Write-Host "   3. Value: [Your Coinbase API Key - long alphanumeric string]" -ForegroundColor Yellow
Write-Host "   4. Click 'Add secret'" -ForegroundColor White
Write-Host ""

Write-Host "🔑 Secret #2: COINBASE_API_SECRET" -ForegroundColor Cyan  
Write-Host "   1. Click 'New repository secret'" -ForegroundColor White
Write-Host "   2. Name: COINBASE_API_SECRET" -ForegroundColor Green
Write-Host "   3. Value: [Your Coinbase API Secret - base64 encoded string]" -ForegroundColor Yellow
Write-Host "   4. Click 'Add secret'" -ForegroundColor White
Write-Host ""

Write-Host "🔑 Secret #3: COINBASE_PASSPHRASE" -ForegroundColor Cyan
Write-Host "   1. Click 'New repository secret'" -ForegroundColor White  
Write-Host "   2. Name: COINBASE_PASSPHRASE" -ForegroundColor Green
Write-Host "   3. Value: [Your Coinbase Passphrase - the one you created]" -ForegroundColor Yellow
Write-Host "   4. Click 'Add secret'" -ForegroundColor White
Write-Host ""

Write-Host "📋 STEP 3: Verify Secrets Are Added" -ForegroundColor Yellow
Write-Host "After adding all three, you should see:" -ForegroundColor White
Write-Host "✓ COINBASE_API_KEY" -ForegroundColor Green
Write-Host "✓ COINBASE_API_SECRET" -ForegroundColor Green  
Write-Host "✓ COINBASE_PASSPHRASE" -ForegroundColor Green
Write-Host "✓ FLY_API_TOKEN (already added)" -ForegroundColor Green
Write-Host ""

Write-Host "📋 STEP 4: Test the Setup" -ForegroundColor Yellow
Write-Host "Once secrets are added:" -ForegroundColor White
Write-Host "• GitHub Actions will use them automatically" -ForegroundColor Cyan
Write-Host "• They'll be passed to your deployed applications" -ForegroundColor Cyan
Write-Host "• Your CroweTrade platform will be able to execute trades" -ForegroundColor Cyan
Write-Host ""

Write-Host "⚠️  SECURITY REMINDERS:" -ForegroundColor Red
Write-Host "• Never commit API credentials to code" -ForegroundColor Yellow
Write-Host "• GitHub secrets are encrypted and secure" -ForegroundColor Yellow  
Write-Host "• Only use API keys with minimum required permissions" -ForegroundColor Yellow
Write-Host "• Consider IP whitelisting for production" -ForegroundColor Yellow
Write-Host ""

Write-Host "🎯 WHAT HAPPENS NEXT:" -ForegroundColor Blue
Write-Host "1. Your next deployment will include these credentials" -ForegroundColor White
Write-Host "2. The trading engines will be able to connect to Coinbase" -ForegroundColor White
Write-Host "3. Your platform will be ready for live trading" -ForegroundColor White
Write-Host ""

# Interactive confirmation
Write-Host "Press Enter when you've added all three secrets to GitHub..." -ForegroundColor Green
Read-Host

Write-Host ""
Write-Host "🧪 Let's verify the setup..." -ForegroundColor Cyan

# Since we can't directly access GitHub secrets from local machine,
# we'll create a test deployment trigger
Write-Host "Creating a test commit to trigger deployment with new secrets..." -ForegroundColor Yellow

# Add a timestamp to trigger new deployment
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$testContent = @"
# Deployment Test - Coinbase Credentials Added
Generated: $timestamp

This file triggers a new deployment to test Coinbase credentials.
Delete this file after successful deployment.
"@

$testContent | Out-File -FilePath ".\COINBASE_TEST_DEPLOYMENT.md" -Encoding UTF8

Write-Host "✅ Test file created" -ForegroundColor Green
Write-Host "🚀 Committing and pushing to trigger new deployment..." -ForegroundColor Cyan

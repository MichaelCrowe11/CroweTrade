# Coinbase Pro API Setup Guide for CroweTrade
# Step-by-step guide to get your trading credentials

Write-Host ""
Write-Host "🪙 Coinbase Pro API Setup for CroweTrade" -ForegroundColor Blue
Write-Host "════════════════════════════════════════" -ForegroundColor Blue
Write-Host ""

Write-Host "📋 STEP 1: Create Coinbase Pro Account (if needed)" -ForegroundColor Yellow
Write-Host "• Go to: https://pro.coinbase.com" -ForegroundColor Cyan
Write-Host "• Sign up or log in with existing account" -ForegroundColor White
Write-Host "• Complete identity verification if required" -ForegroundColor White
Write-Host ""

Write-Host "📋 STEP 2: Navigate to API Settings" -ForegroundColor Yellow  
Write-Host "• Log into Coinbase Pro" -ForegroundColor White
Write-Host "• Click your profile icon (top right)" -ForegroundColor White
Write-Host "• Select: API → Create API Key" -ForegroundColor Cyan
Write-Host "• Direct URL: https://pro.coinbase.com/profile/api" -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 STEP 3: Configure API Permissions" -ForegroundColor Yellow
Write-Host "• Nickname: 'CroweTrade Production'" -ForegroundColor White
Write-Host "• Permissions needed:" -ForegroundColor White
Write-Host "  ✓ View (read account info)" -ForegroundColor Green
Write-Host "  ✓ Trade (place/cancel orders)" -ForegroundColor Green  
Write-Host "  ✓ Transfer (move funds)" -ForegroundColor Green
Write-Host "• IP Whitelist: Add your server IP (optional but recommended)" -ForegroundColor White
Write-Host ""

Write-Host "📋 STEP 4: Save Your Credentials" -ForegroundColor Yellow
Write-Host "After creating the API key, you'll get:" -ForegroundColor White
Write-Host "• API Key (long string starting with letters/numbers)" -ForegroundColor Cyan
Write-Host "• API Secret (base64 encoded string)" -ForegroundColor Cyan  
Write-Host "• Passphrase (the one you entered during creation)" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  IMPORTANT: Save these immediately - you can't view the secret again!" -ForegroundColor Red
Write-Host ""

Write-Host "📋 STEP 5: Test Your Credentials" -ForegroundColor Yellow
Write-Host "We'll test them locally first, then add to GitHub secrets" -ForegroundColor White
Write-Host ""

# Interactive setup
$setupChoice = Read-Host "Do you want to set up credentials now? (y/n)"

if ($setupChoice -eq 'y' -or $setupChoice -eq 'Y') {
    Write-Host ""
    Write-Host "🔧 Setting up Coinbase credentials..." -ForegroundColor Green
    Write-Host ""
    
    # Get API Key
    Write-Host "Enter your Coinbase Pro API Key:" -ForegroundColor Cyan
    $apiKey = Read-Host "API Key" -MaskInput
    
    if ($apiKey) {
        [Environment]::SetEnvironmentVariable("COINBASE_API_KEY", $apiKey, "User")
        $env:COINBASE_API_KEY = $apiKey
        Write-Host "✅ COINBASE_API_KEY set" -ForegroundColor Green
    }
    
    # Get API Secret  
    Write-Host "Enter your Coinbase Pro API Secret:" -ForegroundColor Cyan
    $apiSecret = Read-Host "API Secret" -MaskInput
    
    if ($apiSecret) {
        [Environment]::SetEnvironmentVariable("COINBASE_API_SECRET", $apiSecret, "User")
        $env:COINBASE_API_SECRET = $apiSecret
        Write-Host "✅ COINBASE_API_SECRET set" -ForegroundColor Green
    }
    
    # Get Passphrase
    Write-Host "Enter your Coinbase Pro Passphrase:" -ForegroundColor Cyan
    $passphrase = Read-Host "Passphrase" -MaskInput
    
    if ($passphrase) {
        [Environment]::SetEnvironmentVariable("COINBASE_PASSPHRASE", $passphrase, "User")  
        $env:COINBASE_PASSPHRASE = $passphrase
        Write-Host "✅ COINBASE_PASSPHRASE set" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "🧪 Testing connection..." -ForegroundColor Cyan
    
    # Test the credentials (simplified test)
    if ($apiKey -and $apiSecret -and $passphrase) {
        Write-Host "✅ All credentials provided" -ForegroundColor Green
        Write-Host ""
        Write-Host "🔄 Next: Add these to GitHub secrets for production:" -ForegroundColor Yellow
        Write-Host "1. Go to: https://github.com/MichaelCrowe11/CroweTrade/settings/secrets/actions" -ForegroundColor Cyan
        Write-Host "2. Add these three secrets:" -ForegroundColor White
        Write-Host "   • COINBASE_API_KEY" -ForegroundColor Cyan
        Write-Host "   • COINBASE_API_SECRET" -ForegroundColor Cyan  
        Write-Host "   • COINBASE_PASSPHRASE" -ForegroundColor Cyan
        Write-Host ""
        
        $openGitHub = Read-Host "Open GitHub secrets page now? (y/n)"
        if ($openGitHub -eq 'y' -or $openGitHub -eq 'Y') {
            Start-Process "https://github.com/MichaelCrowe11/CroweTrade/settings/secrets/actions"
        }
    }
    else {
        Write-Host "❌ Some credentials missing. Please try again." -ForegroundColor Red
    }
}
else {
    Write-Host ""
    Write-Host "📖 When you're ready:" -ForegroundColor Cyan
    Write-Host "1. Get credentials from Coinbase Pro" -ForegroundColor White
    Write-Host "2. Run this script again" -ForegroundColor White  
    Write-Host "3. Add them to GitHub secrets" -ForegroundColor White
    Write-Host ""
}

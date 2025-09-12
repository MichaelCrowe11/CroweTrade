# Namecheap API Setup Wizard - Step by Step Guide
param(
    [Parameter()]
    [switch]$CheckStatus = $false
)

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Show-DetailedSetup {
    Write-Host ""
    Write-Host "🔧 Namecheap API Setup - Step by Step" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    Write-Host "STEP 1: Login to Namecheap" -ForegroundColor Yellow
    Write-Host "  • Go to: https://www.namecheap.com" -ForegroundColor White
    Write-Host "  • Login with your Namecheap account credentials" -ForegroundColor White
    Write-Host ""
    
    Write-Host "STEP 2: Navigate to API Settings" -ForegroundColor Yellow
    Write-Host "  • Click your username/profile in the top right" -ForegroundColor White
    Write-Host "  • Select: Profile → Tools → API Access" -ForegroundColor White
    Write-Host "  • Direct URL: https://ap.www.namecheap.com/settings/tools/apiaccess/" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "STEP 3: Enable API Access" -ForegroundColor Yellow
    Write-Host "  • Look for 'API Access' toggle switch" -ForegroundColor White
    Write-Host "  • Switch it to 'ON' (this is the most important step!)" -ForegroundColor Green
    Write-Host "  • If you don't see this option, contact Namecheap support" -ForegroundColor Red
    Write-Host ""
    
    Write-Host "STEP 4: Whitelist Your IP" -ForegroundColor Yellow
    Write-Host "  • Find 'Whitelisted IP Addresses' section" -ForegroundColor White
    Write-Host "  • Add this IP address: 98.186.221.213" -ForegroundColor Cyan
    Write-Host "  • Click 'Add' or 'Save'" -ForegroundColor White
    Write-Host ""
    
    Write-Host "STEP 5: Get Your API Credentials" -ForegroundColor Yellow
    Write-Host "  • API Username: This is your Namecheap username" -ForegroundColor White
    Write-Host "  • API Key: Look for 'Generate' or 'API Key' button" -ForegroundColor White
    Write-Host "  • Copy the generated API key (long string)" -ForegroundColor White
    Write-Host ""
    
    Write-Host "STEP 6: Test the Connection" -ForegroundColor Yellow
    Write-Host "  • Run: .\scripts\namecheap-setup-wizard.ps1 -CheckStatus" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "🚨 COMMON ISSUES:" -ForegroundColor Red
    Write-Host "  • API Access toggle is OFF (most common)" -ForegroundColor Yellow
    Write-Host "  • Wrong API key (make sure to copy the full key)" -ForegroundColor Yellow
    Write-Host "  • Wrong username (use exact Namecheap username)" -ForegroundColor Yellow
    Write-Host "  • Account doesn't have API access privileges" -ForegroundColor Yellow
    Write-Host ""
}

function Test-APIConnection {
    Write-Log "Testing current Namecheap API setup..." -Color Yellow
    
    if (-not $env:NAMECHEAP_API_USER) {
        Write-Log "❌ NAMECHEAP_API_USER environment variable not set" -Color Red
        return $false
    }
    
    if (-not $env:NAMECHEAP_API_KEY) {
        Write-Log "❌ NAMECHEAP_API_KEY environment variable not set" -Color Red
        return $false
    }
    
    Write-Log "✅ Environment variables are set:" -Color Green
    Write-Log "   API User: $env:NAMECHEAP_API_USER" -Color Cyan
    Write-Log "   API Key: $($env:NAMECHEAP_API_KEY.Substring(0,[Math]::Min(8,$env:NAMECHEAP_API_KEY.Length)))..." -Color Cyan
    
    # Get public IP
    try {
        $ip = (Invoke-RestMethod -Uri "https://api.ipify.org?format=json").ip
        Write-Log "   Your IP: $ip" -Color Cyan
    } catch {
        Write-Log "   Could not determine IP" -Color Yellow
        $ip = "0.0.0.0"
    }
    
    # Test API call
    $params = @{
        ApiUser = $env:NAMECHEAP_API_USER
        ApiKey = $env:NAMECHEAP_API_KEY
        UserName = $env:NAMECHEAP_USERNAME ?? $env:NAMECHEAP_API_USER
        ClientIp = $ip
        Command = "namecheap.users.getBalances"
    }
    
    try {
        Write-Log "Testing API connection..." -Color Yellow
        $response = Invoke-RestMethod -Uri "https://api.namecheap.com/xml.response" -Method GET -Body $params
        
        if ($response.ApiResponse.Status -eq "OK") {
            Write-Log "✅ SUCCESS! API connection is working!" -Color Green
            Write-Log "Your Namecheap API is properly configured." -Color Green
            return $true
        } else {
            Write-Log "❌ API Error Details:" -Color Red
            Write-Log "   Status: $($response.ApiResponse.Status)" -Color Red
            
            if ($response.ApiResponse.Errors.Error) {
                foreach ($error in $response.ApiResponse.Errors.Error) {
                    Write-Log "   Error: $($error.InnerText)" -Color Red
                    
                    # Provide specific guidance based on error
                    switch ($error.Number) {
                        "1011102" { 
                            Write-Log "   → This means API access is NOT enabled on your account" -Color Yellow
                            Write-Log "   → Go to Namecheap and turn ON the API Access toggle" -Color Cyan
                        }
                        "1011150" { 
                            Write-Log "   → Your IP address is not whitelisted" -Color Yellow
                            Write-Log "   → Add $ip to your Namecheap IP whitelist" -Color Cyan
                        }
                        "1011101" { 
                            Write-Log "   → Invalid API key or username" -Color Yellow
                            Write-Log "   → Double-check your API credentials" -Color Cyan
                        }
                        default {
                            Write-Log "   → Check the Namecheap API documentation" -Color Yellow
                        }
                    }
                }
            }
            return $false
        }
    } catch {
        Write-Log "❌ Connection Error: $($_.Exception.Message)" -Color Red
        return $false
    }
}

function Interactive-Setup {
    Write-Host ""
    Write-Host "🎯 Interactive API Setup" -ForegroundColor Blue
    Write-Host "═══════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    Write-Log "Let's set up your Namecheap API credentials..." -Color Yellow
    Write-Host ""
    
    # Get API Username
    if ($env:NAMECHEAP_API_USER) {
        Write-Host "Current API User: $env:NAMECHEAP_API_USER" -ForegroundColor Gray
        $useExisting = Read-Host "Keep this username? (Y/n)"
        if ($useExisting -match "^[Nn]") {
            $apiUser = Read-Host "Enter your Namecheap username"
            [Environment]::SetEnvironmentVariable("NAMECHEAP_API_USER", $apiUser, "User")
            $env:NAMECHEAP_API_USER = $apiUser
        }
    } else {
        $apiUser = Read-Host "Enter your Namecheap username"
        [Environment]::SetEnvironmentVariable("NAMECHEAP_API_USER", $apiUser, "User")
        $env:NAMECHEAP_API_USER = $apiUser
    }
    
    # Get API Key
    Write-Host ""
    Write-Host "Now we need your API Key from Namecheap:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://ap.www.namecheap.com/settings/tools/apiaccess/" -ForegroundColor Cyan
    Write-Host "2. Make sure 'API Access' is turned ON" -ForegroundColor Green
    Write-Host "3. Copy your API Key" -ForegroundColor Cyan
    Write-Host ""
    
    $apiKey = Read-Host "Paste your API Key here" -MaskInput
    if ($apiKey) {
        [Environment]::SetEnvironmentVariable("NAMECHEAP_API_KEY", $apiKey, "User")
        $env:NAMECHEAP_API_KEY = $apiKey
        [Environment]::SetEnvironmentVariable("NAMECHEAP_USERNAME", $env:NAMECHEAP_API_USER, "User")
        $env:NAMECHEAP_USERNAME = $env:NAMECHEAP_API_USER
    }
    
    Write-Host ""
    Write-Log "Testing the connection..." -Color Yellow
    Test-APIConnection
}

# Main execution
if ($CheckStatus) {
    Test-APIConnection
} else {
    Show-DetailedSetup
    Write-Host ""
    $setup = Read-Host "Would you like to enter your API credentials now? (y/N)"
    if ($setup -match "^[Yy]") {
        Interactive-Setup
    } else {
        Write-Host ""
        Write-Host "When ready, run: .\scripts\namecheap-setup-wizard.ps1 -CheckStatus" -ForegroundColor Cyan
    }
}

# Namecheap API Test and Setup Helper
param(
    [Parameter(Mandatory=$false)]
    [switch]$Test = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Setup = $false
)

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Get-PublicIP {
    try {
        $ip = (Invoke-RestMethod -Uri "https://api.ipify.org?format=json").ip
        return $ip
    } catch {
        Write-Log "Could not determine public IP" -Color Red
        return $null
    }
}

function Test-NamecheapAPI {
    Write-Log "Testing Namecheap API connection..." -Color Yellow
    
    $ip = Get-PublicIP
    Write-Log "Your public IP: $ip" -Color Cyan
    
    if (-not ($env:NAMECHEAP_API_USER -and $env:NAMECHEAP_API_KEY)) {
        Write-Log "Missing API credentials in environment variables" -Color Red
        return $false
    }
    
    # Test API connection
    $params = @{
        ApiUser = $env:NAMECHEAP_API_USER
        ApiKey = $env:NAMECHEAP_API_KEY  
        UserName = $env:NAMECHEAP_USERNAME ?? $env:NAMECHEAP_API_USER
        ClientIp = $ip
        Command = "namecheap.users.getBalances"
    }
    
    try {
        Write-Log "Testing API call to get account balances..." -Color Yellow
        $response = Invoke-RestMethod -Uri "https://api.namecheap.com/xml.response" -Method GET -Body $params
        
        Write-Log "Raw API Response:" -Color Cyan
        Write-Host $response.OuterXml -ForegroundColor Gray
        
        if ($response.ApiResponse.Status -eq "OK") {
            Write-Log "✅ API connection successful!" -Color Green
            return $true
        } else {
            Write-Log "❌ API Error: $($response.ApiResponse.Errors.Error.InnerText)" -Color Red
            return $false
        }
    } catch {
        Write-Log "❌ API Connection failed: $($_.Exception.Message)" -Color Red
        Write-Log "Response: $($_.Exception.Response)" -Color Red
        return $false
    }
}

function Show-SetupInstructions {
    Write-Host ""
    Write-Host "🔧 Namecheap API Setup Instructions" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    $ip = Get-PublicIP
    
    Write-Host "1. Enable API Access:" -ForegroundColor Yellow
    Write-Host "   • Login to Namecheap.com"
    Write-Host "   • Go to: Profile → Tools → API Access"  
    Write-Host "   • https://ap.www.namecheap.com/settings/tools/apiaccess/"
    Write-Host ""
    
    Write-Host "2. Whitelist Your IP Address:" -ForegroundColor Yellow
    Write-Host "   • Add this IP to the whitelist: $ip" -ForegroundColor Cyan
    Write-Host "   • Note: You may need to update this if your IP changes"
    Write-Host ""
    
    Write-Host "3. Get API Credentials:" -ForegroundColor Yellow
    Write-Host "   • API User: Your Namecheap username"
    Write-Host "   • API Key: Generated in the API Access section"
    Write-Host "   • Username: Same as API User (usually)"
    Write-Host ""
    
    Write-Host "4. Test the API:" -ForegroundColor Yellow
    Write-Host "   • Run: .\scripts\namecheap-test.ps1 -Test"
    Write-Host ""
    
    Write-Host "🔴 Common Issues:" -ForegroundColor Red
    Write-Host "   • IP not whitelisted (most common)"
    Write-Host "   • API access not enabled on account"
    Write-Host "   • Incorrect API key or username"
    Write-Host "   • Using sandbox URL instead of production"
    Write-Host ""
}

if ($Setup) {
    Show-SetupInstructions
} elseif ($Test) {
    Test-NamecheapAPI
} else {
    Write-Host "Usage:"
    Write-Host "  .\scripts\namecheap-test.ps1 -Setup   # Show setup instructions"
    Write-Host "  .\scripts\namecheap-test.ps1 -Test    # Test API connection"
}

# Get Alternative IP Address for Namecheap API Whitelisting
# This script helps when your current IP is already in use by another account

param(
    [Parameter()]
    [switch]$ShowAll = $false
)

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Get-MultipleIPSources {
    Write-Log "Getting IP addresses from multiple sources..." -Color Yellow
    
    $sources = @(
        @{ Name = "ipify.org"; Url = "https://api.ipify.org?format=json"; JSONKey = "ip" },
        @{ Name = "ipapi.co"; Url = "https://ipapi.co/json/"; JSONKey = "ip" },
        @{ Name = "httpbin.org"; Url = "https://httpbin.org/ip"; JSONKey = "origin" },
        @{ Name = "ip-api.com"; Url = "http://ip-api.com/json/"; JSONKey = "query" },
        @{ Name = "whatismyipaddress"; Url = "https://api.whatismyipaddress.com/ip"; JSONKey = $null }
    )
    
    $results = @{}
    
    foreach ($source in $sources) {
        try {
            Write-Host "  Checking $($source.Name)..." -NoNewline
            
            if ($source.JSONKey) {
                $response = Invoke-RestMethod -Uri $source.Url -TimeoutSec 10
                if ($source.JSONKey -eq "origin" -and $response.origin -match "(\d+\.\d+\.\d+\.\d+)") {
                    $ip = $matches[1]
                } else {
                    $ip = $response.($source.JSONKey)
                }
            } else {
                $ip = (Invoke-RestMethod -Uri $source.Url -TimeoutSec 10).Trim()
            }
            
            $results[$source.Name] = $ip
            Write-Host " $ip" -ForegroundColor Green
        }
        catch {
            Write-Host " Failed" -ForegroundColor Red
            $results[$source.Name] = "Failed"
        }
    }
    
    return $results
}

function Get-VPNOptions {
    Write-Host ""
    Write-Host "🔄 VPN/Proxy Options to Get Different IP:" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    Write-Host "Option 1 - Mobile Hotspot:" -ForegroundColor Yellow
    Write-Host "  • Use your phone's mobile hotspot"
    Write-Host "  • This will give you a different IP from your cellular carrier"
    Write-Host "  • Usually the easiest option"
    Write-Host ""
    
    Write-Host "Option 2 - Free VPN Services:" -ForegroundColor Yellow
    Write-Host "  • ProtonVPN (free tier available)"
    Write-Host "  • Windscribe (free tier available)"
    Write-Host "  • TunnelBear (free tier available)"
    Write-Host ""
    
    Write-Host "Option 3 - Browser VPN Extensions:" -ForegroundColor Yellow
    Write-Host "  • Setup Namecheap API through browser with VPN extension"
    Write-Host "  • Then use that IP for the API calls"
    Write-Host ""
    
    Write-Host "Option 4 - Cloud Instance:" -ForegroundColor Yellow
    Write-Host "  • Use a cloud VM (AWS, Azure, Google Cloud free tiers)"
    Write-Host "  • Run the deployment from there"
    Write-Host ""
}

function Show-CurrentNetworkInfo {
    Write-Log "Current Network Information:" -Color Cyan
    
    try {
        # Get network adapter info
        $adapters = Get-NetAdapter | Where-Object { $_.Status -eq "Up" }
        foreach ($adapter in $adapters) {
            $config = Get-NetIPConfiguration -InterfaceIndex $adapter.InterfaceIndex -ErrorAction SilentlyContinue
            if ($config -and $config.IPv4Address) {
                Write-Host "  $($adapter.Name): $($config.IPv4Address.IPAddress)" -ForegroundColor Gray
            }
        }
        
        # Get default gateway
        $gateway = (Get-NetRoute -DestinationPrefix "0.0.0.0/0").NextHop | Select-Object -First 1
        Write-Host "  Gateway: $gateway" -ForegroundColor Gray
        
    } catch {
        Write-Host "  Could not retrieve network info" -ForegroundColor Red
    }
    Write-Host ""
}

# Main execution
Write-Host ""
Write-Host "🌐 CroweTrade Alternative IP Address Helper" -ForegroundColor Blue
Write-Host "════════════════════════════════════════════" -ForegroundColor Blue
Write-Host ""

Write-Log "Your current IP (98.186.221.213) is already in use by another Namecheap account" -Color Yellow
Write-Host ""

Show-CurrentNetworkInfo

$ips = Get-MultipleIPSources

Write-Host ""
Write-Host "📊 IP Address Results:" -ForegroundColor Cyan
$uniqueIPs = $ips.Values | Where-Object { $_ -ne "Failed" } | Sort-Object -Unique

if ($uniqueIPs.Count -eq 1) {
    Write-Host "  All sources return the same IP: $($uniqueIPs[0])" -ForegroundColor Yellow
    Write-Host "  This confirms your current public IP address." -ForegroundColor Yellow
} else {
    Write-Host "  Multiple IPs detected:" -ForegroundColor Green
    foreach ($ip in $uniqueIPs) {
        Write-Host "    • $ip" -ForegroundColor Cyan
    }
}

if ($ShowAll) {
    Write-Host ""
    Write-Host "Detailed Results:" -ForegroundColor Gray
    foreach ($result in $ips.GetEnumerator()) {
        Write-Host "  $($result.Key): $($result.Value)" -ForegroundColor Gray
    }
}

Get-VPNOptions

Write-Host ""
Write-Host "💡 Quick Solution:" -ForegroundColor Green
Write-Host "1. Enable mobile hotspot on your phone"
Write-Host "2. Connect your computer to the hotspot"
Write-Host "3. Run: .\scripts\namecheap-test.ps1 -Setup"
Write-Host "4. Use the new IP address shown"
Write-Host "5. Complete the Namecheap API setup with that IP"
Write-Host ""
Write-Host "After setup, you can switch back to your regular internet." -ForegroundColor Yellow

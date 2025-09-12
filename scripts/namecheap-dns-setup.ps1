# Namecheap DNS Configuration Script for CroweTrade
# Automates DNS record updates for crowetrade.com deployment

param(
    [Parameter(Mandatory=$false)]
    [string]$Domain = "crowetrade.com",
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Get-FlyAppHostnames {
    Write-Log "Retrieving Fly.io app hostnames..." -Color Yellow
    
    $hostnames = @{}
    
    try {
        # Get web frontend hostname
        $webStatus = flyctl status -a crowetrade-web --json 2>$null | ConvertFrom-Json
        $hostnames["web"] = $webStatus.Hostname ?? "crowetrade-web.fly.dev"
        
        # Get execution service hostname
        $execStatus = flyctl status -a crowetrade-execution --json 2>$null | ConvertFrom-Json
        $hostnames["api"] = $execStatus.Hostname ?? "crowetrade-execution.fly.dev"
        
        # Get portfolio service hostname
        $portfolioStatus = flyctl status -a crowetrade-portfolio --json 2>$null | ConvertFrom-Json
        $hostnames["portfolio"] = $portfolioStatus.Hostname ?? "crowetrade-portfolio.fly.dev"
    }
    catch {
        Write-Log "Could not retrieve live hostnames, using defaults" -Color Yellow
        $hostnames = @{
            "web" = "crowetrade-web.fly.dev"
            "api" = "crowetrade-execution.fly.dev"  
            "portfolio" = "crowetrade-portfolio.fly.dev"
        }
    }
    
    return $hostnames
}

function Show-NamecheapInstructions {
    param([hashtable]$Hostnames)
    
    Write-Host ""
    Write-Host "🌐 Namecheap DNS Configuration Instructions" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    Write-Host "1. Login to Namecheap:" -ForegroundColor Yellow
    Write-Host "   • Go to: https://ap.www.namecheap.com/"
    Write-Host "   • Navigate to Domain List > Manage > Advanced DNS"
    Write-Host ""
    
    Write-Host "2. Remove Existing Records:" -ForegroundColor Red
    Write-Host "   ❌ Remove: CNAME Record 'www' -> 'parkingpage.namecheap.com'"
    Write-Host "   ❌ Remove: URL Redirect Record '@' -> 'http://www.crowetrade.com/'"
    Write-Host ""
    
    Write-Host "3. Add New DNS Records:" -ForegroundColor Green
    Write-Host "   ┌─────────────┬──────────┬─────────────────────────────────┬─────────┐"
    Write-Host "   │ Type        │ Host     │ Value                           │ TTL     │"
    Write-Host "   ├─────────────┼──────────┼─────────────────────────────────┼─────────┤"
    Write-Host "   │ CNAME       │ @        │ $($Hostnames.web)               │ 30 min  │"
    Write-Host "   │ CNAME       │ www      │ $($Hostnames.web)               │ 30 min  │"
    Write-Host "   │ CNAME       │ api      │ $($Hostnames.api)               │ 30 min  │"
    Write-Host "   │ CNAME       │ portfolio│ $($Hostnames.portfolio)         │ 30 min  │"
    Write-Host "   └─────────────┴──────────┴─────────────────────────────────┴─────────┘"
    Write-Host ""
    
    Write-Host "4. Keep Existing Records:" -ForegroundColor Cyan
    Write-Host "   ✅ Keep: TXT Record '@' -> 'v=spf1 include:spf.efwd.registrar-servers.com ~all'"
    Write-Host ""
    
    Write-Host "5. DNS Propagation:" -ForegroundColor Yellow
    Write-Host "   • Changes take 5-30 minutes to propagate globally"
    Write-Host "   • Test with: nslookup crowetrade.com"
    Write-Host "   • Verify SSL certificates auto-provision after DNS resolves"
    Write-Host ""
}

function Test-DnsResolution {
    param([string]$Domain, [hashtable]$ExpectedTargets)
    
    Write-Log "Testing DNS resolution..." -Color Yellow
    
    $results = @{}
    
    foreach ($subdomain in @("", "www", "api", "portfolio")) {
        $fqdn = if ($subdomain -eq "") { $Domain } else { "$subdomain.$Domain" }
        
        try {
            $result = Resolve-DnsName -Name $fqdn -Type CNAME -ErrorAction SilentlyContinue
            if ($result) {
                $results[$fqdn] = $result[0].NameHost
                Write-Host "   ✅ $fqdn -> $($result[0].NameHost)" -ForegroundColor Green
            } else {
                $results[$fqdn] = "Not resolved"
                Write-Host "   ⚠️  $fqdn -> Not resolved" -ForegroundColor Yellow
            }
        }
        catch {
            $results[$fqdn] = "Error"
            Write-Host "   ❌ $fqdn -> DNS Error" -ForegroundColor Red
        }
    }
    
    return $results
}

function Export-DnsConfig {
    param([hashtable]$Hostnames, [string]$OutputPath = "dns-config.json")
    
    $config = @{
        "domain" = $Domain
        "timestamp" = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        "records" = @{
            "@" = @{
                "type" = "CNAME"
                "value" = $Hostnames.web
                "ttl" = "1800"
            }
            "www" = @{
                "type" = "CNAME" 
                "value" = $Hostnames.web
                "ttl" = "1800"
            }
            "api" = @{
                "type" = "CNAME"
                "value" = $Hostnames.api
                "ttl" = "1800"
            }
            "portfolio" = @{
                "type" = "CNAME"
                "value" = $Hostnames.portfolio
                "ttl" = "1800"
            }
        }
    }
    
    $config | ConvertTo-Json -Depth 3 | Out-File -FilePath $OutputPath -Encoding UTF8
    Write-Log "DNS configuration exported to: $OutputPath" -Color Green
}

# Main execution
Write-Log "🌐 CroweTrade Namecheap DNS Setup" -Color Blue
Write-Log "Domain: $Domain" -Color Cyan

if ($DryRun) {
    Write-Log "DRY RUN MODE - No actual changes will be made" -Color Yellow
}

# Get Fly.io hostnames
$hostnames = Get-FlyAppHostnames

Write-Log "Retrieved hostnames:" -Color Green
foreach ($key in $hostnames.Keys) {
    Write-Log "  $key -> $($hostnames[$key])" -Color Cyan
}

# Show manual configuration instructions
Show-NamecheapInstructions -Hostnames $hostnames

# Export configuration
Export-DnsConfig -Hostnames $hostnames

# Test current DNS resolution
Write-Host ""
Test-DnsResolution -Domain $Domain -ExpectedTargets $hostnames

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Blue
Write-Host "1. Follow the instructions above to update DNS records in Namecheap"
Write-Host "2. Wait 5-30 minutes for DNS propagation"
Write-Host "3. Run this script again to verify DNS resolution"
Write-Host "4. Visit https://crowetrade.com to confirm deployment"
Write-Host ""
Write-Host "💡 Pro Tip: You can automate this with Namecheap API in the future!" -ForegroundColor Cyan

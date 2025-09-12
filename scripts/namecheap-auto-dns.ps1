# Automated Namecheap DNS Configuration for CroweTrade
# Uses Namecheap API to automatically update DNS records for crowetrade.com

param(
    [Parameter(Mandatory=$false)]
    [string]$Domain = "crowetrade.com",
    
    [Parameter(Mandatory=$false)]
    [switch]$Apply = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verify = $false
)

$ErrorActionPreference = "Stop"

# Namecheap API Configuration
$NAMECHEAP_CONFIG = @{
    ApiUrl = "https://api.namecheap.com/xml.response"
    SandboxUrl = "https://api.sandbox.namecheap.com/xml.response"
    UseSandbox = $false
}

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
        Write-Log "Could not determine public IP, using fallback" -Color Yellow
        return "127.0.0.1"
    }
}

function Get-NamecheapCredentials {
    Write-Log "Namecheap API credentials are required for automation" -Color Yellow
    Write-Log "Get these from: https://ap.www.namecheap.com/settings/tools/apiaccess/" -Color Cyan
    Write-Host ""
    
    if (-not $env:NAMECHEAP_API_USER) {
        $apiUser = Read-Host "Enter Namecheap API Username"
        if ($apiUser) {
            [Environment]::SetEnvironmentVariable("NAMECHEAP_API_USER", $apiUser, "User")
            $env:NAMECHEAP_API_USER = $apiUser
        }
    }
    
    if (-not $env:NAMECHEAP_API_KEY) {
        $apiKey = Read-Host "Enter Namecheap API Key" -MaskInput
        if ($apiKey) {
            [Environment]::SetEnvironmentVariable("NAMECHEAP_API_KEY", $apiKey, "User")
            $env:NAMECHEAP_API_KEY = $apiKey
        }
    }
    
    if (-not $env:NAMECHEAP_USERNAME) {
        $username = Read-Host "Enter Namecheap Username (same as API user usually)"
        if (-not $username) { $username = $env:NAMECHEAP_API_USER }
        [Environment]::SetEnvironmentVariable("NAMECHEAP_USERNAME", $username, "User")
        $env:NAMECHEAP_USERNAME = $username
    }
    
    return @{
        ApiUser = $env:NAMECHEAP_API_USER
        ApiKey = $env:NAMECHEAP_API_KEY
        Username = $env:NAMECHEAP_USERNAME
        ClientIP = Get-PublicIP
    }
}

function Invoke-NamecheapAPI {
    param(
        [hashtable]$Credentials,
        [string]$Command,
        [hashtable]$Parameters = @{}
    )
    
    $baseParams = @{
        ApiUser = $Credentials.ApiUser
        ApiKey = $Credentials.ApiKey
        UserName = $Credentials.Username
        ClientIp = $Credentials.ClientIP
        Command = $Command
    }
    
    $allParams = $baseParams + $Parameters
    $url = $NAMECHEAP_CONFIG.ApiUrl
    if ($NAMECHEAP_CONFIG.UseSandbox) {
        $url = $NAMECHEAP_CONFIG.SandboxUrl
    }
    
    try {
        $response = Invoke-RestMethod -Uri $url -Method GET -Body $allParams
        return $response
    } catch {
        Write-Log "Namecheap API Error: $($_.Exception.Message)" -Color Red
        throw
    }
}

function Get-CurrentDNSRecords {
    param([hashtable]$Credentials, [string]$Domain)
    
    Write-Log "Fetching current DNS records for $Domain..." -Color Yellow
    
    $domainParts = $Domain.Split('.')
    $sld = $domainParts[0]  # Second Level Domain (crowetrade)
    $tld = $domainParts[1]  # Top Level Domain (com)
    
    $params = @{
        SLD = $sld
        TLD = $tld
    }
    
    try {
        $response = Invoke-NamecheapAPI -Credentials $Credentials -Command "namecheap.domains.dns.getHosts" -Parameters $params
        return $response.ApiResponse.CommandResponse.DomainDNSGetHostsResult.host
    } catch {
        Write-Log "Could not fetch current DNS records: $($_.Exception.Message)" -Color Red
        return @()
    }
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
        
        Write-Log "✅ Retrieved Fly.io hostnames" -Color Green
        foreach ($key in $hostnames.Keys) {
            Write-Log "   $key -> $($hostnames[$key])" -Color Cyan
        }
    } catch {
        Write-Log "Could not retrieve live hostnames, using defaults" -Color Yellow
        $hostnames = @{
            "web" = "crowetrade-web.fly.dev"
            "api" = "crowetrade-execution.fly.dev"
            "portfolio" = "crowetrade-portfolio.fly.dev"
        }
    }
    
    return $hostnames
}

function Set-NamecheapDNSRecords {
    param(
        [hashtable]$Credentials,
        [string]$Domain,
        [hashtable]$Hostnames
    )
    
    Write-Log "Updating DNS records for $Domain..." -Color Yellow
    
    $domainParts = $Domain.Split('.')
    $sld = $domainParts[0]
    $tld = $domainParts[1]
    
    # Build the new DNS records
    $records = @(
        @{ Name = "@"; Type = "CNAME"; Address = $Hostnames.web; TTL = "1800" }
        @{ Name = "www"; Type = "CNAME"; Address = $Hostnames.web; TTL = "1800" }
        @{ Name = "api"; Type = "CNAME"; Address = $Hostnames.api; TTL = "1800" }
        @{ Name = "portfolio"; Type = "CNAME"; Address = $Hostnames.portfolio; TTL = "1800" }
    )
    
    # Keep existing TXT records (like SPF)
    $currentRecords = Get-CurrentDNSRecords -Credentials $Credentials -Domain $Domain
    $txtRecords = $currentRecords | Where-Object { $_.Type -eq "TXT" }
    
    # Add TXT records to our update
    foreach ($txtRecord in $txtRecords) {
        $records += @{
            Name = $txtRecord.Name
            Type = "TXT"
            Address = $txtRecord.Address
            TTL = $txtRecord.TTL ?? "1800"
        }
    }
    
    # Build API parameters for batch update
    $params = @{
        SLD = $sld
        TLD = $tld
    }
    
    # Add each record as numbered parameters
    for ($i = 0; $i -lt $records.Count; $i++) {
        $record = $records[$i]
        $params["HostName$($i+1)"] = $record.Name
        $params["RecordType$($i+1)"] = $record.Type
        $params["Address$($i+1)"] = $record.Address
        $params["TTL$($i+1)"] = $record.TTL
    }
    
    try {
        Write-Log "Applying DNS changes..." -Color Yellow
        $response = Invoke-NamecheapAPI -Credentials $Credentials -Command "namecheap.domains.dns.setHosts" -Parameters $params
        
        if ($response.ApiResponse.CommandResponse.DomainDNSSetHostsResult.IsSuccess -eq "true") {
            Write-Log "✅ DNS records updated successfully!" -Color Green
            return $true
        } else {
            Write-Log "❌ DNS update failed" -Color Red
            return $false
        }
    } catch {
        Write-Log "Error updating DNS: $($_.Exception.Message)" -Color Red
        return $false
    }
}

function Test-DNSPropagation {
    param([string]$Domain, [hashtable]$ExpectedHostnames)
    
    Write-Log "Testing DNS propagation..." -Color Yellow
    
    $subdomains = @("", "www", "api", "portfolio")
    $results = @{}
    
    foreach ($subdomain in $subdomains) {
        $fqdn = if ($subdomain -eq "") { $Domain } else { "$subdomain.$Domain" }
        $expected = switch ($subdomain) {
            "" { $ExpectedHostnames.web }
            "www" { $ExpectedHostnames.web }
            "api" { $ExpectedHostnames.api }  
            "portfolio" { $ExpectedHostnames.portfolio }
        }
        
        try {
            $result = Resolve-DnsName -Name $fqdn -Type CNAME -ErrorAction SilentlyContinue
            if ($result -and $result[0].NameHost -eq $expected) {
                Write-Host "   ✅ $fqdn -> $($result[0].NameHost)" -ForegroundColor Green
                $results[$fqdn] = "OK"
            } elseif ($result) {
                Write-Host "   ⚠️  $fqdn -> $($result[0].NameHost) (expected: $expected)" -ForegroundColor Yellow
                $results[$fqdn] = "Mismatch"
            } else {
                Write-Host "   ⏳ $fqdn -> Not resolved yet" -ForegroundColor Yellow  
                $results[$fqdn] = "Pending"
            }
        } catch {
            Write-Host "   ❌ $fqdn -> DNS Error" -ForegroundColor Red
            $results[$fqdn] = "Error"
        }
    }
    
    return $results
}

function Show-SSLStatus {
    param([string]$Domain)
    
    Write-Log "Checking SSL certificate status..." -Color Yellow
    
    $urls = @(
        "https://$Domain",
        "https://www.$Domain",
        "https://api.$Domain",
        "https://portfolio.$Domain"
    )
    
    foreach ($url in $urls) {
        try {
            $response = Invoke-WebRequest -Uri $url -Method HEAD -TimeoutSec 10 -UseBasicParsing -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "   ✅ $url - SSL OK" -ForegroundColor Green
            }
        } catch {
            Write-Host "   ⏳ $url - Not ready yet" -ForegroundColor Yellow
        }
    }
}

# Main execution
function Start-AutomatedDNSSetup {
    Write-Host ""
    Write-Host "🚀 CroweTrade Automated DNS Setup via Namecheap API" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    # Get Fly.io hostnames
    $hostnames = Get-FlyAppHostnames
    
    if (-not $Apply -and -not $Verify) {
        Write-Log "DRY RUN MODE - Use -Apply to make actual changes" -Color Yellow
        Write-Host ""
        Write-Host "Proposed DNS Changes:" -ForegroundColor Cyan
        Write-Host "┌─────────────┬─────────────────────────────────┐"
        Write-Host "│ Record      │ Target                          │"
        Write-Host "├─────────────┼─────────────────────────────────┤"
        Write-Host "│ @           │ $($hostnames.web)               │"
        Write-Host "│ www         │ $($hostnames.web)               │"  
        Write-Host "│ api         │ $($hostnames.api)               │"
        Write-Host "│ portfolio   │ $($hostnames.portfolio)         │"
        Write-Host "└─────────────┴─────────────────────────────────┘"
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Run with -Apply to update DNS records"
        Write-Host "2. Run with -Verify to check propagation"
        return
    }
    
    if ($Verify) {
        $results = Test-DNSPropagation -Domain $Domain -ExpectedHostnames $hostnames
        Show-SSLStatus -Domain $Domain
        
        $allOK = ($results.Values | Where-Object { $_ -eq "OK" }).Count -eq $results.Count
        if ($allOK) {
            Write-Host ""
            Write-Host "🎉 DNS propagation complete! Your site is ready:" -ForegroundColor Green
            Write-Host "   • https://$Domain"
            Write-Host "   • https://api.$Domain"  
            Write-Host "   • https://portfolio.$Domain"
        } else {
            Write-Host ""
            Write-Host "⏳ DNS still propagating. Check again in 5-10 minutes." -ForegroundColor Yellow
        }
        return
    }
    
    if ($Apply) {
        # Get Namecheap credentials
        $credentials = Get-NamecheapCredentials
        
        if (-not ($credentials.ApiUser -and $credentials.ApiKey)) {
            Write-Log "Missing Namecheap API credentials. Cannot proceed." -Color Red
            Write-Log "Get API access at: https://ap.www.namecheap.com/settings/tools/apiaccess/" -Color Cyan
            return
        }
        
        Write-Host ""
        Write-Host "🔄 Applying DNS changes..." -ForegroundColor Yellow
        
        $success = Set-NamecheapDNSRecords -Credentials $credentials -Domain $Domain -Hostnames $hostnames
        
        if ($success) {
            Write-Host ""
            Write-Host "✅ DNS records updated successfully!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Next steps:" -ForegroundColor Cyan
            Write-Host "1. Wait 5-30 minutes for DNS propagation"
            Write-Host "2. Run: .\scripts\namecheap-auto-dns.ps1 -Verify"
            Write-Host "3. Visit: https://$Domain"
        } else {
            Write-Log "DNS update failed. Check API credentials and try again." -Color Red
        }
    }
}

# Run the main function
Start-AutomatedDNSSetup

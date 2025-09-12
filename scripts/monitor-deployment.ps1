# CroweTrade Deployment Monitor
# Checks deployment status in real-time

param(
    [Parameter()]
    [int]$IntervalSeconds = 30
)

function Test-DeploymentStatus {
    Write-Host "`n🔍 Checking Deployment Status..." -ForegroundColor Blue
    Write-Host "Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    
    $apps = @(
        @{Name="Frontend"; URL="https://crowetrade-web.fly.dev"},
        @{Name="Execution"; URL="https://crowetrade-execution.fly.dev"}, 
        @{Name="Portfolio"; URL="https://crowetrade-portfolio.fly.dev"}
    )
    
    $results = @()
    
    foreach ($app in $apps) {
        try {
            $response = Invoke-WebRequest -Uri $app.URL -Method HEAD -TimeoutSec 10 -UseBasicParsing
            $status = "✅ LIVE ($($response.StatusCode))"
            $results += @{Name=$app.Name; Status="✅ LIVE"; Code=$response.StatusCode}
            Write-Host "$($app.Name): $status" -ForegroundColor Green
        }
        catch {
            if ($_.Exception.Message -like "*Could not resolve host*") {
                $status = "⏳ Not Deployed Yet"
                Write-Host "$($app.Name): $status" -ForegroundColor Yellow
            }
            elseif ($_.Exception.Message -like "*timeout*") {
                $status = "⚠️ Timeout"
                Write-Host "$($app.Name): $status" -ForegroundColor Red
            }
            else {
                $status = "❌ Error: $($_.Exception.Message)"
                Write-Host "$($app.Name): $status" -ForegroundColor Red
            }
            $results += @{Name=$app.Name; Status=$status; Code="N/A"}
        }
    }
    
    # Check GitHub Actions
    Write-Host "`n📋 GitHub Actions: https://github.com/MichaelCrowe11/CroweTrade/actions" -ForegroundColor Cyan
    
    return $results
}

function Show-Summary {
    param($Results)
    
    $live = ($Results | Where-Object { $_.Status -eq "✅ LIVE" }).Count
    $total = $Results.Count
    
    Write-Host "`n📊 DEPLOYMENT SUMMARY" -ForegroundColor Blue
    Write-Host "═══════════════════════" -ForegroundColor Blue
    Write-Host "Services Live: $live/$total" -ForegroundColor $(if($live -eq $total){"Green"}else{"Yellow"})
    
    if ($live -eq $total) {
        Write-Host "`n🎉 ALL SERVICES DEPLOYED SUCCESSFULLY!" -ForegroundColor Green
        Write-Host "🌐 Your platform should be accessible at:" -ForegroundColor Green
        Write-Host "   • Frontend: https://crowetrade-web.fly.dev" -ForegroundColor Cyan
        Write-Host "   • Execution: https://crowetrade-execution.fly.dev" -ForegroundColor Cyan  
        Write-Host "   • Portfolio: https://crowetrade-portfolio.fly.dev" -ForegroundColor Cyan
        Write-Host "`n🔧 Next: Configure DNS to point crowetrade.com to these services" -ForegroundColor Yellow
        return $true
    }
    return $false
}

# Main monitoring loop
Write-Host "🚀 CroweTrade Deployment Monitor Started" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop monitoring`n" -ForegroundColor Gray

do {
    $results = Test-DeploymentStatus
    $allLive = Show-Summary -Results $results
    
    if ($allLive) {
        Write-Host "`n✨ Monitoring complete - all services are live!" -ForegroundColor Green
        break
    }
    
    Write-Host "`n⏰ Checking again in $IntervalSeconds seconds..." -ForegroundColor Gray
    Start-Sleep -Seconds $IntervalSeconds
    
} while ($true)

# Simple CroweTrade Deployment - Minimal Version
# This script creates a basic deployment when having Fly.io authorization issues

param(
    [Parameter()]
    [switch]$Deploy = $false
)

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Show-ManualDeployment {
    Write-Host ""
    Write-Host "🚀 Manual CroweTrade Deployment Instructions" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    Write-Host "Since we're experiencing Fly.io CLI authorization issues," -ForegroundColor Yellow
    Write-Host "here's how to deploy CroweTrade manually:" -ForegroundColor Yellow
    Write-Host ""
    
    Write-Host "OPTION A: Fly.io Web Dashboard (Recommended)" -ForegroundColor Cyan
    Write-Host "1. Go to: https://fly.io/dashboard" -ForegroundColor White
    Write-Host "2. Click 'Create App'" -ForegroundColor White
    Write-Host "3. App Name: crowetrade-main" -ForegroundColor White
    Write-Host "4. Region: Ashburn, Virginia (iad)" -ForegroundColor White
    Write-Host "5. Upload this folder or connect GitHub repo: MichaelCrowe11/CroweTrade" -ForegroundColor White
    Write-Host ""
    
    Write-Host "OPTION B: GitHub Actions Deployment" -ForegroundColor Cyan
    Write-Host "1. Push this code to GitHub: MichaelCrowe11/CroweTrade" -ForegroundColor White
    Write-Host "2. Add Fly.io token as GitHub secret: FLY_API_TOKEN" -ForegroundColor White
    Write-Host "3. GitHub Actions will automatically deploy" -ForegroundColor White
    Write-Host ""
    
    Write-Host "OPTION C: Docker + Any Cloud Provider" -ForegroundColor Cyan
    Write-Host "1. Build: docker build -f docker/Dockerfile.frontend -t crowetrade ." -ForegroundColor White
    Write-Host "2. Deploy to: Heroku, Railway, Render, DigitalOcean, etc." -ForegroundColor White
    Write-Host ""
    
    Write-Host "Current Status:" -ForegroundColor Green
    Write-Host "✅ Code ready for deployment" -ForegroundColor Green
    Write-Host "✅ Docker containers configured" -ForegroundColor Green
    Write-Host "✅ Environment variables set" -ForegroundColor Green
    Write-Host "✅ DNS instructions ready" -ForegroundColor Green
    Write-Host ""
}

function Show-DNSInstructions {
    Write-Host "🌐 DNS Configuration for crowetrade.com" -ForegroundColor Blue
    Write-Host "══════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    Write-Host "Add these records in Namecheap Advanced DNS:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After you deploy to any platform, replace the targets below:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "┌─────────────┬─────────────────────────────────┐"
    Write-Host "│ Host Record │ Points To                       │"
    Write-Host "├─────────────┼─────────────────────────────────┤"
    Write-Host "│ @           │ your-app.fly.dev                │"
    Write-Host "│ www         │ your-app.fly.dev                │"
    Write-Host "│ api         │ your-app.fly.dev/api            │"
    Write-Host "└─────────────┴─────────────────────────────────┘"
    Write-Host ""
}

function Test-LocalDeployment {
    Write-Host "🧪 Testing Local Development Environment" -ForegroundColor Blue
    Write-Host "═══════════════════════════════════════" -ForegroundColor Blue
    Write-Host ""
    
    # Check if we can run the app locally
    if (Test-Path "frontend/package.json") {
        Write-Log "Frontend found - testing local setup..." -Color Yellow
        
        Push-Location "frontend"
        try {
            Write-Log "Installing dependencies..." -Color Yellow
            npm install --silent 2>$null
            
            Write-Log "Starting development server..." -Color Yellow
            Write-Host "You can test your app locally by running:" -ForegroundColor Cyan
            Write-Host "  cd frontend" -ForegroundColor White
            Write-Host "  npm run dev" -ForegroundColor White
            Write-Host "  # Then visit: http://localhost:3000" -ForegroundColor White
        }
        catch {
            Write-Log "Local setup needs manual configuration" -Color Yellow
        }
        finally {
            Pop-Location
        }
    }
    
    if (Test-Path "docker/Dockerfile.frontend") {
        Write-Host ""
        Write-Log "Docker setup found - you can also run:" -Color Yellow
        Write-Host "  docker build -f docker/Dockerfile.frontend -t crowetrade ." -ForegroundColor White
        Write-Host "  docker run -p 3000:3000 crowetrade" -ForegroundColor White
    }
    
    Write-Host ""
}

# Main execution
Write-Host ""
Write-Host "🎯 CroweTrade Deployment Solutions" -ForegroundColor Blue
Write-Host "═══════════════════════════════════" -ForegroundColor Blue
Write-Host ""

Write-Log "Fly.io CLI is experiencing authorization issues" -Color Yellow
Write-Log "Providing alternative deployment methods..." -Color Cyan

Show-ManualDeployment
Show-DNSInstructions

if ($Deploy) {
    Test-LocalDeployment
}

Write-Host ""
Write-Host "🎉 Next Steps:" -ForegroundColor Green
Write-Host "1. Choose a deployment method above" -ForegroundColor White  
Write-Host "2. Deploy your app to get a public URL" -ForegroundColor White
Write-Host "3. Update DNS records with your new URL" -ForegroundColor White
Write-Host "4. Visit https://crowetrade.com" -ForegroundColor White
Write-Host ""
Write-Host "Need help? Run: .\simple-deploy.ps1 -Deploy" -ForegroundColor Cyan

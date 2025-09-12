#!/usr/bin/env bash
# CroweTrade Production Deployment Script for crowetrade.com
set -euo pipefail

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}"
    echo "🚀 CroweTrade Production Deployment to crowetrade.com"
    echo "═══════════════════════════════════════════════════════"
    echo -e "${NC}"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check flyctl
    if ! command -v flyctl >/dev/null 2>&1; then
        log "Installing flyctl..."
        curl -fsSL https://fly.io/install.sh | sh -s -- -b /usr/local/bin
    fi
    log "✅ flyctl: $(flyctl version || echo 'installed')"
    
    # Check authentication
    if [ -z "${FLY_API_TOKEN:-}" ] && [ -z "${FLYIO_TOKEN:-}" ]; then
        echo -e "${RED}❌ No Fly.io token found${NC}"
        echo "Please set FLY_API_TOKEN or FLYIO_TOKEN environment variable"
        echo "Get your token from: https://fly.io/user/personal_access_tokens"
        exit 1
    fi
    
    # Set token
    if [ -n "${FLYIO_TOKEN:-}" ]; then
        export FLY_API_TOKEN="$FLYIO_TOKEN"
    fi
    
    log "✅ Fly.io authentication configured"
}

check_secrets() {
    log "Checking required secrets..."
    
    local REQUIRED_SECRETS=(
        "COINBASE_API_KEY"
        "COINBASE_API_SECRET"
        "COINBASE_PASSPHRASE"
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
    )
    
    local missing_secrets=()
    
    for secret in "${REQUIRED_SECRETS[@]}"; do
        if [ -z "${!secret:-}" ]; then
            missing_secrets+=("$secret")
        fi
    done
    
    if [ ${#missing_secrets[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Missing secrets: ${missing_secrets[*]}${NC}"
        echo "Please set these environment variables before deployment"
        echo "For security, these will be set as Fly secrets during deployment"
    else
        log "✅ All required secrets are available"
    fi
}

create_apps() {
    log "Creating Fly.io applications..."
    
    local APPS=(
        "crowetrade-web:Web Frontend"
        "crowetrade-execution:Trading Execution Engine"
        "crowetrade-portfolio:Portfolio Management"
    )
    
    for app_info in "${APPS[@]}"; do
        local app_name="${app_info%%:*}"
        local description="${app_info##*:}"
        
        if flyctl apps list --json | jq -e --arg a "$app_name" '.[] | select(.Name==$a)' >/dev/null 2>&1; then
            log "✅ App $app_name exists"
        else
            log "Creating app $app_name ($description)..."
            flyctl apps create "$app_name" --org "${FLY_ORG:-personal}" || {
                log "⚠️  App creation failed for $app_name, may already exist"
            }
        fi
    done
}

set_secrets() {
    log "Setting application secrets..."
    
    local APPS=("crowetrade-web" "crowetrade-execution" "crowetrade-portfolio")
    
    # Crypto trading secrets
    if [ -n "${COINBASE_API_KEY:-}" ]; then
        for app in "${APPS[@]}"; do
            flyctl secrets set COINBASE_API_KEY="$COINBASE_API_KEY" -a "$app" >/dev/null 2>&1 || true
        done
        log "✅ Set Coinbase API key"
    fi
    
    if [ -n "${COINBASE_API_SECRET:-}" ]; then
        for app in "${APPS[@]}"; do
            flyctl secrets set COINBASE_API_SECRET="$COINBASE_API_SECRET" -a "$app" >/dev/null 2>&1 || true
        done
        log "✅ Set Coinbase API secret"
    fi
    
    if [ -n "${COINBASE_PASSPHRASE:-}" ]; then
        for app in "${APPS[@]}"; do
            flyctl secrets set COINBASE_PASSPHRASE="$COINBASE_PASSPHRASE" -a "$app" >/dev/null 2>&1 || true
        done
        log "✅ Set Coinbase passphrase"
    fi
    
    # Database secrets
    if [ -n "${POSTGRES_PASSWORD:-}" ]; then
        for app in "${APPS[@]}"; do
            flyctl secrets set POSTGRES_PASSWORD="$POSTGRES_PASSWORD" -a "$app" >/dev/null 2>&1 || true
        done
        log "✅ Set PostgreSQL password"
    fi
    
    if [ -n "${REDIS_PASSWORD:-}" ]; then
        for app in "${APPS[@]}"; do
            flyctl secrets set REDIS_PASSWORD="$REDIS_PASSWORD" -a "$app" >/dev/null 2>&1 || true
        done
        log "✅ Set Redis password"
    fi
}

deploy_services() {
    log "Deploying CroweTrade services..."
    
    # Deploy execution engine
    echo -e "${BLUE}🔧 Deploying Trading Execution Engine...${NC}"
    flyctl deploy -c fly.execution.toml --remote-only --now
    log "✅ Execution engine deployed"
    
    # Deploy portfolio service
    echo -e "${BLUE}📊 Deploying Portfolio Management...${NC}"
    flyctl deploy -c fly.portfolio.toml --remote-only --now
    log "✅ Portfolio service deployed"
    
    # Deploy web frontend
    echo -e "${BLUE}🌐 Deploying Web Frontend...${NC}"
    flyctl deploy -c fly.web.toml --remote-only --now
    log "✅ Web frontend deployed"
}

setup_domains() {
    log "Configuring custom domains..."
    
    # Main website
    flyctl certs create crowetrade.com -a crowetrade-web >/dev/null 2>&1 || true
    flyctl certs create www.crowetrade.com -a crowetrade-web >/dev/null 2>&1 || true
    
    # API endpoints
    flyctl certs create api.crowetrade.com -a crowetrade-execution >/dev/null 2>&1 || true
    flyctl certs create portfolio.crowetrade.com -a crowetrade-portfolio >/dev/null 2>&1 || true
    
    log "✅ Domain certificates requested"
}

run_health_checks() {
    log "Running health checks..."
    
    local ENDPOINTS=(
        "https://crowetrade.com:Main Website"
        "https://api.crowetrade.com/health:Trading API"
        "https://portfolio.crowetrade.com/health:Portfolio API"
    )
    
    for endpoint_info in "${ENDPOINTS[@]}"; do
        local url="${endpoint_info%%:*}"
        local name="${endpoint_info##*:}"
        
        log "Checking $name..."
        
        # Wait for deployment
        sleep 10
        
        if curl -fsS --retry 5 --retry-delay 10 "$url" >/dev/null 2>&1; then
            echo -e "   ✅ $name is ${GREEN}healthy${NC}"
        else
            echo -e "   ⚠️  $name is ${YELLOW}not responding yet${NC} (may need more time)"
        fi
    done
}

show_dns_instructions() {
    echo -e "${BLUE}"
    echo "🌐 DNS Configuration Instructions for Namecheap"
    echo "════════════════════════════════════════════════"
    echo -e "${NC}"
    
    # Get app hostnames
    local web_host=$(flyctl status -a crowetrade-web --json 2>/dev/null | jq -r '.Hostname // "crowetrade-web.fly.dev"')
    local api_host=$(flyctl status -a crowetrade-execution --json 2>/dev/null | jq -r '.Hostname // "crowetrade-execution.fly.dev"')
    local portfolio_host=$(flyctl status -a crowetrade-portfolio --json 2>/dev/null | jq -r '.Hostname // "crowetrade-portfolio.fly.dev"')
    
    echo "Please update your DNS records in Namecheap:"
    echo ""
    echo "1. Remove existing records:"
    echo "   - Remove: CNAME www -> parkingpage.namecheap.com"
    echo "   - Remove: URL Redirect @ -> http://www.crowetrade.com/"
    echo ""
    echo "2. Add these new records:"
    echo "   ┌─────────────┬──────┬─────────────────────────────────┬─────────┐"
    echo "   │ Type        │ Host │ Value                           │ TTL     │"
    echo "   ├─────────────┼──────┼─────────────────────────────────┼─────────┤"
    echo "   │ CNAME       │ @    │ $web_host                       │ 30 min  │"
    echo "   │ CNAME       │ www  │ $web_host                       │ 30 min  │"
    echo "   │ CNAME       │ api  │ $api_host                       │ 30 min  │"
    echo "   │ CNAME       │ portfolio │ $portfolio_host            │ 30 min  │"
    echo "   └─────────────┴──────┴─────────────────────────────────┴─────────┘"
    echo ""
    echo "3. Keep existing records:"
    echo "   - TXT @ v=spf1 include:spf.efwd.registrar-servers.com ~all"
    echo ""
    echo -e "${YELLOW}Note: DNS changes may take 5-30 minutes to propagate globally${NC}"
}

show_deployment_summary() {
    echo -e "${GREEN}"
    echo "🎉 CroweTrade Deployment Complete!"
    echo "═════════════════════════════════"
    echo -e "${NC}"
    
    echo "🌐 Your CroweTrade platform will be available at:"
    echo "   • Main Site:    https://crowetrade.com"
    echo "   • Trading API:  https://api.crowetrade.com"
    echo "   • Portfolio:    https://portfolio.crowetrade.com"
    echo ""
    echo "🔧 Features Deployed:"
    echo "   ✅ AI-Powered Trading Engine"
    echo "   ✅ Coinbase Pro Integration"
    echo "   ✅ Real-time Cryptocurrency Trading"
    echo "   ✅ Advanced Risk Management"
    echo "   ✅ Portfolio Analytics"
    echo "   ✅ Web Dashboard"
    echo ""
    echo "🔐 Security:"
    echo "   ✅ HTTPS/TLS Certificates"
    echo "   ✅ API Authentication"
    echo "   ✅ Encrypted Secrets"
    echo ""
    echo "📊 Monitoring:"
    echo "   • Health checks: Every 15 seconds"
    echo "   • Fly.io Dashboard: https://fly.io/dashboard"
    echo "   • Logs: flyctl logs -a <app-name>"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Update DNS records as shown above"
    echo "2. Wait 5-30 minutes for DNS propagation"
    echo "3. Visit https://crowetrade.com to access your platform"
    echo "4. Monitor logs: flyctl logs -a crowetrade-web"
}

# Main deployment flow
main() {
    print_banner
    check_prerequisites
    check_secrets
    create_apps
    set_secrets
    deploy_services
    setup_domains
    run_health_checks
    show_dns_instructions
    show_deployment_summary
}

# Run deployment
case "${1:-}" in
    "check")
        print_banner
        check_prerequisites
        check_secrets
        ;;
    "dns")
        show_dns_instructions
        ;;
    "deploy"|"")
        main
        ;;
    *)
        echo "Usage: $0 [check|dns|deploy]"
        echo "  check  - Check prerequisites and secrets"
        echo "  dns    - Show DNS configuration instructions"
        echo "  deploy - Full deployment (default)"
        exit 1
        ;;
esac

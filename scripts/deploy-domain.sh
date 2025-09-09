#!/bin/bash

# CroweTrade Domain Deployment Script
# Deploy CroweTrade to crowetrade.com using various cloud platforms

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
cat << 'EOF'
   ____                     _____             _      
  / ___|_ __ _____      __ |_   _| __ __ _  __| | ___ 
 | |   | '__/ _ \ \ /\ / /   | || '__/ _` |/ _` |/ _ \
 | |___| | | (_) \ V  V /    | || | | (_| | (_| |  __/
  \____|_|  \___/ \_/\_/     |_||_|  \__,_|\__,_|\___|
                                                      
EOF
echo -e "${NC}"
echo -e "${GREEN}Domain Deployment to crowetrade.com${NC}"
echo ""

PLATFORM=${1:-"fly"}

# Check prerequisites
check_prerequisites() {
    case "$PLATFORM" in
        "fly")
            if ! command -v flyctl &> /dev/null; then
                echo -e "${RED}❌ flyctl not found. Install from: https://fly.io/docs/getting-started/installing-flyctl/${NC}"
                exit 1
            fi
            ;;
        "render")
            if ! command -v git &> /dev/null; then
                echo -e "${RED}❌ git not found${NC}"
                exit 1
            fi
            ;;
        "railway")
            if ! command -v railway &> /dev/null; then
                echo -e "${RED}❌ railway CLI not found. Install from: https://railway.app/cli${NC}"
                exit 1
            fi
            ;;
        *)
            echo -e "${RED}❌ Unknown platform: $PLATFORM${NC}"
            echo "Supported platforms: fly, render, railway"
            exit 1
            ;;
    esac
}

# Deploy to Fly.io
deploy_fly() {
    echo -e "${YELLOW}🚀 Deploying to Fly.io (crowetrade.com)...${NC}"
    
    # Copy fly config to root
    cp deploy/fly.io/fly.toml .
    
    # Launch app if it doesn't exist
    if ! flyctl status --app crowetrade &> /dev/null; then
        echo -e "${YELLOW}📦 Creating new Fly.io app...${NC}"
        flyctl launch --no-deploy --name crowetrade --region ord
    fi
    
    # Set production secrets
    echo -e "${YELLOW}🔐 Setting production secrets...${NC}"
    
    # Generate secure secrets
    JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    
    flyctl secrets set \
        JWT_SECRET_KEY="$JWT_SECRET" \
        ENCRYPTION_KEY="$ENCRYPTION_KEY" \
        ENVIRONMENT=production \
        TRADING_MODE=PAPER \
        --app crowetrade
    
    # Deploy
    echo -e "${YELLOW}🚀 Deploying to Fly.io...${NC}"
    flyctl deploy --app crowetrade
    
    # Set up custom domain
    echo -e "${YELLOW}🌐 Setting up custom domain...${NC}"
    if ! flyctl certs list --app crowetrade | grep -q crowetrade.com; then
        flyctl certs create crowetrade.com --app crowetrade
        flyctl certs create www.crowetrade.com --app crowetrade
    fi
    
    echo -e "${GREEN}✅ Fly.io deployment complete!${NC}"
    echo -e "${BLUE}🔗 Your app will be available at: https://crowetrade.com${NC}"
    echo -e "${YELLOW}📋 Configure your DNS:${NC}"
    echo "   A record: @ -> $(flyctl ips list --app crowetrade | grep -E 'v4.*global' | awk '{print $2}' | head -1)"
    echo "   CNAME record: www -> crowetrade.com"
}

# Deploy to Render
deploy_render() {
    echo -e "${YELLOW}🚀 Deploying to Render.com (crowetrade.com)...${NC}"
    echo ""
    echo -e "${BLUE}📋 Manual steps required for Render deployment:${NC}"
    echo ""
    echo "1. 🌐 Go to https://render.com"
    echo "2. 📁 Connect your GitHub repository"
    echo "3. 📋 Import the render.yaml file from deploy/render/render.yaml"
    echo "4. 🔧 Configure environment variables:"
    echo "   - JWT_SECRET_KEY (generate secure value)"
    echo "   - ENCRYPTION_KEY (generate secure value)"
    echo "5. 🌍 Add custom domain: crowetrade.com"
    echo "6. 🔒 SSL certificate will be auto-generated"
    echo ""
    echo -e "${GREEN}✅ Render configuration ready in deploy/render/render.yaml${NC}"
}

# Deploy to Railway
deploy_railway() {
    echo -e "${YELLOW}🚀 Deploying to Railway (crowetrade.com)...${NC}"
    
    # Login check
    if ! railway whoami &> /dev/null; then
        echo -e "${YELLOW}🔐 Please login to Railway...${NC}"
        railway login
    fi
    
    # Deploy
    railway link
    railway up
    
    # Set environment variables
    echo -e "${YELLOW}🔧 Setting environment variables...${NC}"
    
    JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    
    railway variables set \
        JWT_SECRET_KEY="$JWT_SECRET" \
        ENCRYPTION_KEY="$ENCRYPTION_KEY" \
        ENVIRONMENT=production \
        TRADING_MODE=PAPER
    
    echo -e "${GREEN}✅ Railway deployment complete!${NC}"
    echo -e "${BLUE}🔗 Configure custom domain in Railway dashboard${NC}"
}

# Health check
health_check() {
    echo -e "${YELLOW}🏥 Running health check...${NC}"
    
    local url=""
    case "$PLATFORM" in
        "fly")
            url="https://crowetrade.fly.dev/health"
            ;;
        "render")
            echo -e "${YELLOW}⏳ Health check after deployment via Render dashboard${NC}"
            return 0
            ;;
        "railway")
            url=$(railway status --json | jq -r '.deployments[0].url')/health
            ;;
    esac
    
    if [[ -n "$url" ]]; then
        echo -e "${BLUE}🔍 Checking: $url${NC}"
        for i in {1..10}; do
            if curl -f -s "$url" > /dev/null; then
                echo -e "${GREEN}✅ Health check passed!${NC}"
                return 0
            fi
            echo -e "${YELLOW}⏳ Attempt $i/10 - waiting for deployment...${NC}"
            sleep 30
        done
        echo -e "${RED}❌ Health check failed after 10 attempts${NC}"
    fi
}

# Create GitHub Actions workflow for CD
create_github_actions() {
    echo -e "${YELLOW}⚙️ Creating GitHub Actions workflow...${NC}"
    
    mkdir -p .github/workflows
    
    cat > .github/workflows/deploy.yml << 'EOF'
name: Deploy CroweTrade

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    - name: Run tests
      run: |
        python -m pytest tests/ -x --tb=short

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Setup Fly.io
      uses: superfly/flyctl-actions/setup-flyctl@master
    - name: Deploy to Fly.io
      run: flyctl deploy --remote-only
      env:
        FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
EOF

    echo -e "${GREEN}✅ GitHub Actions workflow created${NC}"
    echo -e "${YELLOW}📋 Add FLY_API_TOKEN to GitHub secrets${NC}"
}

# Main deployment
main() {
    echo -e "${YELLOW}Platform: $PLATFORM${NC}"
    echo ""
    
    check_prerequisites
    
    case "$PLATFORM" in
        "fly")
            deploy_fly
            health_check
            ;;
        "render")
            deploy_render
            ;;
        "railway")
            deploy_railway
            health_check
            ;;
    esac
    
    create_github_actions
    
    echo ""
    echo -e "${GREEN}🎉 Deployment process complete!${NC}"
    echo ""
    echo -e "${BLUE}📋 Next steps:${NC}"
    echo "1. 🌐 Configure DNS for crowetrade.com"
    echo "2. 🔒 SSL certificate will be auto-generated"
    echo "3. 📊 Monitor deployment in platform dashboard"
    echo "4. 🧪 Test the API at https://crowetrade.com/health"
    echo "5. 📚 View docs at https://crowetrade.com/docs"
    echo ""
    echo -e "${YELLOW}⚠️ Remember: Starting in PAPER trading mode for safety${NC}"
    echo ""
}

# Show usage
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [PLATFORM]"
    echo ""
    echo "Platforms:"
    echo "  fly      - Deploy to Fly.io (recommended)"
    echo "  render   - Deploy to Render.com"
    echo "  railway  - Deploy to Railway"
    echo ""
    echo "Example: $0 fly"
    exit 0
fi

main "$@"
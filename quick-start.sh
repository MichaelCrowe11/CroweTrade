#!/bin/bash

# CroweTrade Quick Start Script
# Automated setup and launch for new users

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
echo -e "${GREEN}Professional Quantitative Trading System${NC}"
echo -e "${BLUE}Quick Start Setup${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $1 is available${NC}"
        return 0
    fi
}

# Check required tools
MISSING_TOOLS=0
check_command "docker" || MISSING_TOOLS=1
check_command "docker-compose" || MISSING_TOOLS=1
check_command "python3" || check_command "python" || MISSING_TOOLS=1
check_command "git" || MISSING_TOOLS=1

if [ $MISSING_TOOLS -eq 1 ]; then
    echo -e "${RED}Please install missing tools before continuing.${NC}"
    echo ""
    echo "Installation guides:"
    echo "Docker: https://docs.docker.com/get-docker/"
    echo "Python: https://www.python.org/downloads/"
    echo "Git: https://git-scm.com/downloads"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites met${NC}"
echo ""

# Generate secure configuration
echo -e "${YELLOW}Generating secure configuration...${NC}"

if [ ! -f ".env.production" ]; then
    # Generate secure secrets
    JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    DB_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")
    GRAFANA_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(12))")
    
    cat > .env.production << EOF
# CroweTrade Production Configuration - Generated $(date)

# Security (Auto-generated secure values)
JWT_SECRET_KEY=$JWT_SECRET
ENCRYPTION_KEY=$ENCRYPTION_KEY

# Database
DB_PASSWORD=$DB_PASSWORD

# Monitoring
GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# Trading Configuration
TRADING_MODE=PAPER
DEFAULT_RISK_BUDGET=0.02
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=50000
MAX_DRAWDOWN=0.20

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# API Keys (add your keys here)
ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
POLYGON_API_KEY=
IEX_API_KEY=
EOF
    
    echo -e "${GREEN}✅ Configuration file created (.env.production)${NC}"
else
    echo -e "${YELLOW}⚠️ Configuration file already exists${NC}"
fi

echo ""

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"

# Check if we should create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
pip install fastapi uvicorn[standard] psutil cryptography bcrypt PyJWT

echo -e "${GREEN}✅ Python dependencies installed${NC}"
echo ""

# Create required directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data/features data/models logs backups
mkdir -p monitoring/grafana/dashboards monitoring/grafana/datasources
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Build and start services
echo -e "${YELLOW}Building and starting services...${NC}"
echo "This may take a few minutes on first run..."

# Make scripts executable
chmod +x scripts/deploy.sh

# Start the system
./scripts/deploy.sh local

echo ""
echo -e "${GREEN}🎉 CroweTrade is now running!${NC}"
echo ""
echo -e "${BLUE}Access URLs:${NC}"
echo "🔗 Trading API:     http://localhost:8080"
echo "📊 API Docs:        http://localhost:8080/docs"
echo "💹 Grafana:         http://localhost:3000 (admin/$GRAFANA_PASSWORD)"
echo "📈 Prometheus:      http://localhost:9090"
echo ""
echo -e "${BLUE}Quick Health Check:${NC}"
curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || echo "Starting up..."

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. 📖 Read the PRODUCTION_LAUNCH_GUIDE.md"
echo "2. 🔑 Add your API keys to .env.production"
echo "3. 📊 Check Grafana dashboards"
echo "4. 🧪 Run tests: python -m pytest tests/"
echo "5. 🚀 When ready, switch TRADING_MODE from PAPER to LIVE"
echo ""
echo -e "${GREEN}Happy Trading! 🚀${NC}"
echo ""

# Save important info for user
cat > QUICK_START_INFO.txt << EOF
CroweTrade Quick Start - Generated $(date)

🔐 IMPORTANT: Save these passwords safely!
Grafana Password: $GRAFANA_PASSWORD
Database Password: $DB_PASSWORD

🔗 Access URLs:
Trading API: http://localhost:8080
API Documentation: http://localhost:8080/docs  
Grafana Dashboards: http://localhost:3000
Prometheus Metrics: http://localhost:9090

📁 Important Files:
.env.production - Main configuration file
PRODUCTION_LAUNCH_GUIDE.md - Complete documentation
QUICK_START_INFO.txt - This file (keep it safe!)

⚠️  REMEMBER:
- System is in PAPER trading mode (safe for testing)
- Add your API keys to .env.production
- Change TRADING_MODE to LIVE only when ready for real trading
- Always test thoroughly before live trading

🚀 To restart: docker-compose -f docker-compose.yml up -d
🛑 To stop: docker-compose -f docker-compose.yml down
EOF

echo -e "${GREEN}💾 Important info saved to QUICK_START_INFO.txt${NC}"
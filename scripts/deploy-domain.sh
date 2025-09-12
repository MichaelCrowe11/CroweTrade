#!/bin/bash

# CroweTrade Domain Deployment Script
# This script deploys the CroweTrade platform and connects your domain

set -e

echo "🚀 Starting CroweTrade Domain Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    print_error "flyctl is not installed. Please install it first:"
    echo "  curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    print_error "You're not logged in to Fly.io. Please run: flyctl auth login"
    exit 1
fi

print_status "Building and deploying frontend application..."
flyctl deploy --config fly.toml --remote-only

print_status "Building and deploying execution service..."
flyctl deploy --config fly.execution.toml --remote-only

print_status "Building and deploying portfolio service..."
flyctl deploy --config fly.portfolio.toml --remote-only

# Set up custom domain
print_status "Configuring custom domain: crowetrade.com"
flyctl certs add crowetrade.com --app crowetrade-main
flyctl certs add www.crowetrade.com --app crowetrade-main

# Set up subdomains for APIs
print_status "Configuring API subdomain: api.crowetrade.com"
flyctl certs add api.crowetrade.com --app crowetrade-execution

print_status "Configuring portfolio API subdomain: portfolio.crowetrade.com"
flyctl certs add portfolio.crowetrade.com --app crowetrade-portfolio

print_success "Deployment completed!"

echo ""
print_status "Domain Configuration Required:"
echo "  Add the following DNS records to your crowetrade.com domain:"
echo ""
echo "  Type    | Name      | Value"
echo "  --------|-----------|--------------------------------"
echo "  CNAME   | @         | crowetrade-main.fly.dev"
echo "  CNAME   | www       | crowetrade-main.fly.dev"  
echo "  CNAME   | api       | crowetrade-execution.fly.dev"
echo "  CNAME   | portfolio | crowetrade-portfolio.fly.dev"
echo ""

print_status "Checking certificate status..."
flyctl certs show crowetrade.com --app crowetrade-main
flyctl certs show api.crowetrade.com --app crowetrade-execution

print_success "🎉 CroweTrade platform deployed successfully!"
print_status "Your trading platform will be available at: https://crowetrade.com"
print_status "API endpoints will be available at: https://api.crowetrade.com"
print_status "Portfolio API will be available at: https://portfolio.crowetrade.com"

echo ""
print_warning "Next Steps:"
echo "1. Configure DNS records as shown above"
echo "2. Wait for SSL certificates to be issued (can take up to 10 minutes)"
echo "3. Test your domain: https://crowetrade.com"
echo "4. Monitor deployment: flyctl logs --app crowetrade-main"

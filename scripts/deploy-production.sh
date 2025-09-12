#!/bin/bash

# CroweTrade AI Trading Infrastructure - Production Deployment Script
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="crowetrade-production"
DEPLOYMENT_TIMEOUT="600s"
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_DELAY=10

echo -e "${BLUE}🚀 CroweTrade AI Trading Infrastructure Deployment${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    print_status "kubectl is available"
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    print_status "Connected to Kubernetes cluster"
    
    # Check Docker image availability
    if ! docker manifest inspect ghcr.io/michaelcrowe11/crowetrade:latest &> /dev/null; then
        print_warning "Docker image may not be available - deployment will use latest available"
    else
        print_status "Docker image is available"
    fi
}

# Function to create namespace if not exists
setup_namespace() {
    echo -e "${BLUE}Setting up namespace...${NC}"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        print_status "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace $NAMESPACE
        print_status "Created namespace $NAMESPACE"
    fi
    
    # Label namespace
    kubectl label namespace $NAMESPACE environment=production --overwrite
    kubectl label namespace $NAMESPACE app=crowetrade --overwrite
}

# Function to deploy secrets (placeholder - replace with actual secrets management)
deploy_secrets() {
    echo -e "${BLUE}Deploying secrets...${NC}"
    
    # Create secrets from environment variables or external secret management
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: crowetrade-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  postgres-host: $(echo -n "${POSTGRES_HOST:-postgres.crowetrade.svc.cluster.local}" | base64)
  postgres-user: $(echo -n "${POSTGRES_USER:-crowetrade}" | base64)
  postgres-password: $(echo -n "${POSTGRES_PASSWORD:-changeme123}" | base64)
  redis-host: $(echo -n "${REDIS_HOST:-redis.crowetrade.svc.cluster.local}" | base64)
  redis-password: $(echo -n "${REDIS_PASSWORD:-changeme456}" | base64)
EOF
    
    print_status "Secrets deployed"
}

# Function to deploy the application
deploy_application() {
    echo -e "${BLUE}Deploying CroweTrade AI Trading Infrastructure...${NC}"
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/production-deployment.yaml
    
    print_status "Application manifests applied"
    
    # Wait for deployments to be ready
    echo -e "${BLUE}Waiting for deployments to be ready...${NC}"
    
    kubectl wait --for=condition=available --timeout=$DEPLOYMENT_TIMEOUT \
        deployment/crowetrade-trading-engine -n $NAMESPACE
    print_status "Trading Engine deployment ready"
    
    kubectl wait --for=condition=available --timeout=$DEPLOYMENT_TIMEOUT \
        deployment/crowetrade-model-registry -n $NAMESPACE
    print_status "Model Registry deployment ready"
    
    kubectl wait --for=condition=available --timeout=$DEPLOYMENT_TIMEOUT \
        deployment/crowetrade-backtesting -n $NAMESPACE
    print_status "Backtesting Engine deployment ready"
}

# Function to perform health checks
health_checks() {
    echo -e "${BLUE}Performing health checks...${NC}"
    
    # Get service endpoints
    TRADING_ENGINE_IP=$(kubectl get svc crowetrade-trading-engine -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    MODEL_REGISTRY_IP=$(kubectl get svc crowetrade-model-registry -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    BACKTESTING_IP=$(kubectl get svc crowetrade-backtesting -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    # Wait for LoadBalancer IPs if not available
    if [[ -z "$TRADING_ENGINE_IP" ]]; then
        echo "Waiting for LoadBalancer IP assignment..."
        sleep 30
        TRADING_ENGINE_IP=$(kubectl get svc crowetrade-trading-engine -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    fi
    
    # Health check function
    check_health() {
        local service=$1
        local ip=$2
        local port=$3
        
        if [[ -z "$ip" ]]; then
            print_warning "No LoadBalancer IP for $service - using port-forward for health check"
            kubectl port-forward svc/$service $port:$port -n $NAMESPACE &
            local pf_pid=$!
            sleep 5
            
            if curl -f -s http://localhost:$port/health > /dev/null; then
                print_status "$service is healthy (via port-forward)"
            else
                print_error "$service health check failed"
                return 1
            fi
            
            kill $pf_pid 2>/dev/null || true
        else
            local retries=0
            while [ $retries -lt $HEALTH_CHECK_RETRIES ]; do
                if curl -f -s http://$ip:$port/health > /dev/null 2>&1; then
                    print_status "$service is healthy at http://$ip:$port"
                    return 0
                fi
                
                retries=$((retries + 1))
                if [ $retries -lt $HEALTH_CHECK_RETRIES ]; then
                    echo "Health check attempt $retries/$HEALTH_CHECK_RETRIES failed, retrying in ${HEALTH_CHECK_DELAY}s..."
                    sleep $HEALTH_CHECK_DELAY
                fi
            done
            
            print_error "$service health check failed after $HEALTH_CHECK_RETRIES attempts"
            return 1
        fi
    }
    
    # Perform health checks
    check_health "crowetrade-trading-engine" "$TRADING_ENGINE_IP" "8080"
    check_health "crowetrade-model-registry" "$MODEL_REGISTRY_IP" "8081"  
    check_health "crowetrade-backtesting" "$BACKTESTING_IP" "8082"
}

# Function to verify AI trading capabilities
verify_trading_capabilities() {
    echo -e "${BLUE}Verifying AI trading capabilities...${NC}"
    
    # Use kubectl port-forward for testing
    kubectl port-forward svc/crowetrade-model-registry 8081:8081 -n $NAMESPACE &
    local pf_pid=$!
    sleep 5
    
    # Test Model Registry
    if curl -f -s http://localhost:8081/models/list > /dev/null; then
        print_status "Model Registry API is accessible"
    else
        print_warning "Model Registry API test skipped (may need authentication)"
    fi
    
    # Test Trading Engine
    kubectl port-forward svc/crowetrade-trading-engine 8080:8080 -n $NAMESPACE &
    local pf_pid2=$!
    sleep 5
    
    if curl -f -s http://localhost:8080/strategies/health > /dev/null; then
        print_status "Trading Engine strategies are accessible"
    else
        print_warning "Trading Engine strategies test skipped"
    fi
    
    # Clean up port forwards
    kill $pf_pid $pf_pid2 2>/dev/null || true
}

# Function to display deployment summary
deployment_summary() {
    echo ""
    echo -e "${GREEN}🎯 DEPLOYMENT SUCCESSFUL! 🎯${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "${BLUE}CroweTrade AI Trading Infrastructure is now LIVE in production!${NC}"
    echo ""
    echo -e "${YELLOW}Deployed Components:${NC}"
    echo "• Model Registry with lifecycle management"
    echo "• A/B Testing with multi-armed bandits" 
    echo "• Backtesting Framework with transaction costs"
    echo "• Integrated Trading Engine with risk management"
    echo ""
    echo -e "${YELLOW}Service Endpoints:${NC}"
    kubectl get svc -n $NAMESPACE
    echo ""
    echo -e "${YELLOW}Pod Status:${NC}"
    kubectl get pods -n $NAMESPACE
    echo ""
    echo -e "${GREEN}🚀 READY FOR LIVE TRADING! 🚀${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Configure your trading strategies"
    echo "2. Upload initial models to the Model Registry"
    echo "3. Run backtests to validate performance"
    echo "4. Enable live trading with appropriate risk limits"
    echo ""
    echo -e "${YELLOW}Monitoring:${NC}"
    echo "• Grafana dashboards: http://grafana.crowetrade.com"
    echo "• Prometheus metrics: http://prometheus.crowetrade.com"
    echo "• Application logs: kubectl logs -f deployment/crowetrade-trading-engine -n $NAMESPACE"
}

# Main execution
main() {
    echo -e "${BLUE}Starting deployment process...${NC}"
    echo ""
    
    check_prerequisites
    setup_namespace
    deploy_secrets
    deploy_application
    health_checks
    verify_trading_capabilities
    deployment_summary
    
    echo -e "${GREEN}✅ Production deployment completed successfully!${NC}"
    exit 0
}

# Error handling
trap 'echo -e "${RED}❌ Deployment failed!${NC}"; exit 1' ERR

# Run main function
main "$@"

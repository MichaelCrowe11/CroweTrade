#!/bin/bash

# CroweTrade Production Deployment Script
# Usage: ./scripts/deploy.sh [environment] [options]

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default values
ENVIRONMENT=${1:-staging}
DRY_RUN=${DRY_RUN:-false}
SKIP_TESTS=${SKIP_TESTS:-false}
BACKUP_DB=${BACKUP_DB:-true}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    local required_tools=("docker" "docker-compose" "git")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check if environment file exists
    if [[ "$ENVIRONMENT" == "production" && ! -f "$PROJECT_ROOT/.env.production" ]]; then
        log_error "Production environment file .env.production not found"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests (SKIP_TESTS=true)"
        return 0
    fi
    
    log_info "Running tests..."
    cd "$PROJECT_ROOT"
    
    # Install test dependencies if needed
    if [[ ! -d "venv" ]]; then
        python -m venv venv
        source venv/bin/activate || source venv/Scripts/activate
        pip install -e .[dev]
    else
        source venv/bin/activate || source venv/Scripts/activate
    fi
    
    # Run tests with coverage
    python -m pytest tests/ -x --tb=short
    
    if [[ $? -eq 0 ]]; then
        log_success "Tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT"
    
    # Build main application image
    docker build -f docker/Dockerfile -t crowetrade:$TIMESTAMP .
    docker tag crowetrade:$TIMESTAMP crowetrade:latest
    
    log_success "Docker images built successfully"
}

# Backup database
backup_database() {
    if [[ "$BACKUP_DB" == "false" ]]; then
        log_warning "Skipping database backup"
        return 0
    fi
    
    log_info "Creating database backup..."
    
    # Create backup directory
    mkdir -p "$PROJECT_ROOT/backups"
    
    # Backup PostgreSQL database
    if docker ps --format "table {{.Names}}" | grep -q postgres; then
        docker exec -t $(docker ps -q --filter name=postgres) pg_dumpall -c -U crowetrade > "$PROJECT_ROOT/backups/db_backup_$TIMESTAMP.sql"
        log_success "Database backup created: backups/db_backup_$TIMESTAMP.sql"
    else
        log_warning "PostgreSQL container not running, skipping backup"
    fi
}

# Deploy to environment
deploy() {
    log_info "Deploying to $ENVIRONMENT environment..."
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE - No actual deployment will occur"
        return 0
    fi
    
    # Load environment variables
    if [[ -f ".env.$ENVIRONMENT" ]]; then
        export $(grep -v '^#' .env.$ENVIRONMENT | xargs)
    fi
    
    case "$ENVIRONMENT" in
        "local"|"development")
            log_info "Starting local development environment..."
            docker-compose -f docker-compose.yml up -d
            ;;
        "staging")
            log_info "Deploying to staging environment..."
            docker-compose -f docker-compose.staging.yml up -d
            ;;
        "production")
            log_info "Deploying to production environment..."
            docker-compose -f docker-compose.production.yml up -d
            ;;
        "fly")
            deploy_fly
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # Wait for services to be healthy
    wait_for_health_check
    
    log_success "Deployment to $ENVIRONMENT completed successfully"
}

# Deploy to Fly.io
deploy_fly() {
    log_info "Deploying to Fly.io..."
    
    # Check if flyctl is installed
    if ! command -v flyctl &> /dev/null; then
        log_error "flyctl is not installed. Please install from https://fly.io/docs/getting-started/installing-flyctl/"
        exit 1
    fi
    
    # Deploy portfolio service
    if [[ -f "fly.portfolio.toml" ]]; then
        flyctl deploy --config fly.portfolio.toml
    fi
    
    # Deploy execution service  
    if [[ -f "fly.execution.toml" ]]; then
        flyctl deploy --config fly.execution.toml
    fi
    
    log_success "Fly.io deployment completed"
}

# Health check
wait_for_health_check() {
    log_info "Waiting for services to become healthy..."
    
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Attempt $((attempt + 1))/$max_attempts - waiting for health check..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    exit 1
}

# Rollback function
rollback() {
    local rollback_tag=${1:-"previous"}
    log_warning "Rolling back to $rollback_tag..."
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.production.yml down
        docker tag crowetrade:$rollback_tag crowetrade:latest
        docker-compose -f docker-compose.production.yml up -d
    else
        log_error "Rollback only supported in production environment"
        exit 1
    fi
    
    wait_for_health_check
    log_success "Rollback completed"
}

# Monitoring setup
setup_monitoring() {
    log_info "Setting up monitoring and alerts..."
    
    # Create Grafana dashboards directory
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    
    # Create Prometheus configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'crowetrade'
    static_configs:
      - targets: ['crowetrade-api:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

    # Create basic Grafana datasource
    cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    log_success "Monitoring configuration created"
}

# Cleanup old deployments
cleanup() {
    log_info "Cleaning up old Docker images and containers..."
    
    # Remove old images (keep last 3)
    docker images crowetrade --format "table {{.Tag}}" | grep -v "latest" | tail -n +4 | xargs -r docker rmi crowetrade: 2>/dev/null || true
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old backup files (keep last 7 days)
    find "$PROJECT_ROOT/backups" -name "*.sql" -mtime +7 -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Show usage information
show_usage() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    local       - Local development environment
    staging     - Staging environment  
    production  - Production environment
    fly         - Deploy to Fly.io

OPTIONS:
    --dry-run       - Show what would be deployed without actually deploying
    --skip-tests    - Skip running tests before deployment
    --no-backup     - Skip database backup
    --rollback TAG  - Rollback to specified Docker tag
    --cleanup       - Clean up old deployments
    --help          - Show this help message

EXAMPLES:
    $0 staging
    $0 production --dry-run
    $0 production --rollback previous
    DRY_RUN=true SKIP_TESTS=true $0 staging

EOF
}

# Main deployment flow
main() {
    log_info "Starting CroweTrade deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Timestamp: $TIMESTAMP"
    
    # Parse command line arguments
    while [[ $# -gt 1 ]]; do
        case $2 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --no-backup)
                BACKUP_DB=false
                shift
                ;;
            --rollback)
                rollback "$3"
                exit 0
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $2"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Execute deployment steps
    check_prerequisites
    
    if [[ "$ENVIRONMENT" != "fly" ]]; then
        setup_monitoring
        run_tests
        build_images
        backup_database
    fi
    
    deploy
    
    log_success "🚀 CroweTrade deployment completed successfully!"
    log_info "Access the application at: http://localhost:8080"
    log_info "Grafana dashboard at: http://localhost:3000"
    log_info "Prometheus metrics at: http://localhost:9090"
}

# Handle script arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi
    
    main "$@"
fi
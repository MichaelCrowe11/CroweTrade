# CroweTrade Production Launch Guide

## 🚀 Quick Start

CroweTrade is now production-ready with enterprise-grade security, monitoring, and deployment infrastructure.

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Git
- 4GB+ RAM
- 20GB+ disk space

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   PostgreSQL    │    │   Apache Kafka  │
│   Trading API   │    │   Database      │    │   Message Bus   │
│   Port: 8080    │    │   Port: 5432    │    │   Port: 9092    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │     Redis       │    │     Grafana     │
│   Metrics       │    │    Cache        │    │   Dashboards    │
│   Port: 9090    │    │   Port: 6379    │    │   Port: 3000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### 1. Environment Setup

Copy the example environment file and customize:

```bash
cp .env.example .env.production
```

**Critical Environment Variables:**

```env
# Security (REQUIRED)
JWT_SECRET_KEY=your-256-bit-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# Database
DATABASE_URL=postgresql://crowetrade:your_password@postgres:5432/crowetrade
DB_PASSWORD=secure_database_password

# Risk Management
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=50000
MAX_DRAWDOWN=0.20
DEFAULT_RISK_BUDGET=0.02

# Trading Mode
TRADING_MODE=PAPER  # Use PAPER for safe testing, LIVE for production
```

### 2. API Keys Configuration

Add your data provider API keys:

```env
ALPHA_VANTAGE_API_KEY=your_key
FINNHUB_API_KEY=your_key
POLYGON_API_KEY=your_key
IEX_API_KEY=your_key
```

## 🚀 Deployment Options

### Option 1: Local Production-Like Environment

```bash
# Clone and setup
git clone <your-repo>
cd CroweTrade

# Deploy with all services
./scripts/deploy.sh local
```

### Option 2: Staging Environment

```bash
# Deploy to staging with health checks
./scripts/deploy.sh staging
```

### Option 3: Production Environment

```bash
# Full production deployment
./scripts/deploy.sh production
```

### Option 4: Cloud Deployment (Fly.io)

```bash
# Deploy to Fly.io cloud platform
./scripts/deploy.sh fly
```

## 🔐 Security Features

### Authentication & Authorization
- **JWT tokens** with role-based access control
- **API key authentication** for programmatic access
- **Rate limiting** to prevent abuse
- **Request signing** for secure API communication

### Data Protection
- **Encryption at rest** for sensitive data
- **Secure password hashing** with bcrypt
- **Environment-based secrets** management
- **Non-root container** execution

### Network Security
- **CORS protection** with configurable origins
- **Trusted host validation**
- **TLS/SSL support** via reverse proxy
- **Network isolation** via Docker networks

## 📊 Monitoring & Observability

### Health Checks
- **System metrics**: CPU, memory, disk usage
- **Application metrics**: Trading signals, orders, fills
- **Service dependencies**: Database, cache, message bus

### Metrics Collection
- **Prometheus** integration for metrics
- **Grafana** dashboards for visualization
- **Custom trading metrics** (PnL, risk, execution)
- **Performance tracking** for all operations

### Alerting
- **Health check failures**
- **Risk limit breaches**
- **Performance degradation**
- **System resource exhaustion**

## 📈 Trading System Features

### Risk Management
- **Real-time position monitoring**
- **Dynamic risk budgets** based on volatility
- **Circuit breakers** for automated protection
- **Emergency stop** functionality

### Signal Generation
- **Multi-strategy support** with ensemble methods
- **Uncertainty-aware** predictions
- **Meta-labeling** for trade filtering
- **Regime detection** for market conditions

### Portfolio Management
- **Kelly-tempered sizing** for optimal position sizing
- **Transaction cost awareness**
- **Turnover penalties** to reduce trading costs
- **Exposure limits** by instrument and sector

### Execution
- **Smart order routing** across venues
- **VWAP/TWAP/POV algorithms**
- **Slippage monitoring** and attribution
- **Pre/post-trade compliance** checks

## 🔄 Operational Procedures

### Deployment

```bash
# Standard deployment
./scripts/deploy.sh production

# Dry run (test without deploying)
./scripts/deploy.sh production --dry-run

# Skip tests (faster deployment)
./scripts/deploy.sh production --skip-tests
```

### Backup & Recovery

```bash
# Manual database backup
docker exec -t crowetrade_postgres_1 pg_dumpall -c -U crowetrade > backup.sql

# Restore from backup
cat backup.sql | docker exec -i crowetrade_postgres_1 psql -U crowetrade
```

### Rollback

```bash
# Rollback to previous version
./scripts/deploy.sh production --rollback previous

# Rollback to specific timestamp
./scripts/deploy.sh production --rollback 20250909_143000
```

### Emergency Procedures

```bash
# Emergency stop all trading
curl -X POST http://localhost:8080/api/v1/risk/emergency-stop \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Graceful shutdown
docker-compose -f docker-compose.production.yml down

# Immediate shutdown
docker-compose -f docker-compose.production.yml kill
```

## 📋 API Endpoints

### Health & Monitoring
```
GET  /health              - System health check
GET  /metrics             - Prometheus metrics (auth required)
```

### Trading Operations
```
POST /api/v1/signals      - Generate trading signals
POST /api/v1/positions    - Calculate position targets
GET  /api/v1/risk/status  - Get risk status
POST /api/v1/risk/emergency-stop - Emergency stop
```

### Configuration
```
GET  /api/v1/config       - Get system configuration (admin only)
```

## 🧪 Testing

### Run Test Suite
```bash
# Full test suite
python -m pytest tests/ -v

# Quick smoke tests
python -m pytest tests/unit/ -k "not slow"

# Performance tests
python -m pytest tests/performance/ --benchmark-only
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:8080
```

## 📊 Performance Benchmarks

### Target SLOs
- **API Response Time**: < 50ms p95
- **Signal Generation**: < 20ms p95
- **Order Submission**: < 5ms p95
- **System Availability**: 99.9%
- **Data Freshness**: < 200ms

### Scalability
- **Concurrent Users**: 100+
- **Orders per Second**: 1000+
- **Signals per Second**: 500+
- **Data Throughput**: 10k events/sec

## 🔍 Troubleshooting

### Common Issues

**Service Won't Start**
```bash
# Check logs
docker-compose logs crowetrade-api

# Check configuration
docker exec -it crowetrade_api_1 python -c "from crowetrade.config.production import get_config; print(get_config().to_dict())"
```

**Database Connection Issues**
```bash
# Test database connectivity
docker exec -it crowetrade_postgres_1 psql -U crowetrade -c "SELECT version();"
```

**Authentication Failures**
```bash
# Verify JWT secret is set
echo $JWT_SECRET_KEY

# Test token generation
curl -X POST http://localhost:8080/auth/login -d '{"username":"admin","password":"admin"}'
```

### Log Analysis
```bash
# View real-time logs
docker-compose logs -f crowetrade-api

# Search logs for errors
docker-compose logs crowetrade-api | grep ERROR

# Export logs for analysis
docker-compose logs --no-color > crowetrade-logs-$(date +%Y%m%d).txt
```

## 📞 Support & Maintenance

### Regular Maintenance
- **Weekly**: Review error logs and performance metrics
- **Monthly**: Update dependencies and security patches  
- **Quarterly**: Performance optimization and capacity planning

### Monitoring Alerts
Configure alerts for:
- API response time > 100ms
- Error rate > 1%
- Memory usage > 80%
- Disk usage > 85%
- Risk limit breaches

### Backup Schedule
- **Database**: Daily automated backups (retained 30 days)
- **Configuration**: Version controlled in Git
- **Model artifacts**: Archived in object storage
- **Logs**: Rotated weekly, archived monthly

## 🎯 Next Steps

1. **Configure your trading strategies** in the signal agents
2. **Set up your data feeds** with real market data providers
3. **Configure broker connections** for live trading
4. **Set up monitoring dashboards** in Grafana
5. **Configure alerting rules** for operational issues
6. **Run paper trading** to validate the system
7. **Gradually increase position sizes** as confidence grows

## ⚠️ Important Notices

- **NEVER trade live without extensive paper trading**
- **Always start with minimal position sizes**
- **Monitor risk metrics continuously**
- **Have emergency stop procedures ready**
- **Keep backups of all configurations**
- **Test rollback procedures regularly**

---

**Contact**: For production support, create issues in the project repository.

**License**: MIT - See LICENSE file for details.

**Version**: 1.0.0 - Production Ready

---

*🚀 CroweTrade - Professional Quantitative Trading System*
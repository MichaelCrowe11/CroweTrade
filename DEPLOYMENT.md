# 🚀 CroweTrade Production Deployment Guide

Complete guide for deploying the CroweTrade AI-powered trading platform to **crowetrade.com**.

## 📋 Prerequisites

### Required Accounts & Tools
- ✅ **Fly.io Account**: Sign up at [fly.io](https://fly.io/app/sign-up)
- ✅ **Coinbase Pro Account**: For crypto trading APIs
- ✅ **Domain Access**: Control of crowetrade.com DNS (Namecheap)
- ✅ **flyctl CLI**: Fly.io command-line tool

### Environment Setup

#### 1. Install flyctl (Choose your platform)

**Windows (PowerShell as Administrator):**
```powershell
# Option 1: Using Scoop
scoop install flyctl

# Option 2: Using Chocolatey
choco install flyctl

# Option 3: Manual Download
iwr https://fly.io/install.ps1 -useb | iex
```

**macOS/Linux:**
```bash
curl -fsSL https://fly.io/install.sh | sh
```

#### 2. Authenticate with Fly.io
```bash
flyctl auth login
# Or set token directly:
export FLY_API_TOKEN="your-token-here"
```

#### 3. Set Required Environment Variables

**Critical Secrets (Required for trading):**
```powershell
# Windows PowerShell
$env:COINBASE_API_KEY = "your-coinbase-api-key"
$env:COINBASE_API_SECRET = "your-coinbase-secret" 
$env:COINBASE_PASSPHRASE = "your-coinbase-passphrase"
$env:POSTGRES_PASSWORD = "secure-postgres-password"
$env:REDIS_PASSWORD = "secure-redis-password"
```

```bash
# Linux/macOS Bash
export COINBASE_API_KEY="your-coinbase-api-key"
export COINBASE_API_SECRET="your-coinbase-secret"
export COINBASE_PASSPHRASE="your-coinbase-passphrase" 
export POSTGRES_PASSWORD="secure-postgres-password"
export REDIS_PASSWORD="secure-redis-password"
```

**Optional Configuration:**
```bash
export FLY_ORG="your-fly-org"  # Default: personal
export ENVIRONMENT="production"
```

## 🎯 Quick Deployment

### Option 1: Windows PowerShell
```powershell
# Check prerequisites and secrets
.\deploy-crowetrade.ps1 check

# Deploy everything
.\deploy-crowetrade.ps1 deploy

# Show DNS instructions only
.\deploy-crowetrade.ps1 dns
```

### Option 2: Linux/macOS Bash
```bash
# Make script executable
chmod +x deploy-crowetrade.sh

# Check prerequisites and secrets
./deploy-crowetrade.sh check

# Deploy everything
./deploy-crowetrade.sh deploy

# Show DNS instructions only  
./deploy-crowetrade.sh dns
```

## 🌐 DNS Configuration

After deployment, update your Namecheap DNS settings:

### Remove Existing Records
1. **CNAME** `www` → `parkingpage.namecheap.com` ❌
2. **URL Redirect** `@` → `http://www.crowetrade.com/` ❌

### Add New Records
| Type  | Host      | Value                      | TTL    |
|-------|-----------|----------------------------|--------|
| CNAME | @         | crowetrade-web.fly.dev     | 30 min |
| CNAME | www       | crowetrade-web.fly.dev     | 30 min |
| CNAME | api       | crowetrade-execution.fly.dev| 30 min |
| CNAME | portfolio | crowetrade-portfolio.fly.dev| 30 min |

### Keep Existing Records
- **TXT** `@` `v=spf1 include:spf.efwd.registrar-servers.com ~all` ✅

> 📝 **Note**: DNS propagation takes 5-30 minutes globally.

## 🏗️ Architecture Overview

The deployment creates 3 microservices:

### 1. 🌐 Web Frontend (`crowetrade-web`)
- **Domains**: `crowetrade.com`, `www.crowetrade.com`
- **Purpose**: React/Next.js dashboard and user interface
- **Features**: Portfolio visualization, trading controls, analytics

### 2. 🔧 Trading Engine (`crowetrade-execution`) 
- **Domain**: `api.crowetrade.com`
- **Purpose**: AI-powered trading execution and order management
- **Features**: Coinbase Pro integration, risk management, signal processing

### 3. 📊 Portfolio Service (`crowetrade-portfolio`)
- **Domain**: `portfolio.crowetrade.com`  
- **Purpose**: Portfolio management and analytics
- **Features**: Real-time PnL, risk metrics, position tracking

## 🔐 Security Features

- ✅ **HTTPS/TLS** encryption on all endpoints
- ✅ **API Authentication** with secure token management
- ✅ **Encrypted Secrets** stored in Fly.io vault
- ✅ **Rate Limiting** and DDoS protection
- ✅ **CORS** configured for cross-origin security
- ✅ **Health Monitoring** with automatic restarts

## 📊 Monitoring & Management

### Health Checks
```bash
# Check all services
curl https://crowetrade.com
curl https://api.crowetrade.com/health
curl https://portfolio.crowetrade.com/health
```

### View Logs
```bash
# Web frontend logs
flyctl logs -a crowetrade-web

# Trading engine logs  
flyctl logs -a crowetrade-execution

# Portfolio service logs
flyctl logs -a crowetrade-portfolio
```

### Service Management
```bash
# Scale services
flyctl scale count 2 -a crowetrade-execution

# Restart service
flyctl machine restart -a crowetrade-web

# Check status
flyctl status -a crowetrade-execution
```

## 🔧 Configuration Files

The deployment uses these configuration files:

- `fly.web.toml` - Web frontend configuration
- `fly.execution.toml` - Trading engine configuration  
- `fly.portfolio.toml` - Portfolio service configuration
- `docker/Dockerfile.web` - Web frontend container
- `docker/Dockerfile.execution` - Trading engine container
- `docker/Dockerfile.portfolio` - Portfolio service container

## 🚨 Troubleshooting

### Common Issues

#### 1. DNS Not Working
```bash
# Check DNS propagation
nslookup crowetrade.com
dig crowetrade.com

# Solution: Wait 30 minutes for full propagation
```

#### 2. SSL Certificate Issues
```bash
# Check certificate status
flyctl certs show crowetrade.com -a crowetrade-web

# Force renewal
flyctl certs create crowetrade.com -a crowetrade-web
```

#### 3. Service Not Starting
```bash
# Check logs for errors
flyctl logs -a crowetrade-execution

# Check secrets are set
flyctl secrets list -a crowetrade-execution

# Redeploy if needed
flyctl deploy -c fly.execution.toml --now
```

#### 4. API Connection Issues
```bash
# Verify environment variables
flyctl ssh console -a crowetrade-execution
env | grep COINBASE
```

### Getting Help

1. **Fly.io Dashboard**: https://fly.io/dashboard
2. **Fly.io Docs**: https://fly.io/docs/
3. **CroweTrade Logs**: Use `flyctl logs` commands above
4. **Support**: Check service status at https://status.fly.io/

## 🎉 Success Verification

After deployment and DNS propagation, verify these endpoints:

- ✅ **https://crowetrade.com** - Main dashboard loads
- ✅ **https://api.crowetrade.com/health** - Returns `{"status": "healthy"}`
- ✅ **https://portfolio.crowetrade.com/health** - Returns health status
- ✅ **HTTPS certificates** - No security warnings in browser

## 📈 Next Steps

1. **Monitor Performance**: Set up alerts in Fly.io dashboard
2. **Configure Trading**: Set up trading parameters and risk limits
3. **API Integration**: Connect external tools via API endpoints
4. **Scaling**: Add more instances as trading volume grows
5. **Backup**: Set up database backups and disaster recovery

---

**🎯 Your CroweTrade platform is now live at https://crowetrade.com!**

*Built with AI-powered trading algorithms, secure Coinbase Pro integration, and enterprise-grade infrastructure.*

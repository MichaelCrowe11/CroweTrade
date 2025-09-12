# 🚀 CroweTrade Production Deployment Checklist

Complete this checklist before deploying to **crowetrade.com**.

## ✅ Pre-Deployment Checklist

### 1. 📋 Prerequisites
- [ ] Fly.io account created
- [ ] flyctl CLI installed and working
- [ ] Coinbase Pro account with API access
- [ ] Domain control for crowetrade.com (Namecheap)
- [ ] Git repository is up to date

### 2. 🔐 Environment Variables Setup
Run `.\setup-env.ps1` or manually set these:

**Required Secrets:**
- [ ] `COINBASE_API_KEY` - Your Coinbase Pro API key
- [ ] `COINBASE_API_SECRET` - Your Coinbase Pro API secret  
- [ ] `COINBASE_PASSPHRASE` - Your Coinbase Pro API passphrase
- [ ] `POSTGRES_PASSWORD` - Secure PostgreSQL password
- [ ] `REDIS_PASSWORD` - Secure Redis password
- [ ] `FLY_API_TOKEN` - Your Fly.io API token

**Optional Configuration:**
- [ ] `FLY_ORG` - Your Fly.io organization (default: personal)

### 3. 🛠️ Configuration Files
Verify these files exist and are properly configured:

- [ ] `fly.web.toml` - Web frontend configuration
- [ ] `fly.execution.toml` - Trading engine configuration
- [ ] `fly.portfolio.toml` - Portfolio service configuration
- [ ] `docker/Dockerfile.web` - Web container definition
- [ ] `docker/Dockerfile.execution` - Trading engine container
- [ ] `docker/Dockerfile.portfolio` - Portfolio service container

### 4. 🧪 Pre-Deployment Testing
```powershell
# Check prerequisites
.\deploy-crowetrade.ps1 check

# Verify all secrets are set
Get-ChildItem Env: | Where-Object { $_.Name -match "COINBASE|POSTGRES|REDIS|FLY" }
```

## 🚀 Deployment Process

### Step 1: Deploy Services
```powershell
# Full deployment
.\deploy-crowetrade.ps1 deploy
```

This will:
- ✅ Create Fly.io applications
- ✅ Set encrypted secrets
- ✅ Deploy all 3 services  
- ✅ Configure SSL certificates
- ✅ Run health checks

### Step 2: Configure DNS
After deployment, update Namecheap DNS records:

**Remove these records:**
- [ ] CNAME `www` → `parkingpage.namecheap.com`
- [ ] URL Redirect `@` → `http://www.crowetrade.com/`

**Add these records:**
- [ ] CNAME `@` → `crowetrade-web.fly.dev`
- [ ] CNAME `www` → `crowetrade-web.fly.dev`  
- [ ] CNAME `api` → `crowetrade-execution.fly.dev`
- [ ] CNAME `portfolio` → `crowetrade-portfolio.fly.dev`

### Step 3: Wait for DNS Propagation
- [ ] Wait 5-30 minutes for DNS changes to propagate
- [ ] Test with: `nslookup crowetrade.com`

## ✅ Post-Deployment Verification

### 1. 🌐 Website Access
Test these URLs in your browser:

- [ ] https://crowetrade.com ✅ Main dashboard loads
- [ ] https://www.crowetrade.com ✅ Redirects to main site
- [ ] https://api.crowetrade.com/health ✅ Returns health status
- [ ] https://portfolio.crowetrade.com/health ✅ Returns health status

### 2. 🔐 Security Verification
- [ ] All URLs show valid SSL certificates (🔒 green lock)
- [ ] No mixed content warnings
- [ ] API endpoints require authentication
- [ ] Health checks respond correctly

### 3. 📊 Service Health
```powershell
# Check service status
flyctl status -a crowetrade-web
flyctl status -a crowetrade-execution  
flyctl status -a crowetrade-portfolio

# View logs
flyctl logs -a crowetrade-web --follow
flyctl logs -a crowetrade-execution --follow
flyctl logs -a crowetrade-portfolio --follow
```

### 4. 🔧 Trading System Verification
- [ ] Coinbase Pro connection established
- [ ] WebSocket feeds active
- [ ] Risk management systems online
- [ ] Portfolio tracking functional

## 🚨 Troubleshooting

### Common Issues & Solutions

#### DNS Not Resolving
```powershell
# Check DNS propagation
nslookup crowetrade.com
# Solution: Wait longer, DNS can take up to 48 hours
```

#### SSL Certificate Issues  
```powershell
# Check certificate status
flyctl certs show crowetrade.com -a crowetrade-web
# Solution: DNS must resolve first, then certificates auto-provision
```

#### Service Not Starting
```powershell
# Check logs for errors
flyctl logs -a crowetrade-execution
# Common causes: missing secrets, config errors
```

#### API Connection Failures
```powershell
# Verify secrets are set
flyctl secrets list -a crowetrade-execution
# Check Coinbase API credentials are valid
```

## 🎯 Success Criteria

Your deployment is successful when:

- ✅ **https://crowetrade.com** loads the main dashboard
- ✅ **SSL certificates** are valid (green lock icon)
- ✅ **API health checks** return 200 OK
- ✅ **Coinbase connection** is established
- ✅ **No errors** in service logs
- ✅ **DNS resolves** correctly worldwide

## 📈 Next Steps After Deployment

1. **🔍 Monitor Performance**
   - Set up alerts in Fly.io dashboard
   - Monitor error rates and response times

2. **🛡️ Security Hardening**
   - Review API access patterns
   - Set up rate limiting if needed
   - Configure backup and disaster recovery

3. **📊 Trading Configuration**
   - Configure trading parameters
   - Set risk limits and exposure caps
   - Test with small amounts first

4. **🔄 Continuous Integration**
   - Set up GitHub Actions for automated deployments
   - Configure staging environment
   - Implement proper testing pipeline

5. **📱 User Onboarding**
   - Create user documentation
   - Set up support channels
   - Plan marketing and launch strategy

---

## 🎉 Congratulations!

🚀 **CroweTrade is now live at https://crowetrade.com!**

Your AI-powered cryptocurrency trading platform is deployed with:
- ✅ Enterprise-grade infrastructure
- ✅ Secure Coinbase Pro integration  
- ✅ Real-time portfolio management
- ✅ Advanced risk controls
- ✅ Professional web interface

*Time to start trading with AI! 📈*

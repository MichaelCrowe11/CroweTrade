# 🌐 CroweTrade Domain Deployment Guide

Deploy CroweTrade to **crowetrade.com** using professional cloud platforms.

## 🚀 Quick Deploy Options

### Option 1: Fly.io (Recommended) ⭐
**Cost**: ~$5-15/month | **Setup Time**: 10 minutes

```bash
# Install Fly.io CLI
curl -L https://fly.io/install.sh | sh

# Deploy to crowetrade.com
cd CroweTrade
./scripts/deploy-domain.sh fly
```

**Benefits:**
- ✅ Automatic SSL certificates
- ✅ Global CDN and edge locations
- ✅ Built-in monitoring and metrics
- ✅ Easy domain configuration
- ✅ Production-ready infrastructure

### Option 2: Render.com
**Cost**: $25/month | **Setup Time**: 15 minutes

```bash
cd CroweTrade
./scripts/deploy-domain.sh render
```

Then follow the manual steps displayed.

**Benefits:**
- ✅ Fully managed databases
- ✅ Auto-scaling and zero-downtime deploys
- ✅ Built-in SSL and domain management
- ✅ GitHub integration for CI/CD

### Option 3: Railway
**Cost**: ~$10/month | **Setup Time**: 5 minutes

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
cd CroweTrade
./scripts/deploy-domain.sh railway
```

**Benefits:**
- ✅ Fastest deployment
- ✅ Simple pricing
- ✅ Auto-generated domains
- ✅ Environment variable management

## 📋 Domain Configuration Steps

### 1. Purchase Domain (if needed)
If you don't own crowetrade.com yet:
- **Namecheap**: ~$12/year
- **Cloudflare**: ~$10/year  
- **Google Domains**: ~$12/year
- **GoDaddy**: ~$15/year

### 2. Configure DNS Records

**For Fly.io deployment:**
```
Type    Name    Value                        TTL
A       @       [fly-ip-address]             300
CNAME   www     crowetrade.fly.dev          300
```

**For Render deployment:**
```
Type    Name    Value                        TTL
CNAME   @       crowetrade.onrender.com      300
CNAME   www     crowetrade.onrender.com      300
```

**For Railway deployment:**
```
Type    Name    Value                        TTL
CNAME   @       [railway-domain]             300
CNAME   www     [railway-domain]             300
```

### 3. SSL Certificate Setup
All platforms automatically provide free SSL certificates via Let's Encrypt.

## 🔧 Environment Configuration

### Production Environment Variables
```env
# Security (CRITICAL - Generate unique values)
JWT_SECRET_KEY=your-secure-256-bit-key
ENCRYPTION_KEY=your-secure-encryption-key

# Trading Configuration  
TRADING_MODE=PAPER          # Start with PAPER, change to LIVE when ready
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=50000
MAX_DRAWDOWN=0.20

# Database (Platform managed)
DATABASE_URL=postgresql://...

# API Keys (Add your trading data providers)
ALPHA_VANTAGE_API_KEY=your_key
FINNHUB_API_KEY=your_key
POLYGON_API_KEY=your_key
```

## 🛡️ Security Checklist

### Before Going Live:
- [ ] ✅ Unique JWT_SECRET_KEY generated
- [ ] ✅ Unique ENCRYPTION_KEY generated  
- [ ] ✅ TRADING_MODE set to PAPER initially
- [ ] ✅ Environment variables configured
- [ ] ✅ SSL certificate active
- [ ] ✅ Health checks passing
- [ ] ✅ Monitoring dashboards configured

### Production Hardening:
- [ ] ✅ Database backups enabled
- [ ] ✅ Log aggregation configured  
- [ ] ✅ Error monitoring (Sentry)
- [ ] ✅ Performance monitoring
- [ ] ✅ Uptime monitoring
- [ ] ✅ Rate limiting configured

## 📊 Monitoring Your Deployment

### Health Check Endpoints
- **Health**: https://crowetrade.com/health
- **Metrics**: https://crowetrade.com/metrics
- **API Docs**: https://crowetrade.com/docs

### Key Metrics to Monitor
- **Response Time**: < 100ms average
- **Error Rate**: < 1% 
- **Uptime**: > 99.9%
- **Memory Usage**: < 80%
- **CPU Usage**: < 70%

## 🚀 Deployment Verification

### Automated Tests
```bash
# Health check
curl https://crowetrade.com/health

# API documentation
curl https://crowetrade.com/openapi.json

# Test trading signal (requires auth)
curl -X POST https://crowetrade.com/api/v1/signals \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"instrument":"AAPL","features":{"price_change":0.02}}'
```

### Manual Verification
1. 🌐 Visit https://crowetrade.com/health - should show "healthy": true
2. 📚 Visit https://crowetrade.com/docs - should load API documentation
3. 🔒 Verify SSL certificate is valid (green lock icon)
4. 📱 Test on mobile devices
5. 🌍 Test from different geographic locations

## 💰 Cost Breakdown

### Monthly Operating Costs

**Fly.io Setup:**
- Web service: $5-15/month (1-2GB RAM)
- Database addon: $10/month (PostgreSQL)
- **Total**: ~$15-25/month

**Render Setup:**
- Web service: $25/month (includes database)
- Redis addon: $0 (free tier)
- **Total**: ~$25/month

**Railway Setup:**
- Web service: $10/month
- Database: $5/month
- **Total**: ~$15/month

## 🔄 Continuous Deployment

### GitHub Actions Workflow
Automatically created during deployment. Triggers on:
- ✅ Push to main branch
- ✅ Automated testing
- ✅ Zero-downtime deployment
- ✅ Rollback on failure

### Manual Deployment
```bash
# Update and redeploy
git add .
git commit -m "Update CroweTrade"
git push origin main

# Or direct deployment
./scripts/deploy-domain.sh fly
```

## 🛠️ Troubleshooting

### Common Issues

**SSL Certificate Not Working:**
```bash
# Force SSL renewal (Fly.io)
flyctl certs create crowetrade.com --app crowetrade
```

**Deployment Timeout:**
```bash
# Check deployment status
flyctl status --app crowetrade
flyctl logs --app crowetrade
```

**Environment Variable Issues:**
```bash
# List current variables
flyctl secrets list --app crowetrade

# Update variables
flyctl secrets set TRADING_MODE=PAPER --app crowetrade
```

**Health Check Failures:**
1. Check application logs
2. Verify environment variables
3. Test database connection
4. Verify container startup

### Support Resources
- **Fly.io Docs**: https://fly.io/docs/
- **Render Docs**: https://render.com/docs
- **Railway Docs**: https://docs.railway.app/

## 🎯 Next Steps After Deployment

### 1. Initial Setup (First 24 hours)
- [ ] Verify all endpoints working
- [ ] Set up monitoring alerts
- [ ] Test paper trading functionality
- [ ] Configure backup procedures

### 2. Production Readiness (First week)
- [ ] Load testing with realistic traffic
- [ ] Security audit and penetration testing
- [ ] Performance optimization
- [ ] Database optimization

### 3. Go-Live Preparation (When ready)
- [ ] Switch TRADING_MODE from PAPER to LIVE
- [ ] Configure real broker connections
- [ ] Set up trading strategies
- [ ] Implement proper risk management

## ⚠️ Critical Reminders

### 🛡️ SAFETY FIRST
- **Always start with PAPER trading mode**
- **Never deploy live trading without extensive testing**
- **Monitor risk metrics continuously**
- **Have emergency stop procedures ready**

### 🔒 SECURITY
- **Keep API keys secure and rotated**
- **Monitor for unauthorized access**
- **Regular security updates**
- **Audit trail for all trades**

### 💼 COMPLIANCE
- **Know your regulatory requirements**
- **Implement proper record keeping**
- **Risk disclosure to users**
- **Data protection compliance**

---

## 🚀 Ready to Deploy?

Choose your platform and run the deployment:

```bash
# Recommended: Fly.io
./scripts/deploy-domain.sh fly

# Alternative: Render
./scripts/deploy-domain.sh render

# Alternative: Railway
./scripts/deploy-domain.sh railway
```

Your professional trading platform will be live at **https://crowetrade.com** in minutes!

---

*🌟 CroweTrade - Professional Quantitative Trading Platform*
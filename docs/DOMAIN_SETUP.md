# CroweTrade Domain Configuration Guide

## Overview
This guide helps you connect your `crowetrade.com` domain to your CroweTrade quantitative trading platform.

## Architecture
- **Main Site**: `https://crowetrade.com` - Next.js frontend with quantum agent dashboard
- **API Gateway**: `https://api.crowetrade.com` - Execution service and market data
- **Portfolio API**: `https://portfolio.crowetrade.com` - Portfolio management and risk analytics

## DNS Configuration

### Required DNS Records
Add these CNAME records to your domain registrar's DNS settings:

```
Type    | Name      | Value                        | TTL
--------|-----------|------------------------------|-----
CNAME   | @         | crowetrade-main.fly.dev     | 300
CNAME   | www       | crowetrade-main.fly.dev     | 300
CNAME   | api       | crowetrade-execution.fly.dev | 300
CNAME   | portfolio | crowetrade-portfolio.fly.dev | 300
```

### Alternative A Record Configuration
If your DNS provider doesn't support CNAME records for the root domain (@), use these A records:

```
Type | Name      | Value           | TTL
-----|-----------|-----------------|-----
A    | @         | 66.241.124.44   | 300
A    | @         | 66.241.125.44   | 300
AAAA | @         | 2a09:8280:1::   | 300
```

## Deployment Steps

### 1. Prerequisites
- Install Fly.io CLI: `https://fly.io/docs/getting-started/installing-flyctl/`
- Authenticate: `flyctl auth login`
- Ensure you own the `crowetrade.com` domain

### 2. Deploy Platform
```powershell
# Windows PowerShell
.\scripts\deploy-domain.ps1

# Or deploy individual services
flyctl deploy --config fly.toml
flyctl deploy --config fly.execution.toml  
flyctl deploy --config fly.portfolio.toml
```

### 3. Configure SSL Certificates
```bash
# Add certificates for all domains
flyctl certs add crowetrade.com --app crowetrade-main
flyctl certs add www.crowetrade.com --app crowetrade-main
flyctl certs add api.crowetrade.com --app crowetrade-execution
flyctl certs add portfolio.crowetrade.com --app crowetrade-portfolio
```

### 4. Verify Configuration
```bash
# Check certificate status
flyctl certs list --app crowetrade-main
flyctl certs show crowetrade.com --app crowetrade-main

# Test endpoints
curl -I https://crowetrade.com
curl -I https://api.crowetrade.com/health
curl -I https://portfolio.crowetrade.com/health
```

## Environment Variables

### Frontend (.env.production)
```
NEXT_PUBLIC_API_URL=https://api.crowetrade.com
NEXT_PUBLIC_WS_URL=wss://api.crowetrade.com/ws
NEXT_PUBLIC_PORTFOLIO_URL=https://portfolio.crowetrade.com
NODE_ENV=production
```

### Backend Services
```
CORS_ORIGINS=https://crowetrade.com,https://www.crowetrade.com
ALLOWED_HOSTS=api.crowetrade.com,portfolio.crowetrade.com
SSL_REDIRECT=true
SECURE_HEADERS=true
```

## Security Configuration

### CORS Policy
- Frontend domain: `https://crowetrade.com`
- API cross-origin requests allowed from main domain only
- WebSocket connections secured with origin validation

### SSL/TLS
- Automatic SSL certificate provisioning via Let's Encrypt
- HSTS headers enabled
- HTTP to HTTPS redirects enforced

## Monitoring

### Health Checks
- Frontend: `https://crowetrade.com/` (200 OK)
- API: `https://api.crowetrade.com/health` (JSON health status)
- Portfolio: `https://portfolio.crowetrade.com/health` (JSON health status)

### Logging
```bash
# View application logs
flyctl logs --app crowetrade-main
flyctl logs --app crowetrade-execution
flyctl logs --app crowetrade-portfolio

# Monitor certificate renewal
flyctl certs list
```

## Troubleshooting

### Common Issues

1. **DNS Propagation Delay**
   - Wait 5-15 minutes after DNS changes
   - Use `nslookup crowetrade.com` to verify propagation
   - Check from multiple locations: `https://www.whatsmydns.net/`

2. **Certificate Issues**
   - Verify DNS records are correct
   - Check certificate status: `flyctl certs show crowetrade.com --app crowetrade-main`
   - Certificates can take up to 10 minutes to issue

3. **CORS Errors**
   - Ensure frontend uses exact domain: `https://crowetrade.com` (no trailing slash)
   - Check browser console for detailed error messages
   - Verify CORS_ORIGINS environment variable

### Support Commands
```bash
# Check app status
flyctl status --app crowetrade-main

# Scale if needed
flyctl scale count 2 --app crowetrade-main

# View configuration
flyctl config show --app crowetrade-main
```

## Custom Domain Providers

### Popular DNS Providers
- **Cloudflare**: Proxy through Cloudflare for additional security
- **Route 53**: Use ALIAS records for root domain
- **Namecheap**: Standard CNAME configuration
- **GoDaddy**: May require A records for root domain

### Cloudflare Configuration (Optional)
If using Cloudflare as DNS provider:
1. Set DNS records as CNAME (DNS Only, not proxied initially)
2. Wait for certificates to be issued
3. Enable proxy (orange cloud) for additional security

## Production Checklist

- [ ] DNS records configured
- [ ] SSL certificates issued and valid
- [ ] All services responding to health checks
- [ ] CORS configuration tested
- [ ] WebSocket connections working
- [ ] Environment variables set correctly
- [ ] Monitoring and alerting configured
- [ ] Backup and disaster recovery plan in place

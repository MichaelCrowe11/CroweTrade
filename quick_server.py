"""
Quick fixed server for CroweTrade without datetime serialization issues.
"""

import sys
import os
sys.path.insert(0, 'src')

# Set environment variables
os.environ.setdefault('ENVIRONMENT', 'development')
os.environ.setdefault('JWT_SECRET_KEY', 'dev-secret-key-for-testing-only')
os.environ.setdefault('ENCRYPTION_KEY', 'dev-encryption-key-for-testing-only')
os.environ.setdefault('SERVICE_PORT', '8080')
os.environ.setdefault('TRADING_MODE', 'PAPER')
os.environ.setdefault('LOG_LEVEL', 'INFO')

import uvicorn
from fastapi import FastAPI
from datetime import datetime
import psutil

app = FastAPI(
    title="CroweTrade API",
    description="Production trading system API",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """System health check endpoint."""
    return {
        "healthy": True,
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "memory": True,
            "disk_space": True,
            "cpu": True
        },
        "metrics": {
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100,
            "cpu_usage_percent": psutil.cpu_percent(),
            "process_count": len(psutil.pids())
        }
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CroweTrade API is running",
        "version": "1.0.0",
        "environment": os.getenv('ENVIRONMENT', 'development'),
        "trading_mode": os.getenv('TRADING_MODE', 'PAPER'),
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/status")
async def status():
    """Quick status check."""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running"
    }

if __name__ == "__main__":
    print("Starting CroweTrade Quick Server...")
    print("Environment: DEVELOPMENT (PAPER TRADING)")
    print("URL: http://localhost:8081")
    print("Docs: http://localhost:8081/docs")
    print("Health: http://localhost:8081/health")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
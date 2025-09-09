"""
Production FastAPI server for CroweTrade.
Provides REST API endpoints for trading operations, monitoring, and management.
"""

import asyncio
import logging
import uvicorn
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import traceback

from ..config.production import get_config, ProductionConfig
from ..security.authentication import (
    get_jwt_manager, get_security_middleware, get_api_key_manager,
    User, initialize_security
)
from ..monitoring.metrics import (
    get_health_checker, get_metrics_collector, get_trading_metrics,
    get_performance_tracker, HealthStatus
)
from ..core.contracts import Signal, TargetPosition
from ..live.signal_agent import SignalAgent
from ..live.portfolio_agent import PortfolioAgent
from ..live.risk_guard import RiskGuard

logger = logging.getLogger(__name__)

# Request/Response Models
class HealthResponse(BaseModel):
    """Health check response."""
    healthy: bool
    timestamp: datetime
    checks: Dict[str, bool]
    metrics: Dict[str, float]


class MetricsResponse(BaseModel):
    """Metrics endpoint response."""
    timestamp: datetime
    counters: Dict[str, float]
    gauges: Dict[str, float] 
    histograms: Dict[str, Any]


class SignalRequest(BaseModel):
    """Request to generate trading signal."""
    instrument: str
    features: Dict[str, float]
    strategy: str = "default"


class SignalResponse(BaseModel):
    """Trading signal response."""
    instrument: str
    mu: float
    sigma: float
    prob_edge_pos: float
    policy_id: str
    timestamp: datetime


class PositionRequest(BaseModel):
    """Request to calculate position targets."""
    signals: List[Dict[str, Any]]
    risk_budget: float = 0.02
    volatility: Dict[str, float] = Field(default_factory=dict)


class PositionResponse(BaseModel):
    """Position targets response."""
    targets: Dict[str, float]
    total_exposure: float
    risk_utilization: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    timestamp: datetime


# FastAPI app setup
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = get_config()
    
    app = FastAPI(
        title="CroweTrade API",
        description="Production trading system API",
        version="1.0.0",
        docs_url="/docs" if config.environment != "production" else None,
        redoc_url="/redoc" if config.environment != "production" else None,
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if config.environment == "development" else [
            "localhost", "127.0.0.1", 
            "crowetrade.fly.dev", "crowetrade.com", "www.crowetrade.com"
        ]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Initialize security
    initialize_security(config.security.jwt_secret_key, config.security.encryption_key)
    
    return app


app = create_app()
security = HTTPBearer()


# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
    """Extract and validate user from JWT token or API key."""
    token = credentials.credentials
    
    # Try JWT first
    jwt_manager = get_jwt_manager()
    payload = jwt_manager.verify_token(token)
    
    if payload:
        return User(
            user_id=payload['user_id'],
            username=payload['username'],
            email="",  # Would load from database
            roles=payload['roles']
        )
    
    # Try API key
    api_key_manager = get_api_key_manager()
    api_key = api_key_manager.verify_api_key(token)
    
    if api_key:
        return User(
            user_id=api_key.user_id,
            username="",  # Would load from database
            email="",
            roles=[]  # Would load from database based on user
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials"
    )


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": "",
            "timestamp": datetime.now().isoformat()
        }
    )


# Health and monitoring endpoints
@app.get("/health")
async def health_check():
    """System health check endpoint."""
    health_checker = get_health_checker()
    
    with get_performance_tracker().track_operation("health_check"):
        health_status = await health_checker.check_health()
    
    return {
        "healthy": health_status.healthy,
        "timestamp": health_status.timestamp.isoformat(),
        "checks": health_status.checks,
        "metrics": health_status.metrics
    }


@app.get("/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)):
    """Get system metrics (requires authentication)."""
    if not current_user.has_any_role(["admin", "monitor"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    metrics_collector = get_metrics_collector()
    metrics = metrics_collector.get_metrics()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "counters": metrics['counters'],
        "gauges": metrics['gauges'],
        "histograms": metrics['histograms']
    }


# Trading endpoints
@app.post("/api/v1/signals")
async def generate_signal(
    request: SignalRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate a trading signal."""
    if not current_user.has_any_role(["trader", "admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    trading_metrics = get_trading_metrics()
    
    with get_performance_tracker().track_operation("generate_signal"):
        try:
            # Mock signal agent - in production this would be properly initialized
            signal_agent = SignalAgent(
                model=lambda x: (0.01, 0.02, 0.6),  # Mock model
                policy_id="mock_policy_v1",
                gates={"prob_edge_min": 0.55, "sigma_max": 0.05}
            )
            
            # Create feature vector
            from ..core.contracts import FeatureVector
            feature_vector = FeatureVector(
                instrument=request.instrument,
                asof=datetime.now(),
                horizon="1d",
                values=request.features,
                quality={"coverage": 1.0, "lag_ms": 100}
            )
            
            signal = signal_agent.infer(feature_vector)
            
            if not signal:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No signal generated - failed gate checks"
                )
            
            # Record metrics
            trading_metrics.record_signal_generated(
                instrument=request.instrument,
                strategy=request.strategy
            )
            
            return {
                "instrument": signal.instrument,
                "mu": signal.mu,
                "sigma": signal.sigma,
                "prob_edge_pos": signal.prob_edge_pos,
                "policy_id": signal.policy_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Signal generation failed: {str(e)}"
            )


@app.post("/api/v1/positions")
async def calculate_positions(
    request: PositionRequest,
    current_user: User = Depends(get_current_user)
):
    """Calculate position targets from signals."""
    if not current_user.has_any_role(["trader", "admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    with get_performance_tracker().track_operation("calculate_positions"):
        try:
            # Mock portfolio agent
            portfolio_agent = PortfolioAgent(
                risk_budget=request.risk_budget,
                turnover_penalty=0.01
            )
            
            # Convert signals
            signals = {}
            for sig_dict in request.signals:
                from ..core.contracts import Signal
                signal = Signal(**sig_dict)
                signals[signal.instrument] = signal
            
            # Calculate targets
            targets = portfolio_agent.size(signals, request.volatility)
            
            # Calculate exposure and utilization
            total_exposure = sum(abs(qty) for qty in targets.values())
            risk_utilization = total_exposure / request.risk_budget if request.risk_budget > 0 else 0
            
            return {
                "targets": targets,
                "total_exposure": total_exposure,
                "risk_utilization": risk_utilization,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Position calculation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Position calculation failed: {str(e)}"
            )


# Risk management endpoints
@app.get("/api/v1/risk/status")
async def get_risk_status(current_user: User = Depends(get_current_user)):
    """Get current risk status."""
    if not current_user.has_any_role(["trader", "risk", "admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Mock risk guard
    risk_guard = RiskGuard(dd_limit=0.20, var_limit=0.05)
    
    return {
        "max_drawdown": risk_guard.max_drawdown,
        "current_drawdown": getattr(risk_guard, 'current_dd', 0),
        "pnl_peak": risk_guard.pnl_peak,
        "risk_limits": {
            "drawdown_limit": risk_guard.dd_limit,
            "var_limit": risk_guard.var_limit
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/risk/emergency-stop")
async def emergency_stop(current_user: User = Depends(get_current_user)):
    """Emergency stop all trading."""
    if not current_user.has_any_role(["risk", "admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    logger.critical(f"Emergency stop triggered by {current_user.username}")
    
    # In production, this would trigger actual emergency stop procedures
    get_trading_metrics().record_risk_breach(
        breach_type="emergency_stop",
        value=1,
        limit=0
    )
    
    return {
        "message": "Emergency stop activated",
        "timestamp": datetime.now().isoformat(),
        "triggered_by": current_user.username
    }


# Configuration endpoints
@app.get("/api/v1/config")
async def get_system_config(current_user: User = Depends(get_current_user)):
    """Get system configuration (admin only)."""
    if not current_user.has_role("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    config = get_config()
    return config.to_dict()


# Start server
def start_server():
    """Start the FastAPI server."""
    config = get_config()
    
    uvicorn.run(
        "crowetrade.api.server:app",
        host="0.0.0.0",
        port=config.service_port,
        workers=config.trading.worker_processes if config.environment == "production" else 1,
        log_level=config.monitoring.log_level.lower(),
        access_log=True,
        reload=config.environment == "development"
    )


if __name__ == "__main__":
    start_server()
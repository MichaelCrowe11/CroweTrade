"""Configuration management system for CroweTrade.

Provides hierarchical configuration with environment overrides,
type validation, and runtime reloading support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator


class Environment(str, Enum):
    """Trading environment types."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    PAPER = "paper"
    PRODUCTION = "production"


class MarketDataConfig(BaseModel):
    """Market data ingestion configuration."""
    
    providers: List[str] = Field(default_factory=lambda: ["polygon", "alpaca"])
    tick_buffer_size: int = Field(10000, gt=0)
    snapshot_interval: int = Field(60, gt=0)  # seconds
    max_reconnect_attempts: int = Field(5, gt=0)
    reconnect_delay: float = Field(1.0, gt=0)
    
    class Config:
        extra = "forbid"


class BrokerConfig(BaseModel):
    """Broker/exchange configuration."""
    
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    account_id: Optional[str] = None
    base_url: Optional[str] = None
    paper_trading: bool = True
    rate_limit: int = Field(100, gt=0)  # requests per minute
    
    @validator("api_key", "api_secret", pre=True)
    def resolve_env_vars(cls, v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.environ.get(env_var)
        return v


class RiskConfig(BaseModel):
    """Risk management configuration."""
    
    max_position_size: float = Field(0.1, gt=0, le=1)  # % of portfolio
    max_leverage: float = Field(1.0, gt=0)
    max_drawdown: float = Field(0.05, gt=0, le=1)
    intraday_loss_limit: float = Field(0.02, gt=0, le=1)
    var_limit: float = Field(0.015, gt=0, le=1)
    position_limits: Dict[str, float] = Field(default_factory=dict)
    restricted_symbols: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "forbid"


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    
    default_algo: str = Field("LIMIT", pattern="^(LIMIT|MARKET|TWAP|VWAP|POV|IS)$")
    max_order_size: float = Field(10000, gt=0)
    participation_rate: float = Field(0.1, gt=0, le=1)
    slippage_model: str = Field("linear")
    urgency: str = Field("normal", pattern="^(low|normal|high|urgent)$")
    smart_routing: bool = True
    
    class Config:
        extra = "forbid"


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    
    lookback_periods: List[int] = Field(default_factory=lambda: [5, 10, 20, 50])
    update_frequency: int = Field(60, gt=0)  # seconds
    max_feature_lag: int = Field(300, gt=0)  # seconds
    cache_size: int = Field(10000, gt=0)
    
    class Config:
        extra = "forbid"


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    
    start_date: str
    end_date: str
    initial_capital: float = Field(100000, gt=0)
    commission: float = Field(0.001, ge=0)
    slippage: float = Field(0.0005, ge=0)
    data_frequency: str = Field("1m", pattern="^(1m|5m|15m|30m|1h|1d)$")
    walk_forward_periods: int = Field(12, gt=0)
    
    class Config:
        extra = "forbid"


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    timescale_url: str = Field("postgresql://localhost:5432/crowetrade")
    redis_url: str = Field("redis://localhost:6379/0")
    max_connections: int = Field(20, gt=0)
    connection_timeout: int = Field(30, gt=0)
    
    @validator("timescale_url", "redis_url", pre=True)
    def resolve_env_vars(cls, v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.environ.get(env_var)
        return v


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    prometheus_port: int = Field(9090, gt=0)
    grafana_url: str = Field("http://localhost:3000")
    alert_webhook: Optional[str] = None
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    metrics_interval: int = Field(10, gt=0)  # seconds
    
    class Config:
        extra = "forbid"


class Settings(BaseModel):
    """Main configuration settings."""
    
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    brokers: List[BrokerConfig] = Field(default_factory=list)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    backtest: Optional[BacktestConfig] = None
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    class Config:
        extra = "forbid"
        use_enum_values = True
    
    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> Settings:
        """Load configuration from environment with defaults."""
        config_path = os.environ.get("CROWETRADE_CONFIG")
        if config_path:
            return cls.from_yaml(Path(config_path))
        
        # Build from environment variables
        env = os.environ.get("CROWETRADE_ENV", "development")
        return cls(
            environment=Environment(env),
            debug=os.environ.get("CROWETRADE_DEBUG", "false").lower() == "true",
        )
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


@dataclass
class ConfigManager:
    """Manages configuration lifecycle and reloading."""
    
    settings: Settings = field(default_factory=Settings.from_env)
    config_path: Optional[Path] = None
    _callbacks: List[Any] = field(default_factory=list)
    
    def reload(self) -> None:
        """Reload configuration from source."""
        if self.config_path and self.config_path.exists():
            self.settings = Settings.from_yaml(self.config_path)
        else:
            self.settings = Settings.from_env()
        
        # Notify listeners
        for callback in self._callbacks:
            callback(self.settings)
    
    def register_callback(self, callback: Any) -> None:
        """Register a callback for configuration changes."""
        self._callbacks.append(callback)
    
    def get_broker(self, name: str) -> Optional[BrokerConfig]:
        """Get broker configuration by name."""
        for broker in self.settings.brokers:
            if broker.name == name:
                return broker
        return None


# Global configuration instance
config = ConfigManager()
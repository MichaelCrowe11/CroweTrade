"""
Production-grade configuration management for CroweTrade.
Handles environment variables, secrets, and production settings.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    jwt_secret_key: str
    encryption_key: str
    api_rate_limit: int = 1000
    execution_rate_limit: int = 100
    enable_2fa: bool = True
    session_timeout: int = 3600  # seconds


@dataclass 
class DatabaseConfig:
    """Database connection configuration."""
    url: str
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class MessageBusConfig:
    """Message bus (Kafka) configuration."""
    bootstrap_servers: str
    security_protocol: str = "SASL_SSL"
    sasl_mechanism: str = "PLAIN" 
    sasl_username: str = ""
    sasl_password: str = ""
    consumer_group: str = "crowetrade"
    auto_offset_reset: str = "latest"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float
    max_daily_loss: float
    max_drawdown: float
    var_confidence_level: float = 0.99
    default_risk_budget: float = 0.02
    default_lambda_tempering: float = 0.25
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    prometheus_gateway_url: str
    grafana_url: str
    log_level: str = "INFO"
    sentry_dsn: Optional[str] = None
    health_check_interval: int = 30


@dataclass
class TradingConfig:
    """Trading system configuration."""
    mode: str = "PAPER"  # PAPER or LIVE
    feature_store_path: str = "/data/features"
    model_registry_path: str = "/data/models"
    worker_processes: int = 4
    max_concurrent_requests: int = 100
    

@dataclass
class ProductionConfig:
    """Main production configuration."""
    environment: str
    service_port: int
    security: SecurityConfig
    database: DatabaseConfig
    message_bus: MessageBusConfig  
    risk: RiskConfig
    monitoring: MonitoringConfig
    trading: TradingConfig
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Load configuration from environment variables."""
        
        # Security configuration
        security = SecurityConfig(
            jwt_secret_key=os.getenv('JWT_SECRET_KEY', ''),
            encryption_key=os.getenv('ENCRYPTION_KEY', ''),
            api_rate_limit=int(os.getenv('API_RATE_LIMIT', '1000')),
            execution_rate_limit=int(os.getenv('EXECUTION_RATE_LIMIT', '100')),
        )
        
        # Database configuration  
        database = DatabaseConfig(
            url=os.getenv('DATABASE_URL', 'sqlite:///crowetrade.db'),
            pool_size=int(os.getenv('DB_POOL_SIZE', '20')),
        )
        
        # Message bus configuration
        message_bus = MessageBusConfig(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            security_protocol=os.getenv('KAFKA_SECURITY_PROTOCOL', 'SASL_SSL'),
            sasl_mechanism=os.getenv('KAFKA_SASL_MECHANISM', 'PLAIN'),
            sasl_username=os.getenv('KAFKA_SASL_USERNAME', ''),
            sasl_password=os.getenv('KAFKA_SASL_PASSWORD', ''),
        )
        
        # Risk configuration
        risk = RiskConfig(
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '100000')),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '50000')),
            max_drawdown=float(os.getenv('MAX_DRAWDOWN', '0.20')),
            var_confidence_level=float(os.getenv('VAR_CONFIDENCE_LEVEL', '0.99')),
            default_risk_budget=float(os.getenv('DEFAULT_RISK_BUDGET', '0.02')),
            default_lambda_tempering=float(os.getenv('DEFAULT_LAMBDA_TEMPERING', '0.25')),
        )
        
        # Monitoring configuration
        monitoring = MonitoringConfig(
            prometheus_gateway_url=os.getenv('PROMETHEUS_GATEWAY_URL', 'http://localhost:9091'),
            grafana_url=os.getenv('GRAFANA_URL', 'http://localhost:3000'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            sentry_dsn=os.getenv('SENTRY_DSN'),
            health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', '30')),
        )
        
        # Trading configuration
        trading = TradingConfig(
            mode=os.getenv('TRADING_MODE', 'PAPER'),
            feature_store_path=os.getenv('FEATURE_STORE_PATH', '/data/features'),
            model_registry_path=os.getenv('MODEL_REGISTRY_PATH', '/data/models'),
            worker_processes=int(os.getenv('WORKER_PROCESSES', '4')),
            max_concurrent_requests=int(os.getenv('MAX_CONCURRENT_REQUESTS', '100')),
        )
        
        # API keys
        api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
            'finnhub': os.getenv('FINNHUB_API_KEY', ''),
            'quandl': os.getenv('QUANDL_API_KEY', ''),
            'iex': os.getenv('IEX_API_KEY', ''),
            'polygon': os.getenv('POLYGON_API_KEY', ''),
        }
        
        return cls(
            environment=os.getenv('ENVIRONMENT', 'development'),
            service_port=int(os.getenv('SERVICE_PORT', '8080')),
            security=security,
            database=database,
            message_bus=message_bus,
            risk=risk,
            monitoring=monitoring,
            trading=trading,
            api_keys=api_keys,
        )
        
    def validate(self) -> bool:
        """Validate configuration for production readiness."""
        errors = []
        
        # Security validation
        if not self.security.jwt_secret_key:
            errors.append("JWT_SECRET_KEY is required")
        if not self.security.encryption_key:
            errors.append("ENCRYPTION_KEY is required")
            
        # Database validation  
        if not self.database.url or self.database.url == 'sqlite:///crowetrade.db':
            if self.environment == 'production':
                errors.append("Production database URL is required")
                
        # Trading mode validation
        if self.trading.mode not in ['PAPER', 'LIVE']:
            errors.append("TRADING_MODE must be PAPER or LIVE")
            
        # Risk limits validation
        if self.risk.max_drawdown >= 1.0:
            errors.append("MAX_DRAWDOWN must be less than 1.0")
            
        if errors:
            logger.error("Configuration validation errors: %s", errors)
            return False
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding secrets)."""
        config_dict = {
            'environment': self.environment,
            'service_port': self.service_port,
            'trading_mode': self.trading.mode,
            'risk_limits': {
                'max_position_size': self.risk.max_position_size,
                'max_daily_loss': self.risk.max_daily_loss,
                'max_drawdown': self.risk.max_drawdown,
            },
            'monitoring': {
                'log_level': self.monitoring.log_level,
                'health_check_interval': self.monitoring.health_check_interval,
            }
        }
        return config_dict


def load_config() -> ProductionConfig:
    """Load and validate production configuration."""
    config = ProductionConfig.from_env()
    
    if not config.validate():
        raise ValueError("Invalid configuration - see logs for details")
        
    logger.info("Configuration loaded successfully for environment: %s", config.environment)
    return config


# Global configuration instance
CONFIG: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get the global configuration instance."""
    global CONFIG
    if CONFIG is None:
        CONFIG = load_config()
    return CONFIG
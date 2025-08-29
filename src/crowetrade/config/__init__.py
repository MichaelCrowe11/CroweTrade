from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Environment(Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class CloudProvider(Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class MessageBus(Enum):
    KAFKA = "kafka"
    NATS = "nats"
    PUBSUB = "pubsub"
    EVENTHUB = "eventhub"


@dataclass
class TradingConfig:
    env: Environment = Environment.DEV
    cloud: CloudProvider = CloudProvider.AWS
    region: str = "us-east-1"
    message_bus: MessageBus = MessageBus.KAFKA
    
    data_lake: str = "s3://crowetrade-data-lake"
    online_feature_store: str = "redis://feature-store:6379"
    model_registry: str = "s3://crowetrade-models"
    
    asset_classes: list[str] = None
    venues: list[str] = None
    timezone: str = "UTC"
    symbols_source: str = "universe.json"
    
    strategy_families: list[str] = None
    
    p50_latency_ms: float = 1.0
    p99_latency_ms: float = 5.0
    
    availability_slo: float = 99.99
    slo_window_hours: int = 24
    rto_minutes: int = 10
    rpo_minutes: int = 5
    
    regulatory_rulesets: list[str] = None
    secrets_manager: str = "vault"
    
    def __post_init__(self):
        if self.asset_classes is None:
            self.asset_classes = ["equity", "futures", "options", "fx", "crypto"]
        if self.venues is None:
            self.venues = ["nasdaq", "nyse", "cboe", "cme", "ice"]
        if self.strategy_families is None:
            self.strategy_families = ["momentum", "stat_arb", "market_making", "mean_reversion"]
        if self.regulatory_rulesets is None:
            self.regulatory_rulesets = ["SEC", "FINRA", "MiFID_II", "RegNMS"]


config = TradingConfig()
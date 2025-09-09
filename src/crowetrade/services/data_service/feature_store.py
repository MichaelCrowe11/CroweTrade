"""Feature Store Module - Online and Offline Feature Materialization"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Feature:
    name: str
    value: Any
    timestamp: datetime
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class FeatureVector:
    entity_id: str
    features: dict[str, Feature]
    timestamp: datetime
    version: str = "v1.0"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "timestamp": self.timestamp.isoformat(),
            "version": self.version
        }


class FeatureStore:
    def __init__(self, online_config: dict[str, Any], offline_config: dict[str, Any]):
        self.online_config = online_config
        self.offline_config = offline_config
        self.online_store = {}
        self.offline_store = {}
        
    def register_feature(self, name: str, compute_fn: Any, ttl_seconds: int = 300) -> None:
        self.online_store[name] = {
            "compute_fn": compute_fn,
            "ttl_seconds": ttl_seconds
        }
        
    def materialize_online(self, entity_id: str, feature_names: list[str]) -> FeatureVector:
        features = {}
        timestamp = datetime.utcnow()
        
        for name in feature_names:
            if name in self.online_store:
                feature_spec = self.online_store[name]
                value = feature_spec["compute_fn"](entity_id)
                features[name] = Feature(
                    name=name,
                    value=value,
                    timestamp=timestamp
                )
        
        return FeatureVector(
            entity_id=entity_id,
            features=features,
            timestamp=timestamp
        )
        
    def materialize_offline(self, entity_id: str, feature_names: list[str], 
                           start_time: datetime, end_time: datetime) -> list[FeatureVector]:
        return []
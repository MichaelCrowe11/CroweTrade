"""Data Labeling Module for Training"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class LabelType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    SEQUENCE = "sequence"


@dataclass
class LabelingConfig:
    label_type: LabelType
    lookback_window: int
    lookahead_window: int
    threshold: float | None = None
    custom_fn: Callable | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "label_type": self.label_type.value,
            "lookback_window": self.lookback_window,
            "lookahead_window": self.lookahead_window,
            "threshold": self.threshold
        }


@dataclass
class LabeledData:
    timestamp: datetime
    features: dict[str, Any]
    label: Any
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "label": self.label,
            "metadata": self.metadata
        }


class DataLabeler:
    def __init__(self):
        self.labeling_functions = {}
        
    def register_labeling_function(self, name: str, fn: Callable) -> None:
        self.labeling_functions[name] = fn
        
    def label_data(self, data: list[dict[str, Any]], config: LabelingConfig) -> list[LabeledData]:
        labeled_data = []
        
        for i in range(config.lookback_window, len(data) - config.lookahead_window):
            lookback = data[i - config.lookback_window:i]
            current = data[i]
            lookahead = data[i:i + config.lookahead_window]
            
            label = self._generate_label(lookback, current, lookahead, config)
            
            labeled_data.append(LabeledData(
                timestamp=datetime.fromisoformat(current["timestamp"]),
                features=current.get("features", {}),
                label=label,
                metadata={"labeling_config": config.to_dict()}
            ))
        
        return labeled_data
    
    def _generate_label(self, lookback: list[dict], current: dict, 
                       lookahead: list[dict], config: LabelingConfig) -> Any:
        if config.custom_fn:
            return config.custom_fn(lookback, current, lookahead)
        
        if config.label_type == LabelType.REGRESSION:
            return self._regression_label(lookahead, config)
        elif config.label_type == LabelType.CLASSIFICATION:
            return self._classification_label(lookahead, config)
        else:
            return self._sequence_label(lookahead, config)
    
    def _regression_label(self, lookahead: list[dict], config: LabelingConfig) -> float:
        if not lookahead:
            return 0.0
        return lookahead[-1].get("price", 0.0) - lookahead[0].get("price", 0.0)
    
    def _classification_label(self, lookahead: list[dict], config: LabelingConfig) -> int:
        if not lookahead or not config.threshold:
            return 0
        
        price_change = lookahead[-1].get("price", 0.0) - lookahead[0].get("price", 0.0)
        
        if price_change > config.threshold:
            return 1
        elif price_change < -config.threshold:
            return -1
        else:
            return 0
    
    def _sequence_label(self, lookahead: list[dict], config: LabelingConfig) -> list[Any]:
        return [d.get("price", 0.0) for d in lookahead]
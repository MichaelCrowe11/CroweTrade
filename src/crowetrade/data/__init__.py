from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Protocol


class DataQuality(Enum):
    REALTIME = "realtime"
    DELAYED = "delayed"
    STALE = "stale"
    MISSING = "missing"


class MarketDataType(Enum):
    TICK = "tick"
    L1 = "l1"
    L2 = "l2"
    TRADE = "trade"
    QUOTE = "quote"
    IMBALANCE = "imbalance"
    NEWS = "news"
    REFERENCE = "reference"


@dataclass
class MarketData:
    instrument: str
    timestamp: datetime
    data_type: MarketDataType
    venue: str
    data: dict[str, Any]
    quality: DataQuality
    lag_ms: float
    sequence_num: int | None = None
    
    def is_valid(self) -> bool:
        return self.quality in (DataQuality.REALTIME, DataQuality.DELAYED)


@dataclass
class OrderBook:
    instrument: str
    venue: str
    timestamp: datetime
    bids: list[tuple[float, float, int]]  # price, size, count
    asks: list[tuple[float, float, int]]  # price, size, count
    mid_price: float | None = None
    spread: float | None = None
    imbalance: float | None = None
    
    def calculate_metrics(self):
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            self.mid_price = (best_bid + best_ask) / 2
            self.spread = best_ask - best_bid
            
            bid_volume = sum(b[1] for b in self.bids[:5])
            ask_volume = sum(a[1] for a in self.asks[:5])
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                self.imbalance = (bid_volume - ask_volume) / total_volume


class DataIngester(ABC):
    @abstractmethod
    async def connect(self) -> None:
        pass
    
    @abstractmethod
    async def subscribe(self, instruments: list[str], data_types: list[MarketDataType]) -> None:
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        pass
    
    @abstractmethod
    async def get_snapshot(self, instrument: str) -> MarketData:
        pass


class DataValidator(Protocol):
    def validate(self, data: MarketData) -> tuple[bool, str | None]:
        ...


from .ingestion import FeatureAgent, FeatureConfig, MarketDataAgent, MarketDataConfig

__all__ = [
    "DataQuality",
    "MarketDataType",
    "MarketData",
    "OrderBook",
    "DataIngester",
    "DataValidator",
    "MarketDataAgent",
    "MarketDataConfig", 
    "FeatureAgent",
    "FeatureConfig"
]
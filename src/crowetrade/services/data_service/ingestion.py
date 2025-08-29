"""Data Ingestion Module"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float | None = None
    ask: float | None = None
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "metadata": self.metadata
        }


class DataIngestion:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.sources = {}
        
    def register_source(self, name: str, adapter: Any) -> None:
        self.sources[name] = adapter
        
    def ingest(self, source: str, params: dict[str, Any]) -> MarketData:
        if source not in self.sources:
            raise ValueError(f"Unknown data source: {source}")
        
        adapter = self.sources[source]
        raw_data = adapter.fetch(params)
        return self._normalize(raw_data, source)
    
    def _normalize(self, raw_data: dict[str, Any], source: str) -> MarketData:
        return MarketData(
            symbol=raw_data.get("symbol"),
            timestamp=datetime.fromisoformat(raw_data.get("timestamp")),
            price=float(raw_data.get("price")),
            volume=float(raw_data.get("volume")),
            bid=raw_data.get("bid"),
            ask=raw_data.get("ask"),
            metadata={"source": source}
        )
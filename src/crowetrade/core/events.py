from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import AgentId, OrderId, RequestId, Symbol, Venue


@dataclass
class Event:
    event_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: AgentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass  
class MarketData:
    event_id: str
    symbol: Symbol
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    venue: Venue
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: AgentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBook:
    event_id: str
    symbol: Symbol
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    venue: Venue
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: AgentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskState:
    event_id: str
    pnl: float
    exposure: float
    var_estimate: float
    max_drawdown: float
    positions: dict[Symbol, float]
    risk_budget_remaining: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: AgentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeState:
    event_id: str
    regime: str
    confidence: float
    turbulence_index: float
    vol_regime: str
    risk_multiplier: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: AgentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildOrder:
    event_id: str
    instrument: Symbol
    side: str
    qty: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: AgentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tif: str = "DAY"
    limit_px: float | None = None
    route: Venue | None = None
    parent_id: str | None = None


@dataclass
class ExecutionReport:
    event_id: str
    order_id: OrderId
    symbol: Symbol
    side: str
    qty: float
    price: float
    status: str
    filled_qty: float
    avg_price: float
    venue: Venue
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: AgentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: RequestId | None = None
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    PEG = "peg"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"  # good till cancelled
    IOC = "ioc"  # immediate or cancel
    FOK = "fok"  # fill or kill
    GTD = "gtd"  # good till date
    ATO = "ato"  # at the open
    ATC = "atc"  # at the close


class ExecutionAlgo(Enum):
    VWAP = "vwap"
    TWAP = "twap"
    POV = "pov"  # percentage of volume
    IS = "is"  # implementation shortfall
    LIMIT_LADDER = "limit_ladder"
    SMART = "smart"
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"


@dataclass
class Order:
    order_id: str
    instrument: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    venue: str | None = None
    algo: ExecutionAlgo | None = None
    algo_params: dict[str, Any] = field(default_factory=dict)
    
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = None
    acknowledged_at: datetime | None = None
    completed_at: datetime | None = None
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                              OrderStatus.REJECTED, OrderStatus.EXPIRED)
    
    def update_fill(self, fill_qty: float, fill_price: float) -> None:
        prev_value = self.filled_quantity * self.average_price
        self.filled_quantity += fill_qty
        new_value = prev_value + (fill_qty * fill_price)
        
        if self.filled_quantity > 0:
            self.average_price = new_value / self.filled_quantity
        
        if abs(self.filled_quantity - self.quantity) < 1e-9:
            self.status = OrderStatus.FILLED
            self.completed_at = datetime.utcnow()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass
class ExecutionReport:
    order_id: str
    exec_id: str
    instrument: str
    side: OrderSide
    quantity: float
    price: float
    venue: str
    timestamp: datetime
    fees: float = 0.0
    
    slippage: float | None = None
    market_impact: float | None = None
    latency_us: int | None = None
    
    metadata: dict[str, Any] = field(default_factory=dict)


class OrderRouter(ABC):
    @abstractmethod
    async def route(self, order: Order) -> str:
        pass
    
    @abstractmethod
    async def cancel(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    async def modify(self, order_id: str, new_qty: float | None = None,
                    new_price: float | None = None) -> bool:
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        pass


class SmartOrderRouter:
    def __init__(self, venue_scorecards: dict[str, float]):
        self.venue_scorecards = venue_scorecards
        self.routers: dict[str, OrderRouter] = {}
    
    async def route(self, order: Order) -> list[str]:
        if order.algo == ExecutionAlgo.SMART:
            return await self._smart_route(order)
        elif order.venue:
            router = self.routers.get(order.venue)
            if router:
                order_id = await router.route(order)
                return [order_id]
        
        best_venue = self._select_best_venue(order)
        router = self.routers.get(best_venue)
        if router:
            order.venue = best_venue
            order_id = await router.route(order)
            return [order_id]
        
        return []
    
    async def _smart_route(self, order: Order) -> list[str]:
        child_orders = self._split_order(order)
        order_ids = []
        
        for child in child_orders:
            venue = self._select_best_venue(child)
            router = self.routers.get(venue)
            if router:
                child.venue = venue
                order_id = await router.route(child)
                order_ids.append(order_id)
        
        return order_ids
    
    def _split_order(self, order: Order) -> list[Order]:
        participation_rate = order.algo_params.get("participation_rate", 0.1)
        min_size = order.algo_params.get("min_child_size", 100)
        
        total_qty = order.quantity
        child_orders = []
        remaining = total_qty
        
        for venue, score in sorted(self.venue_scorecards.items(),
                                  key=lambda x: x[1], reverse=True):
            if remaining <= 0:
                break
            
            child_qty = min(remaining, max(min_size, total_qty * participation_rate))
            
            child = Order(
                order_id=f"{order.order_id}_{venue}",
                instrument=order.instrument,
                side=order.side,
                quantity=child_qty,
                order_type=order.order_type,
                price=order.price,
                time_in_force=order.time_in_force,
                venue=venue,
                metadata={"parent_id": order.order_id}
            )
            
            child_orders.append(child)
            remaining -= child_qty
        
        return child_orders
    
    def _select_best_venue(self, order: Order) -> str:
        return max(self.venue_scorecards.items(), key=lambda x: x[1])[0]


class TCAEngine(Protocol):
    def calculate_slippage(self, order: Order, fills: list[ExecutionReport]) -> float:
        ...
    
    def calculate_market_impact(self, order: Order, fills: list[ExecutionReport]) -> float:
        ...
    
    def generate_report(self, orders: list[Order], fills: list[ExecutionReport]) -> dict[str, Any]:
        ...
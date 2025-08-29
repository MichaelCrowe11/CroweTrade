from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..core.types import (
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    PolicyId,
    RequestId,
    Symbol,
    TimeInForce,
    Venue,
)


@dataclass
class Order:
    order_id: OrderId
    symbol: Symbol
    side: OrderSide
    qty: float
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    venue: Venue | None = None
    request_id: RequestId | None = None
    policy_id: PolicyId | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    @property
    def remaining_qty(self) -> float:
        return self.qty - self.filled_qty
    
    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
    
    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
    
    def update_fill(self, fill_qty: float, fill_price: float, fees: float = 0.0):
        if self.filled_qty == 0:
            self.avg_fill_price = fill_price
        else:
            total_value = self.filled_qty * self.avg_fill_price + fill_qty * fill_price
            self.filled_qty += fill_qty
            self.avg_fill_price = total_value / self.filled_qty
        
        self.fees += fees
        self.updated_at = datetime.utcnow()
        
        if self.filled_qty >= self.qty:
            self.status = OrderStatus.FILLED
        elif self.filled_qty > 0:
            self.status = OrderStatus.PARTIAL
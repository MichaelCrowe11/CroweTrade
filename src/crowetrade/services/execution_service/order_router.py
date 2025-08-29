"""Order Router Module - Idempotent Order Management"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: OrderType
    price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "DAY"
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "metadata": self.metadata
        }
    
    def get_idempotency_key(self) -> str:
        key_data = f"{self.client_order_id}_{self.symbol}_{self.side}_{self.quantity}"
        return hashlib.sha256(key_data.encode()).hexdigest()


@dataclass
class OrderResult:
    client_order_id: str
    exchange_order_id: str | None
    status: OrderStatus
    filled_quantity: float
    average_price: float | None
    timestamp: datetime
    venue: str
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "timestamp": self.timestamp.isoformat(),
            "venue": self.venue,
            "metadata": self.metadata
        }


class OrderRouter:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.venues = {}
        self.order_cache = {}
        self.routing_rules = []
        
    def register_venue(self, name: str, adapter: Any) -> None:
        self.venues[name] = adapter
        
    def add_routing_rule(self, rule: Any) -> None:
        self.routing_rules.append(rule)
        
    def route_order(self, order: Order) -> OrderResult:
        idempotency_key = order.get_idempotency_key()
        
        if idempotency_key in self.order_cache:
            cached_result = self.order_cache[idempotency_key]
            if self._is_cache_valid(cached_result):
                return cached_result
        
        venue = self._select_venue(order)
        
        if not venue:
            return self._create_rejection_result(order, "No suitable venue found")
        
        venue_adapter = self.venues[venue]
        
        try:
            result = venue_adapter.submit_order(order)
            
            order_result = OrderResult(
                client_order_id=order.client_order_id,
                exchange_order_id=result.get("exchange_order_id"),
                status=OrderStatus(result.get("status", "pending")),
                filled_quantity=result.get("filled_quantity", 0.0),
                average_price=result.get("average_price"),
                timestamp=datetime.utcnow(),
                venue=venue,
                metadata={"idempotency_key": idempotency_key}
            )
            
            self.order_cache[idempotency_key] = order_result
            
            return order_result
            
        except Exception as e:
            return self._create_rejection_result(order, str(e))
    
    def cancel_order(self, client_order_id: str) -> OrderResult:
        for venue_name, venue_adapter in self.venues.items():
            try:
                result = venue_adapter.cancel_order(client_order_id)
                if result.get("success"):
                    return OrderResult(
                        client_order_id=client_order_id,
                        exchange_order_id=result.get("exchange_order_id"),
                        status=OrderStatus.CANCELLED,
                        filled_quantity=result.get("filled_quantity", 0.0),
                        average_price=result.get("average_price"),
                        timestamp=datetime.utcnow(),
                        venue=venue_name
                    )
            except:
                continue
        
        return self._create_rejection_result(
            Order(client_order_id=client_order_id, symbol="", side="", 
                 quantity=0, order_type=OrderType.MARKET),
            "Order not found"
        )
    
    def get_order_status(self, client_order_id: str) -> OrderResult | None:
        for idempotency_key, cached_result in self.order_cache.items():
            if cached_result.client_order_id == client_order_id:
                return cached_result
        
        for venue_name, venue_adapter in self.venues.items():
            try:
                status = venue_adapter.get_order_status(client_order_id)
                if status:
                    return OrderResult(
                        client_order_id=client_order_id,
                        exchange_order_id=status.get("exchange_order_id"),
                        status=OrderStatus(status.get("status", "unknown")),
                        filled_quantity=status.get("filled_quantity", 0.0),
                        average_price=status.get("average_price"),
                        timestamp=datetime.utcnow(),
                        venue=venue_name
                    )
            except:
                continue
        
        return None
    
    def _select_venue(self, order: Order) -> str | None:
        for rule in self.routing_rules:
            venue = rule.evaluate(order, self.venues)
            if venue:
                return venue
        
        for venue_name in self.venues:
            return venue_name
        
        return None
    
    def _is_cache_valid(self, cached_result: OrderResult) -> bool:
        age_seconds = (datetime.utcnow() - cached_result.timestamp).total_seconds()
        max_cache_age = self.config.get("max_cache_age_seconds", 60)
        
        if age_seconds > max_cache_age:
            return False
        
        if cached_result.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return True
        
        return False
    
    def _create_rejection_result(self, order: Order, reason: str) -> OrderResult:
        return OrderResult(
            client_order_id=order.client_order_id,
            exchange_order_id=None,
            status=OrderStatus.REJECTED,
            filled_quantity=0.0,
            average_price=None,
            timestamp=datetime.utcnow(),
            venue="NONE",
            metadata={"rejection_reason": reason}
        )
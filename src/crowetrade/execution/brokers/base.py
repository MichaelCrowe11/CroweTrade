"""Base broker adapter interface and common types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from crowetrade.core.types import OrderId


class OrderStatus(Enum):
    """Order status enumeration."""
    
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration."""
    
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force enumeration."""
    
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good till date


@dataclass
class Order:
    """Order representation."""
    
    order_id: OrderId
    symbol: str
    quantity: float
    order_type: OrderType
    side: str  # "buy" or "sell"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Position representation."""
    
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float


@dataclass
class AccountInfo:
    """Account information."""
    
    account_id: str
    buying_power: float
    cash_balance: float
    portfolio_value: float
    positions: List[Position]
    daily_pnl: float
    total_pnl: float


class BrokerAdapter(ABC):
    """Abstract base class for broker adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize broker adapter with configuration."""
        self.config = config
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to broker API."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker API."""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> OrderId:
        """Submit an order to the broker."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: OrderId) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: OrderId) -> Order:
        """Get current status of an order."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, float]:
        """Get current market data for a symbol."""
        pass
    
    async def modify_order(
        self,
        order_id: OrderId,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> bool:
        """Modify an existing order (optional implementation)."""
        # Default: cancel and replace
        await self.cancel_order(order_id)
        original = await self.get_order_status(order_id)
        
        new_order = Order(
            order_id=OrderId(f"{order_id}_mod"),
            symbol=original.symbol,
            quantity=quantity or original.quantity,
            order_type=original.order_type,
            side=original.side,
            limit_price=limit_price or original.limit_price,
            stop_price=stop_price or original.stop_price,
            time_in_force=original.time_in_force,
        )
        
        await self.submit_order(new_order)
        return True
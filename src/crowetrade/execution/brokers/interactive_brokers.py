"""Interactive Brokers adapter implementation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from crowetrade.core.types import OrderId
from crowetrade.execution.brokers.base import (
    AccountInfo,
    BrokerAdapter,
    Order,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

logger = logging.getLogger(__name__)


class IBOrderManager:
    """Manages order tracking and state for IB."""
    
    def __init__(self):
        self.orders: Dict[OrderId, Order] = {}
        self.ib_to_internal: Dict[int, OrderId] = {}
        self.internal_to_ib: Dict[OrderId, int] = {}
        self.next_order_id = 1000
    
    def register_order(self, order: Order) -> int:
        """Register order and return IB order ID."""
        ib_id = self.next_order_id
        self.next_order_id += 1
        
        self.orders[order.order_id] = order
        self.ib_to_internal[ib_id] = order.order_id
        self.internal_to_ib[order.order_id] = ib_id
        
        return ib_id
    
    def get_order(self, order_id: OrderId) -> Optional[Order]:
        """Get order by internal ID."""
        return self.orders.get(order_id)
    
    def get_order_by_ib_id(self, ib_id: int) -> Optional[Order]:
        """Get order by IB ID."""
        internal_id = self.ib_to_internal.get(ib_id)
        if internal_id:
            return self.orders.get(internal_id)
        return None


class InteractiveBrokersAdapter(BrokerAdapter):
    """Interactive Brokers adapter for order execution.
    
    This implementation provides a simulation layer for development.
    In production, this would integrate with IB's TWS API via ib_insync or ibapi.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize IB adapter."""
        super().__init__(config)
        self.order_manager = IBOrderManager()
        self.positions: Dict[str, Position] = {}
        self.account_balance = config.get("initial_balance", 100000.0)
        self.buying_power = self.account_balance
        
        # Simulated market data
        self.market_data: Dict[str, Dict[str, float]] = {}
        self._fill_simulator_task = None
        
        # IB-specific configuration
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 7497)  # Paper trading port
        self.client_id = config.get("client_id", 1)
        self.account_id = config.get("account_id")
    
    async def connect(self) -> None:
        """Connect to IB TWS/Gateway."""
        logger.info(f"Connecting to IB at {self.host}:{self.port}")
        
        # In production, this would establish connection to TWS/Gateway
        # For now, simulate connection
        await asyncio.sleep(0.1)
        
        self.is_connected = True
        
        # Start order fill simulator for development
        self._fill_simulator_task = asyncio.create_task(self._simulate_fills())
        
        logger.info("Connected to Interactive Brokers")
    
    async def disconnect(self) -> None:
        """Disconnect from IB."""
        logger.info("Disconnecting from Interactive Brokers")
        
        if self._fill_simulator_task:
            self._fill_simulator_task.cancel()
            try:
                await self._fill_simulator_task
            except asyncio.CancelledError:
                pass
        
        self.is_connected = False
        logger.info("Disconnected from Interactive Brokers")
    
    async def submit_order(self, order: Order) -> OrderId:
        """Submit order to IB."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Interactive Brokers")
        
        # Validate order
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            raise ValueError("Limit order requires limit_price")
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            raise ValueError("Stop order requires stop_price")
        
        # Register order
        ib_id = self.order_manager.register_order(order)
        
        # Update order status
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.utcnow()
        
        logger.info(
            f"Submitted order {order.order_id} (IB ID: {ib_id}) - "
            f"{order.side} {order.quantity} {order.symbol} @ "
            f"{order.order_type.value}"
        )
        
        # In production, would submit to IB API here
        # For now, order will be processed by fill simulator
        
        return order.order_id
    
    async def cancel_order(self, order_id: OrderId) -> bool:
        """Cancel order."""
        order = self.order_manager.get_order(order_id)
        if not order:
            logger.warning(f"Order {order_id} not found")
            return False
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Cancelled order {order_id}")
        
        return True
    
    async def get_order_status(self, order_id: OrderId) -> Order:
        """Get order status."""
        order = self.order_manager.get_order(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        return order
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self.positions.values())
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        positions = list(self.positions.values())
        portfolio_value = self.account_balance
        
        for pos in positions:
            portfolio_value += pos.unrealized_pnl
        
        return AccountInfo(
            account_id=self.account_id or "IB_SIMULATION",
            buying_power=self.buying_power,
            cash_balance=self.account_balance,
            portfolio_value=portfolio_value,
            positions=positions,
            daily_pnl=sum(p.unrealized_pnl for p in positions),
            total_pnl=sum(p.realized_pnl + p.unrealized_pnl for p in positions),
        )
    
    async def get_market_data(self, symbol: str) -> Dict[str, float]:
        """Get market data for symbol."""
        # In production, would fetch from IB market data
        # For simulation, return mock data
        if symbol not in self.market_data:
            self.market_data[symbol] = {
                "bid": 100.0,
                "ask": 100.1,
                "last": 100.05,
                "volume": 1000000,
                "open": 99.5,
                "high": 101.0,
                "low": 99.0,
                "close": 100.05,
            }
        
        return self.market_data[symbol]
    
    async def _simulate_fills(self) -> None:
        """Simulate order fills for development."""
        while True:
            try:
                await asyncio.sleep(1)  # Check orders every second
                
                for order in self.order_manager.orders.values():
                    if order.status != OrderStatus.SUBMITTED:
                        continue
                    
                    # Get market data
                    market = await self.get_market_data(order.symbol)
                    
                    # Determine if order should fill
                    should_fill = False
                    fill_price = 0.0
                    
                    if order.order_type == OrderType.MARKET:
                        should_fill = True
                        fill_price = market["ask"] if order.side == "buy" else market["bid"]
                    
                    elif order.order_type == OrderType.LIMIT:
                        if order.side == "buy" and order.limit_price >= market["ask"]:
                            should_fill = True
                            fill_price = order.limit_price
                        elif order.side == "sell" and order.limit_price <= market["bid"]:
                            should_fill = True
                            fill_price = order.limit_price
                    
                    if should_fill:
                        await self._execute_fill(order, fill_price)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in fill simulator: {e}")
    
    async def _execute_fill(self, order: Order, fill_price: float) -> None:
        """Execute order fill."""
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = order.quantity * 0.001  # $0.001 per share
        
        # Update position
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_cost=0.0,
                current_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                market_value=0.0,
            )
        
        pos = self.positions[symbol]
        
        if order.side == "buy":
            # Update position for buy
            total_cost = pos.quantity * pos.average_cost + order.quantity * fill_price
            pos.quantity += order.quantity
            pos.average_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
            
            # Update account
            self.account_balance -= order.quantity * fill_price + order.commission
            
        else:  # sell
            # Calculate realized PnL
            if pos.quantity > 0:
                realized = order.quantity * (fill_price - pos.average_cost)
                pos.realized_pnl += realized
            
            pos.quantity -= order.quantity
            
            # Update account
            self.account_balance += order.quantity * fill_price - order.commission
        
        # Update position market value and unrealized PnL
        pos.current_price = fill_price
        pos.market_value = pos.quantity * pos.current_price
        pos.unrealized_pnl = pos.quantity * (pos.current_price - pos.average_cost)
        
        # Remove position if fully closed
        if abs(pos.quantity) < 0.0001:
            del self.positions[symbol]
        
        # Update buying power (simplified)
        self.buying_power = self.account_balance
        
        logger.info(
            f"Filled order {order.order_id} - "
            f"{order.side} {order.filled_quantity} {order.symbol} @ {fill_price}"
        )
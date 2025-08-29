"""Adapter implementations for execution routing."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from crowetrade.core.types import OrderId
from crowetrade.execution.brokers.base import BrokerAdapter, Order, OrderType

logger = logging.getLogger(__name__)


class BrokerExecutionAdapter:
    """Adapts broker interface for execution router.
    
    Converts target positions to orders and manages execution.
    """
    
    def __init__(
        self,
        broker: BrokerAdapter,
        order_type: OrderType = OrderType.LIMIT,
        limit_offset_bps: float = 5.0,  # basis points from mid
    ):
        """Initialize adapter.
        
        Args:
            broker: Underlying broker adapter
            order_type: Default order type to use
            limit_offset_bps: Offset in basis points for limit orders
        """
        self.broker = broker
        self.order_type = order_type
        self.limit_offset_bps = limit_offset_bps
        self.current_positions: Dict[str, float] = {}
        self.pending_orders: Dict[str, OrderId] = {}
    
    async def submit_targets(
        self,
        targets: Dict[str, float],
        prices: Dict[str, float],
    ) -> None:
        """Submit target positions as orders.
        
        Args:
            targets: Target positions by symbol
            prices: Current prices by symbol
        """
        # Update current positions from broker
        positions = await self.broker.get_positions()
        self.current_positions = {
            pos.symbol: pos.quantity for pos in positions
        }
        
        # Calculate required trades
        trades = self._calculate_trades(targets)
        
        # Submit orders for each trade
        for symbol, quantity in trades.items():
            if abs(quantity) < 0.001:  # Skip tiny trades
                continue
            
            # Cancel existing pending order if any
            if symbol in self.pending_orders:
                await self.broker.cancel_order(self.pending_orders[symbol])
                del self.pending_orders[symbol]
            
            # Determine order side
            side = "buy" if quantity > 0 else "sell"
            order_quantity = abs(quantity)
            
            # Calculate limit price
            limit_price = None
            if self.order_type == OrderType.LIMIT and symbol in prices:
                price = prices[symbol]
                offset = price * self.limit_offset_bps / 10000
                if side == "buy":
                    limit_price = price + offset
                else:
                    limit_price = price - offset
            
            # Create and submit order
            order = Order(
                order_id=OrderId(f"EXR_{symbol}_{id(self)}"),
                symbol=symbol,
                quantity=order_quantity,
                order_type=self.order_type,
                side=side,
                limit_price=limit_price,
            )
            
            try:
                order_id = await self.broker.submit_order(order)
                self.pending_orders[symbol] = order_id
                logger.info(
                    f"Submitted {side} order for {order_quantity} {symbol} "
                    f"(target: {targets.get(symbol, 0)}, current: {self.current_positions.get(symbol, 0)})"
                )
            except Exception as e:
                logger.error(f"Failed to submit order for {symbol}: {e}")
    
    def _calculate_trades(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Calculate required trades to reach targets.
        
        Args:
            targets: Target positions
            
        Returns:
            Required trades (positive = buy, negative = sell)
        """
        trades = {}
        
        # Calculate trades for target positions
        for symbol, target in targets.items():
            current = self.current_positions.get(symbol, 0.0)
            trade = target - current
            if abs(trade) > 0.001:  # Only include non-zero trades
                trades[symbol] = trade
        
        # Close positions not in targets
        for symbol, current in self.current_positions.items():
            if symbol not in targets and abs(current) > 0.001:
                trades[symbol] = -current  # Close position
        
        return trades
    
    async def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        orders = []
        for order_id in self.pending_orders.values():
            try:
                order = await self.broker.get_order_status(order_id)
                orders.append(order)
            except Exception as e:
                logger.error(f"Failed to get order status for {order_id}: {e}")
        return orders
    
    async def cancel_all_orders(self) -> None:
        """Cancel all pending orders."""
        for symbol, order_id in list(self.pending_orders.items()):
            try:
                await self.broker.cancel_order(order_id)
                del self.pending_orders[symbol]
                logger.info(f"Cancelled order {order_id} for {symbol}")
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
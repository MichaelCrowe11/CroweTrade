from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.agent import AgentConfig, BaseAgent
from ..core.contracts import TargetPosition
from ..core.events import ExecutionReport, MarketData
from ..core.types import OrderId, OrderSide, OrderStatus, OrderType, Symbol, TimeInForce, Venue
from .order import Order


@dataclass
class ExecutionConfig(AgentConfig):
    venues: list[Venue] = field(default_factory=list)
    default_algo: str = "VWAP"
    max_participation: float = 0.1
    min_order_size: float = 100.0
    max_order_size: float = 10000.0
    slippage_bps: float = 5.0


class ExecutionRouter(BaseAgent):
    def __init__(self, config: ExecutionConfig):
        super().__init__(config)
        self.config: ExecutionConfig = config
        self._orders: dict[OrderId, Order] = {}
        self._positions: dict[Symbol, float] = {}
        self._market_prices: dict[Symbol, tuple[float, float]] = {}
        self._running = False
    
    async def on_start(self):
        self._running = True
        self.subscribe(MarketData, self._on_market_data)
        self.subscribe(TargetPosition, self._on_target_position)
        asyncio.create_task(self._monitor_orders())
    
    async def on_stop(self):
        self._running = False
        await self._cancel_all_orders()
    
    async def _on_market_data(self, event: MarketData):
        self._market_prices[event.symbol] = (event.bid, event.ask)
    
    async def _on_target_position(self, event: TargetPosition):
        current_pos = self._positions.get(event.instrument, 0.0)
        target_qty = event.qty_target - current_pos
        
        if abs(target_qty) < self.config.min_order_size:
            return
        
        order = await self._create_order(
            symbol=event.instrument,
            qty=target_qty,
            policy_id=event.policy_id,
            max_participation=event.max_child_participation
        )
        
        if order:
            await self._route_order(order)
    
    async def _create_order(
        self,
        symbol: Symbol,
        qty: float,
        policy_id: str,
        max_participation: float
    ) -> Order | None:
        if symbol not in self._market_prices:
            return None
        
        bid, ask = self._market_prices[symbol]
        
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        order_qty = min(abs(qty), self.config.max_order_size)
        
        order = Order(
            order_id=OrderId(str(uuid.uuid4())),
            symbol=symbol,
            side=side,
            qty=order_qty,
            order_type=OrderType.LIMIT,
            price=bid if side == OrderSide.BUY else ask,
            time_in_force=TimeInForce.DAY,
            policy_id=policy_id,
            metadata={"max_participation": max_participation}
        )
        
        self._orders[order.order_id] = order
        return order
    
    async def _route_order(self, order: Order):
        best_venue = await self._select_venue(order)
        order.venue = best_venue
        order.status = OrderStatus.SUBMITTED
        
        await self._simulate_execution(order)
    
    async def _select_venue(self, order: Order) -> Venue:
        if not self.config.venues:
            return "DEFAULT"
        
        scores = {}
        for venue in self.config.venues:
            liquidity_score = np.random.uniform(0.5, 1.0)
            fee_score = np.random.uniform(0.7, 1.0)
            latency_score = np.random.uniform(0.6, 1.0)
            
            scores[venue] = 0.4 * liquidity_score + 0.3 * fee_score + 0.3 * latency_score
        
        return max(scores, key=scores.get)
    
    async def _simulate_execution(self, order: Order):
        await asyncio.sleep(np.random.uniform(0.1, 0.5))
        
        if np.random.random() < 0.95:
            bid, ask = self._market_prices.get(order.symbol, (100.0, 100.1))
            
            if order.side == OrderSide.BUY:
                fill_price = ask * (1 + self.config.slippage_bps / 10000)
            else:
                fill_price = bid * (1 - self.config.slippage_bps / 10000)
            
            fill_qty = order.qty * np.random.uniform(0.8, 1.0)
            fees = fill_qty * fill_price * 0.0002
            
            order.update_fill(fill_qty, fill_price, fees)
            
            if order.side == OrderSide.BUY:
                self._positions[order.symbol] = self._positions.get(order.symbol, 0) + fill_qty
            else:
                self._positions[order.symbol] = self._positions.get(order.symbol, 0) - fill_qty
            
            report = ExecutionReport(
                event_id=f"exec_{order.order_id}_{datetime.utcnow().isoformat()}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                qty=order.qty,
                price=order.price or 0,
                status=order.status.value,
                filled_qty=order.filled_qty,
                avg_price=order.avg_fill_price,
                venue=order.venue or "DEFAULT",
                request_id=order.request_id
            )
            
            await self.emit(report)
        else:
            order.status = OrderStatus.REJECTED
    
    async def _monitor_orders(self):
        while self._running:
            active_orders = [
                o for o in self._orders.values()
                if o.is_active
            ]
            
            for order in active_orders:
                if order.status == OrderStatus.SUBMITTED:
                    await self._check_order_status(order)
            
            await asyncio.sleep(1.0)
    
    async def _check_order_status(self, order: Order):
        pass
    
    async def _cancel_all_orders(self):
        for order in self._orders.values():
            if order.is_active:
                order.status = OrderStatus.CANCELLED
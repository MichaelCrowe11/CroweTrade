"""Venue Adapter - FIX/Binary Protocol Interface"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import os
from typing import Protocol

from crowetrade.core.contracts import Fill
from crowetrade.core.events import ChildOrder


@dataclass
class VenueConfig:
    """Configuration for venue connectivity"""
    name: str
    host: str
    port: int
    username: str
    password: str
    target_comp_id: str
    sender_comp_id: str
    heartbeat_interval: int = 30
    timeout_ms: int = 5000


class VenueProtocol(Protocol):
    """Protocol for venue communication"""
    
    async def connect(self) -> bool: ...
    async def disconnect(self) -> None: ...
    async def send_order(self, order: ChildOrder) -> str: ...
    async def cancel_order(self, order_id: str) -> bool: ...


class VenueAdapter(ABC):
    """Base venue adapter for order routing"""
    
    def __init__(self, config: VenueConfig):
        self.config = config
        self.connected = False
        self.order_state: dict[str, str] = {}  # order_id -> status
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to venue"""
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to venue"""
    
    @abstractmethod
    async def send_order(self, order: ChildOrder) -> str:
        """Send order to venue, return order_id"""
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order at venue"""
    
    @abstractmethod
    async def handle_fills(self) -> list[Fill]:
        """Process incoming fills"""


class MockVenueAdapter(VenueAdapter):
    """Mock venue adapter for testing"""
    
    def __init__(self, config: VenueConfig, latency_ms: int = 10):
        super().__init__(config)
        self.latency_ms = latency_ms
        self.fill_queue: list[Fill] = []
        
    async def connect(self) -> bool:
        await asyncio.sleep(self.latency_ms / 1000)
        self.connected = True
        return True
        
    async def disconnect(self) -> None:
        self.connected = False
        
    async def send_order(self, order: ChildOrder) -> str:
        if not self.connected:
            raise RuntimeError("Not connected to venue")
            
        await asyncio.sleep(self.latency_ms / 1000)
        order_id = f"MOCK_{order.instrument}_{order.side}_{hash(order)}"
        self.order_state[order_id] = "NEW"
        
        # Simulate fill after short delay
        asyncio.create_task(self._simulate_fill(order, order_id))
        return order_id
        
    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.order_state:
            self.order_state[order_id] = "CANCELLED"
            return True
        return False
        
    async def handle_fills(self) -> list[Fill]:
        fills = self.fill_queue.copy()
        self.fill_queue.clear()
        return fills
        
    async def _simulate_fill(self, order: ChildOrder, order_id: str) -> None:
        """Simulate a fill with realistic delay"""
        await asyncio.sleep((self.latency_ms + 50) / 1000)
        
        if self.order_state.get(order_id) == "NEW":
            fill = Fill(
                instrument=order.instrument,
                qty=order.qty,
                price=order.limit_px or 100.0,  # Use limit or mock price
                ts=datetime.utcnow(),
                venue=self.config.name
            )
            self.fill_queue.append(fill)
            self.order_state[order_id] = "FILLED"


class FIXVenueAdapter(VenueAdapter):
    """FIX protocol venue adapter (stub for production)"""
    
    def __init__(self, config: VenueConfig):
        super().__init__(config)
        # TODO: Initialize FIX engine (QuickFIX, etc.)
        
    async def connect(self) -> bool:
        # TODO: Implement FIX connection
        raise NotImplementedError("FIX adapter not implemented")
        
    async def disconnect(self) -> None:
        # TODO: Implement FIX disconnection
        pass
        
    async def send_order(self, order: ChildOrder) -> str:
        # TODO: Convert to FIX NewOrderSingle and send
        raise NotImplementedError("FIX send_order not implemented")
        
    async def cancel_order(self, order_id: str) -> bool:
        # TODO: Send FIX OrderCancelRequest
        raise NotImplementedError("FIX cancel_order not implemented")
        
    async def handle_fills(self) -> list[Fill]:
        # TODO: Process FIX ExecutionReports
        raise NotImplementedError("FIX handle_fills not implemented")


def create_venue_adapter(config: VenueConfig) -> VenueAdapter:
    """Factory selecting adapter based on env.

    - RUN_MODE=production and PAPER_MODE=true -> NoTradeVenueAdapter (safe, no live fills)
    - RUN_MODE=production and PAPER_MODE!=true -> raise until real adapter exists
    - Else -> MockVenueAdapter for local/dev
    """
    run_mode = os.environ.get("RUN_MODE", "development").lower()
    paper_mode = os.environ.get("PAPER_MODE", "false").lower() in {"1", "true", "yes"}
    if run_mode == "production":
        if paper_mode:
            return NoTradeVenueAdapter(config)
        # Real live trading requires a real adapter implementation
        raise NotImplementedError(
            "Live trading adapter not implemented. Set PAPER_MODE=true or provide a real VenueAdapter."
        )
    return MockVenueAdapter(config)


class NoTradeVenueAdapter(VenueAdapter):
    """Production-safe adapter that acknowledges orders but never routes or fills."""

    def __init__(self, config: VenueConfig):
        super().__init__(config)

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.connected = False

    async def send_order(self, order: ChildOrder) -> str:
        if not self.connected:
            raise RuntimeError("Not connected to venue")
        order_id = f"NT_{order.instrument}_{order.side}_{hash(order)}"
        self.order_state[order_id] = "ACCEPTED"
        # No routing, no fills in paper/no-trade mode
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.order_state:
            self.order_state[order_id] = "CANCELLED"
            return True
        return False

    async def handle_fills(self) -> list[Fill]:
        # No fills are generated in no-trade mode
        return []

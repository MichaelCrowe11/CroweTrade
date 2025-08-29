"""Base market data provider interface."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from crowetrade.data.market_data import (
    Bar,
    DataType,
    MarketDataProcessor,
    OrderBook,
    Quote,
    Trade,
)

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider connection status."""
    
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ProviderConfig:
    """Market data provider configuration."""
    
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    heartbeat_interval: int = 30
    rate_limit: Optional[int] = None


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    def __init__(
        self,
        config: ProviderConfig,
        processor: MarketDataProcessor,
    ):
        """Initialize provider.
        
        Args:
            config: Provider configuration
            processor: Data processor for handling updates
        """
        self.config = config
        self.processor = processor
        self.status = ProviderStatus.DISCONNECTED
        
        # Subscriptions
        self.trade_subscriptions: Set[str] = set()
        self.quote_subscriptions: Set[str] = set()
        self.book_subscriptions: Set[str] = set()
        self.bar_subscriptions: Dict[str, Set[str]] = {}  # symbol -> timeframes
        
        # Connection management
        self.reconnect_attempts = 0
        self.last_heartbeat = datetime.utcnow()
        self._connection_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Error handling
        self.error_callbacks: List[Callable] = []
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to data provider."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data provider."""
        pass
    
    @abstractmethod
    async def subscribe_trades(self, symbols: List[str]) -> None:
        """Subscribe to trade data.
        
        Args:
            symbols: List of symbols to subscribe
        """
        pass
    
    @abstractmethod
    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """Subscribe to quote (L1) data.
        
        Args:
            symbols: List of symbols to subscribe
        """
        pass
    
    @abstractmethod
    async def subscribe_book(self, symbols: List[str], depth: int = 10) -> None:
        """Subscribe to order book (L2) data.
        
        Args:
            symbols: List of symbols to subscribe
            depth: Book depth to receive
        """
        pass
    
    @abstractmethod
    async def subscribe_bars(
        self,
        symbols: List[str],
        timeframe: str = "1m",
    ) -> None:
        """Subscribe to bar data.
        
        Args:
            symbols: List of symbols to subscribe
            timeframe: Bar timeframe
        """
        pass
    
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all data."""
        self.trade_subscriptions.clear()
        self.quote_subscriptions.clear()
        self.book_subscriptions.clear()
        self.bar_subscriptions.clear()
    
    async def start(self) -> None:
        """Start provider and connect."""
        if self.status != ProviderStatus.DISCONNECTED:
            logger.warning(f"Provider {self.config.name} already started")
            return
        
        self.status = ProviderStatus.CONNECTING
        self._connection_task = asyncio.create_task(self._connection_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self) -> None:
        """Stop provider and disconnect."""
        logger.info(f"Stopping provider {self.config.name}")
        
        # Cancel tasks
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect
        await self.disconnect()
        self.status = ProviderStatus.DISCONNECTED
    
    async def _connection_loop(self) -> None:
        """Manage connection with automatic reconnection."""
        while True:
            try:
                if self.status in [ProviderStatus.CONNECTING, ProviderStatus.RECONNECTING]:
                    await self.connect()
                    self.status = ProviderStatus.CONNECTED
                    self.reconnect_attempts = 0
                    logger.info(f"Connected to {self.config.name}")
                    
                    # Resubscribe to all data
                    await self._resubscribe_all()
                
                # Wait for disconnection
                while self.status == ProviderStatus.CONNECTED:
                    await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Connection error in {self.config.name}: {e}")
                self.status = ProviderStatus.ERROR
                
                # Handle reconnection
                if self.reconnect_attempts < self.config.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    delay = self.config.reconnect_delay * (2 ** self.reconnect_attempts)
                    logger.info(
                        f"Reconnecting {self.config.name} in {delay}s "
                        f"(attempt {self.reconnect_attempts})"
                    )
                    self.status = ProviderStatus.RECONNECTING
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Max reconnection attempts reached for {self.config.name}"
                    )
                    break
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.status == ProviderStatus.CONNECTED:
                    await self._send_heartbeat()
                    
                    # Check for stale connection
                    time_since_heartbeat = (
                        datetime.utcnow() - self.last_heartbeat
                    ).total_seconds()
                    
                    if time_since_heartbeat > self.config.heartbeat_interval * 3:
                        logger.warning(
                            f"No heartbeat from {self.config.name} for {time_since_heartbeat}s"
                        )
                        self.status = ProviderStatus.RECONNECTING
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error in {self.config.name}: {e}")
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat to provider (override if needed)."""
        self.last_heartbeat = datetime.utcnow()
    
    async def _resubscribe_all(self) -> None:
        """Resubscribe to all data after reconnection."""
        if self.trade_subscriptions:
            await self.subscribe_trades(list(self.trade_subscriptions))
        
        if self.quote_subscriptions:
            await self.subscribe_quotes(list(self.quote_subscriptions))
        
        if self.book_subscriptions:
            await self.subscribe_book(list(self.book_subscriptions))
        
        for symbol, timeframes in self.bar_subscriptions.items():
            for timeframe in timeframes:
                await self.subscribe_bars([symbol], timeframe)
    
    async def handle_trade(self, trade: Trade) -> None:
        """Handle incoming trade data.
        
        Args:
            trade: Trade data
        """
        await self.processor.process_trade(trade)
    
    async def handle_quote(self, quote: Quote) -> None:
        """Handle incoming quote data.
        
        Args:
            quote: Quote data
        """
        await self.processor.process_quote(quote)
    
    async def handle_book(self, book: OrderBook) -> None:
        """Handle incoming order book data.
        
        Args:
            book: Order book data
        """
        await self.processor.process_book(book)
    
    async def handle_bar(self, bar: Bar) -> None:
        """Handle incoming bar data.
        
        Args:
            bar: Bar data
        """
        # Add to buffer
        self.processor.buffer.add_bar(bar)
        
        # Execute callbacks
        for callback in self.processor.callbacks.get(DataType.BAR, []):
            await callback(bar)
        
        # Publish event
        await self.processor.event_bus.publish("market.bar", bar)
    
    def add_error_callback(self, callback: Callable) -> None:
        """Add error callback.
        
        Args:
            callback: Error callback function
        """
        self.error_callbacks.append(callback)
    
    async def handle_error(self, error: Exception) -> None:
        """Handle provider error.
        
        Args:
            error: Exception that occurred
        """
        logger.error(f"Error in {self.config.name}: {error}")
        
        for callback in self.error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get provider status.
        
        Returns:
            Status dictionary
        """
        return {
            "name": self.config.name,
            "status": self.status.value,
            "reconnect_attempts": self.reconnect_attempts,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "subscriptions": {
                "trades": len(self.trade_subscriptions),
                "quotes": len(self.quote_subscriptions),
                "books": len(self.book_subscriptions),
                "bars": sum(len(tf) for tf in self.bar_subscriptions.values()),
            },
        }
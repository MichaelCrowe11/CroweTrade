"""Market data ingestion pipeline with L1/L2 support."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set

import numpy as np

from crowetrade.core.bus import EventBus
from crowetrade.core.types import Timestamp

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Market data types."""
    
    TRADE = "trade"
    QUOTE = "quote"
    BOOK = "book"
    BAR = "bar"
    IMBALANCE = "imbalance"
    HALT = "halt"


@dataclass
class Trade:
    """Trade tick data."""
    
    symbol: str
    price: float
    size: float
    timestamp: datetime
    exchange: str = ""
    conditions: List[str] = field(default_factory=list)
    trade_id: Optional[str] = None


@dataclass
class Quote:
    """Level 1 quote data."""
    
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    exchange: str = ""
    
    @property
    def mid(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Calculate spread."""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if self.mid > 0:
            return (self.spread / self.mid) * 10000
        return 0.0


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    
    price: float
    size: float
    orders: int = 1
    exchange: str = ""


@dataclass
class OrderBook:
    """Level 2 order book data."""
    
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime
    sequence: Optional[int] = None
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Get best bid level."""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Get best ask level."""
        return self.asks[0] if self.asks else None
    
    def get_mid_price(self) -> float:
        """Calculate mid price from best levels."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return 0.0
    
    def get_spread(self) -> float:
        """Calculate spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return 0.0
    
    def get_imbalance(self, levels: int = 5) -> float:
        """Calculate order imbalance.
        
        Args:
            levels: Number of levels to consider
            
        Returns:
            Imbalance ratio (-1 to 1, negative = bid pressure)
        """
        bid_volume = sum(
            level.size for level in self.bids[:levels]
        )
        ask_volume = sum(
            level.size for level in self.asks[:levels]
        )
        
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0.0


@dataclass
class Bar:
    """OHLCV bar data."""
    
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    vwap: Optional[float] = None
    trades: int = 0
    
    @property
    def range(self) -> float:
        """Calculate bar range."""
        return self.high - self.low
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price."""
        return (self.high + self.low + self.close) / 3


class MarketDataBuffer:
    """Circular buffer for market data."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize buffer.
        
        Args:
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.trades: Deque[Trade] = deque(maxlen=max_size)
        self.quotes: Deque[Quote] = deque(maxlen=max_size)
        self.books: Deque[OrderBook] = deque(maxlen=max_size)
        self.bars: Dict[str, Deque[Bar]] = {}  # By timeframe
    
    def add_trade(self, trade: Trade) -> None:
        """Add trade to buffer."""
        self.trades.append(trade)
    
    def add_quote(self, quote: Quote) -> None:
        """Add quote to buffer."""
        self.quotes.append(quote)
    
    def add_book(self, book: OrderBook) -> None:
        """Add order book to buffer."""
        self.books.append(book)
    
    def add_bar(self, bar: Bar, timeframe: str = "1m") -> None:
        """Add bar to buffer."""
        if timeframe not in self.bars:
            self.bars[timeframe] = deque(maxlen=self.max_size)
        self.bars[timeframe].append(bar)
    
    def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Trade]:
        """Get recent trades for symbol."""
        return [
            t for t in list(self.trades)[-limit:]
            if t.symbol == symbol
        ]
    
    def get_recent_quotes(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Quote]:
        """Get recent quotes for symbol."""
        return [
            q for q in list(self.quotes)[-limit:]
            if q.symbol == symbol
        ]
    
    def calculate_vwap(
        self,
        symbol: str,
        window: int = 100,
    ) -> float:
        """Calculate VWAP from recent trades.
        
        Args:
            symbol: Symbol to calculate for
            window: Number of trades to consider
            
        Returns:
            VWAP value
        """
        trades = self.get_recent_trades(symbol, window)
        if not trades:
            return 0.0
        
        total_value = sum(t.price * t.size for t in trades)
        total_volume = sum(t.size for t in trades)
        
        if total_volume > 0:
            return total_value / total_volume
        return 0.0


class MarketDataProcessor:
    """Processes raw market data into features."""
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize processor.
        
        Args:
            buffer_size: Size of data buffers
        """
        self.buffer = MarketDataBuffer(buffer_size)
        self.event_bus = EventBus()
        
        # Microstructure metrics
        self.tick_rule: Dict[str, int] = {}  # 1 for uptick, -1 for downtick
        self.signed_volume: Dict[str, float] = {}
        
        # Callbacks
        self.callbacks: Dict[DataType, List[Callable]] = {
            DataType.TRADE: [],
            DataType.QUOTE: [],
            DataType.BOOK: [],
            DataType.BAR: [],
        }
    
    async def process_trade(self, trade: Trade) -> None:
        """Process trade tick.
        
        Args:
            trade: Trade data
        """
        # Add to buffer
        self.buffer.add_trade(trade)
        
        # Update tick rule
        if trade.symbol in self.tick_rule:
            last_trades = self.buffer.get_recent_trades(trade.symbol, 2)
            if len(last_trades) >= 2:
                if last_trades[-1].price > last_trades[-2].price:
                    self.tick_rule[trade.symbol] = 1
                elif last_trades[-1].price < last_trades[-2].price:
                    self.tick_rule[trade.symbol] = -1
        else:
            self.tick_rule[trade.symbol] = 1
        
        # Update signed volume (for order flow)
        tick_sign = self.tick_rule.get(trade.symbol, 0)
        if trade.symbol not in self.signed_volume:
            self.signed_volume[trade.symbol] = 0
        self.signed_volume[trade.symbol] += tick_sign * trade.size
        
        # Execute callbacks
        for callback in self.callbacks[DataType.TRADE]:
            await callback(trade)
        
        # Publish event
        await self.event_bus.publish("market.trade", trade)
    
    async def process_quote(self, quote: Quote) -> None:
        """Process quote update.
        
        Args:
            quote: Quote data
        """
        # Add to buffer
        self.buffer.add_quote(quote)
        
        # Execute callbacks
        for callback in self.callbacks[DataType.QUOTE]:
            await callback(quote)
        
        # Publish event
        await self.event_bus.publish("market.quote", quote)
    
    async def process_book(self, book: OrderBook) -> None:
        """Process order book update.
        
        Args:
            book: Order book data
        """
        # Add to buffer
        self.buffer.add_book(book)
        
        # Calculate microstructure features
        features = self.calculate_microstructure_features(book)
        
        # Execute callbacks
        for callback in self.callbacks[DataType.BOOK]:
            await callback(book)
        
        # Publish events
        await self.event_bus.publish("market.book", book)
        await self.event_bus.publish("market.microstructure", features)
    
    def calculate_microstructure_features(
        self,
        book: OrderBook,
    ) -> Dict[str, float]:
        """Calculate microstructure features from order book.
        
        Args:
            book: Order book snapshot
            
        Returns:
            Dictionary of features
        """
        features = {
            "symbol": book.symbol,
            "timestamp": book.timestamp,
        }
        
        # Basic metrics
        features["mid_price"] = book.get_mid_price()
        features["spread"] = book.get_spread()
        features["spread_bps"] = (
            features["spread"] / features["mid_price"] * 10000
            if features["mid_price"] > 0 else 0
        )
        
        # Depth metrics
        features["bid_depth_1"] = sum(
            level.size for level in book.bids[:1]
        )
        features["ask_depth_1"] = sum(
            level.size for level in book.asks[:1]
        )
        features["bid_depth_5"] = sum(
            level.size for level in book.bids[:5]
        )
        features["ask_depth_5"] = sum(
            level.size for level in book.asks[:5]
        )
        
        # Imbalance
        features["imbalance_1"] = book.get_imbalance(1)
        features["imbalance_5"] = book.get_imbalance(5)
        
        # Weighted mid price (microprice)
        if book.bids and book.asks:
            best_bid = book.bids[0]
            best_ask = book.asks[0]
            total_size = best_bid.size + best_ask.size
            if total_size > 0:
                features["microprice"] = (
                    best_bid.price * best_ask.size +
                    best_ask.price * best_bid.size
                ) / total_size
            else:
                features["microprice"] = features["mid_price"]
        else:
            features["microprice"] = features["mid_price"]
        
        # Book pressure (weighted by distance from mid)
        bid_pressure = 0.0
        ask_pressure = 0.0
        mid = features["mid_price"]
        
        if mid > 0:
            for level in book.bids[:10]:
                weight = 1.0 / (1.0 + abs(level.price - mid) / mid)
                bid_pressure += level.size * weight
            
            for level in book.asks[:10]:
                weight = 1.0 / (1.0 + abs(level.price - mid) / mid)
                ask_pressure += level.size * weight
        
        total_pressure = bid_pressure + ask_pressure
        if total_pressure > 0:
            features["book_pressure"] = (
                (bid_pressure - ask_pressure) / total_pressure
            )
        else:
            features["book_pressure"] = 0.0
        
        # Signed volume (order flow)
        features["signed_volume"] = self.signed_volume.get(book.symbol, 0.0)
        
        return features
    
    def register_callback(
        self,
        data_type: DataType,
        callback: Callable,
    ) -> None:
        """Register callback for data type.
        
        Args:
            data_type: Type of data
            callback: Callback function
        """
        self.callbacks[data_type].append(callback)


class MarketDataAggregator:
    """Aggregates tick data into bars."""
    
    def __init__(self):
        """Initialize aggregator."""
        self.current_bars: Dict[Tuple[str, str], Bar] = {}
        self.bar_callbacks: List[Callable] = []
    
    async def aggregate_trade(
        self,
        trade: Trade,
        timeframe: str = "1m",
    ) -> Optional[Bar]:
        """Aggregate trade into bars.
        
        Args:
            trade: Trade to aggregate
            timeframe: Bar timeframe
            
        Returns:
            Completed bar if period ended
        """
        key = (trade.symbol, timeframe)
        
        # Get bar period
        period_seconds = self._parse_timeframe(timeframe)
        bar_time = self._get_bar_time(trade.timestamp, period_seconds)
        
        if key not in self.current_bars:
            # Start new bar
            self.current_bars[key] = Bar(
                symbol=trade.symbol,
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.size,
                timestamp=bar_time,
                trades=1,
            )
        else:
            current_bar = self.current_bars[key]
            
            # Check if we need to start a new bar
            if current_bar.timestamp != bar_time:
                # Complete current bar
                completed_bar = current_bar
                
                # Calculate VWAP if we have volume
                if completed_bar.volume > 0:
                    # Simplified - would track properly in production
                    completed_bar.vwap = (
                        completed_bar.high + completed_bar.low +
                        completed_bar.close
                    ) / 3
                
                # Start new bar
                self.current_bars[key] = Bar(
                    symbol=trade.symbol,
                    open=trade.price,
                    high=trade.price,
                    low=trade.price,
                    close=trade.price,
                    volume=trade.size,
                    timestamp=bar_time,
                    trades=1,
                )
                
                # Execute callbacks
                for callback in self.bar_callbacks:
                    await callback(completed_bar)
                
                return completed_bar
            else:
                # Update current bar
                current_bar.high = max(current_bar.high, trade.price)
                current_bar.low = min(current_bar.low, trade.price)
                current_bar.close = trade.price
                current_bar.volume += trade.size
                current_bar.trades += 1
        
        return None
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string (e.g., "1m", "5m", "1h")
            
        Returns:
            Number of seconds
        """
        if timeframe.endswith("s"):
            return int(timeframe[:-1])
        elif timeframe.endswith("m"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 3600
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 86400
        else:
            return 60  # Default to 1 minute
    
    def _get_bar_time(
        self,
        timestamp: datetime,
        period_seconds: int,
    ) -> datetime:
        """Get bar timestamp for given time.
        
        Args:
            timestamp: Current timestamp
            period_seconds: Bar period in seconds
            
        Returns:
            Bar timestamp
        """
        epoch = timestamp.timestamp()
        bar_epoch = (epoch // period_seconds) * period_seconds
        return datetime.fromtimestamp(bar_epoch)
    
    def register_bar_callback(self, callback: Callable) -> None:
        """Register callback for completed bars.
        
        Args:
            callback: Callback function
        """
        self.bar_callbacks.append(callback)
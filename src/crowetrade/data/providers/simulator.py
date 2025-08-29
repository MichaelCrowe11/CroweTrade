"""Simulated market data provider for testing and development."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from crowetrade.data.market_data import (
    Bar,
    OrderBook,
    OrderBookLevel,
    Quote,
    Trade,
)
from crowetrade.data.providers.base import MarketDataProvider, ProviderConfig

logger = logging.getLogger(__name__)


class PriceSimulator:
    """Simulates realistic price movements."""
    
    def __init__(
        self,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001,
        mean_reversion: float = 0.01,
    ):
        """Initialize price simulator.
        
        Args:
            initial_price: Starting price
            volatility: Daily volatility
            drift: Trend component
            mean_reversion: Mean reversion strength
        """
        self.price = initial_price
        self.initial_price = initial_price
        self.volatility = volatility
        self.drift = drift
        self.mean_reversion = mean_reversion
        
        # Microstructure parameters
        self.spread_bps = 2.0  # 2 basis points
        self.tick_size = 0.01
        
        # Order flow simulation
        self.order_flow_momentum = 0.0
        self.imbalance = 0.0
    
    def next_price(self, dt: float = 1.0) -> float:
        """Generate next price using stochastic process.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Next price
        """
        # Convert to fraction of day
        dt_day = dt / 86400
        
        # Mean reversion component
        mean_pull = self.mean_reversion * (self.initial_price - self.price) * dt_day
        
        # Random walk with drift
        random_component = np.random.normal(0, 1) * self.volatility * np.sqrt(dt_day)
        drift_component = self.drift * dt_day
        
        # Order flow momentum (creates short-term trends)
        self.order_flow_momentum *= 0.95  # Decay
        self.order_flow_momentum += np.random.normal(0, 0.001)
        
        # Update price
        self.price *= (1 + drift_component + random_component + mean_pull)
        self.price += self.order_flow_momentum
        
        # Ensure positive price
        self.price = max(self.tick_size, self.price)
        
        # Round to tick size
        self.price = round(self.price / self.tick_size) * self.tick_size
        
        return self.price
    
    def get_quote(self) -> Quote:
        """Generate quote with realistic spread.
        
        Returns:
            Quote with bid/ask
        """
        # Dynamic spread based on volatility
        spread = self.price * self.spread_bps / 10000
        spread = max(self.tick_size, round(spread / self.tick_size) * self.tick_size)
        
        # Add some randomness to spread
        spread *= (1 + np.random.uniform(-0.2, 0.2))
        
        bid = self.price - spread / 2
        ask = self.price + spread / 2
        
        # Round to tick size
        bid = round(bid / self.tick_size) * self.tick_size
        ask = round(ask / self.tick_size) * self.tick_size
        
        # Generate realistic sizes
        bid_size = np.random.lognormal(8, 1)  # Log-normal distribution
        ask_size = np.random.lognormal(8, 1)
        
        # Add imbalance
        if self.imbalance > 0:
            bid_size *= (1 + self.imbalance)
        else:
            ask_size *= (1 - self.imbalance)
        
        return Quote(
            symbol="",  # Will be set by provider
            bid=bid,
            ask=ask,
            bid_size=int(bid_size),
            ask_size=int(ask_size),
            timestamp=datetime.utcnow(),
        )
    
    def get_order_book(self, levels: int = 10) -> OrderBook:
        """Generate order book with multiple levels.
        
        Args:
            levels: Number of levels per side
            
        Returns:
            Order book
        """
        quote = self.get_quote()
        
        bids = []
        asks = []
        
        # Generate levels with increasing spread
        for i in range(levels):
            # Price steps away from best
            price_step = self.tick_size * (i + 1)
            
            # Size decreases with distance from best (power law)
            size_multiplier = 1.0 / ((i + 1) ** 1.5)
            
            # Bid levels
            bid_price = quote.bid - price_step * i
            bid_size = int(quote.bid_size * size_multiplier * np.random.uniform(0.5, 1.5))
            bids.append(OrderBookLevel(
                price=bid_price,
                size=bid_size,
                orders=np.random.poisson(3) + 1,
            ))
            
            # Ask levels
            ask_price = quote.ask + price_step * i
            ask_size = int(quote.ask_size * size_multiplier * np.random.uniform(0.5, 1.5))
            asks.append(OrderBookLevel(
                price=ask_price,
                size=ask_size,
                orders=np.random.poisson(3) + 1,
            ))
        
        return OrderBook(
            symbol="",  # Will be set by provider
            bids=bids,
            asks=asks,
            timestamp=datetime.utcnow(),
        )
    
    def get_trade(self) -> Trade:
        """Generate trade at current price.
        
        Returns:
            Trade data
        """
        quote = self.get_quote()
        
        # Trades happen at bid or ask (or sometimes in between)
        if np.random.random() < 0.45:
            price = quote.bid
        elif np.random.random() < 0.9:
            price = quote.ask
        else:
            # Mid trade (rare)
            price = (quote.bid + quote.ask) / 2
            price = round(price / self.tick_size) * self.tick_size
        
        # Trade size (log-normal)
        size = int(np.random.lognormal(5, 1.5))
        
        # Update order flow momentum based on trade
        if price >= quote.ask:
            self.order_flow_momentum += 0.0001  # Buy pressure
            self.imbalance = min(0.5, self.imbalance + 0.01)
        elif price <= quote.bid:
            self.order_flow_momentum -= 0.0001  # Sell pressure
            self.imbalance = max(-0.5, self.imbalance - 0.01)
        
        return Trade(
            symbol="",  # Will be set by provider
            price=price,
            size=size,
            timestamp=datetime.utcnow(),
            exchange="SIM",
        )


class SimulatedMarketData(MarketDataProvider):
    """Simulated market data provider."""
    
    def __init__(
        self,
        config: ProviderConfig,
        processor,
        tick_rate: int = 100,  # Ticks per second
    ):
        """Initialize simulated provider.
        
        Args:
            config: Provider configuration
            processor: Data processor
            tick_rate: Rate of data generation
        """
        super().__init__(config, processor)
        self.tick_rate = tick_rate
        self.simulators: Dict[str, PriceSimulator] = {}
        self._simulation_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self) -> None:
        """Connect to simulated data source."""
        logger.info(f"Connecting to simulated market data")
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.last_heartbeat = datetime.utcnow()
    
    async def disconnect(self) -> None:
        """Disconnect from simulated data source."""
        logger.info(f"Disconnecting from simulated market data")
        
        # Cancel all simulation tasks
        for task in self._simulation_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._simulation_tasks.clear()
        self.simulators.clear()
    
    async def subscribe_trades(self, symbols: List[str]) -> None:
        """Subscribe to trade data.
        
        Args:
            symbols: List of symbols to subscribe
        """
        for symbol in symbols:
            if symbol not in self.trade_subscriptions:
                self.trade_subscriptions.add(symbol)
                await self._start_simulation(symbol)
                logger.info(f"Subscribed to trades for {symbol}")
    
    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """Subscribe to quote data.
        
        Args:
            symbols: List of symbols to subscribe
        """
        for symbol in symbols:
            if symbol not in self.quote_subscriptions:
                self.quote_subscriptions.add(symbol)
                await self._start_simulation(symbol)
                logger.info(f"Subscribed to quotes for {symbol}")
    
    async def subscribe_book(self, symbols: List[str], depth: int = 10) -> None:
        """Subscribe to order book data.
        
        Args:
            symbols: List of symbols to subscribe
            depth: Book depth
        """
        for symbol in symbols:
            if symbol not in self.book_subscriptions:
                self.book_subscriptions.add(symbol)
                await self._start_simulation(symbol)
                logger.info(f"Subscribed to L2 book for {symbol}")
    
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
        for symbol in symbols:
            if symbol not in self.bar_subscriptions:
                self.bar_subscriptions[symbol] = set()
            self.bar_subscriptions[symbol].add(timeframe)
            await self._start_simulation(symbol)
            logger.info(f"Subscribed to {timeframe} bars for {symbol}")
    
    async def _start_simulation(self, symbol: str) -> None:
        """Start simulation for a symbol.
        
        Args:
            symbol: Symbol to simulate
        """
        if symbol not in self.simulators:
            # Create simulator with random initial price
            initial_price = np.random.uniform(50, 500)
            volatility = np.random.uniform(0.01, 0.05)
            
            self.simulators[symbol] = PriceSimulator(
                initial_price=initial_price,
                volatility=volatility,
            )
        
        if symbol not in self._simulation_tasks:
            self._simulation_tasks[symbol] = asyncio.create_task(
                self._simulate_symbol(symbol)
            )
    
    async def _simulate_symbol(self, symbol: str) -> None:
        """Simulate data for a symbol.
        
        Args:
            symbol: Symbol to simulate
        """
        simulator = self.simulators[symbol]
        tick_interval = 1.0 / self.tick_rate
        
        # Bar aggregation
        bar_trades = []
        bar_start_time = datetime.utcnow()
        bar_open = simulator.price
        bar_high = simulator.price
        bar_low = simulator.price
        
        while True:
            try:
                # Update price
                simulator.next_price(tick_interval)
                
                # Generate trade
                if symbol in self.trade_subscriptions:
                    if np.random.random() < 0.3:  # 30% chance of trade
                        trade = simulator.get_trade()
                        trade.symbol = symbol
                        await self.handle_trade(trade)
                        
                        # Collect for bar
                        bar_trades.append(trade)
                        bar_high = max(bar_high, trade.price)
                        bar_low = min(bar_low, trade.price)
                
                # Generate quote
                if symbol in self.quote_subscriptions:
                    if np.random.random() < 0.8:  # 80% chance of quote update
                        quote = simulator.get_quote()
                        quote.symbol = symbol
                        await self.handle_quote(quote)
                
                # Generate book
                if symbol in self.book_subscriptions:
                    if np.random.random() < 0.5:  # 50% chance of book update
                        book = simulator.get_order_book()
                        book.symbol = symbol
                        await self.handle_book(book)
                
                # Check for bar completion
                if symbol in self.bar_subscriptions:
                    current_time = datetime.utcnow()
                    elapsed = (current_time - bar_start_time).total_seconds()
                    
                    # Check each timeframe
                    for timeframe in self.bar_subscriptions[symbol]:
                        period = self._parse_timeframe(timeframe)
                        
                        if elapsed >= period:
                            # Complete bar
                            if bar_trades:
                                volume = sum(t.size for t in bar_trades)
                                vwap = sum(t.price * t.size for t in bar_trades) / volume
                            else:
                                volume = 0
                                vwap = simulator.price
                            
                            bar = Bar(
                                symbol=symbol,
                                open=bar_open,
                                high=bar_high,
                                low=bar_low,
                                close=simulator.price,
                                volume=volume,
                                timestamp=bar_start_time,
                                vwap=vwap,
                                trades=len(bar_trades),
                            )
                            
                            await self.handle_bar(bar)
                            
                            # Reset for next bar
                            bar_trades = []
                            bar_start_time = current_time
                            bar_open = simulator.price
                            bar_high = simulator.price
                            bar_low = simulator.price
                
                # Sleep for next tick
                await asyncio.sleep(tick_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error simulating {symbol}: {e}")
                await asyncio.sleep(1)
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe to seconds.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Seconds
        """
        if timeframe.endswith("s"):
            return int(timeframe[:-1])
        elif timeframe.endswith("m"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 3600
        else:
            return 60
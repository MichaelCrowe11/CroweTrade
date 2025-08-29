from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.agent import AgentConfig, BaseAgent
from ..core.events import MarketData
from ..core.types import Symbol, Venue


@dataclass
class MarketDataConfig(AgentConfig):
    symbols: list[Symbol] = field(default_factory=list)
    venues: list[Venue] = field(default_factory=list)
    update_interval: float = 0.1
    simulate: bool = True


class MarketDataAgent(BaseAgent):
    def __init__(self, config: MarketDataConfig):
        super().__init__(config)
        self.config: MarketDataConfig = config
        self._prices: dict[Symbol, float] = {}
        self._volatilities: dict[Symbol, float] = {}
        self._running = False
    
    async def on_start(self):
        self._running = True
        if self.config.simulate:
            self._initialize_simulation()
            asyncio.create_task(self._simulate_market_data())
    
    async def on_stop(self):
        self._running = False
    
    def _initialize_simulation(self):
        for symbol in self.config.symbols:
            self._prices[symbol] = 100.0 + random.uniform(-50, 50)
            self._volatilities[symbol] = 0.02 + random.uniform(0, 0.03)
    
    async def _simulate_market_data(self):
        while self._running:
            for symbol in self.config.symbols:
                price = self._prices[symbol]
                vol = self._volatilities[symbol]
                
                ret = np.random.normal(0, vol * np.sqrt(self.config.update_interval))
                new_price = price * (1 + ret)
                self._prices[symbol] = new_price
                
                spread = 0.01 + random.uniform(0, 0.02)
                bid = new_price * (1 - spread/2)
                ask = new_price * (1 + spread/2)
                
                for venue in self.config.venues:
                    event = MarketData(
                        event_id=f"md_{symbol}_{venue}_{datetime.utcnow().isoformat()}",
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        bid_size=random.uniform(100, 10000),
                        ask_size=random.uniform(100, 10000),
                        last_price=new_price,
                        volume=random.uniform(100000, 10000000),
                        venue=venue
                    )
                    await self.emit(event)
            
            await asyncio.sleep(self.config.update_interval)


@dataclass
class FeatureConfig(AgentConfig):
    lookback_periods: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    update_interval: float = 1.0


class FeatureAgent(BaseAgent):
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.config: FeatureConfig = config
        self._price_history: dict[Symbol, list[float]] = {}
        self._features: dict[Symbol, dict[str, float]] = {}
        self._running = False
    
    async def on_start(self):
        self._running = True
        self.subscribe(MarketData, self._on_market_data)
        asyncio.create_task(self._compute_features())
    
    async def on_stop(self):
        self._running = False
    
    async def _on_market_data(self, event: MarketData):
        symbol = event.symbol
        
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        
        self._price_history[symbol].append(event.last_price)
        
        max_lookback = max(self.config.lookback_periods)
        if len(self._price_history[symbol]) > max_lookback * 2:
            self._price_history[symbol] = self._price_history[symbol][-max_lookback:]
    
    async def _compute_features(self):
        while self._running:
            for symbol, prices in self._price_history.items():
                if len(prices) < 2:
                    continue
                
                features = {}
                prices_arr = np.array(prices)
                
                returns = np.diff(prices_arr) / prices_arr[:-1]
                features['return_1'] = returns[-1] if len(returns) > 0 else 0
                
                for period in self.config.lookback_periods:
                    if len(prices) >= period:
                        features[f'sma_{period}'] = np.mean(prices[-period:])
                        features[f'std_{period}'] = np.std(prices[-period:])
                        
                        if prices[-1] > 0 and features[f'sma_{period}'] > 0:
                            features[f'zscore_{period}'] = (
                                (prices[-1] - features[f'sma_{period}']) / 
                                max(features[f'std_{period}'], 1e-6)
                            )
                
                if len(prices) >= 20:
                    features['rsi'] = self._calculate_rsi(prices, 14)
                    features['momentum'] = (prices[-1] / prices[-10] - 1) if prices[-10] > 0 else 0
                
                self._features[symbol] = features
            
            await asyncio.sleep(self.config.update_interval)
    
    def _calculate_rsi(self, prices: list[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
"""Mean reversion trading strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from crowetrade.features.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionSignal:
    """Mean reversion strategy signal."""
    
    symbol: str
    z_score: float
    deviation_pct: float
    signal_strength: float
    side: str  # "long" or "short"
    target_price: float
    stop_loss: float
    metadata: Dict


class MeanReversionStrategy:
    """Mean reversion strategy using statistical arbitrage.
    
    Identifies assets that have deviated significantly from their mean
    and trades expecting reversion.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,  # Z-score threshold
        exit_threshold: float = 0.5,   # Z-score for exit
        stop_loss_pct: float = 0.05,   # 5% stop loss
        max_holding_period: int = 10,  # Maximum days to hold
        use_bollinger: bool = True,
        use_rsi: bool = True,
        position_size: float = 0.02,   # 2% per position
    ):
        """Initialize mean reversion strategy.
        
        Args:
            lookback_period: Period for calculating mean
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            stop_loss_pct: Stop loss percentage
            max_holding_period: Maximum holding period
            use_bollinger: Use Bollinger Bands
            use_rsi: Use RSI for confirmation
            position_size: Position size per trade
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_period = max_holding_period
        self.use_bollinger = use_bollinger
        self.use_rsi = use_rsi
        self.position_size = position_size
        
        self.active_positions: Dict[str, Dict] = {}
    
    def calculate_z_scores(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate z-scores for all assets.
        
        Args:
            prices: Price DataFrame
            
        Returns:
            DataFrame of z-scores
        """
        # Calculate rolling mean and std
        rolling_mean = prices.rolling(self.lookback_period).mean()
        rolling_std = prices.rolling(self.lookback_period).std()
        
        # Calculate z-scores
        z_scores = (prices - rolling_mean) / (rolling_std + 1e-6)
        
        return z_scores
    
    def identify_opportunities(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> List[MeanReversionSignal]:
        """Identify mean reversion opportunities.
        
        Args:
            prices: Historical prices
            volumes: Historical volumes
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Calculate z-scores
        z_scores = self.calculate_z_scores(prices)
        current_z = z_scores.iloc[-1]
        
        # Calculate additional indicators if enabled
        rsi_values = {}
        bb_positions = {}
        
        for symbol in prices.columns:
            price_array = prices[symbol].values
            
            if self.use_rsi:
                rsi = TechnicalIndicators.rsi(price_array)
                rsi_values[symbol] = rsi[-1] if len(rsi) > 0 else 50
            
            if self.use_bollinger:
                upper, middle, lower = TechnicalIndicators.bollinger_bands(
                    price_array,
                    self.lookback_period,
                )
                if not np.isnan(upper[-1]):
                    bb_position = (price_array[-1] - lower[-1]) / (upper[-1] - lower[-1])
                    bb_positions[symbol] = bb_position
        
        # Generate signals
        for symbol in prices.columns:
            z_score = current_z[symbol]
            
            if np.isnan(z_score):
                continue
            
            # Skip if already in position
            if symbol in self.active_positions:
                # Check exit conditions
                position = self.active_positions[symbol]
                holding_days = (datetime.utcnow() - position["entry_time"]).days
                
                should_exit = (
                    abs(z_score) < self.exit_threshold or
                    holding_days >= self.max_holding_period
                )
                
                if should_exit:
                    del self.active_positions[symbol]
                continue
            
            # Check entry conditions
            signal_strength = 0.0
            side = None
            
            # Oversold condition (buy signal)
            if z_score < -self.entry_threshold:
                signal_strength = min(abs(z_score) / 3.0, 1.0)
                
                # Confirm with RSI if enabled
                if self.use_rsi and symbol in rsi_values:
                    if rsi_values[symbol] < 30:
                        signal_strength *= 1.2
                    elif rsi_values[symbol] > 50:
                        signal_strength *= 0.8
                
                # Confirm with Bollinger Bands if enabled
                if self.use_bollinger and symbol in bb_positions:
                    if bb_positions[symbol] < 0.2:
                        signal_strength *= 1.2
                
                side = "long"
            
            # Overbought condition (sell signal)
            elif z_score > self.entry_threshold:
                signal_strength = min(abs(z_score) / 3.0, 1.0)
                
                # Confirm with RSI if enabled
                if self.use_rsi and symbol in rsi_values:
                    if rsi_values[symbol] > 70:
                        signal_strength *= 1.2
                    elif rsi_values[symbol] < 50:
                        signal_strength *= 0.8
                
                # Confirm with Bollinger Bands if enabled
                if self.use_bollinger and symbol in bb_positions:
                    if bb_positions[symbol] > 0.8:
                        signal_strength *= 1.2
                
                side = "short"
            
            if side and signal_strength > 0.3:  # Minimum signal strength
                current_price = prices[symbol].iloc[-1]
                mean_price = prices[symbol].rolling(self.lookback_period).mean().iloc[-1]
                
                # Calculate targets
                if side == "long":
                    target_price = mean_price
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                else:
                    target_price = mean_price
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                
                signal = MeanReversionSignal(
                    symbol=symbol,
                    z_score=z_score,
                    deviation_pct=(current_price / mean_price - 1) * 100,
                    signal_strength=signal_strength,
                    side=side,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    metadata={
                        "strategy": "mean_reversion",
                        "rsi": rsi_values.get(symbol),
                        "bb_position": bb_positions.get(symbol),
                        "lookback": self.lookback_period,
                    },
                )
                
                signals.append(signal)
                
                # Track position
                self.active_positions[symbol] = {
                    "entry_time": datetime.utcnow(),
                    "entry_price": current_price,
                    "side": side,
                    "z_score": z_score,
                }
        
        return signals
    
    def calculate_pairs_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: Optional[float] = None,
    ) -> pd.Series:
        """Calculate spread for pairs trading.
        
        Args:
            price1: First asset prices
            price2: Second asset prices
            hedge_ratio: Hedge ratio (calculated if None)
            
        Returns:
            Spread series
        """
        if hedge_ratio is None:
            # Calculate hedge ratio using OLS
            x = price2.values.reshape(-1, 1)
            y = price1.values
            
            # Remove NaN
            mask = ~(np.isnan(x.flatten()) | np.isnan(y))
            x_clean = x[mask].reshape(-1, 1)
            y_clean = y[mask]
            
            if len(x_clean) > 0:
                hedge_ratio = np.linalg.lstsq(x_clean, y_clean, rcond=None)[0][0]
            else:
                hedge_ratio = 1.0
        
        spread = price1 - hedge_ratio * price2
        return spread
    
    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        threshold: float = 0.05,
    ) -> List[Tuple[str, str, float]]:
        """Find cointegrated pairs for pairs trading.
        
        Args:
            prices: Price DataFrame
            threshold: P-value threshold
            
        Returns:
            List of (symbol1, symbol2, p-value) tuples
        """
        from statsmodels.tsa.stattools import coint
        
        pairs = []
        symbols = prices.columns.tolist()
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Test for cointegration
                try:
                    _, p_value, _ = coint(
                        prices[symbol1].dropna(),
                        prices[symbol2].dropna(),
                    )
                    
                    if p_value < threshold:
                        pairs.append((symbol1, symbol2, p_value))
                except:
                    continue
        
        # Sort by p-value
        pairs.sort(key=lambda x: x[2])
        
        return pairs
    
    def generate_pairs_signals(
        self,
        prices: pd.DataFrame,
    ) -> List[MeanReversionSignal]:
        """Generate signals for pairs trading.
        
        Args:
            prices: Historical prices
            
        Returns:
            List of pairs trading signals
        """
        signals = []
        
        # Find cointegrated pairs
        pairs = self.find_cointegrated_pairs(prices)
        
        for symbol1, symbol2, p_value in pairs[:5]:  # Top 5 pairs
            # Calculate spread
            spread = self.calculate_pairs_spread(
                prices[symbol1],
                prices[symbol2],
            )
            
            # Calculate z-score of spread
            spread_mean = spread.rolling(self.lookback_period).mean()
            spread_std = spread.rolling(self.lookback_period).std()
            z_score = (spread.iloc[-1] - spread_mean.iloc[-1]) / (spread_std.iloc[-1] + 1e-6)
            
            if abs(z_score) > self.entry_threshold:
                # Generate paired signals
                if z_score > self.entry_threshold:
                    # Spread too high: short symbol1, long symbol2
                    signals.append(MeanReversionSignal(
                        symbol=symbol1,
                        z_score=z_score,
                        deviation_pct=0,
                        signal_strength=min(abs(z_score) / 3.0, 1.0),
                        side="short",
                        target_price=prices[symbol1].iloc[-1] * 0.98,
                        stop_loss=prices[symbol1].iloc[-1] * 1.05,
                        metadata={"pair": symbol2, "p_value": p_value},
                    ))
                    
                    signals.append(MeanReversionSignal(
                        symbol=symbol2,
                        z_score=-z_score,
                        deviation_pct=0,
                        signal_strength=min(abs(z_score) / 3.0, 1.0),
                        side="long",
                        target_price=prices[symbol2].iloc[-1] * 1.02,
                        stop_loss=prices[symbol2].iloc[-1] * 0.95,
                        metadata={"pair": symbol1, "p_value": p_value},
                    ))
                
                elif z_score < -self.entry_threshold:
                    # Spread too low: long symbol1, short symbol2
                    signals.append(MeanReversionSignal(
                        symbol=symbol1,
                        z_score=z_score,
                        deviation_pct=0,
                        signal_strength=min(abs(z_score) / 3.0, 1.0),
                        side="long",
                        target_price=prices[symbol1].iloc[-1] * 1.02,
                        stop_loss=prices[symbol1].iloc[-1] * 0.95,
                        metadata={"pair": symbol2, "p_value": p_value},
                    ))
                    
                    signals.append(MeanReversionSignal(
                        symbol=symbol2,
                        z_score=-z_score,
                        deviation_pct=0,
                        signal_strength=min(abs(z_score) / 3.0, 1.0),
                        side="short",
                        target_price=prices[symbol2].iloc[-1] * 0.98,
                        stop_loss=prices[symbol2].iloc[-1] * 1.05,
                        metadata={"pair": symbol1, "p_value": p_value},
                    ))
        
        return signals
"""Cross-sectional momentum trading strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from crowetrade.core.contracts import Signal
from crowetrade.features.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class MomentumSignal:
    """Momentum strategy signal."""
    
    symbol: str
    momentum_score: float
    rank_percentile: float
    signal_strength: float
    side: str  # "long" or "short"
    metadata: Dict


class CrossSectionalMomentum:
    """Cross-sectional momentum strategy.
    
    Ranks assets by momentum and goes long top performers,
    short bottom performers.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        holding_period: int = 5,
        top_percentile: float = 0.2,
        bottom_percentile: float = 0.2,
        min_momentum: float = 0.02,
        max_position_size: float = 0.1,
        use_volatility_scaling: bool = True,
    ):
        """Initialize momentum strategy.
        
        Args:
            lookback_period: Days to calculate momentum
            holding_period: Days to hold positions
            top_percentile: Top % to go long
            bottom_percentile: Bottom % to go short
            min_momentum: Minimum momentum to trade
            max_position_size: Maximum position size per asset
            use_volatility_scaling: Scale by inverse volatility
        """
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.min_momentum = min_momentum
        self.max_position_size = max_position_size
        self.use_volatility_scaling = use_volatility_scaling
        
        self.last_rebalance: Optional[datetime] = None
        self.current_positions: Dict[str, float] = {}
    
    def calculate_momentum(
        self,
        prices: pd.DataFrame,
        method: str = "return",
    ) -> pd.Series:
        """Calculate momentum scores for all assets.
        
        Args:
            prices: DataFrame with assets as columns
            method: Momentum calculation method
            
        Returns:
            Series of momentum scores
        """
        if method == "return":
            # Simple return momentum
            returns = prices.pct_change(self.lookback_period)
            momentum = returns.iloc[-1]
        
        elif method == "regression":
            # Time-series regression momentum
            momentum = pd.Series(index=prices.columns, dtype=float)
            
            for col in prices.columns:
                if prices[col].isna().sum() > len(prices) * 0.1:
                    continue
                
                y = np.log(prices[col].dropna())
                x = np.arange(len(y))
                
                if len(y) >= self.lookback_period:
                    # Linear regression slope
                    coef = np.polyfit(x[-self.lookback_period:], 
                                     y[-self.lookback_period:], 1)[0]
                    momentum[col] = coef
        
        elif method == "risk_adjusted":
            # Sharpe ratio momentum
            returns = prices.pct_change()
            rolling_mean = returns.rolling(self.lookback_period).mean()
            rolling_std = returns.rolling(self.lookback_period).std()
            
            sharpe = rolling_mean / (rolling_std + 1e-6)
            momentum = sharpe.iloc[-1]
        
        else:
            # Default to simple returns
            momentum = prices.pct_change(self.lookback_period).iloc[-1]
        
        return momentum
    
    def rank_assets(
        self,
        momentum_scores: pd.Series,
    ) -> Tuple[List[str], List[str]]:
        """Rank assets and select long/short candidates.
        
        Args:
            momentum_scores: Momentum scores for all assets
            
        Returns:
            Tuple of (long symbols, short symbols)
        """
        # Remove NaN and filter by minimum momentum
        valid_scores = momentum_scores.dropna()
        valid_scores = valid_scores[abs(valid_scores) >= self.min_momentum]
        
        if len(valid_scores) < 2:
            return [], []
        
        # Rank assets
        ranked = valid_scores.rank(pct=True)
        
        # Select top and bottom
        long_symbols = ranked[ranked >= (1 - self.top_percentile)].index.tolist()
        short_symbols = ranked[ranked <= self.bottom_percentile].index.tolist()
        
        return long_symbols, short_symbols
    
    def calculate_position_sizes(
        self,
        long_symbols: List[str],
        short_symbols: List[str],
        prices: pd.DataFrame,
        volatilities: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Calculate position sizes with risk parity.
        
        Args:
            long_symbols: Symbols to go long
            short_symbols: Symbols to go short
            prices: Current prices
            volatilities: Asset volatilities for scaling
            
        Returns:
            Position sizes by symbol
        """
        positions = {}
        
        # Calculate volatilities if not provided
        if self.use_volatility_scaling and volatilities is None:
            returns = prices.pct_change()
            volatilities = returns.rolling(20).std().iloc[-1]
        
        # Long positions
        if long_symbols:
            if self.use_volatility_scaling and volatilities is not None:
                # Inverse volatility weighting
                long_vols = volatilities[long_symbols]
                inv_vols = 1 / (long_vols + 1e-6)
                weights = inv_vols / inv_vols.sum()
            else:
                # Equal weighting
                weights = pd.Series(1.0 / len(long_symbols), index=long_symbols)
            
            for symbol in long_symbols:
                size = weights[symbol] * (1 - self.bottom_percentile)
                positions[symbol] = min(size, self.max_position_size)
        
        # Short positions
        if short_symbols:
            if self.use_volatility_scaling and volatilities is not None:
                short_vols = volatilities[short_symbols]
                inv_vols = 1 / (short_vols + 1e-6)
                weights = inv_vols / inv_vols.sum()
            else:
                weights = pd.Series(1.0 / len(short_symbols), index=short_symbols)
            
            for symbol in short_symbols:
                size = weights[symbol] * self.bottom_percentile
                positions[symbol] = -min(size, self.max_position_size)
        
        return positions
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        current_time: datetime,
    ) -> List[MomentumSignal]:
        """Generate momentum signals for all assets.
        
        Args:
            prices: Historical prices DataFrame
            current_time: Current timestamp
            
        Returns:
            List of momentum signals
        """
        # Check if rebalance is needed
        if self.last_rebalance:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.holding_period:
                return []  # No rebalance needed
        
        # Calculate momentum
        momentum_scores = self.calculate_momentum(prices, method="risk_adjusted")
        
        # Rank and select
        long_symbols, short_symbols = self.rank_assets(momentum_scores)
        
        # Calculate position sizes
        positions = self.calculate_position_sizes(
            long_symbols,
            short_symbols,
            prices,
        )
        
        # Generate signals
        signals = []
        
        for symbol, position in positions.items():
            # Calculate signal strength based on momentum magnitude
            momentum = momentum_scores[symbol]
            rank_pct = momentum_scores.rank(pct=True)[symbol]
            
            signal = MomentumSignal(
                symbol=symbol,
                momentum_score=momentum,
                rank_percentile=rank_pct,
                signal_strength=abs(momentum) / (abs(momentum_scores).max() + 1e-6),
                side="long" if position > 0 else "short",
                metadata={
                    "strategy": "cross_sectional_momentum",
                    "lookback": self.lookback_period,
                    "position_size": abs(position),
                    "rebalance_date": current_time.isoformat(),
                },
            )
            
            signals.append(signal)
        
        # Update state
        self.last_rebalance = current_time
        self.current_positions = positions
        
        return signals
    
    def backtest(
        self,
        prices: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Backtest the momentum strategy.
        
        Args:
            prices: Historical prices
            start_date: Backtest start
            end_date: Backtest end
            
        Returns:
            DataFrame with backtest results
        """
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
        
        # Initialize results
        results = []
        positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        # Simulate rebalancing
        for i in range(self.lookback_period + 1, len(prices), self.holding_period):
            date = prices.index[i]
            
            # Get data up to current date
            historical = prices.iloc[:i]
            
            # Generate signals
            momentum = self.calculate_momentum(historical)
            long_symbols, short_symbols = self.rank_assets(momentum)
            
            # Calculate positions
            position_sizes = self.calculate_position_sizes(
                long_symbols,
                short_symbols,
                historical,
            )
            
            # Update positions for holding period
            for j in range(self.holding_period):
                if i + j < len(prices):
                    for symbol, size in position_sizes.items():
                        positions.iloc[i + j][symbol] = size
            
            # Record rebalance
            results.append({
                "date": date,
                "num_long": len(long_symbols),
                "num_short": len(short_symbols),
                "avg_momentum_long": momentum[long_symbols].mean() if long_symbols else 0,
                "avg_momentum_short": momentum[short_symbols].mean() if short_symbols else 0,
            })
        
        # Calculate returns
        asset_returns = prices.pct_change()
        strategy_returns = (positions.shift(1) * asset_returns).sum(axis=1)
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        logger.info(f"Momentum Backtest Results:")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Volatility: {volatility:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return pd.DataFrame({
            "positions": positions.sum(axis=1),
            "returns": strategy_returns,
            "cumulative": cumulative,
            "drawdown": drawdown,
        })
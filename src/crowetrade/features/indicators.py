"""Technical indicators for momentum, volatility, and other signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Result from indicator calculation."""
    
    value: float
    signal: Optional[str] = None  # "buy", "sell", "neutral"
    strength: float = 0.0  # Signal strength 0-1
    metadata: Optional[dict] = None


class TechnicalIndicators:
    """Collection of technical indicators for trading signals."""
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average.
        
        Args:
            prices: Price array
            period: Period for SMA
            
        Returns:
            SMA values
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        sma = np.convolve(prices, np.ones(period) / period, mode='same')
        sma[:period-1] = np.nan
        return sma
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average.
        
        Args:
            prices: Price array
            period: Period for EMA
            
        Returns:
            EMA values
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        ema[:period-1] = np.nan
        return ema
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index.
        
        Args:
            prices: Price array
            period: RSI period
            
        Returns:
            RSI values (0-100)
        """
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(gains))
        avg_losses = np.zeros(len(losses))
        
        # Initial averages
        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])
        
        # Smooth averages
        for i in range(period, len(gains)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i]) / period
        
        rs = np.divide(avg_gains, avg_losses, where=avg_losses != 0)
        rsi = 100 - (100 / (1 + rs))
        
        # Add NaN for initial period
        rsi_full = np.full(len(prices), np.nan)
        rsi_full[period:] = rsi[period-1:]
        
        return rsi_full
    
    @staticmethod
    def macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD indicator.
        
        Args:
            prices: Price array
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line[~np.isnan(macd_line)], signal)
        
        # Align signal line
        signal_full = np.full_like(macd_line, np.nan)
        valid_idx = ~np.isnan(macd_line)
        signal_full[valid_idx] = signal_line
        
        histogram = macd_line - signal_full
        
        return macd_line, signal_full, histogram
    
    @staticmethod
    def bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands.
        
        Args:
            prices: Price array
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = TechnicalIndicators.sma(prices, period)
        
        # Calculate rolling standard deviation
        std = np.zeros_like(prices)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        std[:period-1] = np.nan
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Average True Range (volatility indicator).
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: ATR period
            
        Returns:
            ATR values
        """
        if len(highs) < period + 1:
            return np.full(len(highs), np.nan)
        
        # True Range calculation
        tr = np.zeros(len(highs))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)
        
        # ATR calculation (EMA of TR)
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        atr[:period-1] = np.nan
        return atr
    
    @staticmethod
    def stochastic(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator.
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: Lookback period
            smooth_k: %K smoothing period
            smooth_d: %D smoothing period
            
        Returns:
            Tuple of (%K, %D)
        """
        if len(closes) < period:
            return np.full(len(closes), np.nan), np.full(len(closes), np.nan)
        
        k_raw = np.zeros(len(closes))
        
        for i in range(period - 1, len(closes)):
            highest = np.max(highs[i - period + 1:i + 1])
            lowest = np.min(lows[i - period + 1:i + 1])
            
            if highest - lowest > 0:
                k_raw[i] = 100 * (closes[i] - lowest) / (highest - lowest)
            else:
                k_raw[i] = 50
        
        k_raw[:period-1] = np.nan
        
        # Smooth %K
        k_smooth = TechnicalIndicators.sma(k_raw[~np.isnan(k_raw)], smooth_k)
        k_full = np.full_like(k_raw, np.nan)
        k_full[~np.isnan(k_raw)] = k_smooth
        
        # Calculate %D
        d = TechnicalIndicators.sma(k_full[~np.isnan(k_full)], smooth_d)
        d_full = np.full_like(k_full, np.nan)
        valid_idx = ~np.isnan(k_full)
        d_full[valid_idx] = d
        
        return k_full, d_full
    
    @staticmethod
    def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Price momentum.
        
        Args:
            prices: Price array
            period: Lookback period
            
        Returns:
            Momentum values
        """
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        momentum = np.full(len(prices), np.nan)
        momentum[period:] = prices[period:] - prices[:-period]
        
        return momentum
    
    @staticmethod
    def roc(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Rate of Change.
        
        Args:
            prices: Price array
            period: Lookback period
            
        Returns:
            ROC values (percentage)
        """
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        roc = np.full(len(prices), np.nan)
        roc[period:] = ((prices[period:] - prices[:-period]) / prices[:-period]) * 100
        
        return roc
    
    @staticmethod
    def volatility(
        returns: np.ndarray,
        period: int = 20,
        annualize: bool = True,
    ) -> np.ndarray:
        """Historical volatility.
        
        Args:
            returns: Return array
            period: Lookback period
            annualize: Whether to annualize (252 trading days)
            
        Returns:
            Volatility values
        """
        if len(returns) < period:
            return np.full(len(returns), np.nan)
        
        vol = np.zeros(len(returns))
        
        for i in range(period - 1, len(returns)):
            vol[i] = np.std(returns[i - period + 1:i + 1])
        
        vol[:period-1] = np.nan
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free: float = 0.02,
        period: int = 252,
    ) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Return array
            risk_free: Risk-free rate (annualized)
            period: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns) * period
        std_return = np.std(returns) * np.sqrt(period)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free) / std_return
    
    @staticmethod
    def max_drawdown(prices: np.ndarray) -> Tuple[float, int, int]:
        """Calculate maximum drawdown.
        
        Args:
            prices: Price array
            
        Returns:
            Tuple of (max drawdown %, peak index, trough index)
        """
        if len(prices) < 2:
            return 0.0, 0, 0
        
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        
        max_dd = np.min(drawdown)
        trough_idx = np.argmin(drawdown)
        
        # Find the peak before the trough
        peak_idx = np.argmax(prices[:trough_idx+1])
        
        return max_dd * 100, peak_idx, trough_idx
    
    @staticmethod
    def z_score(
        values: np.ndarray,
        period: int = 20,
    ) -> np.ndarray:
        """Calculate rolling z-score.
        
        Args:
            values: Value array
            period: Lookback period
            
        Returns:
            Z-score values
        """
        if len(values) < period:
            return np.full(len(values), np.nan)
        
        z_scores = np.zeros(len(values))
        
        for i in range(period - 1, len(values)):
            window = values[i - period + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                z_scores[i] = (values[i] - mean) / std
            else:
                z_scores[i] = 0
        
        z_scores[:period-1] = np.nan
        return z_scores
    
    @staticmethod
    def hurst_exponent(prices: np.ndarray, max_lag: int = 20) -> float:
        """Calculate Hurst exponent (trend persistence).
        
        Args:
            prices: Price array
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent (< 0.5 = mean reverting, > 0.5 = trending)
        """
        if len(prices) < max_lag * 2:
            return 0.5
        
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            price_diff = prices[lag:] - prices[:-lag]
            tau.append(np.std(price_diff))
        
        # Linear regression of log(tau) on log(lag)
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return reg[0]
    
    @staticmethod
    def generate_signals(
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
    ) -> dict:
        """Generate comprehensive trading signals.
        
        Args:
            prices: Close prices
            volumes: Volume data
            highs: High prices
            lows: Low prices
            
        Returns:
            Dictionary of signals and indicators
        """
        signals = {}
        
        # Trend indicators
        sma_20 = TechnicalIndicators.sma(prices, 20)
        sma_50 = TechnicalIndicators.sma(prices, 50)
        ema_12 = TechnicalIndicators.ema(prices, 12)
        ema_26 = TechnicalIndicators.ema(prices, 26)
        
        signals["sma_20"] = sma_20[-1] if len(sma_20) > 0 else np.nan
        signals["sma_50"] = sma_50[-1] if len(sma_50) > 0 else np.nan
        signals["price_vs_sma20"] = (prices[-1] / sma_20[-1] - 1) * 100 if not np.isnan(sma_20[-1]) else np.nan
        
        # Momentum
        rsi = TechnicalIndicators.rsi(prices)
        signals["rsi"] = rsi[-1] if len(rsi) > 0 else np.nan
        
        momentum = TechnicalIndicators.momentum(prices)
        signals["momentum"] = momentum[-1] if len(momentum) > 0 else np.nan
        
        # MACD
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(prices)
        signals["macd"] = macd[-1] if len(macd) > 0 else np.nan
        signals["macd_signal"] = macd_signal[-1] if len(macd_signal) > 0 else np.nan
        signals["macd_histogram"] = macd_hist[-1] if len(macd_hist) > 0 else np.nan
        
        # Volatility
        if highs is not None and lows is not None:
            atr = TechnicalIndicators.atr(highs, lows, prices)
            signals["atr"] = atr[-1] if len(atr) > 0 else np.nan
            signals["atr_pct"] = (atr[-1] / prices[-1] * 100) if not np.isnan(atr[-1]) else np.nan
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(prices)
        signals["bb_upper"] = bb_upper[-1] if len(bb_upper) > 0 else np.nan
        signals["bb_lower"] = bb_lower[-1] if len(bb_lower) > 0 else np.nan
        signals["bb_position"] = ((prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])) if not np.isnan(bb_upper[-1]) else np.nan
        
        # Generate composite signal
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if not np.isnan(signals["rsi"]):
            if signals["rsi"] < 30:
                buy_signals += 1
            elif signals["rsi"] > 70:
                sell_signals += 1
        
        # MACD signals
        if not np.isnan(signals["macd_histogram"]):
            if signals["macd_histogram"] > 0:
                buy_signals += 1
            else:
                sell_signals += 1
        
        # Bollinger Band signals
        if not np.isnan(signals["bb_position"]):
            if signals["bb_position"] < 0.2:
                buy_signals += 1
            elif signals["bb_position"] > 0.8:
                sell_signals += 1
        
        # Determine overall signal
        if buy_signals > sell_signals:
            signals["signal"] = "buy"
            signals["signal_strength"] = buy_signals / 3.0
        elif sell_signals > buy_signals:
            signals["signal"] = "sell"
            signals["signal_strength"] = sell_signals / 3.0
        else:
            signals["signal"] = "neutral"
            signals["signal_strength"] = 0.0
        
        return signals
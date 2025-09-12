"""
Crypto-Specific Risk Controls for CroweTrade AI Trading Infrastructure

This module provides specialized risk management for cryptocurrency trading with
unique considerations for digital assets including volatility, liquidity, and market structure.

Features:
- Crypto volatility-adjusted position sizing
- Cross-exchange price monitoring
- Liquidity risk assessment
- Flash loan and MEV protection
- Stablecoin depeg monitoring
- DeFi protocol risk controls
- Regulatory compliance tracking
"""

import asyncio
import math
import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

import numpy as np
from crowetrade.core.contracts import FeatureVector
from crowetrade.core.types import Signal
from crowetrade.risk.base_risk import RiskGuard


class CryptoRiskLevel(Enum):
    """Cryptocurrency risk levels"""
    LOW = "low"           # Major coins (BTC, ETH)
    MEDIUM = "medium"     # Top 20 altcoins  
    HIGH = "high"         # Small cap altcoins
    EXTREME = "extreme"   # Meme coins, new launches


@dataclass
class CryptoAssetProfile:
    """Risk profile for cryptocurrency asset"""
    symbol: str
    risk_level: CryptoRiskLevel
    market_cap_usd: Decimal
    daily_volume_usd: Decimal
    volatility_90d: Decimal
    max_position_size_usd: Decimal
    max_position_pct: Decimal  # % of portfolio
    min_liquidity_threshold: Decimal
    correlation_with_btc: Optional[Decimal] = None
    is_stablecoin: bool = False
    defi_protocol_risk: bool = False
    

@dataclass
class CryptoMarketRegime:
    """Current cryptocurrency market regime"""
    regime_type: str  # "bull", "bear", "crab", "crash"
    volatility_regime: str  # "low", "medium", "high", "extreme"
    correlation_regime: str  # "decoupled", "coupled", "synchronized"
    liquidity_regime: str  # "deep", "normal", "thin", "dry"
    fear_greed_index: int
    btc_dominance: Decimal
    total_market_cap_usd: Decimal
    confidence_score: Decimal  # 0-1


@dataclass
class CryptoRiskMetrics:
    """Risk metrics for cryptocurrency portfolio"""
    timestamp: datetime
    
    # Portfolio-level
    total_crypto_exposure_usd: Decimal
    crypto_pct_of_portfolio: Decimal
    portfolio_var_1d: Decimal
    portfolio_cvar_1d: Decimal
    max_drawdown_7d: Decimal
    
    # Asset concentration
    single_asset_max_pct: Decimal
    top_5_concentration_pct: Decimal
    
    # Liquidity metrics
    avg_bid_ask_spread_bps: Decimal
    weighted_volume_ratio: Decimal  # Current volume vs 30d average
    
    # Cross-exchange
    price_dispersion_max_bps: Decimal
    exchange_concentration_risk: Decimal
    
    # Crypto-specific
    stablecoin_exposure_pct: Decimal
    defi_protocol_exposure_pct: Decimal
    impermanent_loss_risk: Decimal


class CryptoRiskController:
    """Advanced cryptocurrency risk management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk limits
        self.max_crypto_portfolio_pct = Decimal(config.get("max_crypto_portfolio_pct", "30"))  # 30% max crypto
        self.max_single_crypto_pct = Decimal(config.get("max_single_crypto_pct", "10"))  # 10% max single coin
        self.crypto_var_limit_1d = Decimal(config.get("crypto_var_limit_1d", "5"))  # 5% daily VaR
        self.max_drawdown_crypto = Decimal(config.get("max_drawdown_crypto", "15"))  # 15% max drawdown
        
        # Volatility adjustments
        self.vol_target = Decimal(config.get("vol_target", "20"))  # 20% target volatility
        self.vol_lookback_days = config.get("vol_lookback_days", 30)
        
        # Asset profiles
        self.asset_profiles: Dict[str, CryptoAssetProfile] = {}
        self._initialize_asset_profiles()
        
        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.cross_exchange_prices: Dict[str, Dict[str, Decimal]] = defaultdict(dict)
        
        # Risk state
        self.current_regime = CryptoMarketRegime(
            regime_type="normal",
            volatility_regime="medium", 
            correlation_regime="normal",
            liquidity_regime="normal",
            fear_greed_index=50,
            btc_dominance=Decimal("45"),
            total_market_cap_usd=Decimal("2000000000000"),  # $2T
            confidence_score=Decimal("0.8")
        )
        
        self.risk_metrics = CryptoRiskMetrics(
            timestamp=datetime.now(timezone.utc),
            total_crypto_exposure_usd=Decimal("0"),
            crypto_pct_of_portfolio=Decimal("0"),
            portfolio_var_1d=Decimal("0"),
            portfolio_cvar_1d=Decimal("0"),
            max_drawdown_7d=Decimal("0"),
            single_asset_max_pct=Decimal("0"),
            top_5_concentration_pct=Decimal("0"),
            avg_bid_ask_spread_bps=Decimal("0"),
            weighted_volume_ratio=Decimal("1"),
            price_dispersion_max_bps=Decimal("0"),
            exchange_concentration_risk=Decimal("0"),
            stablecoin_exposure_pct=Decimal("0"),
            defi_protocol_exposure_pct=Decimal("0"),
            impermanent_loss_risk=Decimal("0")
        )
        
        # Circuit breakers
        self.flash_crash_threshold = Decimal("-20")  # 20% drop trigger
        self.volume_spike_threshold = Decimal("5")   # 5x volume spike
        self.correlation_spike_threshold = Decimal("0.95")  # 95% correlation alarm
        
        print("🏗️  Crypto Risk Controller initialized")
    
    def _initialize_asset_profiles(self):
        """Initialize cryptocurrency asset risk profiles"""
        profiles = [
            # Major cryptocurrencies (Low Risk)
            CryptoAssetProfile(
                symbol="BTC-USD", risk_level=CryptoRiskLevel.LOW,
                market_cap_usd=Decimal("1000000000000"), daily_volume_usd=Decimal("30000000000"),
                volatility_90d=Decimal("60"), max_position_size_usd=Decimal("5000000"),
                max_position_pct=Decimal("15"), min_liquidity_threshold=Decimal("10000000"),
                correlation_with_btc=Decimal("1.0")
            ),
            CryptoAssetProfile(
                symbol="ETH-USD", risk_level=CryptoRiskLevel.LOW,
                market_cap_usd=Decimal("400000000000"), daily_volume_usd=Decimal("15000000000"),
                volatility_90d=Decimal("70"), max_position_size_usd=Decimal("3000000"),
                max_position_pct=Decimal("12"), min_liquidity_threshold=Decimal("5000000"),
                correlation_with_btc=Decimal("0.8")
            ),
            
            # Large altcoins (Medium Risk)
            CryptoAssetProfile(
                symbol="ADA-USD", risk_level=CryptoRiskLevel.MEDIUM,
                market_cap_usd=Decimal("50000000000"), daily_volume_usd=Decimal("1000000000"),
                volatility_90d=Decimal("90"), max_position_size_usd=Decimal("1000000"),
                max_position_pct=Decimal("8"), min_liquidity_threshold=Decimal("500000"),
                correlation_with_btc=Decimal("0.7")
            ),
            CryptoAssetProfile(
                symbol="SOL-USD", risk_level=CryptoRiskLevel.MEDIUM,
                market_cap_usd=Decimal("40000000000"), daily_volume_usd=Decimal("2000000000"),
                volatility_90d=Decimal("100"), max_position_size_usd=Decimal("800000"),
                max_position_pct=Decimal("6"), min_liquidity_threshold=Decimal("400000"),
                correlation_with_btc=Decimal("0.6"), defi_protocol_risk=True
            ),
            
            # Stablecoins (Low Risk - different profile)
            CryptoAssetProfile(
                symbol="USDC-USD", risk_level=CryptoRiskLevel.LOW,
                market_cap_usd=Decimal("50000000000"), daily_volume_usd=Decimal("5000000000"),
                volatility_90d=Decimal("5"), max_position_size_usd=Decimal("10000000"),
                max_position_pct=Decimal("25"), min_liquidity_threshold=Decimal("1000000"),
                correlation_with_btc=Decimal("0.1"), is_stablecoin=True
            )
        ]
        
        for profile in profiles:
            self.asset_profiles[profile.symbol] = profile
    
    async def evaluate_crypto_signal(self, signal: Signal, current_positions: Dict[str, Decimal],
                                   portfolio_value: Decimal) -> Tuple[bool, str, Optional[Decimal]]:
        """
        Evaluate crypto trading signal against risk controls
        Returns: (approved, reason, adjusted_size)
        """
        try:
            symbol = signal.instrument
            profile = self.asset_profiles.get(symbol)
            
            if not profile:
                return False, f"Unknown crypto asset: {symbol}", None
            
            # Update market regime
            await self._update_market_regime()
            
            # Calculate proposed position size
            base_size = self._calculate_base_position_size(signal, portfolio_value, profile)
            
            # Apply risk adjustments
            size_after_vol_adj = self._apply_volatility_adjustment(base_size, symbol, profile)
            size_after_regime = self._apply_regime_adjustment(size_after_vol_adj, profile)
            size_after_liquidity = await self._apply_liquidity_adjustment(size_after_regime, symbol, profile)
            
            final_size = size_after_liquidity
            
            # Risk checks
            checks = [
                await self._check_position_limits(symbol, final_size, current_positions, portfolio_value),
                await self._check_concentration_limits(symbol, final_size, current_positions, portfolio_value),
                await self._check_liquidity_requirements(symbol, final_size, profile),
                await self._check_correlation_limits(symbol, final_size, current_positions),
                await self._check_market_regime_filters(profile),
                await self._check_price_dispersion(symbol),
                await self._check_stablecoin_depeg_risk(symbol) if profile.is_stablecoin else (True, "N/A")
            ]
            
            # Evaluate all checks
            failed_checks = [reason for passed, reason in checks if not passed]
            
            if failed_checks:
                return False, f"Risk check failed: {', '.join(failed_checks)}", None
            
            # Size adjustment warnings
            if final_size < base_size * Decimal("0.5"):
                return True, f"Position size reduced by {((1 - final_size/base_size) * 100):.1f}% due to risk controls", final_size
            
            return True, "All risk checks passed", final_size
            
        except Exception as e:
            return False, f"Risk evaluation error: {e}", None
    
    def _calculate_base_position_size(self, signal: Signal, portfolio_value: Decimal, 
                                    profile: CryptoAssetProfile) -> Decimal:
        """Calculate base position size before risk adjustments"""
        
        # Kelly criterion with crypto modifications
        expected_return = Decimal(str(signal.mu))
        signal_strength = Decimal(str(signal.prob_edge_pos))
        confidence = Decimal(str(1 - signal.sigma)) if signal.sigma else Decimal("0.8")
        
        # Estimate variance from volatility
        asset_vol = profile.volatility_90d / 100
        
        # Kelly fraction with confidence adjustment
        if asset_vol > 0:
            kelly_fraction = (expected_return - Decimal("0.02")) / (asset_vol ** 2)  # Risk-free rate 2%
            kelly_fraction *= signal_strength * confidence  # Confidence scaling
        else:
            kelly_fraction = Decimal("0.01")
        
        # Apply crypto-specific tempering
        crypto_tempering = Decimal("0.25") if profile.risk_level == CryptoRiskLevel.LOW else Decimal("0.15")
        tempered_fraction = kelly_fraction * crypto_tempering
        
        # Convert to dollar size
        max_size_from_portfolio = portfolio_value * profile.max_position_pct / 100
        max_size_from_limit = profile.max_position_size_usd
        
        target_size = portfolio_value * tempered_fraction
        final_size = min(target_size, max_size_from_portfolio, max_size_from_limit)
        
        return max(final_size, Decimal("0"))
    
    def _apply_volatility_adjustment(self, base_size: Decimal, symbol: str, 
                                   profile: CryptoAssetProfile) -> Decimal:
        """Adjust position size based on realized volatility"""
        
        # Get recent price data
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return base_size * Decimal("0.5")  # Conservative if no data
        
        prices = list(self.price_history[symbol])[-30:]  # Last 30 data points
        returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        
        if len(returns) < 10:
            return base_size * Decimal("0.7")
        
        # Calculate realized volatility
        realized_vol = Decimal(str(np.std(returns) * math.sqrt(365 * 24)))  # Annualized
        
        # Volatility targeting
        vol_ratio = self.vol_target / (realized_vol * 100) if realized_vol > 0 else Decimal("0.5")
        vol_ratio = min(max(vol_ratio, Decimal("0.1")), Decimal("2.0"))  # Clamp between 10% and 200%
        
        return base_size * vol_ratio
    
    def _apply_regime_adjustment(self, size: Decimal, profile: CryptoAssetProfile) -> Decimal:
        """Adjust position size based on market regime"""
        
        regime_multipliers = {
            ("bull", "low"): Decimal("1.2"),      # Bull + low vol = increase
            ("bull", "medium"): Decimal("1.0"),
            ("bull", "high"): Decimal("0.8"),
            ("bull", "extreme"): Decimal("0.5"),
            ("bear", "low"): Decimal("0.8"),
            ("bear", "medium"): Decimal("0.6"),
            ("bear", "high"): Decimal("0.4"),
            ("bear", "extreme"): Decimal("0.2"),
            ("crab", "low"): Decimal("1.0"),
            ("crab", "medium"): Decimal("0.9"),
            ("crab", "high"): Decimal("0.7"),
            ("crab", "extreme"): Decimal("0.5"),
            ("crash", "low"): Decimal("0.1"),     # Crash = minimal exposure
            ("crash", "medium"): Decimal("0.05"),
            ("crash", "high"): Decimal("0.02"),
            ("crash", "extreme"): Decimal("0.01")
        }
        
        key = (self.current_regime.regime_type, self.current_regime.volatility_regime)
        multiplier = regime_multipliers.get(key, Decimal("0.5"))
        
        # Additional adjustment for risk level
        risk_multipliers = {
            CryptoRiskLevel.LOW: Decimal("1.0"),
            CryptoRiskLevel.MEDIUM: Decimal("0.8"),
            CryptoRiskLevel.HIGH: Decimal("0.6"),
            CryptoRiskLevel.EXTREME: Decimal("0.3")
        }
        
        risk_multiplier = risk_multipliers[profile.risk_level]
        
        return size * multiplier * risk_multiplier
    
    async def _apply_liquidity_adjustment(self, size: Decimal, symbol: str, 
                                        profile: CryptoAssetProfile) -> Decimal:
        """Adjust position size based on liquidity conditions"""
        
        # Check recent volume
        if symbol in self.volume_history and len(self.volume_history[symbol]) > 0:
            recent_volume = sum(list(self.volume_history[symbol])[-10:]) / 10  # Average recent volume
            
            # Compare to minimum threshold
            if recent_volume < float(profile.min_liquidity_threshold):
                liquidity_ratio = recent_volume / float(profile.min_liquidity_threshold)
                liquidity_adjustment = Decimal(str(math.sqrt(liquidity_ratio)))  # Square root dampening
                size *= liquidity_adjustment
        
        # Market impact consideration
        # Don't trade more than 1% of recent volume in a single order
        if symbol in self.volume_history and len(self.volume_history[symbol]) > 0:
            recent_volume = sum(list(self.volume_history[symbol])[-5:]) / 5
            max_size_from_volume = Decimal(str(recent_volume * 0.01))  # 1% of volume
            size = min(size, max_size_from_volume)
        
        return size
    
    # Risk Check Methods
    
    async def _check_position_limits(self, symbol: str, size: Decimal, current_positions: Dict[str, Decimal],
                                   portfolio_value: Decimal) -> Tuple[bool, str]:
        """Check individual position limits"""
        
        profile = self.asset_profiles[symbol]
        current_position = current_positions.get(symbol, Decimal("0"))
        new_position = abs(current_position + size)
        
        # Check absolute size limit
        if new_position > profile.max_position_size_usd:
            return False, f"Position size exceeds limit: ${new_position} > ${profile.max_position_size_usd}"
        
        # Check percentage of portfolio limit
        position_pct = (new_position / portfolio_value) * 100
        if position_pct > profile.max_position_pct:
            return False, f"Position exceeds portfolio %: {position_pct:.1f}% > {profile.max_position_pct}%"
        
        return True, "Position limits OK"
    
    async def _check_concentration_limits(self, symbol: str, size: Decimal, current_positions: Dict[str, Decimal],
                                        portfolio_value: Decimal) -> Tuple[bool, str]:
        """Check portfolio concentration limits"""
        
        # Calculate total crypto exposure
        total_crypto_value = sum(abs(pos) for pos in current_positions.values() if self._is_crypto_symbol(pos))
        total_crypto_value += abs(size)  # Add new position
        
        crypto_pct = (total_crypto_value / portfolio_value) * 100 if portfolio_value > 0 else Decimal("0")
        
        if crypto_pct > self.max_crypto_portfolio_pct:
            return False, f"Total crypto exposure exceeds limit: {crypto_pct:.1f}% > {self.max_crypto_portfolio_pct}%"
        
        return True, "Concentration limits OK"
    
    async def _check_liquidity_requirements(self, symbol: str, size: Decimal, 
                                          profile: CryptoAssetProfile) -> Tuple[bool, str]:
        """Check liquidity requirements"""
        
        # Check if we have sufficient volume data
        if symbol not in self.volume_history or len(self.volume_history[symbol]) == 0:
            return False, f"Insufficient volume data for {symbol}"
        
        # Check minimum liquidity threshold
        recent_volume = sum(list(self.volume_history[symbol])[-5:]) / 5
        if recent_volume < float(profile.min_liquidity_threshold):
            return False, f"Insufficient liquidity: ${recent_volume:,.0f} < ${profile.min_liquidity_threshold}"
        
        return True, "Liquidity requirements met"
    
    async def _check_correlation_limits(self, symbol: str, size: Decimal, 
                                      current_positions: Dict[str, Decimal]) -> Tuple[bool, str]:
        """Check correlation-based risk limits"""
        
        # Get BTC correlation for this asset
        profile = self.asset_profiles.get(symbol)
        if not profile or not profile.correlation_with_btc:
            return True, "No correlation data"
        
        btc_correlation = profile.correlation_with_btc
        
        # If high correlation with BTC and we already have BTC exposure, reduce size
        btc_position = abs(current_positions.get("BTC-USD", Decimal("0")))
        
        if btc_correlation > Decimal("0.8") and btc_position > Decimal("1000000"):  # $1M BTC position
            return False, f"High correlation with existing BTC position ({btc_correlation:.2f})"
        
        return True, "Correlation limits OK"
    
    async def _check_market_regime_filters(self, profile: CryptoAssetProfile) -> Tuple[bool, str]:
        """Check market regime-based filters"""
        
        # During crash regime, only allow stablecoins
        if self.current_regime.regime_type == "crash" and not profile.is_stablecoin:
            return False, "Crash regime: only stablecoins allowed"
        
        # During extreme volatility, reduce exposure to high-risk assets
        if (self.current_regime.volatility_regime == "extreme" and 
            profile.risk_level in [CryptoRiskLevel.HIGH, CryptoRiskLevel.EXTREME]):
            return False, "Extreme volatility: high-risk assets restricted"
        
        # Check fear & greed index
        if self.current_regime.fear_greed_index < 10:  # Extreme fear
            if profile.risk_level != CryptoRiskLevel.LOW:
                return False, f"Extreme fear (FG: {self.current_regime.fear_greed_index}): only low-risk assets"
        
        return True, "Market regime filters passed"
    
    async def _check_price_dispersion(self, symbol: str) -> Tuple[bool, str]:
        """Check cross-exchange price dispersion"""
        
        if symbol not in self.cross_exchange_prices:
            return True, "No cross-exchange data"
        
        exchange_prices = list(self.cross_exchange_prices[symbol].values())
        
        if len(exchange_prices) < 2:
            return True, "Single exchange only"
        
        # Calculate price dispersion
        prices_float = [float(p) for p in exchange_prices]
        mean_price = statistics.mean(prices_float)
        max_deviation = max(abs(p - mean_price) / mean_price for p in prices_float)
        
        max_deviation_bps = max_deviation * 10000
        
        if max_deviation_bps > 500:  # 5% dispersion threshold
            return False, f"High price dispersion: {max_deviation_bps:.0f} bps > 500 bps"
        
        return True, f"Price dispersion OK: {max_deviation_bps:.0f} bps"
    
    async def _check_stablecoin_depeg_risk(self, symbol: str) -> Tuple[bool, str]:
        """Check stablecoin depeg risk"""
        
        if symbol not in self.price_history or len(self.price_history[symbol]) == 0:
            return True, "No price data"
        
        current_price = list(self.price_history[symbol])[-1]
        target_price = Decimal("1.0")  # Assume USD stablecoin
        
        deviation = abs(current_price - target_price) / target_price
        deviation_bps = deviation * 10000
        
        if deviation_bps > 100:  # 1% depeg threshold
            return False, f"Stablecoin depegged: {deviation_bps:.0f} bps from peg"
        
        return True, f"Stablecoin peg stable: {deviation_bps:.0f} bps deviation"
    
    # Market Monitoring
    
    async def _update_market_regime(self):
        """Update current cryptocurrency market regime"""
        
        try:
            # In production, this would analyze:
            # - BTC price action and volatility
            # - Total market cap changes
            # - Fear & Greed index from APIs
            # - Cross-asset correlations
            # - Funding rates
            # - Open interest
            
            # Placeholder logic for demo
            if len(self.price_history.get("BTC-USD", [])) > 20:
                btc_prices = list(self.price_history["BTC-USD"])[-20:]
                recent_change = (btc_prices[-1] - btc_prices[0]) / btc_prices[0]
                
                if recent_change > Decimal("0.1"):
                    regime_type = "bull"
                elif recent_change < Decimal("-0.1"):
                    regime_type = "bear"
                elif recent_change < Decimal("-0.2"):
                    regime_type = "crash"
                else:
                    regime_type = "crab"
                
                # Update regime
                self.current_regime.regime_type = regime_type
            
        except Exception as e:
            print(f"⚠️ Market regime update failed: {e}")
    
    def update_price_data(self, symbol: str, price: Decimal, volume: Decimal, exchange: str = ""):
        """Update price and volume data for risk calculations"""
        
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        
        if exchange:
            self.cross_exchange_prices[symbol][exchange] = price
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency"""
        crypto_suffixes = ["-USD", "-USDT", "-USDC", "-EUR"]
        crypto_prefixes = ["BTC", "ETH", "ADA", "SOL", "MATIC", "AVAX", "DOT", "LINK"]
        
        for prefix in crypto_prefixes:
            if symbol.startswith(prefix):
                return True
        
        for suffix in crypto_suffixes:
            if symbol.endswith(suffix) and not symbol.startswith(("EUR", "GBP", "JPY")):
                return True
        
        return False
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        await self._update_market_regime()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_regime": {
                "type": self.current_regime.regime_type,
                "volatility": self.current_regime.volatility_regime,
                "fear_greed": self.current_regime.fear_greed_index,
                "btc_dominance": float(self.current_regime.btc_dominance)
            },
            "risk_limits": {
                "max_crypto_portfolio_pct": float(self.max_crypto_portfolio_pct),
                "max_single_crypto_pct": float(self.max_single_crypto_pct),
                "crypto_var_limit_1d": float(self.crypto_var_limit_1d)
            },
            "current_metrics": {
                "total_crypto_exposure": float(self.risk_metrics.total_crypto_exposure_usd),
                "crypto_portfolio_pct": float(self.risk_metrics.crypto_pct_of_portfolio),
                "portfolio_var_1d": float(self.risk_metrics.portfolio_var_1d),
                "max_drawdown_7d": float(self.risk_metrics.max_drawdown_7d)
            },
            "asset_count": len(self.asset_profiles)
        }


# Factory function
def create_crypto_risk_controller(config: Dict[str, Any]) -> CryptoRiskController:
    """Factory function to create crypto risk controller"""
    return CryptoRiskController(config)

"""Market regime detection using statistical and ML methods"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_QUIET = "bull_quiet"      # Uptrend with low volatility
    BULL_VOLATILE = "bull_volatile" # Uptrend with high volatility
    BEAR_QUIET = "bear_quiet"      # Downtrend with low volatility
    BEAR_VOLATILE = "bear_volatile" # Downtrend with high volatility
    RANGING = "ranging"             # Sideways market
    TRANSITIONING = "transitioning" # Regime change in progress


@dataclass
class RegimeFeatures:
    """Features used for regime detection"""
    returns_mean: float          # Average returns
    returns_std: float           # Volatility
    returns_skew: float          # Skewness
    returns_kurt: float          # Kurtosis
    trend_strength: float        # Trend indicator (-1 to 1)
    volatility_ratio: float      # Current vol / historical vol
    volume_ratio: float          # Current volume / avg volume
    correlation_breakdown: float  # Cross-asset correlation change
    vix_level: Optional[float] = None  # VIX or volatility index
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        features = [
            self.returns_mean,
            self.returns_std,
            self.returns_skew,
            self.returns_kurt,
            self.trend_strength,
            self.volatility_ratio,
            self.volume_ratio,
            self.correlation_breakdown
        ]
        if self.vix_level is not None:
            features.append(self.vix_level)
        return np.array(features)


@dataclass
class RegimeDetectionResult:
    """Result from regime detection"""
    current_regime: MarketRegime
    regime_probability: float
    regime_scores: Dict[MarketRegime, float]
    transition_probability: float
    features: RegimeFeatures
    timestamp: float


class RegimeDetector:
    """
    Detects market regimes using a combination of:
    - Statistical measures (volatility, skewness, kurtosis)
    - Trend analysis
    - Hidden Markov Models or Gaussian Mixture Models
    - Correlation breakdown detection
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        vol_threshold_low: float = 0.15,
        vol_threshold_high: float = 0.25,
        trend_threshold: float = 0.02,
        n_regimes: int = 4,
        use_gmm: bool = True
    ):
        self.lookback_window = lookback_window
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.trend_threshold = trend_threshold
        self.n_regimes = n_regimes
        self.use_gmm = use_gmm
        
        # ML models
        self.gmm_model: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # History for regime tracking
        self.regime_history: List[MarketRegime] = []
        self.feature_history: List[RegimeFeatures] = []
        
        # Transition matrix for regime changes
        self.transition_matrix = self._initialize_transition_matrix()
    
    def detect_regime(
        self,
        returns: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        correlations: Optional[np.ndarray] = None,
        vix: Optional[float] = None
    ) -> RegimeDetectionResult:
        """
        Detect current market regime
        
        Args:
            returns: Array of returns (at least lookback_window length)
            volumes: Optional volume data
            correlations: Optional correlation matrix
            vix: Optional volatility index value
        
        Returns:
            RegimeDetectionResult with regime classification
        """
        
        # Extract features
        features = self._extract_features(returns, volumes, correlations, vix)
        
        # Detect regime using rule-based or ML approach
        if self.use_gmm and self.is_fitted:
            regime, scores = self._detect_regime_ml(features)
        else:
            regime, scores = self._detect_regime_rules(features)
        
        # Calculate transition probability
        transition_prob = self._calculate_transition_probability(regime)
        
        # Update history
        self.regime_history.append(regime)
        self.feature_history.append(features)
        
        # Keep history bounded
        if len(self.regime_history) > 1000:
            self.regime_history.pop(0)
            self.feature_history.pop(0)
        
        return RegimeDetectionResult(
            current_regime=regime,
            regime_probability=scores[regime],
            regime_scores=scores,
            transition_probability=transition_prob,
            features=features,
            timestamp=np.datetime64('now').astype(float)
        )
    
    def train_gmm(
        self,
        historical_returns: np.ndarray,
        historical_volumes: Optional[np.ndarray] = None,
        n_samples: int = 1000
    ) -> None:
        """
        Train Gaussian Mixture Model on historical data
        
        Args:
            historical_returns: Historical return series
            historical_volumes: Optional historical volumes
            n_samples: Number of samples to generate for training
        """
        
        if len(historical_returns) < self.lookback_window * 2:
            logger.warning("Insufficient data for GMM training")
            return
        
        # Generate training features
        training_features = []
        
        for i in range(self.lookback_window, len(historical_returns)):
            window_returns = historical_returns[i-self.lookback_window:i]
            window_volumes = None
            if historical_volumes is not None:
                window_volumes = historical_volumes[i-self.lookback_window:i]
            
            features = self._extract_features(window_returns, window_volumes)
            training_features.append(features.to_array())
        
        if len(training_features) < 10:
            logger.warning("Too few samples for GMM training")
            return
        
        # Convert to array and scale
        X = np.array(training_features)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train GMM
        self.gmm_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.gmm_model.fit(X_scaled)
        
        self.is_fitted = True
        logger.info(f"GMM trained with {len(X)} samples, {self.n_regimes} components")
    
    def _extract_features(
        self,
        returns: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        correlations: Optional[np.ndarray] = None,
        vix: Optional[float] = None
    ) -> RegimeFeatures:
        """Extract regime detection features"""
        
        # Use recent window
        recent_returns = returns[-self.lookback_window:] if len(returns) > self.lookback_window else returns
        
        # Basic statistics
        returns_mean = np.mean(recent_returns)
        returns_std = np.std(recent_returns)
        returns_skew = stats.skew(recent_returns)
        returns_kurt = stats.kurtosis(recent_returns)
        
        # Trend strength (simple linear regression slope)
        x = np.arange(len(recent_returns))
        slope, _ = np.polyfit(x, recent_returns, 1)
        trend_strength = np.clip(slope / (returns_std + 1e-8), -1, 1)
        
        # Volatility ratio (recent vs historical)
        if len(returns) > self.lookback_window * 2:
            hist_std = np.std(returns[-self.lookback_window*2:-self.lookback_window])
            volatility_ratio = returns_std / (hist_std + 1e-8)
        else:
            volatility_ratio = 1.0
        
        # Volume ratio
        volume_ratio = 1.0
        if volumes is not None and len(volumes) > self.lookback_window:
            recent_volume = np.mean(volumes[-self.lookback_window:])
            hist_volume = np.mean(volumes[:-self.lookback_window])
            volume_ratio = recent_volume / (hist_volume + 1e-8)
        
        # Correlation breakdown (simplified)
        correlation_breakdown = 0.0
        if correlations is not None and len(correlations) > 1:
            # Measure correlation instability
            recent_corr = correlations[-1] if len(correlations.shape) == 1 else np.mean(correlations[-1])
            hist_corr = np.mean(correlations[:-1])
            correlation_breakdown = abs(recent_corr - hist_corr)
        
        return RegimeFeatures(
            returns_mean=returns_mean,
            returns_std=returns_std,
            returns_skew=returns_skew,
            returns_kurt=returns_kurt,
            trend_strength=trend_strength,
            volatility_ratio=volatility_ratio,
            volume_ratio=volume_ratio,
            correlation_breakdown=correlation_breakdown,
            vix_level=vix
        )
    
    def _detect_regime_rules(
        self,
        features: RegimeFeatures
    ) -> Tuple[MarketRegime, Dict[MarketRegime, float]]:
        """Rule-based regime detection"""
        
        scores = {regime: 0.0 for regime in MarketRegime}
        
        # Classify based on trend and volatility
        is_bullish = features.trend_strength > self.trend_threshold
        is_bearish = features.trend_strength < -self.trend_threshold
        is_ranging = abs(features.trend_strength) <= self.trend_threshold
        
        is_low_vol = features.returns_std < self.vol_threshold_low
        is_high_vol = features.returns_std > self.vol_threshold_high
        
        # Assign scores based on rules
        if is_bullish:
            if is_low_vol:
                regime = MarketRegime.BULL_QUIET
                scores[MarketRegime.BULL_QUIET] = 0.8
                scores[MarketRegime.BULL_VOLATILE] = 0.2
            else:
                regime = MarketRegime.BULL_VOLATILE
                scores[MarketRegime.BULL_VOLATILE] = 0.8
                scores[MarketRegime.BULL_QUIET] = 0.2
        elif is_bearish:
            if is_low_vol:
                regime = MarketRegime.BEAR_QUIET
                scores[MarketRegime.BEAR_QUIET] = 0.8
                scores[MarketRegime.BEAR_VOLATILE] = 0.2
            else:
                regime = MarketRegime.BEAR_VOLATILE
                scores[MarketRegime.BEAR_VOLATILE] = 0.8
                scores[MarketRegime.BEAR_QUIET] = 0.2
        else:
            regime = MarketRegime.RANGING
            scores[MarketRegime.RANGING] = 0.7
            scores[MarketRegime.TRANSITIONING] = 0.3
        
        # Check for transition signals
        if features.correlation_breakdown > 0.3 or features.volatility_ratio > 2.0:
            scores[MarketRegime.TRANSITIONING] += 0.3
            # Normalize scores
            total = sum(scores.values())
            scores = {k: v/total for k, v in scores.items()}
            
            if scores[MarketRegime.TRANSITIONING] > 0.4:
                regime = MarketRegime.TRANSITIONING
        
        return regime, scores
    
    def _detect_regime_ml(
        self,
        features: RegimeFeatures
    ) -> Tuple[MarketRegime, Dict[MarketRegime, float]]:
        """ML-based regime detection using GMM"""
        
        if not self.is_fitted or self.gmm_model is None:
            return self._detect_regime_rules(features)
        
        # Prepare features
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probs = self.gmm_model.predict_proba(X_scaled)[0]
        
        # Map GMM components to regimes
        # This mapping would be learned during training
        regime_mapping = {
            0: MarketRegime.BULL_QUIET,
            1: MarketRegime.BULL_VOLATILE,
            2: MarketRegime.BEAR_QUIET,
            3: MarketRegime.BEAR_VOLATILE
        }
        
        if self.n_regimes > 4:
            regime_mapping[4] = MarketRegime.RANGING
        if self.n_regimes > 5:
            regime_mapping[5] = MarketRegime.TRANSITIONING
        
        # Create scores
        scores = {}
        for component_idx, prob in enumerate(probs):
            if component_idx in regime_mapping:
                regime = regime_mapping[component_idx]
                scores[regime] = prob
        
        # Fill missing regimes with zero probability
        for regime in MarketRegime:
            if regime not in scores:
                scores[regime] = 0.0
        
        # Get highest probability regime
        regime = max(scores.items(), key=lambda x: x[1])[0]
        
        return regime, scores
    
    def _calculate_transition_probability(
        self,
        current_regime: MarketRegime
    ) -> float:
        """Calculate probability of regime transition"""
        
        if len(self.regime_history) < 2:
            return 0.0
        
        # Check if regime changed
        prev_regime = self.regime_history[-2] if len(self.regime_history) > 1 else current_regime
        
        if prev_regime != current_regime:
            return 0.9  # High probability we're transitioning
        
        # Look at feature changes to predict transition
        if len(self.feature_history) < 2:
            return 0.1
        
        current_features = self.feature_history[-1]
        prev_features = self.feature_history[-2]
        
        # Calculate feature divergence
        vol_change = abs(current_features.returns_std - prev_features.returns_std) / (prev_features.returns_std + 1e-8)
        trend_change = abs(current_features.trend_strength - prev_features.trend_strength)
        corr_change = current_features.correlation_breakdown
        
        # Combine signals
        transition_signal = (
            vol_change * 0.3 +
            trend_change * 0.3 +
            corr_change * 0.4
        )
        
        return min(transition_signal, 0.95)
    
    def _initialize_transition_matrix(self) -> np.ndarray:
        """Initialize regime transition probability matrix"""
        
        # Create transition matrix with higher diagonal (regime persistence)
        n = len(MarketRegime)
        matrix = np.full((n, n), 0.05)  # Low transition probability
        np.fill_diagonal(matrix, 0.7)   # High persistence probability
        
        # Normalize rows
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        
        return matrix
    
    def get_regime_recommendations(
        self,
        regime: MarketRegime
    ) -> Dict[str, Any]:
        """Get portfolio recommendations based on regime"""
        
        recommendations = {
            'risk_level': 'medium',
            'position_sizing': 1.0,
            'asset_allocation': {},
            'strategies': []
        }
        
        if regime == MarketRegime.BULL_QUIET:
            recommendations.update({
                'risk_level': 'high',
                'position_sizing': 1.2,
                'asset_allocation': {'equity': 0.8, 'bonds': 0.1, 'cash': 0.1},
                'strategies': ['momentum', 'growth', 'leverage']
            })
        
        elif regime == MarketRegime.BULL_VOLATILE:
            recommendations.update({
                'risk_level': 'medium-high',
                'position_sizing': 1.0,
                'asset_allocation': {'equity': 0.6, 'bonds': 0.2, 'cash': 0.2},
                'strategies': ['momentum', 'volatility_selling']
            })
        
        elif regime == MarketRegime.BEAR_QUIET:
            recommendations.update({
                'risk_level': 'low',
                'position_sizing': 0.6,
                'asset_allocation': {'equity': 0.3, 'bonds': 0.5, 'cash': 0.2},
                'strategies': ['defensive', 'quality', 'short_bias']
            })
        
        elif regime == MarketRegime.BEAR_VOLATILE:
            recommendations.update({
                'risk_level': 'very_low',
                'position_sizing': 0.4,
                'asset_allocation': {'equity': 0.2, 'bonds': 0.4, 'cash': 0.4},
                'strategies': ['defensive', 'hedging', 'cash_preservation']
            })
        
        elif regime == MarketRegime.RANGING:
            recommendations.update({
                'risk_level': 'medium',
                'position_sizing': 0.8,
                'asset_allocation': {'equity': 0.5, 'bonds': 0.3, 'cash': 0.2},
                'strategies': ['mean_reversion', 'pairs_trading', 'arbitrage']
            })
        
        elif regime == MarketRegime.TRANSITIONING:
            recommendations.update({
                'risk_level': 'low-medium',
                'position_sizing': 0.7,
                'asset_allocation': {'equity': 0.4, 'bonds': 0.3, 'cash': 0.3},
                'strategies': ['neutral', 'wait_and_see', 'reduce_exposure']
            })
        
        return recommendations
"""Tests for market regime detection"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from crowetrade.ml.regime_detector import (
    RegimeDetector,
    MarketRegime,
    RegimeFeatures
)


class TestRegimeDetector:
    """Test regime detection functionality"""
    
    @pytest.fixture
    def detector(self):
        """Create a regime detector"""
        return RegimeDetector(
            lookback_window=20,
            vol_threshold_low=0.10,
            vol_threshold_high=0.20,
            trend_threshold=0.01
        )
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data"""
        np.random.seed(42)
        
        # Bull quiet: uptrend with low vol
        bull_quiet = np.random.normal(0.001, 0.08, 100)
        bull_quiet = np.cumsum(bull_quiet) * 0.001 + np.linspace(0, 0.1, 100)
        
        # Bear volatile: downtrend with high vol
        bear_volatile = np.random.normal(-0.002, 0.25, 100)
        bear_volatile = np.cumsum(bear_volatile) * 0.001 - np.linspace(0, 0.15, 100)
        
        # Ranging: sideways with medium vol
        ranging = np.random.normal(0, 0.15, 100)
        
        return {
            'bull_quiet': bull_quiet,
            'bear_volatile': bear_volatile,
            'ranging': ranging
        }
    
    def test_feature_extraction(self, detector):
        """Test feature extraction from returns"""
        
        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.15, 60)
        volumes = np.random.uniform(1e6, 2e6, 60)
        
        features = detector._extract_features(returns, volumes)
        
        assert features.returns_mean is not None
        assert features.returns_std > 0
        assert -3 < features.returns_skew < 3  # Reasonable skewness
        assert features.trend_strength >= -1 and features.trend_strength <= 1
        assert features.volatility_ratio > 0
        assert features.volume_ratio > 0
    
    def test_bull_quiet_detection(self, detector, sample_returns):
        """Test detection of bull quiet regime"""
        
        returns = sample_returns['bull_quiet']
        result = detector.detect_regime(returns)
        
        # Should detect bullish regime
        assert result.current_regime in [
            MarketRegime.BULL_QUIET,
            MarketRegime.BULL_VOLATILE
        ]
        
        # Check features indicate uptrend
        assert result.features.trend_strength > 0
        assert result.features.returns_mean > 0
    
    def test_bear_volatile_detection(self, detector, sample_returns):
        """Test detection of bear volatile regime"""
        
        returns = sample_returns['bear_volatile']
        result = detector.detect_regime(returns)
        
        # Should detect bearish regime
        assert result.current_regime in [
            MarketRegime.BEAR_QUIET,
            MarketRegime.BEAR_VOLATILE
        ]
        
        # Check features indicate downtrend and high vol
        assert result.features.trend_strength < 0
        assert result.features.returns_std > 0.15
    
    def test_ranging_detection(self, detector, sample_returns):
        """Test detection of ranging market"""
        
        returns = sample_returns['ranging']
        result = detector.detect_regime(returns)
        
        # Should detect ranging or transitioning
        assert result.current_regime in [
            MarketRegime.RANGING,
            MarketRegime.TRANSITIONING
        ]
        
        # Trend should be near zero
        assert abs(result.features.trend_strength) < 0.5
    
    def test_regime_history_tracking(self, detector):
        """Test that regime history is tracked"""
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.15, 100)
        
        # Detect regime multiple times
        for i in range(3):
            window = returns[i*20:(i+1)*20+40]
            detector.detect_regime(window)
        
        assert len(detector.regime_history) == 3
        assert len(detector.feature_history) == 3
    
    def test_transition_probability(self, detector):
        """Test regime transition probability calculation"""
        
        np.random.seed(42)
        
        # Stable returns
        stable_returns = np.random.normal(0.001, 0.10, 60)
        result1 = detector.detect_regime(stable_returns)
        
        # Same regime should have low transition probability
        result2 = detector.detect_regime(stable_returns)
        assert result2.transition_probability < 0.5
        
        # Volatile returns (regime change)
        volatile_returns = np.random.normal(-0.005, 0.30, 60)
        result3 = detector.detect_regime(volatile_returns)
        
        # Different regime should have high transition probability
        if result3.current_regime != result2.current_regime:
            assert result3.transition_probability > 0.5
    
    def test_gmm_training(self, detector):
        """Test GMM model training"""
        
        # Generate historical data
        np.random.seed(42)
        historical_returns = np.concatenate([
            np.random.normal(0.001, 0.10, 200),  # Bull quiet
            np.random.normal(-0.002, 0.25, 200),  # Bear volatile
            np.random.normal(0, 0.15, 200),       # Ranging
        ])
        
        historical_volumes = np.random.uniform(1e6, 2e6, 600)
        
        # Train GMM
        detector.train_gmm(historical_returns, historical_volumes)
        
        assert detector.is_fitted
        assert detector.gmm_model is not None
        
        # Test detection with trained model
        test_returns = np.random.normal(0.001, 0.12, 60)
        result = detector.detect_regime(test_returns)
        
        assert result.current_regime is not None
        assert sum(result.regime_scores.values()) == pytest.approx(1.0, abs=0.01)
    
    def test_regime_recommendations(self, detector):
        """Test portfolio recommendations for each regime"""
        
        recommendations = detector.get_regime_recommendations(MarketRegime.BULL_QUIET)
        assert recommendations['risk_level'] == 'high'
        assert recommendations['position_sizing'] > 1.0
        assert 'momentum' in recommendations['strategies']
        
        recommendations = detector.get_regime_recommendations(MarketRegime.BEAR_VOLATILE)
        assert recommendations['risk_level'] == 'very_low'
        assert recommendations['position_sizing'] < 0.5
        assert 'defensive' in recommendations['strategies']
        
        recommendations = detector.get_regime_recommendations(MarketRegime.RANGING)
        assert 'mean_reversion' in recommendations['strategies']
    
    def test_volume_impact(self, detector):
        """Test that volume changes affect regime detection"""
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.15, 60)
        
        # Normal volume
        normal_volume = np.ones(60) * 1e6
        result_normal = detector.detect_regime(returns, normal_volume)
        
        # Spike in volume (could indicate regime change)
        spike_volume = np.ones(60) * 1e6
        spike_volume[-10:] = 3e6
        result_spike = detector.detect_regime(returns, spike_volume)
        
        # Volume spike should increase transition probability
        assert result_spike.features.volume_ratio > result_normal.features.volume_ratio
    
    def test_correlation_breakdown(self, detector):
        """Test correlation breakdown detection"""
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.15, 60)
        
        # Stable correlations
        stable_corr = np.ones(60) * 0.7
        result_stable = detector.detect_regime(returns, correlations=stable_corr)
        
        # Correlation breakdown
        breakdown_corr = np.ones(60) * 0.7
        breakdown_corr[-5:] = 0.2  # Sudden correlation change
        result_breakdown = detector.detect_regime(returns, correlations=breakdown_corr)
        
        assert result_breakdown.features.correlation_breakdown > result_stable.features.correlation_breakdown
    
    def test_vix_integration(self, detector):
        """Test VIX level integration"""
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.15, 60)
        
        # Low VIX
        result_low_vix = detector.detect_regime(returns, vix=12.0)
        assert result_low_vix.features.vix_level == 12.0
        
        # High VIX (fear gauge)
        result_high_vix = detector.detect_regime(returns, vix=35.0)
        assert result_high_vix.features.vix_level == 35.0
    
    def test_feature_array_conversion(self):
        """Test RegimeFeatures to array conversion"""
        
        features = RegimeFeatures(
            returns_mean=0.001,
            returns_std=0.15,
            returns_skew=0.2,
            returns_kurt=3.0,
            trend_strength=0.5,
            volatility_ratio=1.2,
            volume_ratio=1.1,
            correlation_breakdown=0.1,
            vix_level=20.0
        )
        
        array = features.to_array()
        assert len(array) == 9
        assert array[0] == 0.001
        assert array[-1] == 20.0
        
        # Without VIX
        features_no_vix = RegimeFeatures(
            returns_mean=0.001,
            returns_std=0.15,
            returns_skew=0.2,
            returns_kurt=3.0,
            trend_strength=0.5,
            volatility_ratio=1.2,
            volume_ratio=1.1,
            correlation_breakdown=0.1
        )
        
        array_no_vix = features_no_vix.to_array()
        assert len(array_no_vix) == 8
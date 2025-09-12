"""Tests for Prometheus metrics collection"""
import pytest
import time
from unittest.mock import Mock, patch

from crowetrade.monitoring.prometheus_metrics import (
    PrometheusMetrics,
    MetricType,
    initialize_metrics,
    timing_decorator,
    counter_decorator
)


class TestPrometheusMetrics:
    """Test Prometheus metrics functionality"""
    
    @pytest.fixture
    def metrics(self):
        """Create a metrics instance for testing"""
        return PrometheusMetrics(
            enable_http_server=False,  # Don't start HTTP server in tests
            prefix="test"
        )
    
    def test_metric_registration(self, metrics):
        """Test metric registration"""
        
        # Register counter
        counter = metrics.register_counter(
            'test_counter',
            'Test counter metric',
            ['label1', 'label2']
        )
        assert counter is not None
        
        # Register gauge
        gauge = metrics.register_gauge(
            'test_gauge',
            'Test gauge metric',
            ['label1']
        )
        assert gauge is not None
        
        # Register histogram
        histogram = metrics.register_histogram(
            'test_histogram',
            'Test histogram metric',
            ['label1'],
            buckets=[0.1, 0.5, 1.0, 5.0]
        )
        assert histogram is not None
    
    def test_counter_operations(self, metrics):
        """Test counter metric operations"""
        
        metrics.increment_counter('trades_total', 1, {
            'symbol': 'AAPL',
            'side': 'buy'
        })
        
        metrics.increment_counter('trades_total', 2, {
            'symbol': 'GOOGL',
            'side': 'sell'
        })
        
        # Verify counter exists
        counter = metrics.get_metric('trades_total')
        assert counter is not None
    
    def test_gauge_operations(self, metrics):
        """Test gauge metric operations"""
        
        metrics.set_gauge('portfolio_value_usd', 1000000.0, {
            'strategy': 'momentum'
        })
        
        metrics.set_gauge('portfolio_value_usd', 1100000.0, {
            'strategy': 'momentum'
        })
        
        # Verify gauge exists
        gauge = metrics.get_metric('portfolio_value_usd')
        assert gauge is not None
    
    def test_histogram_operations(self, metrics):
        """Test histogram metric operations"""
        
        # Observe values
        for value in [0.1, 0.5, 1.2, 2.5]:
            metrics.observe_histogram('trade_execution_duration_seconds', value, {
                'algorithm': 'twap'
            })
        
        # Test timing context manager
        with metrics.time_histogram('trade_execution_duration_seconds', {'algorithm': 'vwap'}):
            time.sleep(0.01)  # Brief sleep
        
        histogram = metrics.get_metric('trade_execution_duration_seconds')
        assert histogram is not None
    
    def test_trade_recording(self, metrics):
        """Test comprehensive trade recording"""
        
        metrics.record_trade(
            symbol='AAPL',
            side='buy',
            quantity=1000,
            price=150.0,
            strategy='momentum',
            venue='NYSE',
            execution_duration=0.5,
            slippage_bps=2.5,
            pnl=500.0
        )
        
        # Verify multiple metrics were updated
        assert metrics.get_metric('trades_total') is not None
        assert metrics.get_metric('trade_size_shares') is not None
        assert metrics.get_metric('trade_execution_duration_seconds') is not None
        assert metrics.get_metric('execution_slippage_bps') is not None
    
    def test_portfolio_recording(self, metrics):
        """Test portfolio metrics recording"""
        
        positions = {
            'AAPL': 1000,
            'GOOGL': 500,
            'MSFT': 750
        }
        
        weights = {
            'AAPL': 0.4,
            'GOOGL': 0.3,
            'MSFT': 0.3
        }
        
        metrics.record_portfolio_update(
            strategy='balanced',
            total_value=1000000.0,
            positions=positions,
            weights=weights,
            drawdown=0.05
        )
        
        assert metrics.get_metric('portfolio_value_usd') is not None
        assert metrics.get_metric('portfolio_positions') is not None
        assert metrics.get_metric('portfolio_weights') is not None
        assert metrics.get_metric('portfolio_drawdown') is not None
    
    def test_risk_metrics_recording(self, metrics):
        """Test risk metrics recording"""
        
        metrics.record_risk_metrics(
            strategy='aggressive',
            var_95=50000.0,
            gross_exposure=1500000.0
        )
        
        metrics.record_risk_breach(
            limit_type='position_size',
            strategy='aggressive'
        )
        
        assert metrics.get_metric('risk_var_95') is not None
        assert metrics.get_metric('risk_exposure_gross') is not None
        assert metrics.get_metric('risk_breaches_total') is not None
    
    def test_market_data_recording(self, metrics):
        """Test market data metrics"""
        
        metrics.record_market_data_update(
            symbol='AAPL',
            data_type='price',
            source='bloomberg',
            latency_ms=15.5
        )
        
        assert metrics.get_metric('market_data_updates_total') is not None
        assert metrics.get_metric('market_data_latency_ms') is not None
    
    def test_agent_status_recording(self, metrics):
        """Test agent status metrics"""
        
        metrics.record_agent_status(
            agent_id='portfolio_manager_1',
            agent_type='portfolio_manager',
            is_healthy=True
        )
        
        metrics.record_agent_status(
            agent_id='risk_guard_1',
            agent_type='risk_guard',
            is_healthy=False
        )
        
        agent_status = metrics.get_metric('agent_status')
        assert agent_status is not None
    
    def test_event_processing_recording(self, metrics):
        """Test event processing metrics"""
        
        metrics.record_event_processed(
            event_type='SIGNAL',
            source='signal_generator',
            status='success',
            processing_duration=0.025
        )
        
        assert metrics.get_metric('events_processed_total') is not None
        assert metrics.get_metric('event_processing_duration_seconds') is not None
    
    def test_model_metrics_recording(self, metrics):
        """Test ML model metrics"""
        
        metrics.record_model_prediction(
            model_name='regime_detector',
            model_version='v1.2.0',
            prediction_type='regime_classification',
            accuracy=0.85,
            inference_duration_ms=12.5
        )
        
        assert metrics.get_metric('model_predictions_total') is not None
        assert metrics.get_metric('model_accuracy') is not None
        assert metrics.get_metric('model_inference_duration_ms') is not None
    
    def test_timing_decorator(self, metrics):
        """Test timing decorator functionality"""
        
        # Initialize global metrics for decorator
        initialize_metrics(enable_http_server=False)
        
        @timing_decorator('function_duration_seconds')
        def sample_function():
            time.sleep(0.01)
            return "result"
        
        result = sample_function()
        assert result == "result"
        
        # Test with error
        @timing_decorator('error_function_duration_seconds')
        def error_function():
            time.sleep(0.01)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            error_function()
    
    def test_counter_decorator(self, metrics):
        """Test counter decorator functionality"""
        
        initialize_metrics(enable_http_server=False)
        
        @counter_decorator('function_calls_total')
        def sample_function(x):
            if x < 0:
                raise ValueError("Negative input")
            return x * 2
        
        # Successful calls
        assert sample_function(5) == 10
        assert sample_function(3) == 6
        
        # Error call
        with pytest.raises(ValueError):
            sample_function(-1)
    
    def test_metrics_text_export(self, metrics):
        """Test Prometheus text format export"""
        
        # Record some metrics
        metrics.increment_counter('test_counter', 1, {'label': 'value'})
        metrics.set_gauge('test_gauge', 42.0)
        
        # Get metrics text
        metrics_text = metrics.get_metrics_text()
        assert isinstance(metrics_text, str)
        # Should contain metric information or warning if prometheus not available
        assert len(metrics_text) > 0
    
    def test_metrics_dict_export(self, metrics):
        """Test metrics dictionary export"""
        
        metrics_dict = metrics.get_metrics_dict()
        assert isinstance(metrics_dict, dict)
        
        # Should contain registered metrics
        assert len(metrics_dict) > 0
        
        # Each metric should have type info
        for metric_name, metric_info in metrics_dict.items():
            assert 'type' in metric_info
    
    def test_duplicate_registration(self, metrics):
        """Test that duplicate metric registration returns same instance"""
        
        counter1 = metrics.register_counter('duplicate_test', 'Test metric')
        counter2 = metrics.register_counter('duplicate_test', 'Test metric')
        
        assert counter1 is counter2
    
    def test_thread_safety(self, metrics):
        """Test thread safety of metrics operations"""
        import threading
        
        def worker():
            for i in range(100):
                metrics.increment_counter('thread_test', 1, {'worker': str(threading.current_thread().ident)})
                metrics.set_gauge('thread_gauge', i)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        assert metrics.get_metric('thread_test') is not None
        assert metrics.get_metric('thread_gauge') is not None
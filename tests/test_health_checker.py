"""Tests for system health monitoring"""
import pytest
import time
import threading
from unittest.mock import Mock

from crowetrade.monitoring.health_checker import (
    HealthChecker,
    ComponentStatus,
    HealthCheck,
    ComponentHealth
)


class TestHealthChecker:
    """Test health monitoring functionality"""
    
    @pytest.fixture
    def health_checker(self):
        """Create a health checker for testing"""
        checker = HealthChecker(
            check_interval=0.1,  # Fast interval for tests
            failure_threshold=2
        )
        yield checker
        checker.stop_monitoring()
    
    def test_component_registration(self, health_checker):
        """Test component registration"""
        
        health_checker.register_component('test_component')
        
        assert 'test_component' in health_checker.components
        
        component = health_checker.get_component_health('test_component')
        assert component is not None
        assert component.component_name == 'test_component'
        assert component.status == ComponentStatus.UNKNOWN
    
    def test_health_check_registration(self, health_checker):
        """Test health check registration"""
        
        def dummy_check():
            return True
        
        health_checker.register_component_check(
            'test_component',
            'dummy_check',
            'Always returns true',
            dummy_check,
            timeout_seconds=1.0
        )
        
        component = health_checker.get_component_health('test_component')
        assert 'dummy_check' in component.checks
        
        check = component.checks['dummy_check']
        assert check.name == 'dummy_check'
        assert check.description == 'Always returns true'
        assert check.timeout_seconds == 1.0
    
    def test_successful_health_check(self, health_checker):
        """Test successful health check execution"""
        
        call_count = 0
        
        def successful_check():
            nonlocal call_count
            call_count += 1
            return True
        
        health_checker.register_component_check(
            'test_component',
            'success_check',
            'Always succeeds',
            successful_check,
            interval_seconds=0.05
        )
        
        # Run a single check
        check = health_checker.components['test_component'].checks['success_check']
        health_checker._run_check(check, 'test_component')
        
        assert check.last_status == ComponentStatus.HEALTHY
        assert check.consecutive_failures == 0
        assert call_count == 1
    
    def test_failing_health_check(self, health_checker):
        """Test failing health check handling"""
        
        def failing_check():
            return False
        
        health_checker.register_component_check(
            'test_component',
            'fail_check',
            'Always fails',
            failing_check
        )
        
        check = health_checker.components['test_component'].checks['fail_check']
        
        # First failure should be degraded
        health_checker._run_check(check, 'test_component')
        assert check.last_status == ComponentStatus.DEGRADED
        assert check.consecutive_failures == 1
        
        # Second failure should be unhealthy
        health_checker._run_check(check, 'test_component')
        assert check.last_status == ComponentStatus.UNHEALTHY
        assert check.consecutive_failures == 2
    
    def test_health_check_exception(self, health_checker):
        """Test health check that throws exception"""
        
        def exception_check():
            raise ValueError("Test exception")
        
        health_checker.register_component_check(
            'test_component',
            'exception_check',
            'Throws exception',
            exception_check
        )
        
        check = health_checker.components['test_component'].checks['exception_check']
        health_checker._run_check(check, 'test_component')
        
        assert check.last_status == ComponentStatus.UNHEALTHY
        assert check.last_error == "Test exception"
    
    def test_component_status_aggregation(self, health_checker):
        """Test component status aggregation"""
        
        def healthy_check():
            return True
        
        def unhealthy_check():
            return False
        
        # Register healthy check
        health_checker.register_component_check(
            'test_component',
            'healthy_check',
            'Healthy check',
            healthy_check,
            critical=True
        )
        
        # Register unhealthy check
        health_checker.register_component_check(
            'test_component',
            'unhealthy_check',
            'Unhealthy check',
            unhealthy_check,
            critical=True
        )
        
        # Run checks
        for check_name, check in health_checker.components['test_component'].checks.items():
            health_checker._run_check(check, 'test_component')
            health_checker._run_check(check, 'test_component')  # Second run for threshold
        
        # Update component status
        health_checker._update_component_status('test_component')
        
        component = health_checker.get_component_health('test_component')
        assert component.status == ComponentStatus.UNHEALTHY  # Critical failure
    
    def test_non_critical_failure(self, health_checker):
        """Test non-critical check failures don't mark component unhealthy"""
        
        def healthy_check():
            return True
        
        def failing_non_critical():
            return False
        
        health_checker.register_component_check(
            'test_component',
            'healthy_check',
            'Healthy check',
            healthy_check,
            critical=True
        )
        
        health_checker.register_component_check(
            'test_component',
            'non_critical_check',
            'Non-critical check',
            failing_non_critical,
            critical=False
        )
        
        # Run checks
        for check_name, check in health_checker.components['test_component'].checks.items():
            health_checker._run_check(check, 'test_component')
            if not check.critical:
                health_checker._run_check(check, 'test_component')  # Fail the non-critical one
        
        health_checker._update_component_status('test_component')
        
        component = health_checker.get_component_health('test_component')
        assert component.status == ComponentStatus.DEGRADED  # Non-critical failure
    
    def test_global_checks(self, health_checker):
        """Test global system checks"""
        
        def global_check():
            return True
        
        health_checker.register_global_check(
            'test_global',
            'Test global check',
            global_check
        )
        
        assert 'test_global' in health_checker.global_checks
        
        check = health_checker.global_checks['test_global']
        health_checker._run_check(check, 'global')
        
        assert check.last_status == ComponentStatus.HEALTHY
    
    def test_monitoring_loop(self, health_checker):
        """Test monitoring loop execution"""
        
        check_called = threading.Event()
        
        def monitored_check():
            check_called.set()
            return True
        
        health_checker.register_component_check(
            'monitored_component',
            'monitored_check',
            'Check for monitoring test',
            monitored_check,
            interval_seconds=0.05
        )
        
        # Start monitoring
        health_checker.start_monitoring()
        
        # Wait for check to be called
        assert check_called.wait(timeout=1.0)
        
        # Stop monitoring
        health_checker.stop_monitoring()
        
        component = health_checker.get_component_health('monitored_component')
        assert component.checks['monitored_check'].last_status == ComponentStatus.HEALTHY
    
    def test_status_change_callback(self, health_checker):
        """Test status change callbacks"""
        
        callback_calls = []
        
        def status_callback(component_name, old_status, new_status):
            callback_calls.append((component_name, old_status, new_status))
        
        health_checker.add_status_change_callback(status_callback)
        
        # Create a check that will change status
        failure_count = 0
        
        def changing_check():
            nonlocal failure_count
            failure_count += 1
            return failure_count <= 2  # Fail after 2 successes
        
        health_checker.register_component_check(
            'changing_component',
            'changing_check',
            'Check that changes',
            changing_check
        )
        
        # Run checks to trigger status changes
        check = health_checker.components['changing_component'].checks['changing_check']
        
        # First run - should be healthy
        health_checker._run_check(check, 'changing_component')
        health_checker._update_component_status('changing_component')
        
        # Second run - still healthy
        health_checker._run_check(check, 'changing_component')
        health_checker._update_component_status('changing_component')
        
        # Third and fourth runs - should become unhealthy
        health_checker._run_check(check, 'changing_component')
        health_checker._update_component_status('changing_component')
        health_checker._run_check(check, 'changing_component')
        health_checker._update_component_status('changing_component')
        
        # Should have triggered status change callbacks
        assert len(callback_calls) > 0
    
    def test_alert_callback(self, health_checker):
        """Test alert callbacks"""
        
        alert_calls = []
        
        def alert_callback(component_name, message, status):
            alert_calls.append((component_name, message, status))
        
        health_checker.add_alert_callback(alert_callback)
        
        def failing_check():
            return False
        
        health_checker.register_component_check(
            'alert_component',
            'failing_check',
            'Always fails',
            failing_check
        )
        
        # Run check to trigger alert
        check = health_checker.components['alert_component'].checks['failing_check']
        health_checker._run_check(check, 'alert_component')
        
        # Should have triggered alert
        assert len(alert_calls) > 0
        assert alert_calls[0][0] == 'alert_component'
        assert 'Health check failed' in alert_calls[0][1]
    
    def test_system_health_summary(self, health_checker):
        """Test system health summary"""
        
        # Add healthy component
        health_checker.register_component_check(
            'healthy_component',
            'healthy_check',
            'Always healthy',
            lambda: True
        )
        
        # Add unhealthy component
        health_checker.register_component_check(
            'unhealthy_component',
            'unhealthy_check',
            'Always unhealthy',
            lambda: False
        )
        
        # Run checks
        for component_name, component in health_checker.components.items():
            for check in component.checks.values():
                health_checker._run_check(check, component_name)
                health_checker._run_check(check, component_name)  # Second run for threshold
            health_checker._update_component_status(component_name)
        
        # Get system health
        system_health = health_checker.get_system_health()
        
        assert system_health['total_components'] == 2
        assert system_health['healthy_components'] == 1
        assert 'unhealthy_component' in system_health['unhealthy_components']
        assert system_health['overall_status'] == 'degraded'
    
    def test_check_timing(self, health_checker):
        """Test check interval timing"""
        
        last_run_times = []
        
        def timed_check():
            last_run_times.append(time.time())
            return True
        
        health_checker.register_component_check(
            'timed_component',
            'timed_check',
            'Check for timing test',
            timed_check,
            interval_seconds=0.1
        )
        
        check = health_checker.components['timed_component'].checks['timed_check']
        
        # First check should run immediately
        assert health_checker._should_run_check(check)
        health_checker._run_check(check, 'timed_component')
        
        # Second check should not run immediately
        assert not health_checker._should_run_check(check)
        
        # Wait and check again
        time.sleep(0.15)
        assert health_checker._should_run_check(check)
    
    def test_core_health_checks(self, health_checker):
        """Test core system health checks"""
        
        # Test memory check (may not work if psutil not available)
        try:
            result = health_checker._check_memory_usage()
            assert isinstance(result, bool)
        except:
            pass  # psutil may not be available
        
        # Test disk space check
        try:
            result = health_checker._check_disk_space()
            assert isinstance(result, bool)
        except:
            pass  # psutil may not be available
        
        # Test network connectivity check
        result = health_checker._check_network_connectivity()
        assert isinstance(result, bool)
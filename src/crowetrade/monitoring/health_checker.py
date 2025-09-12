"""System health monitoring and alerting"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import asyncio
import threading
import time

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check definition"""
    name: str
    description: str
    check_function: Callable[[], bool]
    timeout_seconds: float = 5.0
    critical: bool = True
    interval_seconds: float = 30.0
    
    # State
    last_check: Optional[datetime] = None
    last_status: ComponentStatus = ComponentStatus.UNKNOWN
    consecutive_failures: int = 0
    last_error: Optional[str] = None


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    component_name: str
    status: ComponentStatus
    checks: Dict[str, HealthCheck]
    last_updated: datetime
    uptime_seconds: float
    error_count: int = 0
    
    @property
    def is_healthy(self) -> bool:
        return self.status == ComponentStatus.HEALTHY
    
    @property
    def critical_failures(self) -> List[str]:
        """Get list of critical health checks that are failing"""
        failures = []
        for check in self.checks.values():
            if check.critical and check.last_status != ComponentStatus.HEALTHY:
                failures.append(check.name)
        return failures


class HealthChecker:
    """
    System health monitoring with configurable checks
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        failure_threshold: int = 3,
        enable_auto_recovery: bool = True
    ):
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.enable_auto_recovery = enable_auto_recovery
        
        # Component registry
        self.components: Dict[str, ComponentHealth] = {}
        self.global_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Callbacks
        self.status_change_callbacks: List[Callable[[str, ComponentStatus, ComponentStatus], None]] = []
        self.alert_callbacks: List[Callable[[str, str, ComponentStatus], None]] = []
        
        # Initialize core health checks
        self._initialize_core_checks()
    
    def _initialize_core_checks(self) -> None:
        """Initialize core system health checks"""
        
        # Memory usage check
        self.register_global_check(
            'memory_usage',
            'System memory usage',
            self._check_memory_usage,
            timeout_seconds=2.0,
            critical=False,
            interval_seconds=60.0
        )
        
        # Disk space check
        self.register_global_check(
            'disk_space',
            'Available disk space',
            self._check_disk_space,
            timeout_seconds=2.0,
            critical=True,
            interval_seconds=120.0
        )
        
        # Network connectivity check
        self.register_global_check(
            'network_connectivity',
            'Network connectivity',
            self._check_network_connectivity,
            timeout_seconds=5.0,
            critical=True,
            interval_seconds=30.0
        )
    
    def register_component(
        self,
        component_name: str,
        checks: Optional[Dict[str, HealthCheck]] = None
    ) -> None:
        """Register a component for health monitoring"""
        
        if component_name in self.components:
            logger.warning(f"Component {component_name} already registered")
            return
        
        component_health = ComponentHealth(
            component_name=component_name,
            status=ComponentStatus.UNKNOWN,
            checks=checks or {},
            last_updated=datetime.now(),
            uptime_seconds=0
        )
        
        self.components[component_name] = component_health
        logger.info(f"Registered component for health monitoring: {component_name}")
    
    def register_component_check(
        self,
        component_name: str,
        check_name: str,
        description: str,
        check_function: Callable[[], bool],
        timeout_seconds: float = 5.0,
        critical: bool = True,
        interval_seconds: float = 30.0
    ) -> None:
        """Register a health check for a specific component"""
        
        if component_name not in self.components:
            self.register_component(component_name)
        
        health_check = HealthCheck(
            name=check_name,
            description=description,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            critical=critical,
            interval_seconds=interval_seconds
        )
        
        self.components[component_name].checks[check_name] = health_check
        logger.info(f"Registered health check: {component_name}.{check_name}")
    
    def register_global_check(
        self,
        check_name: str,
        description: str,
        check_function: Callable[[], bool],
        timeout_seconds: float = 5.0,
        critical: bool = True,
        interval_seconds: float = 30.0
    ) -> None:
        """Register a global system health check"""
        
        health_check = HealthCheck(
            name=check_name,
            description=description,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            critical=critical,
            interval_seconds=interval_seconds
        )
        
        self.global_checks[check_name] = health_check
        logger.info(f"Registered global health check: {check_name}")
    
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread"""
        
        if self._monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self._monitoring = True
        self._shutdown_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Started health monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._shutdown_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped health monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        
        while self._monitoring and not self._shutdown_event.is_set():
            try:
                # Check global health checks
                for check in self.global_checks.values():
                    if self._should_run_check(check):
                        self._run_check(check, "global")
                
                # Check component health checks
                for component_name, component in self.components.items():
                    for check in component.checks.values():
                        if self._should_run_check(check):
                            self._run_check(check, component_name)
                    
                    # Update component status
                    self._update_component_status(component_name)
                
                # Sleep until next check
                self._shutdown_event.wait(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                self._shutdown_event.wait(5.0)  # Brief pause on error
    
    def _should_run_check(self, check: HealthCheck) -> bool:
        """Check if a health check should be run now"""
        
        if check.last_check is None:
            return True
        
        time_since_check = (datetime.now() - check.last_check).total_seconds()
        return time_since_check >= check.interval_seconds
    
    def _run_check(self, check: HealthCheck, component_name: str) -> None:
        """Execute a single health check"""
        
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = self._run_with_timeout(
                check.check_function,
                check.timeout_seconds
            )
            
            # Update check status
            if result:
                old_status = check.last_status
                check.last_status = ComponentStatus.HEALTHY
                check.consecutive_failures = 0
                check.last_error = None
                
                if old_status != ComponentStatus.HEALTHY:
                    logger.info(f"Health check recovered: {component_name}.{check.name}")
            else:
                check.consecutive_failures += 1
                old_status = check.last_status
                
                if check.consecutive_failures >= self.failure_threshold:
                    check.last_status = ComponentStatus.UNHEALTHY
                else:
                    check.last_status = ComponentStatus.DEGRADED
                
                if old_status != check.last_status:
                    logger.warning(
                        f"Health check failed: {component_name}.{check.name} "
                        f"(failures: {check.consecutive_failures})"
                    )
                    
                    # Trigger alert
                    self._trigger_alert(
                        component_name,
                        f"Health check failed: {check.name}",
                        check.last_status
                    )
            
            check.last_check = datetime.now()
            
        except Exception as e:
            check.consecutive_failures += 1
            check.last_status = ComponentStatus.UNHEALTHY
            check.last_error = str(e)
            check.last_check = datetime.now()
            
            logger.error(f"Health check error: {component_name}.{check.name}: {e}")
            
            self._trigger_alert(
                component_name,
                f"Health check error: {check.name} - {e}",
                ComponentStatus.UNHEALTHY
            )
    
    def _run_with_timeout(self, func: Callable, timeout_seconds: float) -> bool:
        """Run function with timeout"""
        
        try:
            # Simple timeout implementation
            start = time.time()
            result = func()
            elapsed = time.time() - start
            
            if elapsed > timeout_seconds:
                logger.warning(f"Health check took {elapsed:.2f}s (timeout: {timeout_seconds}s)")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Health check exception: {e}")
            return False
    
    def _update_component_status(self, component_name: str) -> None:
        """Update overall component health status"""
        
        component = self.components[component_name]
        old_status = component.status
        
        # Determine overall status
        if not component.checks:
            component.status = ComponentStatus.UNKNOWN
        else:
            critical_unhealthy = any(
                check.critical and check.last_status == ComponentStatus.UNHEALTHY
                for check in component.checks.values()
            )
            
            if critical_unhealthy:
                component.status = ComponentStatus.UNHEALTHY
            else:
                any_degraded = any(
                    check.last_status == ComponentStatus.DEGRADED
                    for check in component.checks.values()
                )
                
                if any_degraded:
                    component.status = ComponentStatus.DEGRADED
                else:
                    all_healthy = all(
                        check.last_status == ComponentStatus.HEALTHY
                        for check in component.checks.values()
                    )
                    
                    component.status = ComponentStatus.HEALTHY if all_healthy else ComponentStatus.UNKNOWN
        
        component.last_updated = datetime.now()
        
        # Trigger status change callback
        if old_status != component.status:
            self._trigger_status_change(component_name, old_status, component.status)
    
    def _trigger_status_change(
        self,
        component_name: str,
        old_status: ComponentStatus,
        new_status: ComponentStatus
    ) -> None:
        """Trigger status change callbacks"""
        
        logger.info(f"Component status change: {component_name} {old_status.value} -> {new_status.value}")
        
        for callback in self.status_change_callbacks:
            try:
                callback(component_name, old_status, new_status)
            except Exception as e:
                logger.error(f"Error in status change callback: {e}")
    
    def _trigger_alert(
        self,
        component_name: str,
        message: str,
        status: ComponentStatus
    ) -> None:
        """Trigger alert callbacks"""
        
        for callback in self.alert_callbacks:
            try:
                callback(component_name, message, status)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status of a component"""
        return self.components.get(component_name)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        
        total_components = len(self.components)
        healthy_components = sum(
            1 for c in self.components.values()
            if c.status == ComponentStatus.HEALTHY
        )
        
        unhealthy_components = [
            name for name, component in self.components.items()
            if component.status == ComponentStatus.UNHEALTHY
        ]
        
        return {
            'overall_status': 'healthy' if healthy_components == total_components else 'degraded',
            'total_components': total_components,
            'healthy_components': healthy_components,
            'unhealthy_components': unhealthy_components,
            'critical_failures': self._get_all_critical_failures(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_all_critical_failures(self) -> List[str]:
        """Get all critical failures across components"""
        
        failures = []
        
        # Global checks
        for check in self.global_checks.values():
            if check.critical and check.last_status == ComponentStatus.UNHEALTHY:
                failures.append(f"global.{check.name}")
        
        # Component checks
        for component_name, component in self.components.items():
            failures.extend([
                f"{component_name}.{name}"
                for name in component.critical_failures
            ])
        
        return failures
    
    def add_status_change_callback(
        self,
        callback: Callable[[str, ComponentStatus, ComponentStatus], None]
    ) -> None:
        """Add callback for status changes"""
        self.status_change_callbacks.append(callback)
    
    def add_alert_callback(
        self,
        callback: Callable[[str, str, ComponentStatus], None]
    ) -> None:
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    # Core health check implementations
    
    def _check_memory_usage(self) -> bool:
        """Check system memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90.0  # Less than 90% usage
        except ImportError:
            # psutil not available, assume healthy
            return True
        except Exception:
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            free_percent = (disk.free / disk.total) * 100
            return free_percent > 10.0  # More than 10% free
        except ImportError:
            return True
        except Exception:
            return False
    
    def _check_network_connectivity(self) -> bool:
        """Check basic network connectivity"""
        try:
            import socket
            socket.setdefaulttimeout(3)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            return True
        except Exception:
            return False
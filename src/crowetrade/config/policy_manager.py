"""Policy hot-reload manager with validation and audit logging"""
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
import yaml

try:
    from pydantic import BaseModel, Field, ValidationError, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

logger = logging.getLogger(__name__)


class PolicyConfig(BaseModel if PYDANTIC_AVAILABLE else object):
    """Policy configuration with validation"""
    
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., regex=r'^\d+\.\d+\.\d+$')
    enabled: bool = True
    
    # Trading parameters
    gates: Dict[str, float] = Field(default_factory=dict)
    position_limits: Dict[str, float] = Field(default_factory=dict)
    risk_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Strategy configuration
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    indicators: List[str] = Field(default_factory=list)
    
    # Metadata
    description: str = ""
    author: str = ""
    created_at: str = ""
    tags: List[str] = Field(default_factory=list)
    
    @validator('gates')
    def validate_gates(cls, v):
        """Validate gate thresholds"""
        for key, value in v.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Gate {key} must be between 0 and 1")
        return v
    
    @validator('position_limits')
    def validate_position_limits(cls, v):
        """Validate position limits are positive"""
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"Position limit {key} must be non-negative")
        return v
    
    class Config:
        extra = 'allow'  # Allow extra fields


@dataclass
class PolicyAuditEvent:
    """Audit event for policy changes"""
    timestamp: float
    event_type: str  # 'load', 'reload', 'validation_error', 'apply'
    policy_name: str
    policy_version: str
    policy_hash: str
    old_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON for audit log"""
        return json.dumps(asdict(self))


class PolicyFileHandler(FileSystemEventHandler):
    """Handle policy file changes"""
    
    def __init__(self, manager: 'PolicyHotReloadManager'):
        self.manager = manager
        self.debounce_timer = None
        self.debounce_delay = 1.0  # seconds
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.yaml', '.yml')):
            # Debounce rapid changes
            if self.debounce_timer:
                self.debounce_timer.cancel()
            
            self.debounce_timer = threading.Timer(
                self.debounce_delay,
                self.manager.reload_policy,
                args=[event.src_path]
            )
            self.debounce_timer.start()


class PolicyHotReloadManager:
    """Manage policy hot-reloading with validation and auditing"""
    
    def __init__(
        self,
        policy_dir: str = "./policies",
        audit_log_path: str = "./audit/policy_changes.jsonl",
        enable_hot_reload: bool = True,
        validation_strict: bool = True
    ):
        self.policy_dir = Path(policy_dir)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.enable_hot_reload = enable_hot_reload
        self.validation_strict = validation_strict
        
        # Current policies
        self.policies: Dict[str, PolicyConfig] = {}
        self.policy_hashes: Dict[str, str] = {}
        self.policy_lock = threading.RLock()
        
        # Callbacks for policy changes
        self.change_callbacks: List[Callable] = []
        
        # File watcher
        self.observer = None
        if enable_hot_reload:
            self._start_watcher()
        
        # Load initial policies
        self.load_all_policies()
    
    def load_all_policies(self) -> None:
        """Load all policy files from directory"""
        
        policy_files = list(self.policy_dir.glob("*.yaml")) + \
                      list(self.policy_dir.glob("*.yml"))
        
        for file in policy_files:
            try:
                self.load_policy(str(file))
            except Exception as e:
                logger.error(f"Failed to load policy {file}: {e}")
    
    def load_policy(self, file_path: str) -> Optional[PolicyConfig]:
        """Load and validate a policy file"""
        
        file_path = Path(file_path)
        
        try:
            # Read file
            with open(file_path, 'r') as f:
                content = f.read()
                data = yaml.safe_load(content)
            
            # Calculate hash
            policy_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Validate with Pydantic if available
            if PYDANTIC_AVAILABLE:
                policy = PolicyConfig(**data)
            else:
                # Basic validation without Pydantic
                policy = self._basic_validate(data)
            
            # Check for changes
            old_hash = self.policy_hashes.get(policy.name)
            
            # Store policy atomically
            with self.policy_lock:
                self.policies[policy.name] = policy
                self.policy_hashes[policy.name] = policy_hash
            
            # Log audit event
            self._log_audit_event(PolicyAuditEvent(
                timestamp=time.time(),
                event_type='load' if old_hash is None else 'reload',
                policy_name=policy.name,
                policy_version=policy.version,
                policy_hash=policy_hash,
                old_hash=old_hash,
                metadata={'file': str(file_path)}
            ))
            
            # Notify callbacks
            if old_hash and old_hash != policy_hash:
                self._notify_change(policy.name, policy)
            
            logger.info(f"Loaded policy {policy.name} v{policy.version} (hash: {policy_hash})")
            return policy
            
        except (ValidationError, ValueError) as e:
            error_msg = str(e)
            logger.error(f"Policy validation failed for {file_path}: {error_msg}")
            
            # Log validation error
            self._log_audit_event(PolicyAuditEvent(
                timestamp=time.time(),
                event_type='validation_error',
                policy_name=file_path.stem,
                policy_version='unknown',
                policy_hash='',
                error=error_msg,
                metadata={'file': str(file_path)}
            ))
            
            if self.validation_strict:
                raise
            return None
            
        except Exception as e:
            logger.error(f"Failed to load policy {file_path}: {e}")
            raise
    
    def reload_policy(self, file_path: str) -> None:
        """Reload a specific policy file"""
        
        logger.info(f"Reloading policy from {file_path}")
        self.load_policy(file_path)
    
    def get_policy(self, name: str) -> Optional[PolicyConfig]:
        """Get a policy by name"""
        
        with self.policy_lock:
            return self.policies.get(name)
    
    def get_all_policies(self) -> Dict[str, PolicyConfig]:
        """Get all loaded policies"""
        
        with self.policy_lock:
            return dict(self.policies)
    
    def apply_policy(self, name: str, target: Any) -> bool:
        """Apply a policy to a target object"""
        
        policy = self.get_policy(name)
        if not policy:
            logger.warning(f"Policy {name} not found")
            return False
        
        if not policy.enabled:
            logger.info(f"Policy {name} is disabled")
            return False
        
        try:
            # Apply gates
            if hasattr(target, 'gates'):
                target.gates.update(policy.gates)
            
            # Apply position limits
            if hasattr(target, 'position_limits'):
                target.position_limits.update(policy.position_limits)
            
            # Apply risk params
            if hasattr(target, 'risk_params'):
                target.risk_params.update(policy.risk_params)
            
            # Apply strategy params
            if hasattr(target, 'strategy_params'):
                target.strategy_params.update(policy.strategy_params)
            
            # Log application
            self._log_audit_event(PolicyAuditEvent(
                timestamp=time.time(),
                event_type='apply',
                policy_name=policy.name,
                policy_version=policy.version,
                policy_hash=self.policy_hashes[name],
                metadata={'target': str(type(target).__name__)}
            ))
            
            logger.info(f"Applied policy {name} to {type(target).__name__}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply policy {name}: {e}")
            return False
    
    def register_change_callback(self, callback: Callable) -> None:
        """Register a callback for policy changes"""
        
        self.change_callbacks.append(callback)
    
    def _basic_validate(self, data: Dict) -> PolicyConfig:
        """Basic validation without Pydantic"""
        
        # Create a simple object with required fields
        class SimplePolicy:
            def __init__(self, **kwargs):
                self.name = kwargs.get('name', '')
                self.version = kwargs.get('version', '0.0.0')
                self.enabled = kwargs.get('enabled', True)
                self.gates = kwargs.get('gates', {})
                self.position_limits = kwargs.get('position_limits', {})
                self.risk_params = kwargs.get('risk_params', {})
                self.strategy_params = kwargs.get('strategy_params', {})
                self.indicators = kwargs.get('indicators', [])
                self.description = kwargs.get('description', '')
                self.author = kwargs.get('author', '')
                self.created_at = kwargs.get('created_at', '')
                self.tags = kwargs.get('tags', [])
        
        # Basic validation
        if not data.get('name'):
            raise ValueError("Policy name is required")
        
        if not data.get('version'):
            raise ValueError("Policy version is required")
        
        return SimplePolicy(**data)
    
    def _notify_change(self, policy_name: str, policy: PolicyConfig) -> None:
        """Notify callbacks of policy change"""
        
        for callback in self.change_callbacks:
            try:
                callback(policy_name, policy)
            except Exception as e:
                logger.error(f"Callback error for policy {policy_name}: {e}")
    
    def _log_audit_event(self, event: PolicyAuditEvent) -> None:
        """Log an audit event"""
        
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(event.to_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _start_watcher(self) -> None:
        """Start file system watcher"""
        
        self.observer = Observer()
        handler = PolicyFileHandler(self)
        self.observer.schedule(handler, str(self.policy_dir), recursive=False)
        self.observer.start()
        logger.info(f"Started policy file watcher on {self.policy_dir}")
    
    def shutdown(self) -> None:
        """Shutdown the manager"""
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped policy file watcher")
    
    def get_audit_log(self, limit: int = 100) -> List[PolicyAuditEvent]:
        """Read recent audit events"""
        
        events = []
        
        if not self.audit_log_path.exists():
            return events
        
        try:
            with open(self.audit_log_path, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    data = json.loads(line)
                    events.append(PolicyAuditEvent(**data))
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
        
        return events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        
        with self.policy_lock:
            return {
                'loaded_policies': len(self.policies),
                'enabled_policies': sum(1 for p in self.policies.values() if p.enabled),
                'hot_reload_enabled': self.enable_hot_reload,
                'validation_strict': self.validation_strict,
                'policy_dir': str(self.policy_dir),
                'audit_log_path': str(self.audit_log_path)
            }
"""Tests for policy hot-reload manager"""
import json
import time
import tempfile
import shutil
from pathlib import Path
import pytest
import yaml

from crowetrade.config.policy_manager import (
    PolicyHotReloadManager,
    PolicyConfig,
    PolicyAuditEvent
)


class TestPolicyManager:
    """Test policy hot-reload functionality"""
    
    @pytest.fixture
    def temp_manager(self):
        """Create a temporary policy manager"""
        temp_dir = tempfile.mkdtemp()
        audit_dir = tempfile.mkdtemp()
        
        manager = PolicyHotReloadManager(
            policy_dir=temp_dir,
            audit_log_path=f"{audit_dir}/audit.jsonl",
            enable_hot_reload=False,  # Disable for tests
            validation_strict=True
        )
        
        yield manager, temp_dir
        
        # Cleanup
        manager.shutdown()
        shutil.rmtree(temp_dir)
        shutil.rmtree(audit_dir)
    
    def test_load_policy(self, temp_manager):
        """Test loading a policy from file"""
        manager, temp_dir = temp_manager
        
        # Create a policy file
        policy_data = {
            'name': 'test_policy',
            'version': '1.0.0',
            'enabled': True,
            'gates': {
                'min_confidence': 0.7,
                'max_correlation': 0.8
            },
            'position_limits': {
                'AAPL': 50000,
                'default': 25000
            },
            'risk_params': {
                'max_drawdown': 0.05
            }
        }
        
        policy_file = Path(temp_dir) / 'test_policy.yaml'
        with open(policy_file, 'w') as f:
            yaml.dump(policy_data, f)
        
        # Load the policy
        policy = manager.load_policy(str(policy_file))
        
        assert policy is not None
        assert policy.name == 'test_policy'
        assert policy.version == '1.0.0'
        assert policy.enabled is True
        assert policy.gates['min_confidence'] == 0.7
        assert policy.position_limits['AAPL'] == 50000
    
    def test_policy_validation(self, temp_manager):
        """Test policy validation"""
        manager, temp_dir = temp_manager
        
        # Invalid gate value (should be 0-1)
        invalid_policy = {
            'name': 'invalid_policy',
            'version': '1.0.0',
            'gates': {
                'min_confidence': 1.5  # Invalid
            }
        }
        
        policy_file = Path(temp_dir) / 'invalid.yaml'
        with open(policy_file, 'w') as f:
            yaml.dump(invalid_policy, f)
        
        # Should raise validation error in strict mode
        with pytest.raises(Exception):
            manager.load_policy(str(policy_file))
    
    def test_policy_reload(self, temp_manager):
        """Test policy reloading with changes"""
        manager, temp_dir = temp_manager
        
        # Initial policy
        policy_v1 = {
            'name': 'reload_test',
            'version': '1.0.0',
            'gates': {'min_confidence': 0.5}
        }
        
        policy_file = Path(temp_dir) / 'reload_test.yaml'
        with open(policy_file, 'w') as f:
            yaml.dump(policy_v1, f)
        
        # Load initial version
        manager.load_policy(str(policy_file))
        policy = manager.get_policy('reload_test')
        assert policy.gates['min_confidence'] == 0.5
        
        # Update policy
        policy_v2 = {
            'name': 'reload_test',
            'version': '1.0.1',
            'gates': {'min_confidence': 0.7}
        }
        
        with open(policy_file, 'w') as f:
            yaml.dump(policy_v2, f)
        
        # Reload
        manager.reload_policy(str(policy_file))
        policy = manager.get_policy('reload_test')
        assert policy.version == '1.0.1'
        assert policy.gates['min_confidence'] == 0.7
    
    def test_apply_policy(self, temp_manager):
        """Test applying policy to target object"""
        manager, temp_dir = temp_manager
        
        # Create policy
        policy_data = {
            'name': 'apply_test',
            'version': '1.0.0',
            'enabled': True,
            'gates': {'threshold': 0.8},
            'position_limits': {'max': 100000},
            'risk_params': {'var': 0.95}
        }
        
        policy_file = Path(temp_dir) / 'apply_test.yaml'
        with open(policy_file, 'w') as f:
            yaml.dump(policy_data, f)
        
        manager.load_policy(str(policy_file))
        
        # Create target object
        class Target:
            def __init__(self):
                self.gates = {}
                self.position_limits = {}
                self.risk_params = {}
        
        target = Target()
        
        # Apply policy
        success = manager.apply_policy('apply_test', target)
        assert success is True
        assert target.gates['threshold'] == 0.8
        assert target.position_limits['max'] == 100000
        assert target.risk_params['var'] == 0.95
    
    def test_disabled_policy(self, temp_manager):
        """Test that disabled policies are not applied"""
        manager, temp_dir = temp_manager
        
        # Create disabled policy
        policy_data = {
            'name': 'disabled_test',
            'version': '1.0.0',
            'enabled': False,
            'gates': {'threshold': 0.8}
        }
        
        policy_file = Path(temp_dir) / 'disabled_test.yaml'
        with open(policy_file, 'w') as f:
            yaml.dump(policy_data, f)
        
        manager.load_policy(str(policy_file))
        
        # Try to apply
        class Target:
            def __init__(self):
                self.gates = {}
        
        target = Target()
        success = manager.apply_policy('disabled_test', target)
        
        assert success is False
        assert len(target.gates) == 0  # Should not be modified
    
    def test_audit_logging(self, temp_manager):
        """Test audit event logging"""
        manager, temp_dir = temp_manager
        
        # Create and load a policy
        policy_data = {
            'name': 'audit_test',
            'version': '1.0.0',
            'gates': {'min_confidence': 0.6}
        }
        
        policy_file = Path(temp_dir) / 'audit_test.yaml'
        with open(policy_file, 'w') as f:
            yaml.dump(policy_data, f)
        
        manager.load_policy(str(policy_file))
        
        # Apply the policy
        class Target:
            def __init__(self):
                self.gates = {}
        
        target = Target()
        manager.apply_policy('audit_test', target)
        
        # Check audit log
        events = manager.get_audit_log()
        assert len(events) >= 2  # Load + apply events
        
        # Check load event
        load_events = [e for e in events if e.event_type == 'load']
        assert len(load_events) > 0
        assert load_events[0].policy_name == 'audit_test'
        
        # Check apply event
        apply_events = [e for e in events if e.event_type == 'apply']
        assert len(apply_events) > 0
        assert apply_events[0].policy_name == 'audit_test'
    
    def test_change_callbacks(self, temp_manager):
        """Test policy change callbacks"""
        manager, temp_dir = temp_manager
        
        # Track callbacks
        callback_calls = []
        
        def callback(name, policy):
            callback_calls.append((name, policy.version))
        
        manager.register_change_callback(callback)
        
        # Initial load (no callback since no change)
        policy_v1 = {
            'name': 'callback_test',
            'version': '1.0.0',
            'gates': {'threshold': 0.5}
        }
        
        policy_file = Path(temp_dir) / 'callback_test.yaml'
        with open(policy_file, 'w') as f:
            yaml.dump(policy_v1, f)
        
        manager.load_policy(str(policy_file))
        assert len(callback_calls) == 0  # No change yet
        
        # Update policy (should trigger callback)
        policy_v2 = {
            'name': 'callback_test',
            'version': '1.0.1',
            'gates': {'threshold': 0.7}
        }
        
        with open(policy_file, 'w') as f:
            yaml.dump(policy_v2, f)
        
        manager.reload_policy(str(policy_file))
        assert len(callback_calls) == 1
        assert callback_calls[0] == ('callback_test', '1.0.1')
    
    def test_get_stats(self, temp_manager):
        """Test statistics reporting"""
        manager, temp_dir = temp_manager
        
        # Load some policies
        for i in range(3):
            policy_data = {
                'name': f'policy_{i}',
                'version': '1.0.0',
                'enabled': i < 2  # First 2 enabled
            }
            
            policy_file = Path(temp_dir) / f'policy_{i}.yaml'
            with open(policy_file, 'w') as f:
                yaml.dump(policy_data, f)
            
            manager.load_policy(str(policy_file))
        
        stats = manager.get_stats()
        assert stats['loaded_policies'] == 3
        assert stats['enabled_policies'] == 2
        assert stats['hot_reload_enabled'] is False  # Disabled for tests
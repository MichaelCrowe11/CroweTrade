"""A/B Testing System for Model Comparison

Statistical framework for comparing model performance:
- Multi-armed bandit allocation
- Statistical significance testing
- Performance monitoring and analysis
- Automatic traffic splitting
- Champion/challenger evaluation
"""

import logging
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
from collections import defaultdict, deque

try:
    import scipy.stats as stats
except ImportError:
    # Mock scipy.stats for testing without scipy
    class MockStats:
        @staticmethod
        def ttest_ind(a, b):
            return 0.0, 0.5  # Mock t-statistic and p-value
        
        class t:
            @staticmethod
            def interval(confidence, df, loc=0, scale=1):
                return (loc - scale, loc + scale)  # Mock confidence interval
    
    stats = MockStats()

from ..core.agent import BaseAgent
from ..core.types import HealthStatus
from .registry import ModelRegistry, ModelMetadata

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AllocationStrategy(Enum):
    """Traffic allocation strategies"""
    FIXED = "fixed"                  # Fixed percentage allocation
    THOMPSON_SAMPLING = "thompson"   # Thompson sampling (Bayesian)
    UCB = "ucb"                     # Upper Confidence Bound
    EPSILON_GREEDY = "epsilon_greedy" # Epsilon-greedy exploration


@dataclass
class TestArm:
    """Individual test arm configuration"""
    arm_id: str
    model_registry_id: str          # Points to model in registry
    allocation_weight: float = 0.0  # Current allocation percentage
    
    # Performance tracking
    total_requests: int = 0
    total_returns: float = 0.0      # Cumulative returns
    total_sharpe: float = 0.0       # Cumulative Sharpe ratio
    win_rate: float = 0.0           # Win rate vs other arms
    
    # Statistical properties
    mean_return: float = 0.0
    std_return: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Multi-armed bandit parameters
    alpha: float = 1.0              # Beta distribution alpha (successes + 1)
    beta: float = 1.0               # Beta distribution beta (failures + 1)


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    
    # Test parameters
    arms: List[TestArm]
    allocation_strategy: AllocationStrategy
    
    # Statistical parameters
    significance_level: float = 0.05        # Alpha for hypothesis testing
    minimum_sample_size: int = 1000         # Min samples per arm
    minimum_effect_size: float = 0.01       # Min detectable effect (1% return diff)
    
    # Duration controls
    max_duration_days: int = 30
    early_stopping: bool = True             # Stop early if significant result
    
    # Risk controls
    max_allocation_per_arm: float = 0.8     # Max traffic to single arm
    min_allocation_per_arm: float = 0.05    # Min traffic to keep arm active
    
    # Performance thresholds
    stop_loss_threshold: float = -0.05      # Stop arm if returns < -5%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['allocation_strategy'] = self.allocation_strategy.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTestConfig':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['allocation_strategy'] = AllocationStrategy(data['allocation_strategy'])
        # Convert arm dictionaries to TestArm objects
        data['arms'] = [TestArm(**arm_data) for arm_data in data['arms']]
        return cls(**data)


class ABTestEngine:
    """A/B testing engine for model comparison"""
    
    def __init__(self,
                 model_registry: ModelRegistry,
                 test_storage_path: Path,
                 update_interval_minutes: int = 5):
        
        self.model_registry = model_registry
        self.test_storage_path = Path(test_storage_path)
        self.update_interval_minutes = update_interval_minutes
        
        # Create storage directories
        self.tests_path = self.test_storage_path / "tests"
        self.results_path = self.test_storage_path / "results"
        self.tests_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Active tests cache
        self._active_tests: Dict[str, ABTestConfig] = {}
        self._test_results: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance tracking
        self._arm_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        self._load_active_tests()
        
        logger.info(f"ABTestEngine initialized with {len(self._active_tests)} active tests")
    
    def create_test(self,
                   test_config: ABTestConfig) -> str:
        """Create new A/B test"""
        
        try:
            # Validate test configuration
            self._validate_test_config(test_config)
            
            # Initialize equal allocation for new test
            num_arms = len(test_config.arms)
            initial_allocation = 1.0 / num_arms
            
            for arm in test_config.arms:
                arm.allocation_weight = initial_allocation
            
            # Save test configuration
            self._save_test_config(test_config)
            
            # Add to active tests
            self._active_tests[test_config.test_id] = test_config
            
            logger.info(f"Created A/B test: {test_config.test_id} with {num_arms} arms")
            return test_config.test_id
            
        except Exception as e:
            logger.error(f"Error creating A/B test {test_config.test_id}: {e}")
            raise
    
    def get_model_allocation(self, test_id: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get model allocation for request (which arm to use)"""
        
        if test_id not in self._active_tests:
            return None
        
        test_config = self._active_tests[test_id]
        
        # Check if test is still running
        if not self._is_test_active(test_config):
            return None
        
        # Select arm based on allocation strategy
        if test_config.allocation_strategy == AllocationStrategy.FIXED:
            selected_arm = self._select_arm_fixed(test_config.arms)
        elif test_config.allocation_strategy == AllocationStrategy.THOMPSON_SAMPLING:
            selected_arm = self._select_arm_thompson(test_config.arms)
        elif test_config.allocation_strategy == AllocationStrategy.UCB:
            selected_arm = self._select_arm_ucb(test_config.arms)
        elif test_config.allocation_strategy == AllocationStrategy.EPSILON_GREEDY:
            selected_arm = self._select_arm_epsilon_greedy(test_config.arms)
        else:
            selected_arm = self._select_arm_fixed(test_config.arms)
        
        if selected_arm:
            return selected_arm.model_registry_id
        
        return None
    
    def record_result(self,
                     test_id: str,
                     model_registry_id: str,
                     return_value: float,
                     additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """Record result for model in A/B test"""
        
        if test_id not in self._active_tests:
            return
        
        test_config = self._active_tests[test_id]
        
        # Find the arm
        arm = None
        for test_arm in test_config.arms:
            if test_arm.model_registry_id == model_registry_id:
                arm = test_arm
                break
        
        if not arm:
            logger.warning(f"Unknown model {model_registry_id} in test {test_id}")
            return
        
        # Update arm statistics
        arm.total_requests += 1
        arm.total_returns += return_value
        
        # Calculate running statistics
        if arm.total_requests > 0:
            arm.mean_return = arm.total_returns / arm.total_requests
        
        # Store detailed metrics for analysis
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'return': return_value,
            'arm_id': arm.arm_id,
            'model_id': model_registry_id
        }
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self._arm_metrics[f"{test_id}:{arm.arm_id}"].append(metrics)
        
        # Update allocation weights based on strategy
        self._update_allocations(test_config)
        
        # Check for early stopping conditions
        if test_config.early_stopping:
            self._check_early_stopping(test_config)
        
        # Save updated test state
        self._save_test_config(test_config)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get current test results and statistics"""
        
        if test_id not in self._active_tests:
            return {}
        
        test_config = self._active_tests[test_id]
        
        # Calculate comprehensive statistics
        results = {
            'test_id': test_id,
            'name': test_config.name,
            'status': self._get_test_status(test_config),
            'created_at': test_config.created_at.isoformat(),
            'duration_days': (datetime.utcnow() - test_config.created_at).days,
            'arms': []
        }
        
        # Calculate statistics for each arm
        for arm in test_config.arms:
            arm_results = self._calculate_arm_statistics(test_id, arm)
            results['arms'].append(arm_results)
        
        # Statistical significance testing
        if len(test_config.arms) == 2:
            significance_results = self._calculate_significance(test_config.arms)
            results['significance_test'] = significance_results
        
        # Overall test metrics
        total_requests = sum(arm.total_requests for arm in test_config.arms)
        results['total_requests'] = total_requests
        results['requests_per_arm'] = total_requests / len(test_config.arms) if test_config.arms else 0
        
        return results
    
    def stop_test(self, test_id: str, reason: str = "Manual stop") -> bool:
        """Stop an active test"""
        
        if test_id not in self._active_tests:
            return False
        
        test_config = self._active_tests[test_id]
        
        # Save final results
        final_results = self.get_test_results(test_id)
        final_results['stopped_at'] = datetime.utcnow().isoformat()
        final_results['stop_reason'] = reason
        
        results_file = self.results_path / f"{test_id}_final.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Remove from active tests
        del self._active_tests[test_id]
        
        # Archive test configuration
        test_file = self.tests_path / f"{test_id}.json"
        archive_file = self.tests_path / f"{test_id}_archived.json"
        if test_file.exists():
            test_file.rename(archive_file)
        
        logger.info(f"Stopped A/B test: {test_id} - {reason}")
        return True
    
    def _validate_test_config(self, test_config: ABTestConfig) -> None:
        """Validate test configuration"""
        
        if len(test_config.arms) < 2:
            raise ValueError("Test must have at least 2 arms")
        
        if len(test_config.arms) > 10:
            raise ValueError("Test cannot have more than 10 arms")
        
        # Validate model registry IDs
        for arm in test_config.arms:
            metadata = self.model_registry.get_metadata(arm.model_registry_id)
            if not metadata:
                raise ValueError(f"Model not found in registry: {arm.model_registry_id}")
        
        # Validate parameters
        if not (0.01 <= test_config.significance_level <= 0.1):
            raise ValueError("Significance level must be between 0.01 and 0.1")
        
        if test_config.minimum_sample_size < 100:
            raise ValueError("Minimum sample size must be at least 100")
    
    def _select_arm_fixed(self, arms: List[TestArm]) -> Optional[TestArm]:
        """Fixed allocation selection"""
        if not arms:
            return None
        
        # Weighted random selection
        weights = [arm.allocation_weight for arm in arms]
        if sum(weights) == 0:
            return np.random.choice(arms)
        
        return np.random.choice(arms, p=np.array(weights) / sum(weights))
    
    def _select_arm_thompson(self, arms: List[TestArm]) -> Optional[TestArm]:
        """Thompson sampling selection"""
        if not arms:
            return None
        
        # Sample from Beta distribution for each arm
        samples = []
        for arm in arms:
            # Convert returns to success/failure for Beta distribution
            # Positive returns = success, negative = failure
            sample = np.random.beta(arm.alpha, arm.beta)
            samples.append((sample, arm))
        
        # Select arm with highest sample
        return max(samples, key=lambda x: x[0])[1]
    
    def _select_arm_ucb(self, arms: List[TestArm]) -> Optional[TestArm]:
        """Upper Confidence Bound selection"""
        if not arms:
            return None
        
        total_requests = sum(arm.total_requests for arm in arms)
        if total_requests == 0:
            return np.random.choice(arms)
        
        # Calculate UCB values
        ucb_values = []
        for arm in arms:
            if arm.total_requests == 0:
                ucb_value = float('inf')  # Explore unvisited arms first
            else:
                confidence_bonus = np.sqrt(2 * np.log(total_requests) / arm.total_requests)
                ucb_value = arm.mean_return + confidence_bonus
            
            ucb_values.append((ucb_value, arm))
        
        # Select arm with highest UCB
        return max(ucb_values, key=lambda x: x[0])[1]
    
    def _select_arm_epsilon_greedy(self, arms: List[TestArm], epsilon: float = 0.1) -> Optional[TestArm]:
        """Epsilon-greedy selection"""
        if not arms:
            return None
        
        # Exploration vs exploitation
        if np.random.random() < epsilon:
            # Explore: random selection
            return np.random.choice(arms)
        else:
            # Exploit: select best performing arm
            if all(arm.total_requests == 0 for arm in arms):
                return np.random.choice(arms)
            
            best_arm = max(arms, key=lambda arm: arm.mean_return if arm.total_requests > 0 else -float('inf'))
            return best_arm
    
    def _update_allocations(self, test_config: ABTestConfig) -> None:
        """Update allocation weights based on performance"""
        
        if test_config.allocation_strategy == AllocationStrategy.FIXED:
            return  # Fixed allocation doesn't change
        
        # Update Beta distribution parameters for Thompson sampling
        for arm in test_config.arms:
            if arm.total_requests > 0:
                # Convert returns to successes/failures
                positive_returns = sum(1 for metrics in self._arm_metrics[f"{test_config.test_id}:{arm.arm_id}"] 
                                     if metrics['return'] > 0)
                
                arm.alpha = positive_returns + 1
                arm.beta = arm.total_requests - positive_returns + 1
    
    def _calculate_arm_statistics(self, test_id: str, arm: TestArm) -> Dict[str, Any]:
        """Calculate comprehensive statistics for an arm"""
        
        metrics_key = f"{test_id}:{arm.arm_id}"
        metrics = list(self._arm_metrics[metrics_key])
        
        if not metrics:
            return {
                'arm_id': arm.arm_id,
                'model_registry_id': arm.model_registry_id,
                'total_requests': 0,
                'mean_return': 0.0,
                'std_return': 0.0,
                'allocation_weight': arm.allocation_weight,
                'confidence_interval': (0.0, 0.0)
            }
        
        returns = [m['return'] for m in metrics]
        
        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Confidence interval
        n = len(returns)
        if n > 1:
            se = std_return / np.sqrt(n)
            ci = stats.t.interval(0.95, n-1, loc=mean_return, scale=se)
        else:
            ci = (mean_return, mean_return)
        
        return {
            'arm_id': arm.arm_id,
            'model_registry_id': arm.model_registry_id,
            'total_requests': arm.total_requests,
            'mean_return': mean_return,
            'std_return': std_return,
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'allocation_weight': arm.allocation_weight,
            'confidence_interval': ci,
            'sharpe_ratio': mean_return / std_return if std_return > 0 else 0,
            'win_rate': sum(1 for r in returns if r > 0) / len(returns)
        }
    
    def _calculate_significance(self, arms: List[TestArm]) -> Dict[str, Any]:
        """Calculate statistical significance between two arms"""
        
        if len(arms) != 2:
            return {}
        
        arm1, arm2 = arms
        
        # Get returns for each arm
        metrics1 = list(self._arm_metrics[f"{arm1.arm_id}"])
        metrics2 = list(self._arm_metrics[f"{arm2.arm_id}"])
        
        if not metrics1 or not metrics2:
            return {'significant': False, 'p_value': 1.0}
        
        returns1 = [m['return'] for m in metrics1]
        returns2 = [m['return'] for m in metrics2]
        
        # Two-sample t-test
        try:
            t_stat, p_value = stats.ttest_ind(returns1, returns2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(returns1) - 1) * np.var(returns1) + 
                                 (len(returns2) - 1) * np.var(returns2)) / 
                                (len(returns1) + len(returns2) - 2))
            
            effect_size = (np.mean(returns1) - np.mean(returns2)) / pooled_std if pooled_std > 0 else 0
            
            return {
                'significant': p_value < 0.05,
                'p_value': p_value,
                't_statistic': t_stat,
                'effect_size': effect_size,
                'winner': arm1.arm_id if np.mean(returns1) > np.mean(returns2) else arm2.arm_id
            }
            
        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            return {'significant': False, 'p_value': 1.0}
    
    def _check_early_stopping(self, test_config: ABTestConfig) -> None:
        """Check if test should be stopped early"""
        
        # Check minimum sample size
        min_samples_met = all(arm.total_requests >= test_config.minimum_sample_size 
                             for arm in test_config.arms)
        
        if not min_samples_met:
            return
        
        # Check for statistical significance (for 2-arm tests)
        if len(test_config.arms) == 2:
            significance = self._calculate_significance(test_config.arms)
            
            if significance.get('significant', False):
                effect_size = abs(significance.get('effect_size', 0))
                
                if effect_size >= test_config.minimum_effect_size:
                    self.stop_test(test_config.test_id, 
                                 f"Early stopping: Significant result (p={significance['p_value']:.4f})")
                    return
        
        # Check stop-loss thresholds
        for arm in test_config.arms:
            if arm.mean_return < test_config.stop_loss_threshold:
                # Reduce allocation for underperforming arm
                arm.allocation_weight = max(arm.allocation_weight * 0.8, 
                                          test_config.min_allocation_per_arm)
    
    def _is_test_active(self, test_config: ABTestConfig) -> bool:
        """Check if test is still active"""
        
        # Check duration
        duration = datetime.utcnow() - test_config.created_at
        if duration.days >= test_config.max_duration_days:
            self.stop_test(test_config.test_id, "Maximum duration reached")
            return False
        
        return True
    
    def _get_test_status(self, test_config: ABTestConfig) -> str:
        """Get current test status"""
        
        if not self._is_test_active(test_config):
            return "completed"
        
        # Check if test has sufficient data
        total_requests = sum(arm.total_requests for arm in test_config.arms)
        
        if total_requests == 0:
            return "starting"
        elif total_requests < test_config.minimum_sample_size * len(test_config.arms):
            return "warming_up"
        else:
            return "running"
    
    def _save_test_config(self, test_config: ABTestConfig) -> None:
        """Save test configuration to disk"""
        
        test_file = self.tests_path / f"{test_config.test_id}.json"
        with open(test_file, 'w') as f:
            json.dump(test_config.to_dict(), f, indent=2)
    
    def _load_active_tests(self) -> None:
        """Load active tests from disk"""
        
        for test_file in self.tests_path.glob("*.json"):
            if test_file.name.endswith("_archived.json"):
                continue
            
            try:
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                
                test_config = ABTestConfig.from_dict(test_data)
                
                # Only load tests that are still within duration
                if self._is_test_active(test_config):
                    self._active_tests[test_config.test_id] = test_config
                
            except Exception as e:
                logger.error(f"Error loading test config {test_file}: {e}")


def create_ab_test(test_id: str,
                  name: str,
                  description: str,
                  model_registry_ids: List[str],
                  created_by: str,
                  allocation_strategy: AllocationStrategy = AllocationStrategy.THOMPSON_SAMPLING,
                  **kwargs) -> ABTestConfig:
    """Convenience function to create A/B test configuration"""
    
    arms = []
    for i, model_id in enumerate(model_registry_ids):
        arm = TestArm(
            arm_id=f"arm_{i}",
            model_registry_id=model_id
        )
        arms.append(arm)
    
    return ABTestConfig(
        test_id=test_id,
        name=name,
        description=description,
        created_by=created_by,
        created_at=datetime.utcnow(),
        arms=arms,
        allocation_strategy=allocation_strategy,
        significance_level=kwargs.get("significance_level", 0.05),
        minimum_sample_size=kwargs.get("minimum_sample_size", 1000),
        minimum_effect_size=kwargs.get("minimum_effect_size", 0.01),
        max_duration_days=kwargs.get("max_duration_days", 30),
        early_stopping=kwargs.get("early_stopping", True)
    )

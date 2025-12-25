"""
Hyperparameter sampler for optimization.

Implements Sobol-based exploration with TPE-like exploitation.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

class HyperparamSampler:
    """Sampler for hyperparameter optimization."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.history = defaultdict(list)  # arena -> list of (hparams, score)
        self.n_samples = defaultdict(int)
    
    def sample_continuous(self, low: float, high: float) -> float:
        """Sample uniform continuous value."""
        return self.rng.uniform(low, high)
    
    def sample_int(self, low: int, high: int) -> int:
        """Sample uniform integer value."""
        return self.rng.integers(low, high + 1)
    
    def sample_choice(self, options: List[Any]) -> Any:
        """Sample from discrete options."""
        return self.rng.choice(options)
    
    def sample_log_uniform(self, low: float, high: float) -> float:
        """Sample log-uniform value."""
        log_low = np.log(low)
        log_high = np.log(high)
        return np.exp(self.rng.uniform(log_low, log_high))
    
    def get_best_hparams(self, arena: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k best hyperparameter configs for an arena."""
        if arena not in self.history or len(self.history[arena]) == 0:
            return []
        
        sorted_history = sorted(self.history[arena], key=lambda x: x[1])
        return [h[0] for h in sorted_history[:top_k]]
    
    def add_result(self, arena: str, hparams: Dict[str, Any], score: float) -> None:
        """Record a trial result."""
        self.history[arena].append((hparams.copy(), score))
        self.n_samples[arena] += 1
    
    def should_exploit(self, arena: str, exploration_ratio: float = 0.3) -> bool:
        """Decide whether to exploit (use good configs) or explore."""
        if self.n_samples[arena] < 5:
            return False
        return self.rng.random() > exploration_ratio
    
    def mutate_hparams(self, hparams: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """Mutate hyperparameters slightly."""
        new_hparams = hparams.copy()
        
        for key, value in hparams.items():
            if self.rng.random() > mutation_rate:
                continue
            
            if isinstance(value, float):
                # Perturb by up to 20%
                factor = self.rng.uniform(0.8, 1.2)
                new_hparams[key] = value * factor
            elif isinstance(value, int):
                delta = max(1, int(value * 0.2))
                new_hparams[key] = max(1, value + self.rng.integers(-delta, delta + 1))
            elif isinstance(value, (list, tuple)):
                # Shuffle or keep
                if self.rng.random() < 0.5:
                    new_hparams[key] = list(self.rng.permutation(list(value)))
        
        return new_hparams


class ASHAScheduler:
    """Asynchronous Successive Halving Algorithm for early stopping."""
    
    def __init__(self, max_resources: int = 4, reduction_factor: int = 3,
                 min_resources: int = 1):
        self.max_resources = max_resources  # Number of stages
        self.reduction_factor = reduction_factor
        self.min_resources = min_resources
        
        # Track trials at each rung
        self.rungs = defaultdict(list)  # rung -> list of (trial_id, score)
        self.trial_configs = {}  # trial_id -> (arena, hparams)
        self.promoted = set()  # trial_ids that have been promoted
        self.next_trial_id = 0
    
    def get_budget_for_rung(self, rung: int) -> float:
        """Get time budget for a given rung."""
        # Exponential budget: rung 0 = 1x, rung 1 = 3x, rung 2 = 9x, etc.
        return self.min_resources * (self.reduction_factor ** rung)
    
    def register_trial(self, arena: str, hparams: Dict[str, Any]) -> int:
        """Register a new trial and return its ID."""
        trial_id = self.next_trial_id
        self.next_trial_id += 1
        self.trial_configs[trial_id] = (arena, hparams)
        return trial_id
    
    def report(self, trial_id: int, rung: int, score: float) -> bool:
        """
        Report trial result at a rung.
        
        Returns True if trial should continue to next rung.
        """
        self.rungs[rung].append((trial_id, score))
        
        # Check if we have enough trials to promote
        n_at_rung = len(self.rungs[rung])
        n_to_promote = max(1, n_at_rung // self.reduction_factor)
        
        # Sort by score and check if this trial is in top
        sorted_trials = sorted(self.rungs[rung], key=lambda x: x[1])
        top_trials = [t[0] for t in sorted_trials[:n_to_promote]]
        
        should_continue = trial_id in top_trials and rung < self.max_resources - 1
        
        if should_continue:
            self.promoted.add(trial_id)
        
        return should_continue
    
    def get_trials_for_promotion(self, rung: int) -> List[Tuple[int, str, Dict[str, Any]]]:
        """Get trials that should be promoted from this rung."""
        if rung not in self.rungs:
            return []
        
        n_at_rung = len(self.rungs[rung])
        n_to_promote = max(1, n_at_rung // self.reduction_factor)
        
        sorted_trials = sorted(self.rungs[rung], key=lambda x: x[1])
        
        results = []
        for trial_id, score in sorted_trials[:n_to_promote]:
            if trial_id in self.promoted:
                continue
            arena, hparams = self.trial_configs[trial_id]
            results.append((trial_id, arena, hparams))
        
        return results


class MarginalGainAllocator:
    """Allocate compute based on marginal improvement potential."""
    
    def __init__(self):
        self.contributions = {}  # n -> current contribution (s^2/n)
        self.improvement_history = defaultdict(list)  # n -> list of improvements
    
    def update_contribution(self, n: int, contribution: float) -> None:
        """Update contribution for group n."""
        if n in self.contributions:
            improvement = self.contributions[n] - contribution
            self.improvement_history[n].append(improvement)
        self.contributions[n] = contribution
    
    def get_priority_groups(self, top_k: int = 30) -> List[int]:
        """Get groups with highest improvement potential."""
        if not self.contributions:
            return list(range(1, top_k + 1))
        
        # Sort by contribution (higher contribution = more important to optimize)
        sorted_groups = sorted(self.contributions.items(), key=lambda x: -x[1])
        return [g[0] for g in sorted_groups[:top_k]]
    
    def get_marginal_weight(self, n: int, s: float) -> float:
        """Get marginal weight for group n with current bounding square s."""
        # d/ds (s^2/n) = 2s/n
        return 2.0 * s / n if n > 0 else 0.0

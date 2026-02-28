"""
ASHA (Asynchronous Successive Halving Algorithm) scheduler for Santa 2025.

Implements early-stopping/promotion based on intermediate results at different
compute budget levels (rungs).
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class TrialResult:
    """Result from a single trial evaluation."""
    trial_id: str
    family_id: str
    hyperparams: Dict[str, Any]
    seed: int
    rung: int
    score: float  # Proxy score (lower is better)
    runtime_seconds: float
    is_valid: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trial:
    """A trial configuration."""
    trial_id: str
    family_id: str
    hyperparams: Dict[str, Any]
    seed: int
    current_rung: int = 0
    results: List[TrialResult] = field(default_factory=list)
    status: str = 'pending'  # pending, running, completed, stopped


class ASHAScheduler:
    """
    ASHA scheduler for hyperparameter tuning.
    
    Uses staged evaluation:
    - Stage 0 (rung 0): n in [1..30]
    - Stage 1 (rung 1): n in [1..60]
    - Stage 2 (rung 2): n in [1..120]
    - Stage 3 (rung 3): n in [1..200]
    
    At each rung, only top 1/eta fraction of trials are promoted.
    """
    
    def __init__(
        self,
        eta: int = 3,
        max_rungs: int = 4,
        stage_sizes: List[int] = None,
        log_dir: Path = None
    ):
        """
        Initialize ASHA scheduler.
        
        Args:
            eta: Reduction factor (promote top 1/eta trials)
            max_rungs: Number of rungs
            stage_sizes: n ranges for each rung [30, 60, 120, 200]
            log_dir: Directory for logging
        """
        self.eta = eta
        self.max_rungs = max_rungs
        self.stage_sizes = stage_sizes or [30, 60, 120, 200]
        self.log_dir = Path(log_dir) if log_dir else Path('runs')
        
        self.trials: Dict[str, Trial] = {}
        self.results_by_rung: Dict[int, List[TrialResult]] = {r: [] for r in range(max_rungs)}
        
        self._trial_counter = 0
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'asha_results.jsonl'
    
    def get_stage_ns(self, rung: int) -> List[int]:
        """Get the n values to evaluate for a given rung."""
        max_n = self.stage_sizes[min(rung, len(self.stage_sizes) - 1)]
        return list(range(1, max_n + 1))
    
    def create_trial(self, family_id: str, hyperparams: Dict, seed: int) -> Trial:
        """Create a new trial."""
        self._trial_counter += 1
        trial_id = f"trial_{self._trial_counter:04d}"
        
        trial = Trial(
            trial_id=trial_id,
            family_id=family_id,
            hyperparams=hyperparams,
            seed=seed
        )
        
        self.trials[trial_id] = trial
        return trial
    
    def get_pending_trials(self, rung: int) -> List[Trial]:
        """Get trials that need to be evaluated at a given rung."""
        pending = []
        for trial in self.trials.values():
            if trial.status == 'pending' and trial.current_rung == rung:
                pending.append(trial)
        return pending
    
    def record_result(self, result: TrialResult):
        """Record a trial result."""
        trial = self.trials[result.trial_id]
        trial.results.append(result)
        self.results_by_rung[result.rung].append(result)
        
        # Log to file - convert numpy types for JSON serialization
        result_dict = asdict(result)
        # Convert any numpy types to native Python types
        for key, value in result_dict.items():
            if hasattr(value, 'item'):  # numpy scalar
                result_dict[key] = value.item()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(v, 'item'):
                        value[k] = v.item()
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(result_dict) + '\n')
    
    def get_trials_to_promote(self, rung: int) -> List[Trial]:
        """
        Get trials that should be promoted from rung to rung+1.
        
        Returns top 1/eta fraction of completed trials at this rung.
        """
        if rung >= self.max_rungs - 1:
            return []
        
        # Get all completed results at this rung
        results = self.results_by_rung[rung]
        
        if not results:
            return []
        
        # Sort by score (lower is better)
        results_sorted = sorted(results, key=lambda r: r.score)
        
        # Promote top 1/eta
        num_promote = max(1, len(results_sorted) // self.eta)
        to_promote = results_sorted[:num_promote]
        
        # Get corresponding trials
        trials = []
        for result in to_promote:
            trial = self.trials[result.trial_id]
            if trial.status == 'pending' and trial.current_rung == rung:
                trial.current_rung = rung + 1
                trials.append(trial)
        
        return trials
    
    def mark_trial_stopped(self, trial_id: str):
        """Mark a trial as stopped (not promoted)."""
        self.trials[trial_id].status = 'stopped'
    
    def mark_trial_completed(self, trial_id: str):
        """Mark a trial as completed."""
        self.trials[trial_id].status = 'completed'
    
    def get_best_trial(self) -> Optional[Trial]:
        """Get the best trial based on final rung results."""
        final_rung = self.max_rungs - 1
        results = self.results_by_rung[final_rung]
        
        if not results:
            # Fall back to best at any rung
            for rung in range(final_rung - 1, -1, -1):
                if self.results_by_rung[rung]:
                    results = self.results_by_rung[rung]
                    break
        
        if not results:
            return None
        
        best_result = min(results, key=lambda r: r.score)
        return self.trials[best_result.trial_id]
    
    def get_statistics(self) -> Dict:
        """Get scheduler statistics."""
        stats = {
            'total_trials': len(self.trials),
            'trials_by_status': {},
            'trials_by_rung': {},
            'best_scores_by_rung': {}
        }
        
        for trial in self.trials.values():
            stats['trials_by_status'][trial.status] = stats['trials_by_status'].get(trial.status, 0) + 1
            stats['trials_by_rung'][trial.current_rung] = stats['trials_by_rung'].get(trial.current_rung, 0) + 1
        
        for rung, results in self.results_by_rung.items():
            if results:
                stats['best_scores_by_rung'][rung] = min(r.score for r in results)
        
        return stats


def sample_hyperparams(
    family_id: str,
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Sample hyperparameters for a given family.
    
    Args:
        family_id: Solver family (F0, F1, F2, F3, F4)
        rng: Random generator
        
    Returns:
        Dict of hyperparameters
    """
    params = {}
    
    if family_id == 'F0':  # Greedy
        params['num_candidates'] = rng.choice([100, 200, 300, 500])
    
    elif family_id == 'F1':  # Lattice
        params['v1_length'] = rng.uniform(0.8, 1.5)
        params['v1_angle'] = rng.uniform(0, 360)
        params['v2_angle_offset'] = rng.uniform(60, 120)
        params['refine_iterations'] = rng.choice([50, 100, 200])
        params['refine_step_size'] = rng.uniform(0.05, 0.2)
    
    elif family_id == 'F2':  # SA
        params['T0'] = 10 ** rng.uniform(-1, 1)  # 0.1 to 10
        params['T_end'] = 10 ** rng.uniform(-4, -2)  # 0.0001 to 0.01
        params['alpha'] = rng.uniform(0.99, 0.999)
        params['dx0'] = rng.uniform(0.1, 0.5)
        params['dy0'] = rng.uniform(0.1, 0.5)
        params['ddeg0'] = rng.choice([45, 90, 180])
        params['p_translate'] = rng.uniform(0.3, 0.7)
        params['p_rotate'] = rng.uniform(0.1, 0.3)
        params['p_swap'] = rng.uniform(0.05, 0.2)
        params['p_boundary'] = rng.uniform(0.1, 0.3)
        params['iterations'] = rng.choice([5000, 10000, 20000])
    
    elif family_id == 'F3':  # ILS/VNS
        params['kick_strength'] = rng.uniform(0.1, 0.5)
        params['sa_iterations'] = rng.choice([500, 1000, 2000])
        params['ils_iterations'] = rng.choice([20, 50, 100])
    
    elif family_id == 'F4':  # Shrink-repair
        params['initial_delta'] = rng.uniform(0.02, 0.1)
        params['min_delta'] = rng.uniform(0.0005, 0.005)
        params['delta_decay'] = rng.uniform(0.8, 0.95)
    
    return params


def sample_trial_config(rng: np.random.Generator) -> Tuple[str, Dict, int]:
    """
    Sample a trial configuration (family, hyperparams, seed).
    
    Returns:
        (family_id, hyperparams, seed)
    """
    # Choose family
    families = ['F0', 'F1', 'F2', 'F3', 'F4']
    weights = [0.1, 0.1, 0.4, 0.2, 0.2]  # Prefer SA
    
    family_id = rng.choice(families, p=weights)
    hyperparams = sample_hyperparams(family_id, rng)
    seed = rng.integers(0, 2**31)
    
    return family_id, hyperparams, seed

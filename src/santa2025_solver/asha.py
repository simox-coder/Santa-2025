"""
ASHA (Asynchronous Successive Halving Algorithm) implementation.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict

class ASHA:
    """
    ASHA scheduler for hyperparameter optimization.
    
    Implements early stopping based on performance at each rung.
    """
    
    def __init__(self, max_rung: int = 4, eta: int = 3, 
                 base_budget_sec: float = 60.0):
        """
        Args:
            max_rung: Maximum rung (stage) level (0-indexed)
            eta: Reduction factor (keep 1/eta of trials at each rung)
            base_budget_sec: Base time budget for rung 0
        """
        self.max_rung = max_rung
        self.eta = eta
        self.base_budget_sec = base_budget_sec
        
        # Track trials
        self.trials = {}  # trial_id -> {arena, hparams, results_per_rung}
        self.rung_results = defaultdict(list)  # rung -> [(trial_id, score)]
        self.next_trial_id = 0
    
    def get_budget_for_rung(self, rung: int) -> float:
        """Get time budget (seconds) for a rung."""
        return self.base_budget_sec * (self.eta ** rung)
    
    def get_stage_for_rung(self, rung: int) -> int:
        """Map rung to stage (S0-S3)."""
        # rung 0 -> stage 0 (n=1-30)
        # rung 1 -> stage 1 (n=1-60)
        # rung 2 -> stage 2 (n=1-120)
        # rung 3 -> stage 3 (n=1-200)
        return min(rung, 3)
    
    def create_trial(self, arena: str, hparams: Dict[str, Any]) -> int:
        """Create a new trial."""
        trial_id = self.next_trial_id
        self.next_trial_id += 1
        
        self.trials[trial_id] = {
            'arena': arena,
            'hparams': hparams.copy(),
            'results': {},  # rung -> {score, layouts, ...}
            'current_rung': 0,
            'stopped': False
        }
        
        return trial_id
    
    def record_result(self, trial_id: int, rung: int, 
                      result: Dict[str, Any]) -> bool:
        """
        Record result for a trial at a rung.
        
        Returns True if trial should continue to next rung.
        """
        if trial_id not in self.trials:
            raise ValueError(f"Unknown trial {trial_id}")
        
        trial = self.trials[trial_id]
        trial['results'][rung] = result
        trial['current_rung'] = rung
        
        # Add to rung results
        score = result.get('score_stage', float('inf'))
        self.rung_results[rung].append((trial_id, score))
        
        # Check if should promote
        return self._should_promote(trial_id, rung)
    
    def _should_promote(self, trial_id: int, rung: int) -> bool:
        """Check if trial should be promoted to next rung."""
        if rung >= self.max_rung - 1:
            return False
        
        # Get all results at this rung
        results = self.rung_results[rung]
        n_trials = len(results)
        
        if n_trials < 2:
            return True  # Not enough to compare
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x[1])
        
        # Keep top 1/eta
        n_promote = max(1, n_trials // self.eta)
        top_trials = {t[0] for t in sorted_results[:n_promote]}
        
        return trial_id in top_trials
    
    def get_trial(self, trial_id: int) -> Dict[str, Any]:
        """Get trial info."""
        return self.trials.get(trial_id)
    
    def get_best_trials(self, arena: str = None, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get best trials overall or for a specific arena."""
        results = []
        
        for trial_id, trial in self.trials.items():
            if arena and trial['arena'] != arena:
                continue
            
            # Get best score from any rung
            if not trial['results']:
                continue
            
            best_score = min(r.get('score_stage', float('inf')) 
                           for r in trial['results'].values())
            results.append((trial_id, best_score))
        
        return sorted(results, key=lambda x: x[1])[:top_k]
    
    def get_arena_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics per arena."""
        stats = defaultdict(lambda: {'n_trials': 0, 'best_score': float('inf'), 
                                     'avg_score': 0.0, 'scores': []})
        
        for trial_id, trial in self.trials.items():
            arena = trial['arena']
            stats[arena]['n_trials'] += 1
            
            if trial['results']:
                best = min(r.get('score_stage', float('inf')) 
                          for r in trial['results'].values())
                stats[arena]['scores'].append(best)
                if best < stats[arena]['best_score']:
                    stats[arena]['best_score'] = best
        
        for arena in stats:
            if stats[arena]['scores']:
                stats[arena]['avg_score'] = np.mean(stats[arena]['scores'])
        
        return dict(stats)

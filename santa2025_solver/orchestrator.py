"""
Orchestrator for Santa 2025 solver.

Manages parallel trial execution with ASHA scheduling.
"""

import os
import sys
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from .asha import ASHAScheduler, TrialResult, sample_trial_config
from .trial_runner import run_trial, TrialOutput
from .submission import write_submission
from .score import print_public_score


def get_max_workers() -> int:
    """Get number of worker processes."""
    env_workers = os.environ.get('MAX_WORKERS')
    if env_workers:
        return int(env_workers)
    return os.cpu_count() or 4


def get_time_budget_minutes() -> float:
    """Get total time budget in minutes."""
    env_budget = os.environ.get('TOTAL_BUDGET_MIN')
    if env_budget:
        return float(env_budget)
    return 180.0  # Default 3 hours


def worker_run_trial(
    trial_id: str,
    family_id: str,
    hyperparams: Dict,
    seed: int,
    ns: List[int],
    time_budget: float
) -> Dict:
    """
    Worker function to run a single trial.
    
    This runs in a separate process.
    """
    from .trial_runner import run_trial
    
    start = time.time()
    output = run_trial(family_id, hyperparams, seed, ns, time_budget)
    elapsed = time.time() - start
    
    return {
        'trial_id': trial_id,
        'family_id': family_id,
        'seed': seed,
        'score': output.total_score,
        'runtime': elapsed,
        'is_valid': output.is_valid,
        'n_solved': len(output.layouts),
        'scores_per_n': {n: s for n, s in output.scores.items()}
    }


class Orchestrator:
    """
    Main orchestrator for running ASHA-scheduled trials.
    """
    
    def __init__(
        self,
        max_workers: int = None,
        time_budget_minutes: float = None,
        artifacts_dir: str = 'artifacts',
        runs_dir: str = 'runs'
    ):
        self.max_workers = max_workers or get_max_workers()
        self.time_budget_minutes = time_budget_minutes or get_time_budget_minutes()
        self.artifacts_dir = Path(artifacts_dir)
        self.runs_dir = Path(runs_dir)
        
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        self.scheduler = ASHAScheduler(
            eta=3,
            max_rungs=4,
            stage_sizes=[30, 60, 120, 200],
            log_dir=self.runs_dir
        )
        
        self.rng = np.random.default_rng()
        
        self.best_layouts: Dict[int, Any] = {}
        self.best_score = float('inf')
        self.best_config = None
    
    def run(self):
        """Run the full orchestration loop."""
        print(f"Starting orchestrator")
        print(f"  Max workers: {self.max_workers}")
        print(f"  Time budget: {self.time_budget_minutes} minutes")
        print()
        
        start_time = time.time()
        deadline = start_time + self.time_budget_minutes * 60
        
        # Initial trial generation
        num_initial_trials = self.max_workers * 3
        print(f"Generating {num_initial_trials} initial trials...")
        
        for _ in range(num_initial_trials):
            family_id, hyperparams, seed = sample_trial_config(self.rng)
            self.scheduler.create_trial(family_id, hyperparams, seed)
        
        # Main loop
        rung = 0
        iteration = 0
        
        while time.time() < deadline and rung < self.scheduler.max_rungs:
            iteration += 1
            print(f"\n=== Iteration {iteration}, Rung {rung} ===")
            
            # Get pending trials at current rung
            pending = self.scheduler.get_pending_trials(rung)
            
            if not pending:
                # Try to promote trials from previous rung
                if rung > 0:
                    promoted = self.scheduler.get_trials_to_promote(rung - 1)
                    if promoted:
                        print(f"Promoted {len(promoted)} trials to rung {rung}")
                        pending = promoted
                
                if not pending:
                    # Generate new trials if at rung 0
                    if rung == 0:
                        for _ in range(self.max_workers):
                            family_id, hyperparams, seed = sample_trial_config(self.rng)
                            trial = self.scheduler.create_trial(family_id, hyperparams, seed)
                            pending.append(trial)
                    else:
                        # Move to next rung
                        rung += 1
                        continue
            
            if not pending:
                break
            
            # Run trials in parallel
            ns = self.scheduler.get_stage_ns(rung)
            time_per_trial = min(300, (deadline - time.time()) / max(len(pending), 1))
            
            print(f"Running {len(pending)} trials for n in [1..{max(ns)}]")
            
            results = self._run_trials_parallel(pending, ns, time_per_trial)
            
            # Record results
            for result_dict in results:
                trial = self.scheduler.trials[result_dict['trial_id']]
                
                result = TrialResult(
                    trial_id=result_dict['trial_id'],
                    family_id=trial.family_id,
                    hyperparams=trial.hyperparams,
                    seed=trial.seed,
                    rung=rung,
                    score=result_dict['score'],
                    runtime_seconds=result_dict['runtime'],
                    is_valid=result_dict['is_valid']
                )
                
                self.scheduler.record_result(result)
                
                # Track best
                if result_dict['is_valid'] and result_dict['score'] < self.best_score:
                    self.best_score = result_dict['score']
                    self.best_config = {
                        'family_id': trial.family_id,
                        'hyperparams': trial.hyperparams,
                        'seed': trial.seed
                    }
                    print(f"  New best! Score: {self.best_score:.6f}")
            
            # Print progress
            stats = self.scheduler.get_statistics()
            print(f"Stats: {stats['total_trials']} trials, best scores: {stats['best_scores_by_rung']}")
        
        # Final full solve with best config
        print("\n=== Final Solve ===")
        self._final_solve()
        
        # Save results
        self._save_results()
    
    def _run_trials_parallel(
        self,
        trials: List,
        ns: List[int],
        time_per_trial: float
    ) -> List[Dict]:
        """Run trials in parallel using process pool."""
        results = []
        
        # For simplicity, run sequentially if max_workers == 1
        if self.max_workers == 1:
            for trial in trials:
                result = worker_run_trial(
                    trial.trial_id,
                    trial.family_id,
                    trial.hyperparams,
                    trial.seed,
                    ns,
                    time_per_trial
                )
                results.append(result)
        else:
            # Run in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for trial in trials:
                    future = executor.submit(
                        worker_run_trial,
                        trial.trial_id,
                        trial.family_id,
                        trial.hyperparams,
                        trial.seed,
                        ns,
                        time_per_trial
                    )
                    futures[future] = trial
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Trial failed: {e}")
        
        return results
    
    def _final_solve(self):
        """Run final solve with best configuration."""
        if self.best_config is None:
            print("No best config found, using default greedy")
            self.best_config = {
                'family_id': 'F0',
                'hyperparams': {'num_candidates': 300},
                'seed': 42
            }
        
        print(f"Final solve with: {self.best_config['family_id']}")
        
        output = run_trial(
            self.best_config['family_id'],
            self.best_config['hyperparams'],
            self.best_config['seed'],
            list(range(1, 201)),
            time_budget=None
        )
        
        self.best_layouts = output.layouts
        self.best_score = output.total_score
        
        print(f"Final score: {self.best_score:.6f}")
        print(f"Valid: {output.is_valid}")
    
    def _save_results(self):
        """Save best submission and configuration."""
        # Save config
        config_path = self.artifacts_dir / 'best_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.best_config, f)
        print(f"Saved config to: {config_path}")
        
        # Save submission
        submission_path = self.artifacts_dir / 'submission_best.csv'
        write_submission(str(submission_path), self.best_layouts)
        
        # Save score
        score_path = self.artifacts_dir / 'score.txt'
        with open(score_path, 'w') as f:
            f.write(f"{self.best_score}\n")
        
        print()
        print_public_score(self.best_score)


def main():
    """CLI entry point."""
    orchestrator = Orchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()

"""
Santa 2025 Solver - Orchestrator

Orchestrates parallel execution of trials with ASHA scheduling.
"""

import os
import sys
import time
import yaml
import json
import multiprocessing as mp
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from santa2025_solver.system_info import (
    get_system_info, compute_safe_workers, format_system_info_markdown
)
from santa2025_solver.backend_select import select_best_backend
from santa2025_solver.asha import ASHAScheduler, Trial
from santa2025_solver.trial_runner import (
    run_trial, sample_hyperparams, create_trial_batch, TrialResult
)
from santa2025_solver.geometry_fast import Layout, compute_bounding_square_side
from santa2025_solver.submission import write_submission
from santa2025_solver.validate import validate_submission


def worker_run_trial(args: Tuple) -> Tuple[str, TrialResult]:
    """Worker function to run a single trial."""
    trial_id, family_id, hyperparams, seed, n_range, budget_per_n = args
    
    try:
        result = run_trial(
            family_id=family_id,
            hyperparams=hyperparams,
            seed=seed,
            n_range=n_range,
            budget_per_n=budget_per_n
        )
        return trial_id, result
    except Exception as e:
        # Return failed result
        return trial_id, TrialResult(
            proxy_score=float('inf'),
            runtime=0,
            feasible=False,
            layouts={},
            details={'error': str(e)}
        )


class Orchestrator:
    """
    Orchestrates parallel ASHA hyperparameter tuning.
    """
    
    def __init__(
        self,
        budget_minutes: int = 180,
        max_workers: str = "auto",
        collision_backend: str = "auto"
    ):
        self.budget_minutes = budget_minutes
        self.max_workers_setting = max_workers
        self.collision_backend_setting = collision_backend
        
        self.scheduler = ASHAScheduler()
        self.best_layouts: Dict[int, Layout] = {}
        self.best_score: float = float('inf')
        
        # Paths
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.backend: str = "python"
        self.n_workers: int = 1
        self.system_info: Dict = {}
    
    def _initialize(self):
        """Initialize orchestrator: detect system, select backend, etc."""
        print("Initializing orchestrator...")
        
        # Get system info
        self.system_info = get_system_info()
        print(f"  CPU cores: {self.system_info['cpu_count_physical']} physical, "
              f"{self.system_info['cpu_count_logical']} logical")
        print(f"  Memory: {self.system_info['memory_available_gb']:.1f} GB available")
        
        # Select collision backend
        print(f"  Selecting collision backend ({self.collision_backend_setting})...")
        self.backend = select_best_backend(
            force_benchmark=(self.collision_backend_setting == 'auto'),
            verbose=True
        )
        print(f"  Selected backend: {self.backend}")
        
        # Determine worker count
        if self.max_workers_setting == "auto":
            self.n_workers = compute_safe_workers(
                mem_per_worker_gb=0.5,
                reserve_cores=1
            )
        else:
            self.n_workers = int(self.max_workers_setting)
        
        print(f"  Using {self.n_workers} workers")
        
        # Update dashboard
        self.scheduler.update_dashboard({
            'system_info': self.system_info,
            'backend': self.backend,
            'workers': self.n_workers
        })
    
    def _generate_initial_trials(self, n_trials: int = 50) -> List[str]:
        """Generate initial batch of trials."""
        print(f"Generating {n_trials} initial trials...")
        
        rng = np.random.RandomState(42)
        trial_ids = []
        
        # Distribute across families
        family_quotas = {0: 5, 1: 5, 2: 20, 3: 10, 4: 10}
        
        for family_id, quota in family_quotas.items():
            for _ in range(quota):
                seed = rng.randint(0, 2**31)
                hyperparams = sample_hyperparams(family_id, rng)
                trial_id = self.scheduler.add_trial(family_id, hyperparams, seed)
                trial_ids.append(trial_id)
        
        return trial_ids
    
    def _run_stage(self, stage: int, deadline: float):
        """Run all pending trials for a stage."""
        stage_info = ASHAScheduler.STAGES[stage]
        n_range = stage_info['n_range']
        budget_factor = stage_info['budget_factor']
        
        print(f"\n=== Running {stage_info['name']} (n={n_range[0]}..{n_range[1]}) ===")
        
        # Get pending trials
        pending_trials = self.scheduler.get_pending_trials(stage)
        
        # Add promotable trials from previous stage
        if stage > 0:
            promotable = self.scheduler.get_promotable_trials(stage - 1)
            for trial_id in promotable:
                self.scheduler.promote_trial(trial_id)
            pending_trials.extend(promotable)
        
        if not pending_trials:
            print("  No pending trials for this stage")
            return
        
        print(f"  {len(pending_trials)} trials to run")
        
        # Calculate budget per n
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            return
        
        n_count = n_range[1] - n_range[0] + 1
        budget_per_n = min(1.0, remaining_time / (len(pending_trials) * n_count) * 0.8)
        
        # Prepare work items
        work_items = []
        for trial_id in pending_trials:
            trial = self.scheduler.trials[trial_id]
            work_items.append((
                trial_id,
                trial.family_id,
                trial.hyperparams,
                trial.seed,
                n_range,
                budget_per_n
            ))
        
        # Run trials
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(worker_run_trial, item): item[0] 
                      for item in work_items}
            
            for future in as_completed(futures):
                if time.time() > deadline:
                    break
                
                trial_id = futures[future]
                
                try:
                    _, result = future.result(timeout=30)
                    
                    # Report result
                    self.scheduler.report_result(
                        trial_id,
                        proxy_score=result.proxy_score,
                        runtime=result.runtime,
                        feasible=result.feasible
                    )
                    
                    # Update best if improved
                    if result.feasible and result.proxy_score < self.best_score:
                        self.best_score = result.proxy_score
                        self.best_layouts = result.layouts
                        print(f"  New best: {result.proxy_score:.4f} (trial {trial_id})")
                    
                    completed += 1
                    
                except Exception as e:
                    print(f"  Trial {trial_id} failed: {e}")
                    self.scheduler.report_result(
                        trial_id,
                        proxy_score=float('inf'),
                        runtime=0,
                        feasible=False
                    )
        
        print(f"  Completed {completed}/{len(pending_trials)} trials")
        
        # Update dashboard
        self.scheduler.update_dashboard({
            'system_info': self.system_info,
            'backend': self.backend,
            'workers': self.n_workers
        })
    
    def _complete_full_solution(self):
        """
        Complete the solution for all n=1..200.
        
        Uses the best configuration found to fill in missing n values.
        """
        print("\n=== Completing full solution ===")
        
        best_trial = self.scheduler.get_best_trial()
        if best_trial is None:
            print("  No valid trial found, using greedy fallback")
            family_id = 0
            hyperparams = {}
            seed = 42
        else:
            trial_id, trial = best_trial
            print(f"  Using best trial: {trial_id} (score: {trial.best_score:.4f})")
            family_id = trial.family_id
            hyperparams = trial.hyperparams
            seed = trial.seed
        
        # Fill in missing n values
        rng = np.random.RandomState(seed)
        
        from santa2025_solver.trial_runner import get_solver_class
        solver_class = get_solver_class(family_id)
        solver = solver_class(hyperparams)
        
        for n in range(1, 201):
            if n not in self.best_layouts:
                layout = solver.solve(n, 1.0, rng, hyperparams)
                self.best_layouts[n] = layout
        
        # Compute final score
        total_score = 0.0
        for n in range(1, 201):
            s = compute_bounding_square_side(
                self.best_layouts[n].positions,
                self.best_layouts[n].rotations
            )
            total_score += s * s
        
        self.best_score = total_score
        print(f"  Final score: {total_score:.6f}")
    
    def _save_results(self):
        """Save best results to files."""
        print("\n=== Saving results ===")
        
        # Save submission
        submission_path = self.artifacts_dir / "submission_best.csv"
        write_submission(self.best_layouts, str(submission_path))
        print(f"  Saved submission to {submission_path}")
        
        # Save best config
        best_trial = self.scheduler.get_best_trial()
        if best_trial:
            trial_id, trial = best_trial
            config = {
                'family_id': trial.family_id,
                'hyperparams': trial.hyperparams,
                'seed': trial.seed,
                'score': trial.best_score,
            }
            config_path = self.artifacts_dir / "best_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"  Saved config to {config_path}")
        
        # Save score
        score_path = self.artifacts_dir / "score.txt"
        with open(score_path, 'w') as f:
            f.write(f"Public Score: {self.best_score}\n")
        print(f"  Saved score to {score_path}")
        
        # Final dashboard update
        self.scheduler.update_dashboard({
            'system_info': self.system_info,
            'backend': self.backend,
            'workers': self.n_workers
        })
    
    def run(self):
        """Run the full orchestration."""
        print("=" * 60)
        print("Santa 2025 Solver - Hyperparameter Tuning")
        print("=" * 60)
        
        start_time = time.time()
        deadline = start_time + self.budget_minutes * 60
        
        # Initialize
        self._initialize()
        
        # Generate initial trials
        self._generate_initial_trials(n_trials=50)
        
        # Run stages
        for stage in range(4):
            if time.time() > deadline:
                print("\nTime budget exceeded, stopping early")
                break
            self._run_stage(stage, deadline)
        
        # Complete solution
        self._complete_full_solution()
        
        # Save results
        self._save_results()
        
        # Print final score
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Completed in {elapsed/60:.1f} minutes")
        print(f"Public Score: {self.best_score}")
        print("=" * 60)


def run_tuning(budget_minutes: int = 180, max_workers: str = "auto", collision_backend: str = "auto"):
    """Entry point for tuning."""
    orchestrator = Orchestrator(
        budget_minutes=budget_minutes,
        max_workers=max_workers,
        collision_backend=collision_backend
    )
    orchestrator.run()


def generate_from_config(config_path: str = "artifacts/best_config.yaml", 
                         output_path: str = "artifacts/submission_best.csv"):
    """Generate submission from saved config."""
    print(f"Loading config from {config_path}...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    family_id = config['family_id']
    hyperparams = config['hyperparams']
    seed = config['seed']
    
    print(f"Family: {family_id}, Seed: {seed}")
    
    from santa2025_solver.trial_runner import get_solver_class
    solver_class = get_solver_class(family_id)
    solver = solver_class(hyperparams)
    
    rng = np.random.RandomState(seed)
    layouts = {}
    
    for n in range(1, 201):
        if n % 20 == 0:
            print(f"  Solving n={n}...")
        layout = solver.solve(n, 2.0, rng, hyperparams)
        layouts[n] = layout
    
    # Save
    write_submission(layouts, output_path)
    print(f"Saved submission to {output_path}")
    
    # Compute score
    total_score = 0.0
    for n in range(1, 201):
        s = compute_bounding_square_side(layouts[n].positions, layouts[n].rotations)
        total_score += s * s
    
    print(f"Public Score: {total_score}")


if __name__ == "__main__":
    # Default: run tuning
    budget = int(os.environ.get("TOTAL_BUDGET_MIN", "180"))
    workers = os.environ.get("MAX_WORKERS", "auto")
    backend = os.environ.get("COLLISION_BACKEND", "auto")
    
    run_tuning(budget_minutes=budget, max_workers=workers, collision_backend=backend)

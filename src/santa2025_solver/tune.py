"""
Hyperparameter tuning module with ASHA scheduler.

Implements multi-fidelity optimization for finding the best solver configuration.
"""

import os
import sys
import time
import json
import random
import math
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from santa2025_solver.solvers import (
    solve_n, solve_all_n, Layout, SolverConfig, repair_overlaps
)
from santa2025_solver.submission import write_submission
from santa2025_solver.kaggle_ref.metric_ref import calculate_score
from santa2025_solver.collision import strict_validate_layout


@dataclass
class Trial:
    """Represents a hyperparameter trial."""
    trial_id: int
    family: str  # "greedy", "sa", "ils"
    config: dict
    seed: int
    score: float = float('inf')
    rung: int = 0
    elapsed: float = 0.0
    valid: bool = False


def generate_config_space(family: str, rng: random.Random) -> dict:
    """Generate random configuration from search space.
    
    Args:
        family: Solver family name
        rng: Random number generator
        
    Returns:
        Configuration dictionary
    """
    config = {
        'seed': rng.randint(0, 10000),
    }
    
    if family == "greedy":
        config['num_candidates'] = rng.choice([24, 36, 48, 72])
    elif family == "sa":
        config['max_iterations'] = rng.choice([3000, 5000, 8000, 10000])
        config['temperature_init'] = math.exp(rng.uniform(-1, 1))  # 0.37 to 2.7
        config['cooling_rate'] = rng.uniform(0.990, 0.999)
        config['move_step'] = rng.uniform(0.02, 0.1)
        config['rotation_step'] = rng.choice([10, 15, 30, 45, 90])
    elif family == "ils":
        config['max_iterations'] = rng.choice([2000, 3000, 5000])
        config['temperature_init'] = math.exp(rng.uniform(-0.5, 0.5))
        config['cooling_rate'] = rng.uniform(0.990, 0.998)
        config['move_step'] = rng.uniform(0.03, 0.08)
    
    return config


def run_trial_partial(trial: Trial, n_max: int) -> Tuple[float, bool]:
    """Run a trial for n=1 to n_max.
    
    Args:
        trial: Trial configuration
        n_max: Maximum n to solve
        
    Returns:
        Tuple of (total_radius, is_valid)
    """
    solver_config = SolverConfig(
        seed=trial.seed,
        max_iterations=trial.config.get('max_iterations', 3000),
        temperature_init=trial.config.get('temperature_init', 1.0),
        cooling_rate=trial.config.get('cooling_rate', 0.995),
        move_step=trial.config.get('move_step', 0.05),
        rotation_step=trial.config.get('rotation_step', 15.0),
        num_candidates=trial.config.get('num_candidates', 36),
        verbose=False
    )
    
    # Solve
    solutions = solve_all_n(n_max, solver_config, verbose=False)
    
    # Calculate score and validate
    total_radius = 0.0
    all_valid = True
    
    for n, layout in solutions.items():
        total_radius += layout.get_radius()
        if not strict_validate_layout(layout.positions):
            all_valid = False
    
    return total_radius, all_valid


def asha_tune(total_budget_minutes: float = 30,
              max_workers: int = None,
              output_dir: str = "artifacts",
              export_dir: str = "exports") -> Trial:
    """Run ASHA hyperparameter tuning.
    
    ASHA = Asynchronous Successive Halving Algorithm
    
    Args:
        total_budget_minutes: Total time budget in minutes
        max_workers: Number of parallel workers
        output_dir: Directory for artifacts
        export_dir: Directory for exports
        
    Returns:
        Best trial found
    """
    print("=" * 60)
    print("ASHA Hyperparameter Tuning")
    print("=" * 60)
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)
    
    # Create directories
    Path(output_dir).mkdir(exist_ok=True)
    Path(export_dir).mkdir(exist_ok=True)
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # ASHA parameters
    eta = 3  # Halving rate
    rungs = [
        {"n_max": 30, "name": "rung0"},   # Quick eval
        {"n_max": 60, "name": "rung1"},   # Medium
        {"n_max": 120, "name": "rung2"},  # Long
        {"n_max": 200, "name": "rung3"},  # Full
    ]
    
    # Trial tracking
    trials = []
    trial_counter = 0
    rng = random.Random(42)
    
    families = ["greedy", "sa", "ils"]
    
    start_time = time.time()
    deadline = start_time + total_budget_minutes * 60
    
    best_trial = None
    best_score = float('inf')
    
    # Generate initial population
    print(f"\nGenerating initial trials (budget: {total_budget_minutes}min)...")
    for family in families:
        for _ in range(5):  # 5 trials per family
            config = generate_config_space(family, rng)
            trial = Trial(
                trial_id=trial_counter,
                family=family,
                config=config,
                seed=config['seed']
            )
            trials.append(trial)
            trial_counter += 1
    
    print(f"Created {len(trials)} initial trials")
    
    # Run through rungs
    for rung_idx, rung in enumerate(rungs):
        if time.time() > deadline:
            print(f"Time limit reached at rung {rung_idx}")
            break
        
        n_max = rung["n_max"]
        rung_name = rung["name"]
        
        # Get trials that haven't been evaluated at this rung
        pending = [t for t in trials if t.rung == rung_idx]
        
        if not pending:
            continue
        
        print(f"\n--- {rung_name}: n_max={n_max}, {len(pending)} trials ---")
        
        # Evaluate trials at this rung
        for trial in pending:
            if time.time() > deadline:
                break
            
            print(f"  Trial {trial.trial_id} ({trial.family})...", end=" ", flush=True)
            
            start_eval = time.time()
            score, valid = run_trial_partial(trial, n_max)
            elapsed = time.time() - start_eval
            
            trial.score = score
            trial.valid = valid
            trial.elapsed = elapsed
            trial.rung = rung_idx + 1
            
            print(f"score={score:.4f}, valid={valid}, time={elapsed:.1f}s")
            
            # Track best
            if valid and score < best_score:
                best_score = score
                best_trial = trial
                print(f"  *** NEW BEST: {score:.4f} ***")
        
        # Promote top 1/eta trials to next rung
        if rung_idx < len(rungs) - 1:
            evaluated = [t for t in trials if t.rung > rung_idx and t.valid]
            evaluated.sort(key=lambda t: t.score)
            
            n_promote = max(1, len(evaluated) // eta)
            promoted = evaluated[:n_promote]
            
            print(f"  Promoting {n_promote} trials to next rung")
            
            for t in promoted:
                t.rung = rung_idx + 1  # Ready for next rung
    
    print("\n" + "=" * 60)
    
    if best_trial is None:
        print("WARNING: No valid trial found! Using default config.")
        best_trial = Trial(
            trial_id=-1,
            family="ils",
            config={'max_iterations': 5000},
            seed=42,
            score=float('inf'),
            valid=False
        )
    else:
        print(f"Best trial: {best_trial.trial_id} ({best_trial.family})")
        print(f"Best partial score: {best_trial.score:.4f}")
    
    # Run full solve with best config
    print("\nRunning full solve with best configuration...")
    
    solver_config = SolverConfig(
        seed=best_trial.seed,
        max_iterations=best_trial.config.get('max_iterations', 5000),
        temperature_init=best_trial.config.get('temperature_init', 1.0),
        cooling_rate=best_trial.config.get('cooling_rate', 0.995),
        move_step=best_trial.config.get('move_step', 0.05),
        rotation_step=best_trial.config.get('rotation_step', 15.0),
        num_candidates=best_trial.config.get('num_candidates', 36),
        verbose=True
    )
    
    solutions = solve_all_n(200, solver_config, verbose=True)
    
    # Validate and repair
    print("\nValidating solutions...")
    for n, layout in solutions.items():
        if not strict_validate_layout(layout.positions):
            print(f"  Repairing n={n}...")
            solutions[n] = repair_overlaps(layout, solver_config)
    
    # Write submission
    submission_path = Path(output_dir) / "submission_best.csv"
    write_submission(solutions, str(submission_path))
    
    # Calculate final score
    final_score = calculate_score(str(submission_path))
    
    print(f"\nPublic Score: {final_score:.6f}")
    
    # Save artifacts
    score_path = Path(output_dir) / "score.txt"
    with open(score_path, 'w') as f:
        f.write(f"{final_score:.6f}\n")
    
    config_out = {
        'family': best_trial.family,
        'trial_id': best_trial.trial_id,
        'seed': best_trial.seed,
        'score': final_score,
        **best_trial.config
    }
    config_path = Path(output_dir) / "best_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_out, f)
    
    # Export
    import shutil
    shutil.copy(submission_path, Path(export_dir) / "submission_best.csv")
    shutil.copy(score_path, Path(export_dir) / "score.txt")
    shutil.copy(config_path, Path(export_dir) / "best_config.yaml")
    
    # Dashboard
    generate_tuning_dashboard(trials, final_score, output_dir, export_dir)
    
    print(f"\nExported to: {export_dir}/")
    
    return best_trial


def generate_tuning_dashboard(trials: List[Trial], final_score: float,
                              output_dir: str, export_dir: str):
    """Generate tuning dashboard."""
    lines = [
        "# Santa 2025 Tuning Dashboard",
        "",
        f"## Final Score",
        f"**Public Score: {final_score:.6f}**",
        "",
        "## ASHA Trials",
        "",
        "| Trial | Family | Rung | Score | Valid | Time |",
        "|-------|--------|------|-------|-------|------|",
    ]
    
    for t in sorted(trials, key=lambda x: x.score):
        lines.append(
            f"| {t.trial_id} | {t.family} | {t.rung} | {t.score:.4f} | {t.valid} | {t.elapsed:.1f}s |"
        )
    
    lines.extend([
        "",
        "---",
        "Generated by ASHA Tuner"
    ])
    
    dashboard = '\n'.join(lines)
    
    for dir_path in [output_dir, export_dir]:
        with open(Path(dir_path) / "dashboard.md", 'w') as f:
            f.write(dashboard)


def main():
    """Main entry point for tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ASHA hyperparameter tuning")
    parser.add_argument("--budget", type=float, default=30,
                        help="Time budget in minutes (default: 30)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--output-dir", type=str, default="artifacts",
                        help="Output directory")
    parser.add_argument("--export-dir", type=str, default="exports",
                        help="Export directory")
    args = parser.parse_args()
    
    asha_tune(
        total_budget_minutes=args.budget,
        max_workers=args.workers,
        output_dir=args.output_dir,
        export_dir=args.export_dir
    )


if __name__ == "__main__":
    main()

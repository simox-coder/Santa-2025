"""
Trial runner for Santa 2025 solver.

Runs a single trial (family + hyperparams + seed) and returns results.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .geometry_fast import TreeLayout
from .collision_fast import CollisionChecker
from .greedy import greedy_solve, greedy_solve_with_warmstart
from .lattice import lattice_solve
from .sa import sa_full_solve
from .ils_vns import ils_full_solve
from .shrink_repair import shrink_repair_full_solve


@dataclass
class TrialOutput:
    """Output from a trial run."""
    layouts: Dict[int, TreeLayout]
    scores: Dict[int, float]
    total_score: float
    runtime_seconds: float
    is_valid: bool


def run_solver_for_n(
    n: int,
    family_id: str,
    rng: np.random.Generator,
    hyperparams: Dict,
    prev_layout: Optional[TreeLayout] = None,
    time_limit: float = None
) -> Optional[TreeLayout]:
    """
    Run solver for a specific n.
    
    Args:
        n: Number of trees
        family_id: Solver family
        rng: Random generator
        hyperparams: Solver parameters
        prev_layout: Previous layout for warm-starting
        time_limit: Time limit in seconds
        
    Returns:
        TreeLayout or None if failed
    """
    try:
        if family_id == 'F0':
            return greedy_solve_with_warmstart(n, prev_layout, rng, hyperparams)
        
        elif family_id == 'F1':
            return lattice_solve(n, rng, hyperparams)
        
        elif family_id == 'F2':
            init_layout = None
            if prev_layout and prev_layout.n == n - 1:
                # Extend previous layout
                init_layout = TreeLayout(n)
                for i in range(n - 1):
                    init_layout.set_tree(i, *prev_layout.get_tree(i))
                # Place new tree at origin initially
                init_layout.set_tree(n - 1, 0.0, 0.0, 0.0)
            return sa_full_solve(n, rng, hyperparams, init_layout)
        
        elif family_id == 'F3':
            return ils_full_solve(n, rng, hyperparams, prev_layout)
        
        elif family_id == 'F4':
            return shrink_repair_full_solve(n, rng, hyperparams, prev_layout)
        
        else:
            raise ValueError(f"Unknown family: {family_id}")
    
    except Exception as e:
        print(f"Error solving n={n} with {family_id}: {e}")
        return None


def run_trial(
    family_id: str,
    hyperparams: Dict,
    seed: int,
    ns: List[int],
    time_budget: float = None
) -> TrialOutput:
    """
    Run a complete trial for specified n values.
    
    Args:
        family_id: Solver family
        hyperparams: Solver parameters
        seed: Random seed
        ns: List of n values to solve
        time_budget: Total time budget in seconds
        
    Returns:
        TrialOutput with layouts and scores
    """
    start_time = time.time()
    
    rng = np.random.default_rng(seed)
    
    layouts = {}
    scores = {}
    total_score = 0.0
    is_valid = True
    
    prev_layout = None
    
    for n in sorted(ns):
        # Check time budget
        if time_budget:
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                # Run out of time - use simple fallback for remaining
                break
        
        # Run solver
        layout = run_solver_for_n(
            n, family_id, rng, hyperparams, prev_layout
        )
        
        if layout is None:
            # Fallback to greedy
            layout = greedy_solve(n, rng)
        
        if layout is not None:
            layouts[n] = layout
            score = layout.compute_score()
            scores[n] = score
            total_score += score
            
            # Verify validity
            checker = CollisionChecker(layout)
            if not checker.is_layout_valid():
                is_valid = False
            
            prev_layout = layout
        else:
            is_valid = False
    
    runtime = time.time() - start_time
    
    return TrialOutput(
        layouts=layouts,
        scores=scores,
        total_score=total_score,
        runtime_seconds=runtime,
        is_valid=is_valid
    )


def evaluate_trial_at_rung(
    trial_id: str,
    family_id: str,
    hyperparams: Dict,
    seed: int,
    ns: List[int],
    time_budget: float = None
) -> Dict:
    """
    Evaluate a trial at a specific rung.
    
    Returns dict with:
    - score: Total proxy score
    - runtime: Runtime in seconds
    - is_valid: Whether all layouts are valid
    """
    output = run_trial(family_id, hyperparams, seed, ns, time_budget)
    
    return {
        'trial_id': trial_id,
        'score': output.total_score,
        'runtime': output.runtime_seconds,
        'is_valid': output.is_valid,
        'n_solved': len(output.layouts)
    }

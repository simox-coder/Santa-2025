"""
Santa 2025 Solver - Trial Runner

Runs individual trials for the orchestrator.
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from santa2025_solver.geometry_fast import Layout, compute_bounding_square_side
from santa2025_solver.greedy import GreedySolver
from santa2025_solver.lattice import LatticeSolver
from santa2025_solver.sa import SimulatedAnnealingSolver
from santa2025_solver.ils_vns import ILSVNSSolver
from santa2025_solver.shrink_repair import ShrinkRepairSolver


@dataclass
class TrialResult:
    """Result of running a trial."""
    proxy_score: float
    runtime: float
    feasible: bool
    layouts: Dict[int, Layout]
    details: Dict[str, Any]


def get_solver_class(family_id: int):
    """Get solver class by family ID."""
    if family_id == 0:
        return GreedySolver
    elif family_id == 1:
        return LatticeSolver
    elif family_id == 2:
        return SimulatedAnnealingSolver
    elif family_id == 3:
        return ILSVNSSolver
    elif family_id == 4:
        return ShrinkRepairSolver
    else:
        raise ValueError(f"Unknown family ID: {family_id}")


def run_trial(
    family_id: int,
    hyperparams: Dict[str, Any],
    seed: int,
    n_range: Tuple[int, int],
    budget_per_n: float = 1.0,
    warm_start: Optional[Dict[int, Layout]] = None
) -> TrialResult:
    """
    Run a single trial.
    
    Args:
        family_id: Solver family ID
        hyperparams: Hyperparameters for the solver
        seed: Random seed
        n_range: (n_min, n_max) range of n to solve
        budget_per_n: Budget per n in seconds
        warm_start: Optional warm-start layouts from previous stage
    
    Returns:
        TrialResult with score, runtime, feasibility, and layouts
    """
    rng = np.random.RandomState(seed)
    solver_class = get_solver_class(family_id)
    solver = solver_class(hyperparams)
    
    n_min, n_max = n_range
    layouts = {}
    total_score = 0.0
    all_feasible = True
    
    start_time = time.perf_counter()
    
    for n in range(n_min, n_max + 1):
        n_start = time.perf_counter()
        
        # Get initial layout
        if warm_start and n in warm_start:
            initial = warm_start[n].copy()
        else:
            initial = None
        
        # Solve
        if initial is not None and family_id in [2, 3, 4]:
            # Use improvement phase only for optimization solvers
            layout = solver.improve(initial, budget_per_n, rng, hyperparams)
        else:
            # Full solve
            layout = solver.solve(n, budget_per_n, rng, hyperparams)
        
        layouts[n] = layout
        
        # Compute score
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        total_score += s * s
        
        # Check feasibility (no collisions)
        feasible = check_layout_feasibility(layout)
        if not feasible:
            all_feasible = False
    
    total_runtime = time.perf_counter() - start_time
    
    return TrialResult(
        proxy_score=total_score,
        runtime=total_runtime,
        feasible=all_feasible,
        layouts=layouts,
        details={
            'family_id': family_id,
            'seed': seed,
            'n_range': n_range,
        }
    )


def check_layout_feasibility(layout: Layout) -> bool:
    """Check if a layout is feasible (no collisions)."""
    from santa2025_solver.geometry_fast import get_tree_vertices
    from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python
    
    n = layout.n
    for i in range(n):
        vi = get_tree_vertices(
            layout.positions[i, 0], layout.positions[i, 1], layout.rotations[i]
        )
        for j in range(i + 1, n):
            vj = get_tree_vertices(
                layout.positions[j, 0], layout.positions[j, 1], layout.rotations[j]
            )
            if check_tree_overlap_python(vi, vj):
                return False
    return True


def sample_hyperparams(family_id: int, rng: np.random.RandomState) -> Dict[str, Any]:
    """
    Sample hyperparameters for a solver family.
    
    Uses randomized search with appropriate distributions.
    """
    if family_id == 0:  # Greedy
        return {
            'n_candidates': rng.choice([30, 50, 100, 200]),
            'position_strategy': rng.choice(['spiral', 'grid', 'random']),
            'grid_resolution': rng.uniform(0.05, 0.2),
        }
    
    elif family_id == 1:  # Lattice
        return {
            'lattice_type': rng.choice(['hexagonal', 'square', 'brick']),
            'spacing_factor': rng.uniform(0.5, 0.8),
            'rotation_pattern': rng.choice(['alternating', 'random', 'fixed']),
            'jitter_factor': rng.uniform(0, 0.1),
        }
    
    elif family_id == 2:  # SA
        return {
            'T0': 10 ** rng.uniform(-1, 1),  # 0.1 to 10
            'T_end': 10 ** rng.uniform(-4, -2),  # 0.0001 to 0.01
            'cooling_rate': rng.uniform(0.99, 0.9999),
            'p_translate': rng.uniform(0.3, 0.7),
            'p_rotate': rng.uniform(0.1, 0.3),
            'p_swap': rng.uniform(0.1, 0.3),
            'p_boundary_push': rng.uniform(0.05, 0.2),
            'translate_step': rng.uniform(0.05, 0.2),
            'step_decay': rng.uniform(0.999, 0.99999),
        }
    
    elif family_id == 3:  # ILS/VNS
        return {
            'kick_strength': rng.uniform(0.1, 0.4),
            'kick_magnitude': rng.uniform(0.1, 0.5),
            'max_no_improve': rng.randint(3, 10),
            'local_search_budget': rng.uniform(0.3, 0.7),
            'accept_worse_prob': rng.uniform(0.05, 0.2),
        }
    
    elif family_id == 4:  # Shrink-Repair
        return {
            'shrink_delta': rng.uniform(0.005, 0.02),
            'shrink_decay': rng.uniform(0.95, 0.99),
            'repair_iterations': rng.randint(30, 100),
            'repair_step': rng.uniform(0.01, 0.05),
        }
    
    else:
        return {}


def create_trial_batch(
    n_trials: int,
    family_quotas: Dict[int, int] = None,
    base_seed: int = 42
) -> list:
    """
    Create a batch of trials to run.
    
    Args:
        n_trials: Total number of trials
        family_quotas: Optional quotas per family (default: equal distribution)
        base_seed: Base seed for reproducibility
    
    Returns:
        List of (family_id, hyperparams, seed) tuples
    """
    rng = np.random.RandomState(base_seed)
    trials = []
    
    if family_quotas is None:
        # Equal distribution across 5 families
        n_per_family = n_trials // 5
        family_quotas = {i: n_per_family for i in range(5)}
        # Add remainder to SA (family 2)
        family_quotas[2] += n_trials % 5
    
    for family_id, quota in family_quotas.items():
        for i in range(quota):
            seed = rng.randint(0, 2**31)
            hyperparams = sample_hyperparams(family_id, rng)
            trials.append((family_id, hyperparams, seed))
    
    # Shuffle
    rng.shuffle(trials)
    
    return trials

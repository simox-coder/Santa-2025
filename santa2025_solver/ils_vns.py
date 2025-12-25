"""
Iterated Local Search (ILS) and Variable Neighborhood Search (VNS) solver (F3).

Combines perturbation ("kick") with local search to escape local optima.
"""

import numpy as np
from typing import Optional
from .geometry_fast import TreeLayout
from .collision_fast import CollisionChecker
from .sa import sa_solve


def perturbation(
    layout: TreeLayout,
    checker: CollisionChecker,
    rng: np.random.Generator,
    strength: float = 0.3,
    num_trees: int = None
) -> bool:
    """
    Apply perturbation by moving a subset of trees.
    
    Args:
        layout: Layout to perturb
        checker: Collision checker
        rng: Random generator
        strength: Perturbation strength (max displacement)
        num_trees: Number of trees to perturb (default: n/5)
        
    Returns:
        True if perturbation produced valid layout
    """
    if num_trees is None:
        num_trees = max(1, layout.n // 5)
    
    # Select trees to perturb
    indices = rng.choice(layout.n, size=min(num_trees, layout.n), replace=False)
    
    # Save original positions
    original = [(i, layout.get_tree(i)) for i in indices]
    
    # Apply perturbations
    for idx in indices:
        x, y, deg = layout.get_tree(idx)
        dx = rng.uniform(-strength, strength)
        dy = rng.uniform(-strength, strength)
        ddeg = rng.choice([0, 90, 180, 270])
        
        layout.set_tree(idx, x + dx, y + dy, (deg + ddeg) % 360)
        layout.update_cache(idx)
        checker.update_tree(idx)
    
    # Try to resolve collisions
    max_repair = 100
    for _ in range(max_repair):
        collision = checker.find_first_collision()
        if collision is None:
            return True
        
        idx1, idx2 = collision
        
        # Push apart
        x1, y1, deg1 = layout.get_tree(idx1)
        x2, y2, deg2 = layout.get_tree(idx2)
        
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        dx /= dist
        dy /= dist
        
        push = 0.1
        layout.set_tree(idx2, x2 + dx * push, y2 + dy * push, deg2)
        layout.update_cache(idx2)
        checker.update_tree(idx2)
    
    # If still has collisions, revert
    if not checker.is_layout_valid():
        for idx, (x, y, deg) in original:
            layout.set_tree(idx, x, y, deg)
            layout.update_cache(idx)
            checker.update_tree(idx)
        return False
    
    return True


def ils_solve(
    layout: TreeLayout,
    rng: np.random.Generator = None,
    hyperparams: dict = None,
    max_iterations: int = 100,
    time_budget: float = None
) -> TreeLayout:
    """
    Optimize layout using Iterated Local Search.
    
    Args:
        layout: Initial layout (must be collision-free)
        rng: Random generator
        hyperparams: ILS parameters
        max_iterations: Maximum ILS iterations
        time_budget: Time budget in seconds
        
    Returns:
        Optimized layout
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Default parameters
    kick_strength = 0.3
    sa_iterations = 1000
    
    if hyperparams:
        kick_strength = hyperparams.get('kick_strength', kick_strength)
        sa_iterations = hyperparams.get('sa_iterations', sa_iterations)
        max_iterations = hyperparams.get('ils_iterations', max_iterations)
    
    import time
    start_time = time.time()
    
    best_layout = layout.copy()
    best_score = best_layout.compute_score()
    
    current_layout = layout.copy()
    checker = CollisionChecker(current_layout)
    
    for iteration in range(max_iterations):
        if time_budget and (time.time() - start_time) > time_budget:
            break
        
        # Perturbation (kick)
        kicked_layout = current_layout.copy()
        kicked_checker = CollisionChecker(kicked_layout)
        
        if perturbation(kicked_layout, kicked_checker, rng, kick_strength):
            # Local search (short SA)
            sa_params = dict(hyperparams) if hyperparams else {}
            sa_params['iterations'] = sa_iterations
            
            improved_layout = sa_solve(kicked_layout, rng, sa_params)
            improved_score = improved_layout.compute_score()
            
            # Acceptance criterion (greedy in ILS)
            if improved_score < best_score:
                best_score = improved_score
                best_layout = improved_layout.copy()
                current_layout = improved_layout.copy()
                checker = CollisionChecker(current_layout)
    
    return best_layout


def vns_solve(
    layout: TreeLayout,
    rng: np.random.Generator = None,
    hyperparams: dict = None,
    max_iterations: int = 50,
    time_budget: float = None
) -> TreeLayout:
    """
    Optimize layout using Variable Neighborhood Search.
    
    Escalates perturbation strength when no improvement is found.
    
    Args:
        layout: Initial layout
        rng: Random generator
        hyperparams: VNS parameters
        max_iterations: Maximum iterations
        time_budget: Time budget in seconds
        
    Returns:
        Optimized layout
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Neighborhood sizes
    k_max = 5
    strengths = [0.1, 0.2, 0.3, 0.5, 0.8]
    sa_iterations = 1000
    
    if hyperparams:
        k_max = hyperparams.get('k_max', k_max)
        strengths = hyperparams.get('strengths', strengths)
        sa_iterations = hyperparams.get('sa_iterations', sa_iterations)
    
    import time
    start_time = time.time()
    
    best_layout = layout.copy()
    best_score = best_layout.compute_score()
    
    current_layout = layout.copy()
    
    k = 0
    iteration = 0
    
    while iteration < max_iterations:
        if time_budget and (time.time() - start_time) > time_budget:
            break
        
        # Shaking with neighborhood k
        strength = strengths[min(k, len(strengths) - 1)]
        
        shaken_layout = current_layout.copy()
        checker = CollisionChecker(shaken_layout)
        
        if perturbation(shaken_layout, checker, rng, strength):
            # Local search
            sa_params = dict(hyperparams) if hyperparams else {}
            sa_params['iterations'] = sa_iterations
            
            improved_layout = sa_solve(shaken_layout, rng, sa_params)
            improved_score = improved_layout.compute_score()
            
            if improved_score < best_score:
                # Improvement found - restart from neighborhood 0
                best_score = improved_score
                best_layout = improved_layout.copy()
                current_layout = improved_layout.copy()
                k = 0
            else:
                # No improvement - move to next neighborhood
                k = (k + 1) % k_max
        else:
            k = (k + 1) % k_max
        
        iteration += 1
    
    return best_layout


def ils_full_solve(
    n: int,
    rng: np.random.Generator = None,
    hyperparams: dict = None,
    init_layout: TreeLayout = None
) -> Optional[TreeLayout]:
    """
    Full ILS solver: initialize + optimize.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Get initial layout
    if init_layout is not None and init_layout.n == n:
        layout = init_layout.copy()
    else:
        from .greedy import greedy_solve
        layout = greedy_solve(n, rng, hyperparams=hyperparams)
        if layout is None:
            return None
    
    return ils_solve(layout, rng, hyperparams)

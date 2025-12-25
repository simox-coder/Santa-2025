"""
Shrink-and-Repair solver (F4) for Santa 2025.

Maintains a feasible layout and progressively tries to shrink the bounding box,
using repair operations to maintain feasibility.
"""

import numpy as np
from typing import Optional, Tuple
from .geometry_fast import TreeLayout
from .collision_fast import CollisionChecker


def shrink_towards_center(
    layout: TreeLayout,
    checker: CollisionChecker,
    target_side: float,
    rng: np.random.Generator,
    max_repair_attempts: int = 100
) -> Tuple[bool, float]:
    """
    Try to shrink layout to fit within target bounding square side.
    
    Moves trees towards the center of the current bounding box.
    
    Args:
        layout: Layout to shrink
        checker: Collision checker
        target_side: Target bounding square side
        rng: Random generator
        max_repair_attempts: Max attempts to repair collisions
        
    Returns:
        (success, achieved_side) - success indicates if target was achieved
    """
    # Current bounds
    xs, ys = layout.get_all_vertices_flat()
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    
    current_width = max_x - min_x
    current_height = max_y - min_y
    current_side = max(current_width, current_height)
    
    if current_side <= target_side:
        return True, current_side
    
    # Compute center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Compute scale factor
    scale = target_side / current_side
    
    # Move all trees towards center
    for i in range(layout.n):
        x, y, deg = layout.get_tree(i)
        
        # New position: move towards center by (1 - scale)
        new_x = center_x + (x - center_x) * scale
        new_y = center_y + (y - center_y) * scale
        
        layout.set_tree(i, new_x, new_y, deg)
    
    layout.update_all_caches()
    checker.rebuild_grid()
    
    # Repair collisions
    for _ in range(max_repair_attempts):
        collision = checker.find_first_collision()
        if collision is None:
            break
        
        idx1, idx2 = collision
        
        # Push trees apart
        x1, y1, deg1 = layout.get_tree(idx1)
        x2, y2, deg2 = layout.get_tree(idx2)
        
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        dx /= dist
        dy /= dist
        
        push = 0.05 + rng.uniform(0, 0.05)
        
        # Push both trees (in opposite directions)
        layout.set_tree(idx1, x1 - dx * push * 0.5, y1 - dy * push * 0.5, deg1)
        layout.set_tree(idx2, x2 + dx * push * 0.5, y2 + dy * push * 0.5, deg2)
        layout.update_cache(idx1)
        layout.update_cache(idx2)
        checker.update_tree(idx1)
        checker.update_tree(idx2)
    
    # Check result
    is_valid = checker.is_layout_valid()
    achieved = layout.compute_score()
    
    return is_valid and achieved <= target_side * 1.01, achieved


def shrink_repair_solve(
    layout: TreeLayout,
    rng: np.random.Generator = None,
    hyperparams: dict = None,
    max_iterations: int = 100,
    time_budget: float = None
) -> TreeLayout:
    """
    Optimize layout using shrink-and-repair approach.
    
    Args:
        layout: Initial layout (must be collision-free)
        rng: Random generator
        hyperparams: Parameters
        max_iterations: Maximum iterations
        time_budget: Time budget in seconds
        
    Returns:
        Optimized layout
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Parameters
    initial_delta = 0.05  # Initial shrink amount
    min_delta = 0.001
    delta_decay = 0.9
    
    if hyperparams:
        initial_delta = hyperparams.get('initial_delta', initial_delta)
        min_delta = hyperparams.get('min_delta', min_delta)
        delta_decay = hyperparams.get('delta_decay', delta_decay)
    
    import time
    start_time = time.time()
    
    best_layout = layout.copy()
    best_score = best_layout.compute_score()
    
    current_layout = layout.copy()
    checker = CollisionChecker(current_layout)
    
    delta = initial_delta
    consecutive_failures = 0
    
    for iteration in range(max_iterations):
        if time_budget and (time.time() - start_time) > time_budget:
            break
        
        current_score = current_layout.compute_score()
        target_side = current_score * (1 - delta)
        
        # Try to shrink
        trial_layout = current_layout.copy()
        trial_checker = CollisionChecker(trial_layout)
        
        success, achieved = shrink_towards_center(
            trial_layout, trial_checker, target_side, rng
        )
        
        if success and achieved < best_score:
            best_score = achieved
            best_layout = trial_layout.copy()
            current_layout = trial_layout.copy()
            checker = CollisionChecker(current_layout)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            
            # Reduce delta after failures
            if consecutive_failures >= 3:
                delta *= delta_decay
                delta = max(delta, min_delta)
                consecutive_failures = 0
    
    return best_layout


def shrink_repair_full_solve(
    n: int,
    rng: np.random.Generator = None,
    hyperparams: dict = None,
    init_layout: TreeLayout = None
) -> Optional[TreeLayout]:
    """
    Full shrink-repair solver: initialize + optimize.
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
    
    return shrink_repair_solve(layout, rng, hyperparams)

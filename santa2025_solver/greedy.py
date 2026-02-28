"""
Greedy placement solver (F0) for Santa 2025.

Places trees one by one, choosing the position that minimizes the bounding square.
"""

import numpy as np
from typing import List, Tuple, Optional
from .geometry_fast import TreeLayout, TREE_VERTICES_BASE, transform_tree_fast, compute_aabb
from .collision_fast import CollisionChecker


def generate_candidate_positions(
    layout: TreeLayout,
    checker: CollisionChecker,
    num_candidates: int = 100,
    rng: np.random.Generator = None
) -> List[Tuple[float, float, float]]:
    """
    Generate candidate positions for the next tree.
    
    Strategies:
    1. Around existing trees (tangent-ish positions)
    2. Around the boundary of current bounding box
    3. Random positions within a reasonable range
    
    Args:
        layout: Current layout
        checker: Collision checker
        num_candidates: Number of candidates to generate
        rng: Random number generator
        
    Returns:
        List of (x, y, deg) candidates
    """
    if rng is None:
        rng = np.random.default_rng()
    
    candidates = []
    
    # Rotation options
    rotations = [0.0, 90.0, 180.0, 270.0]
    
    if layout.n == 0:
        # First tree: place at origin with various rotations
        for deg in rotations:
            candidates.append((0.0, 0.0, deg))
        return candidates
    
    # Get current bounding box
    xs, ys = layout.get_all_vertices_flat()
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    
    # Tree approximate size
    tree_size = 1.0
    
    # Strategy 1: Around a few existing trees (limit to 3 for speed)
    sample_trees = min(layout.n, 3)
    for i in range(sample_trees):
        tx, ty, _ = layout.get_tree(i)
        
        # Generate offset positions (fewer for speed)
        offsets = [
            (tree_size, 0), (-tree_size, 0),
            (0, tree_size), (0, -tree_size),
        ]
        
        for dx, dy in offsets:
            deg = rng.choice(rotations)
            candidates.append((tx + dx, ty + dy, deg))
    
    # Strategy 2: Around boundary (just 4 corners)
    boundary_positions = [
        (max_x + tree_size * 0.5, (min_y + max_y) / 2),
        (min_x - tree_size * 0.5, (min_y + max_y) / 2),
        ((min_x + max_x) / 2, max_y + tree_size * 0.5),
        ((min_x + max_x) / 2, min_y - tree_size * 0.5),
    ]
    
    for bx, by in boundary_positions:
        deg = rng.choice(rotations)
        candidates.append((bx, by, deg))
    
    # Strategy 3: Random positions to fill remaining
    range_x = max_x - min_x + 2 * tree_size
    range_y = max_y - min_y + 2 * tree_size
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    remaining = num_candidates - len(candidates)
    for _ in range(remaining):
        x = center_x + rng.uniform(-range_x, range_x)
        y = center_y + rng.uniform(-range_y, range_y)
        deg = rng.choice(rotations)
        candidates.append((x, y, deg))
    
    return candidates[:num_candidates]


def evaluate_candidate(
    layout: TreeLayout,
    idx: int,
    x: float, y: float, deg: float,
    checker: CollisionChecker
) -> Tuple[float, bool]:
    """
    Evaluate a candidate position for a tree.
    
    Args:
        layout: Current layout
        idx: Tree index to place
        x, y, deg: Candidate position
        checker: Collision checker
        
    Returns:
        (score, is_valid) - score is bounding square side, is_valid means no collision
    """
    # Temporarily place tree
    old_pos = layout.get_tree(idx)
    layout.set_tree(idx, x, y, deg)
    layout.update_cache(idx)
    checker.update_tree(idx)
    
    # Check collision
    is_valid = not checker.has_any_collision(idx)
    
    if is_valid:
        # Compute new bounding square
        score = layout.compute_score()
    else:
        score = float('inf')
    
    # Restore old position
    layout.set_tree(idx, *old_pos)
    layout.update_cache(idx)
    checker.update_tree(idx)
    
    return score, is_valid


def greedy_place_tree(
    layout: TreeLayout,
    idx: int,
    checker: CollisionChecker,
    rng: np.random.Generator = None,
    num_candidates: int = 200
) -> bool:
    """
    Place a single tree using greedy selection.
    
    Args:
        layout: Layout to modify
        idx: Tree index to place
        checker: Collision checker
        rng: Random generator
        num_candidates: Number of candidates to try
        
    Returns:
        True if tree was placed successfully
    """
    candidates = generate_candidate_positions(layout, checker, num_candidates, rng)
    
    best_score = float('inf')
    best_pos = None
    
    for x, y, deg in candidates:
        score, is_valid = evaluate_candidate(layout, idx, x, y, deg, checker)
        if is_valid and score < best_score:
            best_score = score
            best_pos = (x, y, deg)
    
    if best_pos is not None:
        layout.set_tree(idx, *best_pos)
        layout.update_cache(idx)
        checker.update_tree(idx)
        return True
    
    return False


def greedy_solve(
    n: int,
    rng: np.random.Generator = None,
    num_candidates: int = 200,
    hyperparams: dict = None
) -> Optional[TreeLayout]:
    """
    Solve for n trees using greedy placement.
    
    Args:
        n: Number of trees
        rng: Random generator
        num_candidates: Candidates per tree
        hyperparams: Additional parameters
        
    Returns:
        TreeLayout or None if failed
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if hyperparams:
        num_candidates = hyperparams.get('num_candidates', num_candidates)
    
    layout = TreeLayout(n)
    checker = CollisionChecker(layout)
    
    # Place trees one by one
    for i in range(n):
        success = greedy_place_tree(layout, i, checker, rng, num_candidates)
        if not success:
            # Try more candidates
            success = greedy_place_tree(layout, i, checker, rng, num_candidates * 5)
            if not success:
                return None
    
    return layout


def greedy_solve_with_warmstart(
    n: int,
    prev_layout: Optional[TreeLayout] = None,
    rng: np.random.Generator = None,
    hyperparams: dict = None
) -> Optional[TreeLayout]:
    """
    Solve for n trees, optionally using previous solution for n-1.
    
    Args:
        n: Number of trees
        prev_layout: Previous solution for n-1 trees
        rng: Random generator
        hyperparams: Additional parameters
        
    Returns:
        TreeLayout or None if failed
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_candidates = 200
    if hyperparams:
        num_candidates = hyperparams.get('num_candidates', num_candidates)
    
    if prev_layout is not None and prev_layout.n == n - 1:
        # Copy previous layout and add one tree
        layout = TreeLayout(n)
        for i in range(n - 1):
            x, y, deg = prev_layout.get_tree(i)
            layout.set_tree(i, x, y, deg)
        
        checker = CollisionChecker(layout)
        
        # Place the new tree
        success = greedy_place_tree(layout, n - 1, checker, rng, num_candidates)
        if success:
            return layout
    
    # Fallback to full greedy
    return greedy_solve(n, rng, num_candidates, hyperparams)

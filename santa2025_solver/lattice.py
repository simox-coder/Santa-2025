"""
Lattice-based solver (F1) for Santa 2025.

Initializes trees on a parametric lattice grid, then refines locally.
"""

import numpy as np
from typing import Optional, Tuple
from .geometry_fast import TreeLayout
from .collision_fast import CollisionChecker


def generate_lattice_points(
    n: int,
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    center: Tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """
    Generate n points on a 2D lattice defined by vectors v1 and v2.
    
    Points are arranged in a roughly circular pattern around center.
    
    Args:
        n: Number of points
        v1, v2: Lattice basis vectors
        center: Center point
        
    Returns:
        Nx2 array of lattice points
    """
    # Generate lattice points in a spiral-like order
    points = []
    
    # Start with center
    points.append(center)
    
    if n == 1:
        return np.array(points)
    
    # Add points in expanding shells
    shell = 1
    while len(points) < n:
        # Points in this shell
        for i in range(-shell, shell + 1):
            for j in range(-shell, shell + 1):
                if abs(i) == shell or abs(j) == shell:  # On shell boundary
                    x = center[0] + i * v1[0] + j * v2[0]
                    y = center[1] + i * v1[1] + j * v2[1]
                    points.append((x, y))
                    
                    if len(points) >= n:
                        break
            if len(points) >= n:
                break
        shell += 1
    
    return np.array(points[:n], dtype=np.float64)


def lattice_init(
    n: int,
    rng: np.random.Generator = None,
    hyperparams: dict = None
) -> TreeLayout:
    """
    Initialize layout using a lattice pattern.
    
    Args:
        n: Number of trees
        rng: Random generator
        hyperparams: Dict with lattice parameters:
            - v1_length: Length of first basis vector
            - v1_angle: Angle of first basis vector (degrees)
            - v2_angle_offset: Angle offset of second vector from first
            - rotation_palette: List of rotation angles to use
            
    Returns:
        Initialized TreeLayout
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Default parameters
    v1_length = 1.0
    v1_angle = 0.0
    v2_angle_offset = 90.0
    rotation_palette = [0.0, 90.0, 180.0, 270.0]
    
    if hyperparams:
        v1_length = hyperparams.get('v1_length', v1_length)
        v1_angle = hyperparams.get('v1_angle', v1_angle)
        v2_angle_offset = hyperparams.get('v2_angle_offset', v2_angle_offset)
        rotation_palette = hyperparams.get('rotation_palette', rotation_palette)
    
    # Compute basis vectors
    rad1 = np.radians(v1_angle)
    rad2 = np.radians(v1_angle + v2_angle_offset)
    
    v1 = (v1_length * np.cos(rad1), v1_length * np.sin(rad1))
    v2 = (v1_length * np.cos(rad2), v1_length * np.sin(rad2))
    
    # Generate lattice points
    points = generate_lattice_points(n, v1, v2)
    
    # Create layout
    layout = TreeLayout(n)
    
    for i in range(n):
        x, y = points[i]
        deg = rng.choice(rotation_palette)
        layout.set_tree(i, x, y, deg)
    
    return layout


def local_refine(
    layout: TreeLayout,
    checker: CollisionChecker,
    rng: np.random.Generator,
    iterations: int = 100,
    step_size: float = 0.1,
    hyperparams: dict = None
) -> TreeLayout:
    """
    Locally refine a layout by small perturbations.
    
    Args:
        layout: Layout to refine
        checker: Collision checker
        rng: Random generator
        iterations: Number of refinement iterations
        step_size: Initial step size for perturbations
        hyperparams: Additional parameters
        
    Returns:
        Refined layout
    """
    if hyperparams:
        iterations = hyperparams.get('refine_iterations', iterations)
        step_size = hyperparams.get('refine_step_size', step_size)
    
    best_layout = layout.copy()
    best_score = best_layout.compute_score()
    
    for _ in range(iterations):
        # Pick a random tree
        idx = rng.integers(0, layout.n)
        
        # Current position
        x, y, deg = layout.get_tree(idx)
        
        # Try small perturbations
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        ddeg = rng.choice([0, 90, 180, 270]) if rng.random() < 0.1 else 0
        
        new_x = x + dx
        new_y = y + dy
        new_deg = (deg + ddeg) % 360
        
        # Apply and check
        layout.set_tree(idx, new_x, new_y, new_deg)
        layout.update_cache(idx)
        checker.update_tree(idx)
        
        if checker.has_any_collision(idx):
            # Revert
            layout.set_tree(idx, x, y, deg)
            layout.update_cache(idx)
            checker.update_tree(idx)
        else:
            # Check score improvement
            new_score = layout.compute_score()
            if new_score < best_score:
                best_score = new_score
                best_layout = layout.copy()
    
    return best_layout


def resolve_collisions(
    layout: TreeLayout,
    checker: CollisionChecker,
    rng: np.random.Generator,
    max_attempts: int = 1000
) -> bool:
    """
    Try to resolve all collisions by moving trees apart.
    
    Args:
        layout: Layout to fix
        checker: Collision checker
        rng: Random generator
        max_attempts: Maximum attempts
        
    Returns:
        True if all collisions resolved
    """
    for _ in range(max_attempts):
        collision = checker.find_first_collision()
        if collision is None:
            return True
        
        idx1, idx2 = collision
        
        # Push trees apart
        x1, y1, deg1 = layout.get_tree(idx1)
        x2, y2, deg2 = layout.get_tree(idx2)
        
        # Direction from tree1 to tree2
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        
        # Normalize and push
        dx /= dist
        dy /= dist
        
        push_dist = 0.1 + rng.uniform(0, 0.1)
        
        # Move tree2 away
        layout.set_tree(idx2, x2 + dx * push_dist, y2 + dy * push_dist, deg2)
        layout.update_cache(idx2)
        checker.update_tree(idx2)
    
    return checker.is_layout_valid()


def lattice_solve(
    n: int,
    rng: np.random.Generator = None,
    hyperparams: dict = None
) -> Optional[TreeLayout]:
    """
    Solve for n trees using lattice initialization + refinement.
    
    Args:
        n: Number of trees
        rng: Random generator
        hyperparams: Parameters for lattice and refinement
        
    Returns:
        TreeLayout or None if failed
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize with lattice
    layout = lattice_init(n, rng, hyperparams)
    checker = CollisionChecker(layout)
    
    # Try to resolve collisions
    if not resolve_collisions(layout, checker, rng):
        # If can't resolve, try with larger spacing
        if hyperparams is None:
            hyperparams = {}
        hyperparams['v1_length'] = hyperparams.get('v1_length', 1.0) * 1.5
        
        layout = lattice_init(n, rng, hyperparams)
        checker = CollisionChecker(layout)
        
        if not resolve_collisions(layout, checker, rng):
            return None
    
    # Refine
    layout = local_refine(layout, checker, rng, hyperparams=hyperparams)
    
    return layout

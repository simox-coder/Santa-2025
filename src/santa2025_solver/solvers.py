"""
Solver families for the Christmas tree packing problem.

Implements:
F0) Greedy constructive
F1) Lattice/template init + local refine
F2) Simulated Annealing (SA)
F3) ILS/VNS
F4) Shrink-and-Repair
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import random
import math
from dataclasses import dataclass

from .collision import check_tree_collision, check_any_collision, find_all_collisions, SpatialHash
from .kaggle_ref.geometry_ref import get_bounding_circle_radius, transform_tree, get_tree_aabb


@dataclass
class SolverConfig:
    """Configuration for solvers."""
    seed: int = 42
    max_iterations: int = 10000
    temperature_init: float = 1.0
    temperature_min: float = 0.001
    cooling_rate: float = 0.995
    move_step: float = 0.05
    rotation_step: float = 15.0
    num_candidates: int = 36
    verbose: bool = False


class Layout:
    """Represents a packing layout for n trees."""
    
    def __init__(self, n: int):
        self.n = n
        self.positions = [(0.0, 0.0, 0.0)] * n  # List of (x, y, deg)
    
    def copy(self) -> 'Layout':
        new_layout = Layout(self.n)
        new_layout.positions = list(self.positions)
        return new_layout
    
    def set_position(self, idx: int, x: float, y: float, deg: float):
        self.positions[idx] = (x, y, deg)
    
    def get_position(self, idx: int) -> Tuple[float, float, float]:
        return self.positions[idx]
    
    def get_radius(self) -> float:
        return get_bounding_circle_radius(self.positions)
    
    def has_collision(self) -> bool:
        return check_any_collision(self.positions)
    
    def get_collisions(self) -> List[Tuple[int, int]]:
        return find_all_collisions(self.positions)


# ========== F0: Greedy Constructive ==========

def solve_greedy(n: int, config: SolverConfig = None) -> Layout:
    """Greedy constructive solver.
    
    Places trees one by one, trying to minimize bounding circle radius
    while avoiding collisions.
    """
    if config is None:
        config = SolverConfig()
    
    rng = random.Random(config.seed)
    layout = Layout(n)
    
    if n == 0:
        return layout
    
    # Place first tree at origin with rotation 90 (standard orientation)
    layout.set_position(0, 0.0, 0.0, 90.0)
    
    for tree_idx in range(1, n):
        best_pos = None
        best_radius = float('inf')
        
        # Current bounding radius
        current_radius = layout.get_radius()
        
        # Try placing at various positions around the current pack
        for angle_idx in range(config.num_candidates):
            angle = 2 * math.pi * angle_idx / config.num_candidates
            
            # Try at various distances
            for dist_factor in [1.0, 1.1, 1.2, 1.3, 1.5]:
                dist = current_radius * dist_factor + 0.3
                
                x = dist * math.cos(angle)
                y = dist * math.sin(angle)
                
                # Try different rotations
                for deg in [0, 90, 180, 270]:
                    layout.set_position(tree_idx, x, y, deg)
                    
                    # Check collision with already placed trees
                    has_collision = False
                    for prev_idx in range(tree_idx):
                        if check_tree_collision(layout.get_position(tree_idx), 
                                                layout.get_position(prev_idx)):
                            has_collision = True
                            break
                    
                    if not has_collision:
                        radius = layout.get_radius()
                        if radius < best_radius:
                            best_radius = radius
                            best_pos = (x, y, deg)
        
        if best_pos is not None:
            layout.set_position(tree_idx, *best_pos)
        else:
            # Fallback: place at larger distance
            angle = 2 * math.pi * rng.random()
            dist = current_radius + 1.0
            layout.set_position(tree_idx, dist * math.cos(angle), 
                               dist * math.sin(angle), 90.0)
    
    return layout


# ========== F1: Lattice Init + Local Refine ==========

def init_lattice(n: int, spacing: float = 0.8) -> Layout:
    """Initialize trees in a hexagonal lattice pattern."""
    layout = Layout(n)
    
    if n == 0:
        return layout
    
    # Calculate grid dimensions for roughly circular packing
    rows = int(math.ceil(math.sqrt(n)))
    
    idx = 0
    for row in range(rows * 2):  # Extra rows to ensure enough positions
        for col in range(rows * 2):
            if idx >= n:
                break
            
            # Hexagonal offset
            offset = spacing / 2 if row % 2 else 0
            
            x = (col - rows // 2) * spacing + offset
            y = (row - rows // 2) * spacing * 0.866  # sqrt(3)/2 for hex
            
            # Center the layout
            x -= spacing / 2
            y -= spacing * 0.866 / 2
            
            layout.set_position(idx, x, y, 90.0)
            idx += 1
        
        if idx >= n:
            break
    
    return layout


def local_refine(layout: Layout, config: SolverConfig) -> Layout:
    """Locally refine a layout to reduce radius."""
    rng = random.Random(config.seed)
    best_layout = layout.copy()
    best_radius = layout.get_radius()
    
    for _ in range(config.max_iterations):
        # Pick random tree
        idx = rng.randint(0, layout.n - 1)
        x, y, deg = layout.get_position(idx)
        
        # Try small perturbation
        dx = rng.gauss(0, config.move_step)
        dy = rng.gauss(0, config.move_step)
        
        # Move towards center to reduce radius
        dist = math.sqrt(x*x + y*y)
        if dist > 0.1:
            dx -= 0.3 * x / dist * config.move_step
            dy -= 0.3 * y / dist * config.move_step
        
        new_x = x + dx
        new_y = y + dy
        
        layout.set_position(idx, new_x, new_y, deg)
        
        # Check validity
        has_collision = False
        for other_idx in range(layout.n):
            if other_idx != idx:
                if check_tree_collision(layout.get_position(idx), 
                                        layout.get_position(other_idx)):
                    has_collision = True
                    break
        
        if has_collision:
            layout.set_position(idx, x, y, deg)
        else:
            radius = layout.get_radius()
            if radius < best_radius:
                best_radius = radius
                best_layout = layout.copy()
            elif rng.random() > 0.9:
                # Sometimes revert anyway to explore
                layout.set_position(idx, x, y, deg)
    
    return best_layout


# ========== F2: Simulated Annealing ==========

def solve_sa(n: int, initial_layout: Optional[Layout] = None, 
             config: SolverConfig = None) -> Layout:
    """Simulated Annealing solver."""
    if config is None:
        config = SolverConfig()
    
    rng = random.Random(config.seed)
    
    # Initialize
    if initial_layout is not None:
        layout = initial_layout.copy()
    else:
        layout = solve_greedy(n, config)
    
    best_layout = layout.copy()
    best_radius = layout.get_radius()
    
    temperature = config.temperature_init
    
    for iteration in range(config.max_iterations):
        # Pick random tree
        idx = rng.randint(0, n - 1)
        x, y, deg = layout.get_position(idx)
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Translate
            dx = rng.gauss(0, config.move_step)
            dy = rng.gauss(0, config.move_step)
            new_x, new_y, new_deg = x + dx, y + dy, deg
        elif move_type < 0.8:
            # Rotate
            d_deg = rng.choice([-config.rotation_step, config.rotation_step])
            new_x, new_y, new_deg = x, y, (deg + d_deg) % 360
        else:
            # Move toward center
            dist = math.sqrt(x*x + y*y)
            if dist > 0.1:
                factor = rng.uniform(0.8, 0.99)
                new_x = x * factor
                new_y = y * factor
            else:
                new_x = x + rng.gauss(0, config.move_step * 0.5)
                new_y = y + rng.gauss(0, config.move_step * 0.5)
            new_deg = deg
        
        # Apply move
        layout.set_position(idx, new_x, new_y, new_deg)
        
        # Check collision
        has_collision = False
        for other_idx in range(n):
            if other_idx != idx:
                if check_tree_collision(layout.get_position(idx), 
                                        layout.get_position(other_idx)):
                    has_collision = True
                    break
        
        if has_collision:
            # Revert
            layout.set_position(idx, x, y, deg)
        else:
            # Accept or reject based on radius change
            new_radius = layout.get_radius()
            delta = new_radius - best_radius
            
            if delta < 0:
                # Accept improvement
                best_radius = new_radius
                best_layout = layout.copy()
            elif rng.random() < math.exp(-delta / temperature):
                # Accept worse move with probability
                pass
            else:
                # Reject
                layout.set_position(idx, x, y, deg)
        
        # Cool down
        temperature = max(config.temperature_min, temperature * config.cooling_rate)
    
    return best_layout


# ========== F3: Iterated Local Search ==========

def solve_ils(n: int, config: SolverConfig = None) -> Layout:
    """Iterated Local Search with SA refinement."""
    if config is None:
        config = SolverConfig()
    
    rng = random.Random(config.seed)
    
    # Start with greedy
    best_layout = solve_greedy(n, config)
    best_radius = best_layout.get_radius()
    
    # Refine with SA
    sa_config = SolverConfig(
        seed=config.seed,
        max_iterations=config.max_iterations // 5,
        temperature_init=config.temperature_init,
        cooling_rate=config.cooling_rate,
        move_step=config.move_step
    )
    
    for restart in range(5):
        # Perturb (kick)
        if restart > 0:
            layout = best_layout.copy()
            num_perturb = max(1, n // 10)
            
            for _ in range(num_perturb):
                idx = rng.randint(0, n - 1)
                x, y, deg = layout.get_position(idx)
                
                # Large perturbation
                dx = rng.gauss(0, config.move_step * 3)
                dy = rng.gauss(0, config.move_step * 3)
                new_deg = rng.choice([0, 90, 180, 270])
                
                layout.set_position(idx, x + dx, y + dy, new_deg)
            
            # Repair overlaps if needed
            layout = repair_overlaps(layout, config)
        else:
            layout = best_layout.copy()
        
        # SA refinement
        sa_config.seed = config.seed + restart
        refined = solve_sa(n, layout, sa_config)
        
        if refined.get_radius() < best_radius and not refined.has_collision():
            best_radius = refined.get_radius()
            best_layout = refined
    
    return best_layout


# ========== Overlap Repair ==========

def repair_overlaps(layout: Layout, config: SolverConfig = None, 
                   max_attempts: int = 100) -> Layout:
    """Repair overlapping trees by local push-away.
    
    This is a minimal, quality-preserving repair that only touches
    conflicting trees.
    """
    if config is None:
        config = SolverConfig()
    
    for attempt in range(max_attempts):
        collisions = layout.get_collisions()
        if not collisions:
            return layout
        
        for i, j in collisions:
            xi, yi, degi = layout.get_position(i)
            xj, yj, degj = layout.get_position(j)
            
            # Push apart along centroid direction
            dx = xj - xi
            dy = yj - yi
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-6:
                # Coincident - move in random direction
                angle = random.random() * 2 * math.pi
                dx = math.cos(angle)
                dy = math.sin(angle)
                dist = 1.0
            
            # Normalize
            dx /= dist
            dy /= dist
            
            # Push apart with small step
            step = 0.001 * (1 + attempt * 0.1)
            
            layout.set_position(i, xi - dx * step, yi - dy * step, degi)
            layout.set_position(j, xj + dx * step, yj + dy * step, degj)
    
    return layout


# ========== Main Solver Interface ==========

def solve_n(n: int, config: SolverConfig = None, 
            method: str = "ils",
            initial_layout: Optional[Layout] = None) -> Layout:
    """Solve for n trees using specified method.
    
    Args:
        n: Number of trees
        config: Solver configuration
        method: One of "greedy", "lattice", "sa", "ils"
        initial_layout: Optional initial layout (for warm-start)
        
    Returns:
        Best layout found
    """
    if config is None:
        config = SolverConfig()
    
    if method == "greedy":
        return solve_greedy(n, config)
    elif method == "lattice":
        layout = init_lattice(n)
        return local_refine(layout, config)
    elif method == "sa":
        return solve_sa(n, initial_layout, config)
    elif method == "ils":
        return solve_ils(n, config)
    else:
        raise ValueError(f"Unknown method: {method}")


def solve_all_n(max_n: int = 200, config: SolverConfig = None,
                verbose: bool = True) -> Dict[int, Layout]:
    """Solve for all n from 1 to max_n with warm-start.
    
    Uses solutions from n-1 to warm-start n.
    
    Args:
        max_n: Maximum n to solve (default 200)
        config: Solver configuration
        verbose: Print progress
        
    Returns:
        Dictionary mapping n to best layout
    """
    if config is None:
        config = SolverConfig()
    
    solutions = {}
    
    for n in range(1, max_n + 1):
        if verbose and n % 10 == 0:
            print(f"Solving n={n}...")
        
        # Warm-start from n-1
        if n > 1 and n - 1 in solutions:
            prev_layout = solutions[n - 1]
            # Create new layout with one more tree
            initial = Layout(n)
            for i in range(n - 1):
                initial.set_position(i, *prev_layout.get_position(i))
            
            # Place new tree using greedy placement
            temp_config = SolverConfig(seed=config.seed + n)
            greedy = solve_greedy(n, temp_config)
            initial.set_position(n - 1, *greedy.get_position(n - 1))
            
            # Refine
            layout = solve_sa(n, initial, config)
        else:
            layout = solve_ils(n, config)
        
        # Final repair
        layout = repair_overlaps(layout, config)
        
        solutions[n] = layout
        
        if verbose and n <= 10:
            print(f"  n={n}: radius={layout.get_radius():.4f}")
    
    return solutions


if __name__ == "__main__":
    # Quick test
    config = SolverConfig(max_iterations=1000, verbose=True)
    
    print("Testing n=4...")
    layout = solve_n(4, config, method="ils")
    print(f"  Radius: {layout.get_radius():.4f}")
    print(f"  Has collision: {layout.has_collision()}")
    print(f"  Positions: {layout.positions}")

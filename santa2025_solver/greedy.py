"""
Santa 2025 Solver - Greedy Constructive Solver

Baseline greedy algorithm that places trees one at a time,
minimizing bounding square at each step.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import math

from santa2025_solver.geometry_fast import (
    Layout, get_tree_vertices, get_tree_aabb,
    get_all_tree_vertices, get_all_aabbs, compute_bounding_square_side
)
from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python


class GreedySolver:
    """
    Greedy constructive solver.
    
    Places trees one at a time, choosing positions that minimize
    the bounding square while avoiding collisions.
    """
    
    FAMILY_ID = 0
    FAMILY_NAME = "greedy"
    
    DEFAULT_HYPERPARAMS = {
        'n_candidates': 50,         # Number of candidate positions to try
        'rotation_palette': [0, 90, 180, 270],
        'position_strategy': 'spiral',  # 'spiral', 'grid', 'random'
        'grid_resolution': 0.1,
    }
    
    def __init__(self, hyperparams: Optional[Dict[str, Any]] = None):
        self.hyperparams = {**self.DEFAULT_HYPERPARAMS}
        if hyperparams:
            self.hyperparams.update(hyperparams)
    
    def init_layout(self, n: int, rng: np.random.RandomState) -> Layout:
        """
        Initialize a layout by placing trees greedily.
        
        Args:
            n: Number of trees
            rng: Random state for any randomness
        
        Returns:
            Layout with n trees placed without overlap
        """
        layout = Layout(n)
        
        # Place first tree at origin with rotation 90 (upright)
        layout.set_tree(0, 0.0, 0.0, 90.0)
        
        # Place remaining trees
        for i in range(1, n):
            pos, deg = self._find_best_position(layout, i, rng)
            layout.set_tree(i, pos[0], pos[1], deg)
        
        return layout
    
    def _find_best_position(
        self,
        layout: Layout,
        tree_idx: int,
        rng: np.random.RandomState
    ) -> Tuple[Tuple[float, float], float]:
        """
        Find the best position for a new tree.
        
        Returns:
            ((x, y), deg) for the best position found
        """
        n_placed = tree_idx
        candidates = self._generate_candidates(layout, n_placed, rng)
        
        best_pos = None
        best_deg = None
        best_score = float('inf')
        
        for pos, deg in candidates:
            # Check if position is valid (no collisions)
            if not self._is_valid_position(layout, tree_idx, pos[0], pos[1], deg):
                continue
            
            # Temporarily place tree and compute score
            layout.set_tree(tree_idx, pos[0], pos[1], deg)
            score = compute_bounding_square_side(
                layout.positions[:tree_idx+1],
                layout.rotations[:tree_idx+1]
            )
            
            if score < best_score:
                best_score = score
                best_pos = pos
                best_deg = deg
        
        if best_pos is None:
            # Fallback: place at expanding spiral until valid
            best_pos, best_deg = self._fallback_placement(layout, tree_idx, rng)
        
        return best_pos, best_deg
    
    def _generate_candidates(
        self,
        layout: Layout,
        n_placed: int,
        rng: np.random.RandomState
    ) -> List[Tuple[Tuple[float, float], float]]:
        """Generate candidate positions to try."""
        candidates = []
        n_candidates = self.hyperparams['n_candidates']
        rotations = self.hyperparams['rotation_palette']
        strategy = self.hyperparams['position_strategy']
        
        # Get current bounding box
        if n_placed > 0:
            vertices = get_all_tree_vertices(
                layout.positions[:n_placed],
                layout.rotations[:n_placed]
            )
            max_coord = np.abs(vertices).max()
        else:
            max_coord = 0.5
        
        if strategy == 'spiral':
            # Generate positions in a spiral pattern
            for i in range(n_candidates):
                angle = i * 2.4  # Golden angle for good coverage
                r = 0.3 + 0.1 * math.sqrt(i)  # Expanding radius
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                
                for deg in rotations:
                    candidates.append(((x, y), deg))
        
        elif strategy == 'grid':
            # Generate positions on a grid
            resolution = self.hyperparams['grid_resolution']
            grid_size = max(1.0, max_coord * 1.5)
            
            for x in np.arange(-grid_size, grid_size + resolution, resolution):
                for y in np.arange(-grid_size, grid_size + resolution, resolution):
                    for deg in rotations:
                        candidates.append(((x, y), deg))
                        if len(candidates) >= n_candidates * len(rotations):
                            break
        
        else:  # random
            for _ in range(n_candidates):
                x = rng.uniform(-max_coord * 1.5, max_coord * 1.5)
                y = rng.uniform(-max_coord * 1.5, max_coord * 1.5)
                deg = rng.choice(rotations)
                candidates.append(((x, y), deg))
        
        return candidates
    
    def _is_valid_position(
        self,
        layout: Layout,
        tree_idx: int,
        x: float,
        y: float,
        deg: float
    ) -> bool:
        """Check if a position is valid (no collisions with placed trees)."""
        new_vertices = get_tree_vertices(x, y, deg)
        
        for i in range(tree_idx):
            existing_vertices = get_tree_vertices(
                layout.positions[i, 0],
                layout.positions[i, 1],
                layout.rotations[i]
            )
            if check_tree_overlap_python(new_vertices, existing_vertices):
                return False
        
        return True
    
    def _fallback_placement(
        self,
        layout: Layout,
        tree_idx: int,
        rng: np.random.RandomState
    ) -> Tuple[Tuple[float, float], float]:
        """Fallback placement when no candidate works."""
        rotations = self.hyperparams['rotation_palette']
        
        # Try expanding spiral
        for r in np.arange(0.5, 50, 0.2):
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                
                for deg in rotations:
                    if self._is_valid_position(layout, tree_idx, x, y, deg):
                        return (x, y), deg
        
        # Last resort: random search
        for _ in range(1000):
            x = rng.uniform(-50, 50)
            y = rng.uniform(-50, 50)
            deg = rng.choice(rotations)
            if self._is_valid_position(layout, tree_idx, x, y, deg):
                return (x, y), deg
        
        # Should never get here, but return something
        return (50.0, 50.0), 90.0
    
    def improve(
        self,
        layout: Layout,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """
        Try to improve an existing layout.
        
        For greedy solver, this doesn't do much - just returns the layout.
        Use SA or ILS for actual improvement.
        """
        # Greedy solver doesn't have an improvement phase
        return layout
    
    def solve(
        self,
        n: int,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """
        Full solve: init + improve.
        
        Args:
            n: Number of trees
            budget: Time budget in seconds
            rng: Random state
            hyperparams: Optional hyperparameters
        
        Returns:
            Best layout found
        """
        if hyperparams:
            self.hyperparams.update(hyperparams)
        
        layout = self.init_layout(n, rng)
        layout = self.improve(layout, budget, rng, hyperparams)
        
        return layout


def solve_greedy(n: int, seed: int = 42, **kwargs) -> Layout:
    """Convenience function to solve with greedy."""
    rng = np.random.RandomState(seed)
    solver = GreedySolver(kwargs)
    return solver.init_layout(n, rng)


if __name__ == "__main__":
    # Test greedy solver
    import time
    
    for n in [5, 10, 20, 30]:
        print(f"\nSolving for n={n}...")
        start = time.perf_counter()
        
        layout = solve_greedy(n)
        
        elapsed = time.perf_counter() - start
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Bounding side: {s:.4f}")
        print(f"  Score (s^2): {s*s:.4f}")

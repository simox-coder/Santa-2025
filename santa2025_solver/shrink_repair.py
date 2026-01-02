"""
Santa 2025 Solver - Shrink and Repair

Strategy:
1. Start with a feasible solution
2. Shrink the bounding box (move boundary trees inward)
3. Repair any collisions introduced
4. Repeat until no improvement possible
"""

import numpy as np
import time
from typing import Optional, Dict, Any, List, Tuple

from santa2025_solver.geometry_fast import (
    Layout, compute_bounding_square_side, get_tree_vertices,
    get_all_tree_vertices, get_all_aabbs
)
from santa2025_solver.collision_backends.python_backend import (
    check_tree_overlap_python, PythonCollisionBackend
)


class ShrinkRepairSolver:
    """
    Shrink and Repair solver.
    
    Iteratively shrinks the bounding box and repairs collisions.
    """
    
    FAMILY_ID = 4
    FAMILY_NAME = "shrink_repair"
    
    DEFAULT_HYPERPARAMS = {
        # Shrink parameters
        'shrink_delta': 0.01,       # Amount to shrink each iteration
        'shrink_decay': 0.99,       # Delta decay factor
        'min_delta': 0.001,         # Minimum shrink amount
        
        # Repair parameters
        'repair_iterations': 50,    # Max iterations for repair
        'repair_step': 0.02,        # Repair push step
        'repair_decay': 0.95,       # Repair step decay
        
        # Outer loop
        'max_outer_iterations': 1000,
        'max_no_improve': 50,
    }
    
    def __init__(self, hyperparams: Optional[Dict[str, Any]] = None):
        self.hyperparams = {**self.DEFAULT_HYPERPARAMS}
        if hyperparams:
            self.hyperparams.update(hyperparams)
    
    def _get_boundary_trees(self, layout: Layout, threshold_frac: float = 0.9) -> List[int]:
        """
        Get indices of trees that are near the bounding box boundary.
        
        Args:
            layout: Current layout
            threshold_frac: Fraction of max coordinate to consider as boundary
        
        Returns:
            List of tree indices
        """
        vertices = get_all_tree_vertices(layout.positions, layout.rotations)
        
        # Find max absolute coordinate for each tree
        tree_max_coords = np.abs(vertices).max(axis=(1, 2))
        
        # Overall max
        overall_max = tree_max_coords.max()
        
        # Trees near boundary
        threshold = overall_max * threshold_frac
        boundary_trees = np.where(tree_max_coords >= threshold)[0]
        
        return list(boundary_trees)
    
    def _shrink_tree(
        self,
        layout: Layout,
        idx: int,
        delta: float
    ) -> Tuple[float, float]:
        """
        Shrink (move toward center) a single tree.
        
        Returns:
            (old_x, old_y)
        """
        x, y, deg = layout.get_tree(idx)
        
        # Move toward origin
        dist = np.sqrt(x**2 + y**2)
        if dist > delta:
            factor = (dist - delta) / dist
            new_x = x * factor
            new_y = y * factor
            layout.set_tree(idx, new_x, new_y, deg)
        
        return (x, y)
    
    def _repair_collisions(
        self,
        layout: Layout,
        rng: np.random.RandomState
    ) -> bool:
        """
        Repair collisions by pushing overlapping trees apart.
        
        Returns:
            True if repair successful (no collisions remain)
        """
        repair_iters = self.hyperparams['repair_iterations']
        repair_step = self.hyperparams['repair_step']
        repair_decay = self.hyperparams['repair_decay']
        
        n = layout.n
        
        for iteration in range(repair_iters):
            # Find collisions
            collisions = []
            for i in range(n):
                vi = get_tree_vertices(
                    layout.positions[i, 0], layout.positions[i, 1], layout.rotations[i]
                )
                for j in range(i + 1, n):
                    vj = get_tree_vertices(
                        layout.positions[j, 0], layout.positions[j, 1], layout.rotations[j]
                    )
                    if check_tree_overlap_python(vi, vj):
                        collisions.append((i, j))
            
            if not collisions:
                return True
            
            # Repair each collision
            current_step = repair_step * (repair_decay ** iteration)
            
            for i, j in collisions:
                # Direction from j to i
                direction = layout.positions[i] - layout.positions[j]
                dist = np.linalg.norm(direction)
                
                if dist < 1e-6:
                    # Same position, push randomly
                    direction = rng.randn(2)
                    dist = np.linalg.norm(direction)
                
                direction = direction / dist
                
                # Push apart
                xi, yi, di = layout.get_tree(i)
                xj, yj, dj = layout.get_tree(j)
                
                layout.set_tree(i, xi + direction[0] * current_step, 
                               yi + direction[1] * current_step, di)
                layout.set_tree(j, xj - direction[0] * current_step,
                               yj - direction[1] * current_step, dj)
        
        # Check final state
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
    
    def _try_rotate_to_fix(
        self,
        layout: Layout,
        idx: int,
        rng: np.random.RandomState
    ) -> bool:
        """
        Try different rotations to fix collisions involving a tree.
        
        Returns:
            True if a rotation fixed the collision
        """
        x, y, original_deg = layout.get_tree(idx)
        rotations = [0, 90, 180, 270]
        rng.shuffle(rotations)
        
        for deg in rotations:
            layout.set_tree(idx, x, y, deg)
            
            # Check if this tree now collides
            vi = get_tree_vertices(x, y, deg)
            has_collision = False
            
            for j in range(layout.n):
                if j == idx:
                    continue
                vj = get_tree_vertices(
                    layout.positions[j, 0], layout.positions[j, 1], layout.rotations[j]
                )
                if check_tree_overlap_python(vi, vj):
                    has_collision = True
                    break
            
            if not has_collision:
                return True
        
        # Restore original rotation
        layout.set_tree(idx, x, y, original_deg)
        return False
    
    def improve(
        self,
        layout: Layout,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """
        Improve layout using shrink and repair.
        """
        if hyperparams:
            hp = {**self.hyperparams, **hyperparams}
        else:
            hp = self.hyperparams
        
        shrink_delta = hp['shrink_delta']
        shrink_decay = hp['shrink_decay']
        min_delta = hp['min_delta']
        max_outer = hp['max_outer_iterations']
        max_no_improve = hp['max_no_improve']
        
        # Current best
        current_layout = layout.copy()
        current_score = compute_bounding_square_side(
            current_layout.positions, current_layout.rotations
        )
        
        best_layout = current_layout.copy()
        best_score = current_score
        
        start_time = time.perf_counter()
        no_improve_count = 0
        current_delta = shrink_delta
        
        for outer_iter in range(max_outer):
            # Check time budget
            elapsed = time.perf_counter() - start_time
            if elapsed >= budget:
                break
            
            if no_improve_count >= max_no_improve:
                break
            
            if current_delta < min_delta:
                break
            
            # Get boundary trees
            boundary_trees = self._get_boundary_trees(current_layout)
            
            if not boundary_trees:
                break
            
            # Try shrinking each boundary tree
            improved = False
            
            for idx in boundary_trees:
                # Save state
                trial_layout = current_layout.copy()
                
                # Shrink
                self._shrink_tree(trial_layout, idx, current_delta)
                
                # Try rotation fix first
                self._try_rotate_to_fix(trial_layout, idx, rng)
                
                # Repair collisions
                repair_success = self._repair_collisions(trial_layout, rng)
                
                if repair_success:
                    # Check if improved
                    new_score = compute_bounding_square_side(
                        trial_layout.positions, trial_layout.rotations
                    )
                    
                    if new_score < current_score:
                        current_layout = trial_layout
                        current_score = new_score
                        improved = True
                        
                        if new_score < best_score:
                            best_layout = trial_layout.copy()
                            best_score = new_score
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                current_delta *= shrink_decay
        
        return best_layout
    
    def solve(
        self,
        n: int,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """Full solve: init with lattice, then shrink-repair."""
        from santa2025_solver.lattice import LatticeSolver
        
        lattice = LatticeSolver()
        layout = lattice.solve(n, budget * 0.1, rng)
        
        return self.improve(layout, budget * 0.9, rng, hyperparams)


def solve_shrink_repair(n: int, budget: float = 10.0, seed: int = 42, **kwargs) -> Layout:
    """Convenience function to solve with shrink-repair."""
    rng = np.random.RandomState(seed)
    solver = ShrinkRepairSolver(kwargs)
    return solver.solve(n, budget, rng)


if __name__ == "__main__":
    # Test shrink-repair solver
    for n in [10, 20, 30]:
        print(f"\nSolving for n={n} with Shrink-Repair...")
        start = time.perf_counter()
        
        layout = solve_shrink_repair(n, budget=5.0)
        
        elapsed = time.perf_counter() - start
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Bounding side: {s:.4f}")
        print(f"  Score (s^2): {s*s:.4f}")

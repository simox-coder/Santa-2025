"""
Santa 2025 Solver - Simulated Annealing (SA)

Fast SA solver with multiple move types:
- Translate: Move a tree by a small amount
- Rotate: Change tree rotation
- Swap: Swap positions of two trees
- Boundary push: Push trees toward center
"""

import numpy as np
import time
from typing import Optional, Dict, Any, Tuple, List

from santa2025_solver.geometry_fast import (
    Layout, get_tree_vertices, compute_bounding_square_side,
    get_all_tree_vertices, get_all_aabbs
)
from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python


class SimulatedAnnealingSolver:
    """
    Simulated Annealing solver with multiple move types.
    """
    
    FAMILY_ID = 2
    FAMILY_NAME = "sa"
    
    DEFAULT_HYPERPARAMS = {
        # Temperature schedule
        'T0': 1.0,              # Initial temperature
        'T_end': 0.001,         # Final temperature
        'cooling_rate': 0.999,  # Geometric cooling
        
        # Move probabilities
        'p_translate': 0.5,
        'p_rotate': 0.2,
        'p_swap': 0.2,
        'p_boundary_push': 0.1,
        
        # Move parameters
        'translate_step': 0.1,  # Initial step size
        'step_decay': 0.9999,   # Step size decay
        'min_step': 0.001,      # Minimum step size
        
        # Iterations
        'max_iterations': 100000,
        'iterations_per_temp': 100,
        
        # Rotation palette
        'rotation_palette': [0, 90, 180, 270],
    }
    
    def __init__(self, hyperparams: Optional[Dict[str, Any]] = None):
        self.hyperparams = {**self.DEFAULT_HYPERPARAMS}
        if hyperparams:
            self.hyperparams.update(hyperparams)
    
    def _move_translate(
        self,
        layout: Layout,
        rng: np.random.RandomState,
        step_size: float
    ) -> Tuple[int, Tuple[float, float, float]]:
        """
        Translate a random tree.
        
        Returns:
            (tree_idx, (old_x, old_y, old_deg))
        """
        idx = rng.randint(layout.n)
        old_x, old_y, old_deg = layout.get_tree(idx)
        
        # Random direction
        angle = rng.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        layout.set_tree(idx, old_x + dx, old_y + dy, old_deg)
        
        return idx, (old_x, old_y, old_deg)
    
    def _move_rotate(
        self,
        layout: Layout,
        rng: np.random.RandomState
    ) -> Tuple[int, Tuple[float, float, float]]:
        """
        Rotate a random tree.
        
        Returns:
            (tree_idx, (old_x, old_y, old_deg))
        """
        idx = rng.randint(layout.n)
        old_x, old_y, old_deg = layout.get_tree(idx)
        
        # Choose new rotation
        palette = self.hyperparams['rotation_palette']
        new_deg = rng.choice(palette)
        
        layout.set_tree(idx, old_x, old_y, new_deg)
        
        return idx, (old_x, old_y, old_deg)
    
    def _move_swap(
        self,
        layout: Layout,
        rng: np.random.RandomState
    ) -> Tuple[Tuple[int, int], Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Swap positions of two random trees.
        
        Returns:
            ((idx1, idx2), ((old_x1, old_y1, old_deg1), (old_x2, old_y2, old_deg2)))
        """
        if layout.n < 2:
            return (0, 0), ((0, 0, 0), (0, 0, 0))
        
        idx1, idx2 = rng.choice(layout.n, size=2, replace=False)
        
        old1 = layout.get_tree(idx1)
        old2 = layout.get_tree(idx2)
        
        # Swap positions (keep rotations)
        layout.set_tree(idx1, old2[0], old2[1], old1[2])
        layout.set_tree(idx2, old1[0], old1[1], old2[2])
        
        return (idx1, idx2), (old1, old2)
    
    def _move_boundary_push(
        self,
        layout: Layout,
        rng: np.random.RandomState,
        step_size: float
    ) -> Tuple[int, Tuple[float, float, float]]:
        """
        Push a boundary tree toward center.
        
        Returns:
            (tree_idx, (old_x, old_y, old_deg))
        """
        # Find the tree contributing most to bounding box
        vertices = get_all_tree_vertices(layout.positions, layout.rotations)
        max_coords = np.abs(vertices).max(axis=(1, 2))
        
        # Choose from top contributors with some randomness
        probs = max_coords / max_coords.sum()
        idx = rng.choice(layout.n, p=probs)
        
        old_x, old_y, old_deg = layout.get_tree(idx)
        
        # Push toward center
        dist = np.sqrt(old_x**2 + old_y**2)
        if dist > 1e-6:
            dx = -step_size * old_x / dist
            dy = -step_size * old_y / dist
            layout.set_tree(idx, old_x + dx, old_y + dy, old_deg)
        
        return idx, (old_x, old_y, old_deg)
    
    def _undo_translate(self, layout: Layout, idx: int, old_state: Tuple[float, float, float]):
        """Undo a translate move."""
        layout.set_tree(idx, old_state[0], old_state[1], old_state[2])
    
    def _undo_swap(
        self,
        layout: Layout,
        indices: Tuple[int, int],
        old_states: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ):
        """Undo a swap move."""
        idx1, idx2 = indices
        layout.set_tree(idx1, old_states[0][0], old_states[0][1], old_states[0][2])
        layout.set_tree(idx2, old_states[1][0], old_states[1][1], old_states[1][2])
    
    def _count_collisions_for_tree(self, layout: Layout, idx: int) -> int:
        """Count collisions involving a specific tree."""
        vertices_idx = get_tree_vertices(
            layout.positions[idx, 0],
            layout.positions[idx, 1],
            layout.rotations[idx]
        )
        
        count = 0
        for j in range(layout.n):
            if j == idx:
                continue
            
            vertices_j = get_tree_vertices(
                layout.positions[j, 0],
                layout.positions[j, 1],
                layout.rotations[j]
            )
            
            if check_tree_overlap_python(vertices_idx, vertices_j):
                count += 1
        
        return count
    
    def _has_any_collision(self, layout: Layout) -> bool:
        """Check if layout has any collisions."""
        n = layout.n
        for i in range(n):
            for j in range(i + 1, n):
                vi = get_tree_vertices(
                    layout.positions[i, 0], layout.positions[i, 1], layout.rotations[i]
                )
                vj = get_tree_vertices(
                    layout.positions[j, 0], layout.positions[j, 1], layout.rotations[j]
                )
                if check_tree_overlap_python(vi, vj):
                    return True
        return False
    
    def improve(
        self,
        layout: Layout,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """
        Improve layout using simulated annealing.
        
        Args:
            layout: Initial layout
            budget: Time budget in seconds
            rng: Random state
            hyperparams: Optional hyperparameters
        
        Returns:
            Improved layout
        """
        if hyperparams:
            hp = {**self.hyperparams, **hyperparams}
        else:
            hp = self.hyperparams
        
        # Initialize
        T = hp['T0']
        T_end = hp['T_end']
        cooling = hp['cooling_rate']
        step_size = hp['translate_step']
        step_decay = hp['step_decay']
        min_step = hp['min_step']
        max_iter = hp['max_iterations']
        
        p_translate = hp['p_translate']
        p_rotate = hp['p_rotate']
        p_swap = hp['p_swap']
        # p_boundary = hp['p_boundary_push']  # remainder
        
        # Normalize probabilities
        total_p = p_translate + p_rotate + p_swap + hp['p_boundary_push']
        p_translate /= total_p
        p_rotate /= total_p
        p_swap /= total_p
        
        # Current state
        current_layout = layout.copy()
        current_score = compute_bounding_square_side(
            current_layout.positions, current_layout.rotations
        )
        
        # Best state
        best_layout = current_layout.copy()
        best_score = current_score
        
        # Collision penalty
        collision_penalty = 10.0
        
        start_time = time.perf_counter()
        iteration = 0
        
        while T > T_end and iteration < max_iter:
            # Check time budget
            elapsed = time.perf_counter() - start_time
            if elapsed >= budget:
                break
            
            # Choose move type
            r = rng.random()
            
            if r < p_translate:
                move_type = 'translate'
                undo_info = self._move_translate(current_layout, rng, step_size)
            elif r < p_translate + p_rotate:
                move_type = 'rotate'
                undo_info = self._move_rotate(current_layout, rng)
            elif r < p_translate + p_rotate + p_swap:
                move_type = 'swap'
                undo_info = self._move_swap(current_layout, rng)
            else:
                move_type = 'boundary'
                undo_info = self._move_boundary_push(current_layout, rng, step_size)
            
            # Compute new score
            new_score = compute_bounding_square_side(
                current_layout.positions, current_layout.rotations
            )
            
            # Add collision penalty
            if move_type in ['translate', 'rotate', 'boundary']:
                idx = undo_info[0]
                collisions = self._count_collisions_for_tree(current_layout, idx)
            else:
                idx1, idx2 = undo_info[0]
                collisions = (
                    self._count_collisions_for_tree(current_layout, idx1) +
                    self._count_collisions_for_tree(current_layout, idx2)
                )
            
            new_score += collisions * collision_penalty
            
            # Accept or reject
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Accept
                current_score = new_score - collisions * collision_penalty  # Store actual score
                
                # Update best if no collisions and better
                if collisions == 0 and current_score < best_score:
                    best_layout = current_layout.copy()
                    best_score = current_score
            else:
                # Reject - undo move
                if move_type in ['translate', 'rotate', 'boundary']:
                    self._undo_translate(current_layout, undo_info[0], undo_info[1])
                else:
                    self._undo_swap(current_layout, undo_info[0], undo_info[1])
            
            # Cool down
            T *= cooling
            step_size = max(min_step, step_size * step_decay)
            iteration += 1
        
        return best_layout
    
    def solve(
        self,
        n: int,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """
        Full solve: init with greedy, then SA.
        """
        from santa2025_solver.greedy import GreedySolver
        
        # Initialize with greedy
        greedy = GreedySolver()
        layout = greedy.init_layout(n, rng)
        
        # Improve with SA
        return self.improve(layout, budget, rng, hyperparams)


def solve_sa(n: int, budget: float = 10.0, seed: int = 42, **kwargs) -> Layout:
    """Convenience function to solve with SA."""
    rng = np.random.RandomState(seed)
    solver = SimulatedAnnealingSolver(kwargs)
    return solver.solve(n, budget, rng)


if __name__ == "__main__":
    # Test SA solver
    for n in [5, 10, 20]:
        print(f"\nSolving for n={n} with SA...")
        start = time.perf_counter()
        
        layout = solve_sa(n, budget=5.0)
        
        elapsed = time.perf_counter() - start
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Bounding side: {s:.4f}")
        print(f"  Score (s^2): {s*s:.4f}")

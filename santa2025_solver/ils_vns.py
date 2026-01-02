"""
Santa 2025 Solver - Iterated Local Search (ILS) / Variable Neighborhood Search (VNS)

ILS: Perturb solution + local search
VNS: Try different neighborhood structures
"""

import numpy as np
import time
from typing import Optional, Dict, Any, Tuple

from santa2025_solver.geometry_fast import (
    Layout, compute_bounding_square_side, get_tree_vertices,
    get_all_tree_vertices
)
from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python
from santa2025_solver.sa import SimulatedAnnealingSolver


class ILSVNSSolver:
    """
    Iterated Local Search with Variable Neighborhood Search.
    
    Strategy:
    1. Start from initial solution
    2. Perturb (kick) solution
    3. Apply local search (short SA)
    4. Accept if improved (or by metropolis criterion)
    5. Repeat
    """
    
    FAMILY_ID = 3
    FAMILY_NAME = "ils_vns"
    
    DEFAULT_HYPERPARAMS = {
        # Kick parameters
        'kick_strength': 0.2,      # Fraction of trees to perturb
        'kick_magnitude': 0.3,     # Perturbation size
        
        # Neighborhood escalation
        'neighborhoods': ['translate', 'rotate', 'swap', 'cluster'],
        'max_no_improve': 5,       # Steps before escalating neighborhood
        
        # Local search (SA) parameters
        'local_search_budget': 0.5,  # Fraction of budget for each local search
        'sa_T0': 0.1,
        'sa_T_end': 0.001,
        'sa_cooling': 0.995,
        'sa_iterations': 1000,
        
        # ILS parameters
        'max_outer_iterations': 100,
        'accept_worse_prob': 0.1,  # Probability of accepting worse solution
    }
    
    def __init__(self, hyperparams: Optional[Dict[str, Any]] = None):
        self.hyperparams = {**self.DEFAULT_HYPERPARAMS}
        if hyperparams:
            self.hyperparams.update(hyperparams)
    
    def _kick_translate(self, layout: Layout, rng: np.random.RandomState) -> Layout:
        """Kick by translating random subset of trees."""
        new_layout = layout.copy()
        n = layout.n
        
        kick_strength = self.hyperparams['kick_strength']
        kick_mag = self.hyperparams['kick_magnitude']
        
        n_kick = max(1, int(n * kick_strength))
        indices = rng.choice(n, size=n_kick, replace=False)
        
        for idx in indices:
            x, y, deg = new_layout.get_tree(idx)
            angle = rng.uniform(0, 2 * np.pi)
            dx = kick_mag * np.cos(angle)
            dy = kick_mag * np.sin(angle)
            new_layout.set_tree(idx, x + dx, y + dy, deg)
        
        return new_layout
    
    def _kick_rotate(self, layout: Layout, rng: np.random.RandomState) -> Layout:
        """Kick by rotating random subset of trees."""
        new_layout = layout.copy()
        n = layout.n
        
        kick_strength = self.hyperparams['kick_strength']
        n_kick = max(1, int(n * kick_strength))
        indices = rng.choice(n, size=n_kick, replace=False)
        
        rotations = [0, 90, 180, 270]
        for idx in indices:
            x, y, _ = new_layout.get_tree(idx)
            new_deg = rng.choice(rotations)
            new_layout.set_tree(idx, x, y, new_deg)
        
        return new_layout
    
    def _kick_swap(self, layout: Layout, rng: np.random.RandomState) -> Layout:
        """Kick by swapping positions of tree pairs."""
        new_layout = layout.copy()
        n = layout.n
        
        if n < 2:
            return new_layout
        
        kick_strength = self.hyperparams['kick_strength']
        n_swaps = max(1, int(n * kick_strength / 2))
        
        for _ in range(n_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            xi, yi, di = new_layout.get_tree(i)
            xj, yj, dj = new_layout.get_tree(j)
            new_layout.set_tree(i, xj, yj, di)
            new_layout.set_tree(j, xi, yi, dj)
        
        return new_layout
    
    def _kick_cluster(self, layout: Layout, rng: np.random.RandomState) -> Layout:
        """Kick by moving a cluster of nearby trees together."""
        new_layout = layout.copy()
        n = layout.n
        
        if n < 3:
            return self._kick_translate(new_layout, rng)
        
        # Pick a random tree as cluster center
        center_idx = rng.randint(n)
        center_pos = layout.positions[center_idx]
        
        # Find k nearest neighbors
        k = max(2, int(n * self.hyperparams['kick_strength']))
        distances = np.linalg.norm(layout.positions - center_pos, axis=1)
        nearest = np.argsort(distances)[:k]
        
        # Move cluster
        kick_mag = self.hyperparams['kick_magnitude']
        angle = rng.uniform(0, 2 * np.pi)
        dx = kick_mag * np.cos(angle)
        dy = kick_mag * np.sin(angle)
        
        for idx in nearest:
            x, y, deg = new_layout.get_tree(idx)
            new_layout.set_tree(idx, x + dx, y + dy, deg)
        
        return new_layout
    
    def _kick(self, layout: Layout, rng: np.random.RandomState, neighborhood: str) -> Layout:
        """Apply kick using specified neighborhood."""
        if neighborhood == 'translate':
            return self._kick_translate(layout, rng)
        elif neighborhood == 'rotate':
            return self._kick_rotate(layout, rng)
        elif neighborhood == 'swap':
            return self._kick_swap(layout, rng)
        elif neighborhood == 'cluster':
            return self._kick_cluster(layout, rng)
        else:
            return self._kick_translate(layout, rng)
    
    def _local_search(
        self,
        layout: Layout,
        budget: float,
        rng: np.random.RandomState
    ) -> Layout:
        """Apply local search using short SA."""
        sa_params = {
            'T0': self.hyperparams['sa_T0'],
            'T_end': self.hyperparams['sa_T_end'],
            'cooling_rate': self.hyperparams['sa_cooling'],
            'max_iterations': self.hyperparams['sa_iterations'],
        }
        
        sa = SimulatedAnnealingSolver(sa_params)
        return sa.improve(layout, budget, rng)
    
    def _is_feasible(self, layout: Layout) -> bool:
        """Check if layout has no collisions."""
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
    
    def improve(
        self,
        layout: Layout,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """
        Improve layout using ILS/VNS.
        """
        if hyperparams:
            hp = {**self.hyperparams, **hyperparams}
        else:
            hp = self.hyperparams
        
        neighborhoods = hp['neighborhoods']
        max_no_improve = hp['max_no_improve']
        accept_worse_prob = hp['accept_worse_prob']
        max_outer = hp['max_outer_iterations']
        local_budget_frac = hp['local_search_budget']
        
        # Current best
        current_layout = layout.copy()
        current_score = compute_bounding_square_side(
            current_layout.positions, current_layout.rotations
        )
        
        best_layout = current_layout.copy()
        best_score = current_score
        
        start_time = time.perf_counter()
        no_improve_count = 0
        neighborhood_idx = 0
        
        for outer_iter in range(max_outer):
            # Check time budget
            elapsed = time.perf_counter() - start_time
            if elapsed >= budget:
                break
            
            remaining = budget - elapsed
            local_budget = min(remaining * local_budget_frac, remaining / 2)
            
            # Current neighborhood
            neighborhood = neighborhoods[neighborhood_idx % len(neighborhoods)]
            
            # Kick
            kicked_layout = self._kick(current_layout, rng, neighborhood)
            
            # Local search
            improved_layout = self._local_search(kicked_layout, local_budget, rng)
            
            # Evaluate
            new_score = compute_bounding_square_side(
                improved_layout.positions, improved_layout.rotations
            )
            
            # Check feasibility
            is_feasible = self._is_feasible(improved_layout)
            
            # Accept?
            accept = False
            if is_feasible:
                if new_score < current_score:
                    accept = True
                    no_improve_count = 0
                    neighborhood_idx = 0
                elif rng.random() < accept_worse_prob:
                    accept = True
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if accept:
                current_layout = improved_layout
                current_score = new_score
                
                if new_score < best_score:
                    best_layout = improved_layout.copy()
                    best_score = new_score
            
            # Escalate neighborhood if stuck
            if no_improve_count >= max_no_improve:
                neighborhood_idx += 1
                no_improve_count = 0
        
        return best_layout
    
    def solve(
        self,
        n: int,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """Full solve: init with greedy, then ILS/VNS."""
        from santa2025_solver.greedy import GreedySolver
        
        greedy = GreedySolver()
        layout = greedy.init_layout(n, rng)
        
        return self.improve(layout, budget, rng, hyperparams)


def solve_ils_vns(n: int, budget: float = 10.0, seed: int = 42, **kwargs) -> Layout:
    """Convenience function to solve with ILS/VNS."""
    rng = np.random.RandomState(seed)
    solver = ILSVNSSolver(kwargs)
    return solver.solve(n, budget, rng)


if __name__ == "__main__":
    # Test ILS/VNS solver
    for n in [10, 20, 30]:
        print(f"\nSolving for n={n} with ILS/VNS...")
        start = time.perf_counter()
        
        layout = solve_ils_vns(n, budget=5.0)
        
        elapsed = time.perf_counter() - start
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Bounding side: {s:.4f}")
        print(f"  Score (s^2): {s*s:.4f}")

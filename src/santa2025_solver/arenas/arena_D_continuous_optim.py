"""
Arena D: Continuous Optimization (CMA-ES style / Coordinate Descent).

Goal: Exploit continuous nature of (x, y, deg) with constraint handling.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import time

from ..geometry import transform_tree, get_bounding_square_side
from ..collision.bench import get_collision_backend

class ContinuousOptimArena:
    """Continuous optimization arena."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.backend = get_collision_backend()
    
    def propose_hparams(self) -> Dict[str, Any]:
        """Propose random hyperparameters."""
        return {
            'sigma0': self.rng.uniform(0.1, 1.0),
            'population': self.rng.integers(10, 50),
            'elite_ratio': self.rng.uniform(0.1, 0.4),
            'max_generations': self.rng.integers(50, 200),
            'penalty_weight': self.rng.uniform(10.0, 100.0),
            'projection_strength': self.rng.uniform(0.5, 1.0),
            'safety_gap': self.rng.uniform(0.001, 0.02),
            'coord_descent_iters': self.rng.integers(10, 100),
            'step_size': self.rng.uniform(0.005, 0.05),
        }
    
    def check_feasibility(self, positions: np.ndarray, safety_gap: float) -> bool:
        """Check if layout is feasible."""
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        return not self.backend.has_any_collision(polygons, -safety_gap)
    
    def get_score(self, positions: np.ndarray) -> float:
        """Get bounding square side."""
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        return get_bounding_square_side(polygons)
    
    def count_collisions(self, positions: np.ndarray, safety_gap: float) -> int:
        """Count number of colliding pairs."""
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        collisions = self.backend.check_all_pairs(polygons, -safety_gap)
        return len(collisions)
    
    def fitness_with_penalty(self, positions: np.ndarray, hparams: Dict[str, Any]) -> float:
        """Fitness function with penalty for infeasibility."""
        score = self.get_score(positions)
        n_collisions = self.count_collisions(positions, hparams['safety_gap'])
        return score + hparams['penalty_weight'] * n_collisions
    
    def project_to_feasible(self, positions: np.ndarray, hparams: Dict[str, Any],
                            max_iters: int = 50) -> np.ndarray:
        """Project infeasible solution toward feasibility using MTV."""
        from ..collision.backend_py import sat_collision_mtv
        
        positions = positions.copy()
        
        for _ in range(max_iters):
            polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
            collisions = self.backend.check_all_pairs(polygons, -hparams['safety_gap'])
            
            if not collisions:
                return positions
            
            # Fix first collision using MTV
            i, j = collisions[0]
            mtv = sat_collision_mtv(polygons[i], polygons[j])
            
            if mtv is not None:
                axis, magnitude = mtv
                magnitude = (magnitude + hparams['safety_gap'] + 0.001) * hparams['projection_strength']
                positions[i, 0] -= axis[0] * magnitude * 0.5
                positions[i, 1] -= axis[1] * magnitude * 0.5
                positions[j, 0] += axis[0] * magnitude * 0.5
                positions[j, 1] += axis[1] * magnitude * 0.5
        
        return positions
    
    def coordinate_descent(self, positions: np.ndarray, hparams: Dict[str, Any]) -> np.ndarray:
        """Coordinate descent optimization."""
        positions = positions.copy()
        n = len(positions)
        step = hparams['step_size']
        
        best_score = self.get_score(positions) if self.check_feasibility(positions, hparams['safety_gap']) else float('inf')
        best_positions = positions.copy()
        
        for _ in range(hparams['coord_descent_iters']):
            improved = False
            
            for i in range(n):
                for dim in range(3):  # x, y, deg
                    for direction in [-1, 1]:
                        new_positions = positions.copy()
                        
                        if dim == 2:  # deg
                            new_positions[i, dim] = (new_positions[i, dim] + direction * step * 10) % 360
                        else:  # x, y
                            new_positions[i, dim] += direction * step
                        
                        if self.check_feasibility(new_positions, hparams['safety_gap']):
                            new_score = self.get_score(new_positions)
                            if new_score < best_score:
                                best_score = new_score
                                best_positions = new_positions.copy()
                                positions = new_positions
                                improved = True
                                break
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try shrinking
                shrunk = self.shrink_layout(positions, 0.99)
                if self.check_feasibility(shrunk, hparams['safety_gap']):
                    new_score = self.get_score(shrunk)
                    if new_score < best_score:
                        best_score = new_score
                        best_positions = shrunk.copy()
                        positions = shrunk
                        improved = True
            
            if not improved:
                break
        
        return best_positions
    
    def shrink_layout(self, positions: np.ndarray, factor: float) -> np.ndarray:
        """Shrink layout toward centroid."""
        cx = positions[:, 0].mean()
        cy = positions[:, 1].mean()
        new_positions = positions.copy()
        new_positions[:, 0] = cx + (positions[:, 0] - cx) * factor
        new_positions[:, 1] = cy + (positions[:, 1] - cy) * factor
        return new_positions
    
    def evolution_strategy(self, n_trees: int, hparams: Dict[str, Any],
                           time_budget: float, initial: np.ndarray = None) -> np.ndarray:
        """Simple (1+lambda) evolution strategy."""
        pop_size = hparams['population']
        sigma = hparams['sigma0']
        
        # Initialize
        if initial is not None:
            parent = initial.copy()
        else:
            parent = self.generate_initial(n_trees, hparams)
        
        if parent is None:
            return None
        
        parent_score = self.fitness_with_penalty(parent, hparams)
        best = parent.copy()
        best_score = parent_score if self.check_feasibility(parent, hparams['safety_gap']) else float('inf')
        
        start_time = time.time()
        generation = 0
        
        while generation < hparams['max_generations'] and time.time() - start_time < time_budget:
            # Generate offspring
            offspring_list = []
            for _ in range(pop_size):
                child = parent.copy()
                # Mutate
                child[:, 0] += self.rng.normal(0, sigma, n_trees)
                child[:, 1] += self.rng.normal(0, sigma, n_trees)
                child[:, 2] = (child[:, 2] + self.rng.normal(0, sigma * 10, n_trees)) % 360
                offspring_list.append(child)
            
            # Evaluate
            scores = [self.fitness_with_penalty(c, hparams) for c in offspring_list]
            
            # Select best
            best_idx = np.argmin(scores)
            if scores[best_idx] < parent_score:
                parent = offspring_list[best_idx]
                parent_score = scores[best_idx]
                
                # Check if truly feasible and track best
                if self.check_feasibility(parent, hparams['safety_gap']):
                    real_score = self.get_score(parent)
                    if real_score < best_score:
                        best = parent.copy()
                        best_score = real_score
            
            # Adapt sigma
            sigma *= 0.99
            generation += 1
        
        # Final projection if needed
        if not self.check_feasibility(best, hparams['safety_gap']):
            best = self.project_to_feasible(best, hparams)
        
        return best
    
    def generate_initial(self, n: int, hparams: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate initial feasible layout."""
        from .arena_B_lattice_templates import LatticeArena
        lattice = LatticeArena(self.rng.integers(0, 2**31))
        lattice_hparams = {
            'lattice_type': 'hex',
            'v1_len': 1.0,
            'v2_len': 1.0,
            'lattice_angle': 60,
            'global_rotation': 0,
            'rotation_palette': [0, 90, 180, 270],
            'safety_gap': hparams['safety_gap'],
            'relax_steps': 0,
            'step_scale': 0.05,
        }
        positions = lattice.generate_lattice_positions(n, lattice_hparams)
        return lattice.scale_to_feasible(positions, lattice_hparams)
    
    def run_trial(self, hparams: Dict[str, Any], seed: int,
                  stage: int, time_budget_sec: float,
                  initial_layouts: Dict[int, np.ndarray] = None) -> Dict[str, Any]:
        """Run a trial with given hyperparameters."""
        self.rng = np.random.default_rng(seed)
        
        stage_ranges = {
            0: (1, 30),
            1: (1, 60),
            2: (1, 120),
            3: (1, 200)
        }
        n_min, n_max = stage_ranges.get(stage, (1, 200))
        
        start_time = time.time()
        layouts = {}
        total_score = 0.0
        
        for n in range(n_min, n_max + 1):
            elapsed = time.time() - start_time
            if elapsed > time_budget_sec:
                break
            
            remaining = time_budget_sec - elapsed
            per_n_budget = remaining / max(1, n_max - n + 1)
            
            # Get initial
            initial = None
            if initial_layouts and n in initial_layouts:
                initial = initial_layouts[n].copy()
            
            # Run ES
            optimized = self.evolution_strategy(n, hparams, per_n_budget * 0.7, initial)
            
            if optimized is not None:
                # Coordinate descent refinement
                optimized = self.coordinate_descent(optimized, hparams)
                
                if self.check_feasibility(optimized, hparams['safety_gap']):
                    layouts[n] = optimized
                    s = self.get_score(optimized)
                    total_score += s * s / n
        
        return {
            'score_stage': total_score,
            'best_layouts': layouts,
            'n_solved': len(layouts),
            'time_sec': time.time() - start_time
        }

def run_arena_d(n: int, seed: int = 42, hparams: Dict[str, Any] = None,
                initial: np.ndarray = None) -> Optional[np.ndarray]:
    """Convenience function to run arena D for single n."""
    arena = ContinuousOptimArena(seed)
    if hparams is None:
        hparams = {
            'sigma0': 0.3,
            'population': 20,
            'elite_ratio': 0.2,
            'max_generations': 100,
            'penalty_weight': 50.0,
            'projection_strength': 0.8,
            'safety_gap': 0.005,
            'coord_descent_iters': 50,
            'step_size': 0.02,
        }
    
    result = arena.evolution_strategy(n, hparams, time_budget=30.0, initial=initial)
    if result is not None:
        result = arena.coordinate_descent(result, hparams)
    return result

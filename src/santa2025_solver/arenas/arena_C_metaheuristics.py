"""
Arena C: Metaheuristics (SA / ILS / VNS / LNS).

Goal: Escape local minima and find better raw scores through search.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import time

from ..geometry import transform_tree, get_bounding_square_side
from ..collision.bench import get_collision_backend

class MetaheuristicArena:
    """Metaheuristic optimization arena."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.backend = get_collision_backend()
    
    def propose_hparams(self) -> Dict[str, Any]:
        """Propose random hyperparameters."""
        return {
            'T0': self.rng.uniform(1.0, 10.0),
            'T_end': self.rng.uniform(0.001, 0.1),
            'cooling': self.rng.choice(['exponential', 'linear']),
            'cooling_rate': self.rng.uniform(0.95, 0.999),
            'move_probs': {
                'translate': self.rng.uniform(0.3, 0.5),
                'rotate': self.rng.uniform(0.2, 0.4),
                'swap': self.rng.uniform(0.05, 0.15),
            },
            'translate_scale': self.rng.uniform(0.01, 0.2),
            'rotate_scale': self.rng.uniform(5, 45),
            'safety_gap': self.rng.uniform(0.001, 0.02),
            'max_iters': self.rng.integers(500, 5000),
            'lns_ruin_frac': self.rng.uniform(0.1, 0.4),
        }
    
    def check_feasibility(self, positions: np.ndarray, safety_gap: float) -> bool:
        """Check if layout is feasible."""
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        return not self.backend.has_any_collision(polygons, -safety_gap)
    
    def get_score(self, positions: np.ndarray) -> float:
        """Get bounding square side (smaller is better)."""
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        return get_bounding_square_side(polygons)
    
    def move_translate(self, positions: np.ndarray, idx: int, 
                       scale: float) -> np.ndarray:
        """Apply random translation to one tree."""
        new_pos = positions.copy()
        new_pos[idx, 0] += self.rng.uniform(-scale, scale)
        new_pos[idx, 1] += self.rng.uniform(-scale, scale)
        return new_pos
    
    def move_rotate(self, positions: np.ndarray, idx: int,
                    scale: float) -> np.ndarray:
        """Apply random rotation to one tree."""
        new_pos = positions.copy()
        new_pos[idx, 2] = (new_pos[idx, 2] + self.rng.uniform(-scale, scale)) % 360
        return new_pos
    
    def move_swap(self, positions: np.ndarray, i: int, j: int) -> np.ndarray:
        """Swap positions of two trees."""
        new_pos = positions.copy()
        new_pos[i, :2], new_pos[j, :2] = new_pos[j, :2].copy(), new_pos[i, :2].copy()
        return new_pos
    
    def move_shrink(self, positions: np.ndarray, factor: float) -> np.ndarray:
        """Shrink layout toward centroid."""
        cx = positions[:, 0].mean()
        cy = positions[:, 1].mean()
        new_pos = positions.copy()
        new_pos[:, 0] = cx + (positions[:, 0] - cx) * factor
        new_pos[:, 1] = cy + (positions[:, 1] - cy) * factor
        return new_pos
    
    def simulated_annealing(self, initial_positions: np.ndarray,
                            hparams: Dict[str, Any], time_budget: float) -> np.ndarray:
        """Run simulated annealing."""
        positions = initial_positions.copy()
        n = len(positions)
        
        T = hparams['T0']
        T_end = hparams['T_end']
        cooling_rate = hparams['cooling_rate']
        
        best_positions = positions.copy()
        best_score = self.get_score(positions)
        current_score = best_score
        
        start_time = time.time()
        iterations = 0
        max_iters = hparams['max_iters']
        
        while T > T_end and iterations < max_iters:
            if time.time() - start_time > time_budget:
                break
            
            # Choose move type
            r = self.rng.random()
            probs = hparams['move_probs']
            
            if r < probs['translate']:
                idx = self.rng.integers(0, n)
                new_positions = self.move_translate(positions, idx, hparams['translate_scale'])
            elif r < probs['translate'] + probs['rotate']:
                idx = self.rng.integers(0, n)
                new_positions = self.move_rotate(positions, idx, hparams['rotate_scale'])
            else:
                if n >= 2:
                    i, j = self.rng.choice(n, size=2, replace=False)
                    new_positions = self.move_swap(positions, i, j)
                else:
                    continue
            
            # Check feasibility
            if not self.check_feasibility(new_positions, hparams['safety_gap']):
                iterations += 1
                continue
            
            # Evaluate
            new_score = self.get_score(new_positions)
            delta = new_score - current_score
            
            # Accept or reject
            if delta < 0 or self.rng.random() < math.exp(-delta / T):
                positions = new_positions
                current_score = new_score
                
                if current_score < best_score:
                    best_positions = positions.copy()
                    best_score = current_score
            
            # Cool down
            if hparams['cooling'] == 'exponential':
                T *= cooling_rate
            else:
                T -= (hparams['T0'] - T_end) / max_iters
            
            iterations += 1
        
        return best_positions
    
    def local_search(self, positions: np.ndarray, hparams: Dict[str, Any],
                     max_iters: int = 100) -> np.ndarray:
        """Simple local search (hill climbing)."""
        positions = positions.copy()
        n = len(positions)
        best_score = self.get_score(positions)
        
        for _ in range(max_iters):
            improved = False
            
            for i in range(n):
                # Try small translations
                for dx in [-0.01, 0, 0.01]:
                    for dy in [-0.01, 0, 0.01]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        new_pos = positions.copy()
                        new_pos[i, 0] += dx
                        new_pos[i, 1] += dy
                        
                        if self.check_feasibility(new_pos, hparams['safety_gap']):
                            new_score = self.get_score(new_pos)
                            if new_score < best_score:
                                positions = new_pos
                                best_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                
                # Try small rotations
                if not improved:
                    for ddeg in [-5, 5]:
                        new_pos = positions.copy()
                        new_pos[i, 2] = (new_pos[i, 2] + ddeg) % 360
                        
                        if self.check_feasibility(new_pos, hparams['safety_gap']):
                            new_score = self.get_score(new_pos)
                            if new_score < best_score:
                                positions = new_pos
                                best_score = new_score
                                improved = True
                                break
            
            if not improved:
                break
        
        return positions
    
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
            
            # Get initial layout
            if initial_layouts and n in initial_layouts:
                initial = initial_layouts[n].copy()
            else:
                # Generate random initial layout
                initial = self.generate_random_layout(n, hparams)
                if initial is None:
                    continue
            
            # Run SA
            optimized = self.simulated_annealing(initial, hparams, per_n_budget * 0.8)
            
            # Local search refinement
            optimized = self.local_search(optimized, hparams, max_iters=50)
            
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
    
    def generate_random_layout(self, n: int, hparams: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate a random feasible layout."""
        # Use lattice as starting point
        from .arena_B_lattice_templates import LatticeArena
        lattice = LatticeArena(self.rng.integers(0, 2**31))
        lattice_hparams = {
            'lattice_type': 'hex',
            'v1_len': 1.0,
            'v2_len': 1.0,
            'lattice_angle': 60,
            'global_rotation': self.rng.uniform(0, 360),
            'rotation_palette': [0, 90, 180, 270],
            'safety_gap': hparams['safety_gap'],
            'relax_steps': 0,
            'step_scale': 0.05,
        }
        positions = lattice.generate_lattice_positions(n, lattice_hparams)
        return lattice.scale_to_feasible(positions, lattice_hparams)

def run_arena_c(n: int, seed: int = 42, hparams: Dict[str, Any] = None,
                initial: np.ndarray = None) -> Optional[np.ndarray]:
    """Convenience function to run arena C for single n."""
    arena = MetaheuristicArena(seed)
    if hparams is None:
        hparams = {
            'T0': 5.0,
            'T_end': 0.01,
            'cooling': 'exponential',
            'cooling_rate': 0.99,
            'move_probs': {'translate': 0.4, 'rotate': 0.3, 'swap': 0.1},
            'translate_scale': 0.05,
            'rotate_scale': 15,
            'safety_gap': 0.005,
            'max_iters': 1000,
            'lns_ruin_frac': 0.2,
        }
    
    if initial is None:
        initial = arena.generate_random_layout(n, hparams)
    
    if initial is None:
        return None
    
    result = arena.simulated_annealing(initial, hparams, time_budget=30.0)
    return arena.local_search(result, hparams)

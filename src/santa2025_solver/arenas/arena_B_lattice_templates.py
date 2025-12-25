"""
Arena B: Lattice / Template Packing.

Goal: Use regular lattice patterns for stable, scalable packing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math

from ..geometry import transform_tree, get_bounding_square_side
from ..collision.bench import get_collision_backend

class LatticeArena:
    """Lattice-based packing arena."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.backend = get_collision_backend()
    
    def propose_hparams(self) -> Dict[str, Any]:
        """Propose random hyperparameters."""
        return {
            'lattice_type': self.rng.choice(['square', 'hex', 'triangular']),
            'v1_len': self.rng.uniform(0.5, 1.5),
            'v2_len': self.rng.uniform(0.5, 1.5),
            'lattice_angle': self.rng.uniform(30, 150),
            'global_rotation': self.rng.uniform(0, 360),
            'rotation_palette': list(self.rng.choice([0, 90, 180, 270], size=4)),
            'safety_gap': self.rng.uniform(0.001, 0.05),
            'relax_steps': self.rng.integers(0, 30),
            'step_scale': self.rng.uniform(0.01, 0.1),
        }
    
    def generate_lattice_positions(self, n: int, hparams: Dict[str, Any]) -> np.ndarray:
        """Generate n positions on a lattice."""
        v1_len = hparams['v1_len']
        v2_len = hparams['v2_len']
        angle = math.radians(hparams['lattice_angle'])
        
        # Basis vectors
        v1 = np.array([v1_len, 0.0])
        v2 = np.array([v2_len * math.cos(angle), v2_len * math.sin(angle)])
        
        # Generate enough grid points
        grid_size = int(math.ceil(math.sqrt(n))) + 1
        
        positions = []
        for i in range(-grid_size, grid_size + 1):
            for j in range(-grid_size, grid_size + 1):
                pos = i * v1 + j * v2
                positions.append(pos)
        
        positions = np.array(positions)
        
        # Sort by distance from origin and take n closest
        distances = np.linalg.norm(positions, axis=1)
        indices = np.argsort(distances)[:n]
        positions = positions[indices]
        
        # Apply global rotation
        global_rot = math.radians(hparams['global_rotation'])
        cos_r, sin_r = math.cos(global_rot), math.sin(global_rot)
        rotated = np.empty_like(positions)
        rotated[:, 0] = positions[:, 0] * cos_r - positions[:, 1] * sin_r
        rotated[:, 1] = positions[:, 0] * sin_r + positions[:, 1] * cos_r
        
        # Add rotation angles
        palette = hparams['rotation_palette']
        result = np.zeros((n, 3))
        result[:, :2] = rotated
        for i in range(n):
            result[i, 2] = palette[i % len(palette)]
        
        return result
    
    def check_feasibility(self, positions: np.ndarray, safety_gap: float) -> bool:
        """Check if layout is feasible."""
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        return not self.backend.has_any_collision(polygons, -safety_gap)
    
    def relax_layout(self, positions: np.ndarray, hparams: Dict[str, Any]) -> np.ndarray:
        """Local relaxation to improve layout."""
        positions = positions.copy()
        n = len(positions)
        
        for _ in range(hparams['relax_steps']):
            # Try small moves to reduce bounding square
            polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
            current_score = get_bounding_square_side(polygons)
            
            # Compute centroid
            cx = positions[:, 0].mean()
            cy = positions[:, 1].mean()
            
            improved = False
            for i in range(n):
                # Try moving toward centroid
                for scale in [hparams['step_scale'], -hparams['step_scale']]:
                    new_positions = positions.copy()
                    dx = (cx - positions[i, 0]) * scale
                    dy = (cy - positions[i, 1]) * scale
                    new_positions[i, 0] += dx
                    new_positions[i, 1] += dy
                    
                    if self.check_feasibility(new_positions, hparams['safety_gap']):
                        new_polygons = [transform_tree(p[0], p[1], p[2]) for p in new_positions]
                        new_score = get_bounding_square_side(new_polygons)
                        if new_score < current_score:
                            positions = new_positions
                            current_score = new_score
                            improved = True
                            break
                
                if improved:
                    break
        
        return positions
    
    def scale_to_feasible(self, positions: np.ndarray, hparams: Dict[str, Any]) -> np.ndarray:
        """Scale layout until feasible."""
        positions = positions.copy()
        
        # Start with current scale and increase if needed
        for scale_mult in np.linspace(1.0, 3.0, 20):
            scaled = positions.copy()
            scaled[:, 0] *= scale_mult
            scaled[:, 1] *= scale_mult
            
            if self.check_feasibility(scaled, hparams['safety_gap']):
                return scaled
        
        return None
    
    def run_trial(self, hparams: Dict[str, Any], seed: int,
                  stage: int, time_budget_sec: float) -> Dict[str, Any]:
        """Run a trial with given hyperparameters."""
        import time
        
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
            if time.time() - start_time > time_budget_sec:
                break
            
            # Generate lattice positions
            positions = self.generate_lattice_positions(n, hparams)
            
            # Scale to feasible
            positions = self.scale_to_feasible(positions, hparams)
            
            if positions is None:
                continue
            
            # Relax
            if hparams['relax_steps'] > 0:
                positions = self.relax_layout(positions, hparams)
            
            layouts[n] = positions
            polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
            s = get_bounding_square_side(polygons)
            total_score += s * s / n
        
        return {
            'score_stage': total_score,
            'best_layouts': layouts,
            'n_solved': len(layouts),
            'time_sec': time.time() - start_time
        }

def run_arena_b(n: int, seed: int = 42, hparams: Dict[str, Any] = None) -> Optional[np.ndarray]:
    """Convenience function to run arena B for single n."""
    arena = LatticeArena(seed)
    if hparams is None:
        hparams = {
            'lattice_type': 'hex',
            'v1_len': 0.8,
            'v2_len': 0.8,
            'lattice_angle': 60,
            'global_rotation': 0,
            'rotation_palette': [0, 90, 180, 270],
            'safety_gap': 0.01,
            'relax_steps': 10,
            'step_scale': 0.05,
        }
    
    positions = arena.generate_lattice_positions(n, hparams)
    positions = arena.scale_to_feasible(positions, hparams)
    if positions is not None and hparams['relax_steps'] > 0:
        positions = arena.relax_layout(positions, hparams)
    return positions

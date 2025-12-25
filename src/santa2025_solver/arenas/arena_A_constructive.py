"""
Arena A: Constructive / Contact-Graph methods.

Goal: Build strong feasible initial layouts quickly using greedy placement
with contact/tangent candidates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math

from ..geometry import transform_tree, get_bounding_square_side, aabb, aabb_intersect
from ..collision.bench import get_collision_backend

class ConstructiveArena:
    """Constructive packing arena."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.backend = get_collision_backend()
        
    def propose_hparams(self) -> Dict[str, Any]:
        """Propose random hyperparameters."""
        return {
            'candidate_count': self.rng.integers(8, 32),
            'beam_width': self.rng.integers(1, 5),
            'rotation_steps': self.rng.integers(4, 16),
            'placement_order': self.rng.choice(['random', 'center_first', 'spread']),
            'safety_gap': self.rng.uniform(0.001, 0.05),
            'tighten_iters': self.rng.integers(0, 20),
            'shrink_factor': self.rng.uniform(0.95, 0.999),
        }
    
    def generate_rotation_palette(self, rotation_steps: int) -> np.ndarray:
        """Generate rotation angles to try."""
        return np.linspace(0, 360, rotation_steps, endpoint=False)
    
    def generate_candidates(self, placed_polygons: List[np.ndarray], 
                           placed_positions: np.ndarray,
                           candidate_count: int, safety_gap: float,
                           rotation_palette: np.ndarray) -> List[Tuple[float, float, float]]:
        """Generate candidate positions for next tree."""
        candidates = []
        
        if len(placed_polygons) == 0:
            # First tree: place at origin with various rotations
            for deg in rotation_palette:
                candidates.append((0.0, 0.0, deg))
            return candidates
        
        # Get bounding box of current layout
        all_points = np.vstack(placed_polygons)
        xmin, xmax = all_points[:, 0].min(), all_points[:, 0].max()
        ymin, ymax = all_points[:, 1].min(), all_points[:, 1].max()
        
        # Generate candidates around boundary
        tree_width = 0.7  # Approximate tree width
        tree_height = 1.0  # Approximate tree height
        
        # Candidates on edges of bounding box
        n_edge = max(3, candidate_count // 4)
        
        for deg in rotation_palette[:min(4, len(rotation_palette))]:
            # Left edge
            for i in range(n_edge):
                y = ymin + (ymax - ymin) * i / n_edge
                candidates.append((xmin - tree_width - safety_gap, y, deg))
            
            # Right edge
            for i in range(n_edge):
                y = ymin + (ymax - ymin) * i / n_edge
                candidates.append((xmax + tree_width + safety_gap, y, deg))
            
            # Bottom edge
            for i in range(n_edge):
                x = xmin + (xmax - xmin) * i / n_edge
                candidates.append((x, ymin - tree_height - safety_gap, deg))
            
            # Top edge
            for i in range(n_edge):
                x = xmin + (xmax - xmin) * i / n_edge
                candidates.append((x, ymax + tree_height + safety_gap, deg))
        
        # Candidates near existing trees (contact candidates)
        for idx, poly in enumerate(placed_polygons):
            pos = placed_positions[idx]
            for deg in rotation_palette[:min(4, len(rotation_palette))]:
                for angle in [0, 90, 180, 270]:
                    rad = math.radians(angle)
                    dx = math.cos(rad) * (tree_width + safety_gap)
                    dy = math.sin(rad) * (tree_height + safety_gap)
                    candidates.append((pos[0] + dx, pos[1] + dy, deg))
        
        return candidates[:candidate_count * len(rotation_palette)]
    
    def is_feasible(self, candidate_poly: np.ndarray, 
                    placed_polygons: List[np.ndarray],
                    eps: float = 1e-9) -> bool:
        """Check if candidate doesn't overlap with placed trees."""
        if len(placed_polygons) == 0:
            return True
        
        # Quick AABB check
        c_aabb = aabb(candidate_poly)
        
        for poly in placed_polygons:
            p_aabb = aabb(poly)
            if not aabb_intersect(c_aabb, p_aabb):
                continue
            if self.backend.sat_collision(candidate_poly, poly, eps):
                return False
        
        return True
    
    def evaluate_candidate(self, candidate: Tuple[float, float, float],
                          placed_polygons: List[np.ndarray],
                          safety_gap: float) -> Optional[float]:
        """Evaluate candidate position. Returns bounding square if feasible, None otherwise."""
        x, y, deg = candidate
        poly = transform_tree(x, y, deg)
        
        if not self.is_feasible(poly, placed_polygons, -safety_gap):
            return None
        
        if len(placed_polygons) == 0:
            return get_bounding_square_side([poly])
        
        return get_bounding_square_side(placed_polygons + [poly])
    
    def place_trees_greedy(self, n: int, hparams: Dict[str, Any]) -> Optional[np.ndarray]:
        """Place n trees using greedy construction."""
        rotation_palette = self.generate_rotation_palette(hparams['rotation_steps'])
        
        placed_polygons = []
        placed_positions = []
        
        for tree_idx in range(n):
            candidates = self.generate_candidates(
                placed_polygons, 
                np.array(placed_positions) if placed_positions else np.empty((0, 3)),
                hparams['candidate_count'],
                hparams['safety_gap'],
                rotation_palette
            )
            
            best_score = float('inf')
            best_candidate = None
            best_poly = None
            
            for candidate in candidates:
                score = self.evaluate_candidate(candidate, placed_polygons, hparams['safety_gap'])
                if score is not None and score < best_score:
                    best_score = score
                    best_candidate = candidate
                    best_poly = transform_tree(*candidate)
            
            if best_candidate is None:
                # Try more rotations
                for deg in np.linspace(0, 360, 36):
                    for offset in [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
                        if placed_positions:
                            base_x = np.mean([p[0] for p in placed_positions])
                            base_y = np.mean([p[1] for p in placed_positions])
                        else:
                            base_x, base_y = 0, 0
                        
                        candidate = (base_x + offset[0], base_y + offset[1], deg)
                        score = self.evaluate_candidate(candidate, placed_polygons, hparams['safety_gap'])
                        if score is not None and score < best_score:
                            best_score = score
                            best_candidate = candidate
                            best_poly = transform_tree(*candidate)
                
                if best_candidate is None:
                    return None  # Failed to place
            
            placed_polygons.append(best_poly)
            placed_positions.append(best_candidate)
        
        return np.array(placed_positions)
    
    def tighten_layout(self, positions: np.ndarray, hparams: Dict[str, Any]) -> np.ndarray:
        """Try to shrink layout while maintaining feasibility."""
        n = len(positions)
        if n <= 1:
            return positions
        
        positions = positions.copy()
        
        for _ in range(hparams['tighten_iters']):
            # Compute centroid
            cx = positions[:, 0].mean()
            cy = positions[:, 1].mean()
            
            # Try shrinking toward centroid
            new_positions = positions.copy()
            new_positions[:, 0] = cx + (positions[:, 0] - cx) * hparams['shrink_factor']
            new_positions[:, 1] = cy + (positions[:, 1] - cy) * hparams['shrink_factor']
            
            # Check feasibility
            new_polygons = [transform_tree(p[0], p[1], p[2]) for p in new_positions]
            if not self.backend.has_any_collision(new_polygons, -hparams['safety_gap']):
                positions = new_positions
        
        return positions
    
    def run_trial(self, hparams: Dict[str, Any], seed: int, 
                  stage: int, time_budget_sec: float) -> Dict[str, Any]:
        """Run a trial with given hyperparameters."""
        import time
        
        self.rng = np.random.default_rng(seed)
        
        # Determine n range for stage
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
            
            # Try to place trees
            positions = self.place_trees_greedy(n, hparams)
            
            if positions is not None and hparams['tighten_iters'] > 0:
                positions = self.tighten_layout(positions, hparams)
            
            if positions is not None:
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

def run_arena_a(n: int, seed: int = 42, hparams: Dict[str, Any] = None) -> Optional[np.ndarray]:
    """Convenience function to run arena A for single n."""
    arena = ConstructiveArena(seed)
    if hparams is None:
        hparams = {
            'candidate_count': 16,
            'beam_width': 1,
            'rotation_steps': 8,
            'placement_order': 'random',
            'safety_gap': 0.01,
            'tighten_iters': 10,
            'shrink_factor': 0.99,
        }
    return arena.place_trees_greedy(n, hparams)

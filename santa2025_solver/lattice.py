"""
Santa 2025 Solver - Lattice/Template Initialization

Places trees on a regular lattice pattern, then refines.
Good for larger n where greedy becomes slow.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import math

from santa2025_solver.geometry_fast import (
    Layout, get_tree_vertices, compute_bounding_square_side,
    get_all_tree_vertices, get_all_aabbs
)
from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python


class LatticeSolver:
    """
    Lattice-based solver.
    
    Places trees on a hexagonal or square lattice pattern,
    which provides good packing for large n.
    """
    
    FAMILY_ID = 1
    FAMILY_NAME = "lattice"
    
    DEFAULT_HYPERPARAMS = {
        'lattice_type': 'hexagonal',  # 'hexagonal', 'square', 'brick'
        'spacing_factor': 0.6,        # Spacing between trees (multiplied by tree size)
        'rotation_palette': [0, 90, 180, 270],
        'rotation_pattern': 'alternating',  # 'alternating', 'random', 'fixed'
        'jitter_factor': 0.0,         # Random jitter to add (as fraction of spacing)
        'center_first': True,         # Place first tree at center
    }
    
    # Approximate tree dimensions
    TREE_WIDTH = 0.5   # Max width (foliage)
    TREE_HEIGHT = 0.625  # Total height (tip to trunk bottom)
    
    def __init__(self, hyperparams: Optional[Dict[str, Any]] = None):
        self.hyperparams = {**self.DEFAULT_HYPERPARAMS}
        if hyperparams:
            self.hyperparams.update(hyperparams)
    
    def _get_lattice_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lattice basis vectors."""
        lattice_type = self.hyperparams['lattice_type']
        spacing = self.hyperparams['spacing_factor'] * self.TREE_WIDTH
        
        if lattice_type == 'hexagonal':
            # Hexagonal lattice (optimal circle packing)
            v1 = np.array([spacing, 0.0])
            v2 = np.array([spacing * 0.5, spacing * math.sqrt(3) / 2])
        
        elif lattice_type == 'square':
            # Square lattice
            v1 = np.array([spacing, 0.0])
            v2 = np.array([0.0, spacing])
        
        elif lattice_type == 'brick':
            # Brick pattern (offset rows)
            v1 = np.array([spacing, 0.0])
            v2 = np.array([spacing * 0.5, spacing * 0.9])
        
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
        
        return v1, v2
    
    def _generate_lattice_positions(self, n: int) -> np.ndarray:
        """Generate n positions on a lattice."""
        v1, v2 = self._get_lattice_vectors()
        
        # Estimate grid size needed
        # For hexagonal lattice, area per point ≈ |v1 × v2|
        area_per_point = abs(v1[0] * v2[1] - v1[1] * v2[0])
        total_area = n * area_per_point
        grid_radius = math.sqrt(total_area / math.pi) * 1.2
        
        # Generate positions in concentric rings (spiral from center)
        positions = []
        max_rings = int(grid_radius / min(np.linalg.norm(v1), np.linalg.norm(v2))) + 5
        
        # Start at origin
        positions.append(np.array([0.0, 0.0]))
        
        # Generate in concentric shells
        for ring in range(1, max_rings):
            # Generate points at this distance from center
            for i1 in range(-ring, ring + 1):
                for i2 in range(-ring, ring + 1):
                    # Only include points on the boundary of this ring
                    if max(abs(i1), abs(i2)) != ring:
                        continue
                    
                    pos = i1 * v1 + i2 * v2
                    positions.append(pos)
                    
                    if len(positions) >= n:
                        break
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        
        # Sort by distance from origin (center trees first)
        positions = np.array(positions[:n])
        distances = np.linalg.norm(positions, axis=1)
        sorted_indices = np.argsort(distances)
        positions = positions[sorted_indices]
        
        return positions
    
    def _assign_rotations(self, n: int, positions: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Assign rotations to each position."""
        pattern = self.hyperparams['rotation_pattern']
        palette = self.hyperparams['rotation_palette']
        
        if pattern == 'random':
            return rng.choice(palette, size=n).astype(np.float64)
        
        elif pattern == 'fixed':
            return np.full(n, palette[0], dtype=np.float64)
        
        elif pattern == 'alternating':
            # Alternate rotations based on position parity
            rotations = np.zeros(n, dtype=np.float64)
            for i in range(n):
                # Use checkered pattern
                parity = int(positions[i, 0] * 10) + int(positions[i, 1] * 10)
                rotations[i] = palette[parity % len(palette)]
            return rotations
        
        else:
            return np.full(n, 90.0, dtype=np.float64)
    
    def _add_jitter(self, positions: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Add random jitter to positions."""
        jitter = self.hyperparams['jitter_factor']
        if jitter <= 0:
            return positions
        
        spacing = self.hyperparams['spacing_factor'] * self.TREE_WIDTH
        jitter_magnitude = jitter * spacing
        
        noise = rng.uniform(-jitter_magnitude, jitter_magnitude, size=positions.shape)
        return positions + noise
    
    def init_layout(self, n: int, rng: np.random.RandomState) -> Layout:
        """
        Initialize a layout using lattice placement.
        
        Args:
            n: Number of trees
            rng: Random state
        
        Returns:
            Layout with n trees placed
        """
        layout = Layout(n)
        
        # Generate lattice positions
        positions = self._generate_lattice_positions(n)
        
        # Add jitter if enabled
        positions = self._add_jitter(positions, rng)
        
        # Assign rotations
        rotations = self._assign_rotations(n, positions, rng)
        
        # Set all trees
        layout.positions = positions
        layout.rotations = rotations
        
        return layout
    
    def repair_collisions(self, layout: Layout, rng: np.random.RandomState, max_iters: int = 100) -> Layout:
        """
        Repair collisions by pushing trees apart.
        
        Simple repulsion-based repair.
        """
        from santa2025_solver.collision_backends.python_backend import PythonCollisionBackend
        
        n = layout.n
        if n <= 1:
            return layout
        
        vertices = get_all_tree_vertices(layout.positions, layout.rotations)
        aabbs = get_all_aabbs(layout.positions, layout.rotations)
        
        backend = PythonCollisionBackend(n)
        backend.initialize(vertices, aabbs)
        
        for iteration in range(max_iters):
            collisions = backend.get_all_collisions()
            if not collisions:
                break
            
            # For each collision, push trees apart
            for i, j in collisions:
                # Direction from j to i
                direction = layout.positions[i] - layout.positions[j]
                dist = np.linalg.norm(direction)
                
                if dist < 1e-6:
                    # Trees at same position, push randomly
                    direction = rng.randn(2)
                    dist = np.linalg.norm(direction)
                
                direction = direction / dist
                
                # Push both trees apart
                push_amount = 0.05
                layout.positions[i] += direction * push_amount
                layout.positions[j] -= direction * push_amount
            
            # Update backend
            layout.invalidate_cache()
            vertices = get_all_tree_vertices(layout.positions, layout.rotations)
            aabbs = get_all_aabbs(layout.positions, layout.rotations)
            backend.initialize(vertices, aabbs)
        
        return layout
    
    def improve(
        self,
        layout: Layout,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """Improve layout by repairing collisions."""
        return self.repair_collisions(layout, rng)
    
    def solve(
        self,
        n: int,
        budget: float,
        rng: np.random.RandomState,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Layout:
        """Full solve: init + improve."""
        if hyperparams:
            self.hyperparams.update(hyperparams)
        
        layout = self.init_layout(n, rng)
        layout = self.improve(layout, budget, rng, hyperparams)
        
        return layout


def solve_lattice(n: int, seed: int = 42, **kwargs) -> Layout:
    """Convenience function to solve with lattice."""
    rng = np.random.RandomState(seed)
    solver = LatticeSolver(kwargs)
    return solver.solve(n, 10.0, rng)


if __name__ == "__main__":
    # Test lattice solver
    import time
    
    for n in [10, 30, 50, 100]:
        print(f"\nSolving for n={n}...")
        start = time.perf_counter()
        
        layout = solve_lattice(n)
        
        elapsed = time.perf_counter() - start
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Bounding side: {s:.4f}")
        print(f"  Score (s^2): {s*s:.4f}")

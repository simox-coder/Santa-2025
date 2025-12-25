"""
Fast geometry operations for Santa 2025 solver.

This module provides optimized geometry operations using NumPy and Numba.
All operations use float64 for precision.
"""

import numpy as np
from typing import Tuple, Optional

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: create dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if callable(args[0]):
            return args[0]
        return decorator
    prange = range


# Official tree polygon vertices (15 vertices)
# Symmetric Christmas tree shape
TREE_VERTICES_BASE = np.array([
    [0.0, 0.5],       # Top point
    [-0.25, 0.25],    # Upper left branch
    [-0.125, 0.25],   # Inner upper left
    [-0.375, 0.0],    # Middle left branch
    [-0.1875, 0.0],   # Inner middle left
    [-0.5, -0.25],    # Lower left branch
    [-0.125, -0.25],  # Trunk left
    [-0.125, -0.5],   # Trunk bottom left
    [0.125, -0.5],    # Trunk bottom right
    [0.125, -0.25],   # Trunk right
    [0.5, -0.25],     # Lower right branch
    [0.1875, 0.0],    # Inner middle right
    [0.375, 0.0],     # Middle right branch
    [0.125, 0.25],    # Inner upper right
    [0.25, 0.25],     # Upper right branch
], dtype=np.float64)

NUM_VERTICES = len(TREE_VERTICES_BASE)


@njit(cache=True)
def rotate_vertices_fast(vertices: np.ndarray, deg: float) -> np.ndarray:
    """
    Rotate vertices by deg degrees around origin.
    
    Args:
        vertices: Nx2 array of (x, y) coordinates
        deg: Rotation angle in degrees
        
    Returns:
        Rotated vertices as Nx2 array
    """
    rad = deg * np.pi / 180.0
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    n = len(vertices)
    result = np.empty((n, 2), dtype=np.float64)
    
    for i in range(n):
        x = vertices[i, 0]
        y = vertices[i, 1]
        result[i, 0] = x * cos_a - y * sin_a
        result[i, 1] = x * sin_a + y * cos_a
    
    return result


@njit(cache=True)
def transform_tree_fast(x: float, y: float, deg: float, base_vertices: np.ndarray) -> np.ndarray:
    """
    Transform tree: rotate by deg, then translate by (x, y).
    
    Args:
        x, y: Translation
        deg: Rotation in degrees
        base_vertices: Base tree vertices
        
    Returns:
        Transformed vertices
    """
    rotated = rotate_vertices_fast(base_vertices, deg)
    
    for i in range(len(rotated)):
        rotated[i, 0] += x
        rotated[i, 1] += y
    
    return rotated


@njit(cache=True)
def compute_aabb(vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute axis-aligned bounding box.
    
    Args:
        vertices: Nx2 array
        
    Returns:
        (min_x, min_y, max_x, max_y)
    """
    min_x = vertices[0, 0]
    max_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_y = vertices[0, 1]
    
    for i in range(1, len(vertices)):
        x = vertices[i, 0]
        y = vertices[i, 1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    
    return min_x, min_y, max_x, max_y


@njit(cache=True)
def aabb_overlap(aabb1: Tuple[float, float, float, float], 
                 aabb2: Tuple[float, float, float, float]) -> bool:
    """
    Check if two AABBs overlap.
    
    Args:
        aabb1, aabb2: (min_x, min_y, max_x, max_y)
        
    Returns:
        True if AABBs overlap
    """
    if aabb1[2] < aabb2[0] or aabb2[2] < aabb1[0]:  # x check
        return False
    if aabb1[3] < aabb2[1] or aabb2[3] < aabb1[1]:  # y check
        return False
    return True


@njit(cache=True)
def compute_bounding_square_side_fast(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Compute bounding square side from arrays of all x and y coordinates.
    
    Args:
        xs: 1D array of all x coordinates
        ys: 1D array of all y coordinates
        
    Returns:
        Side length of bounding square
    """
    min_x = xs[0]
    max_x = xs[0]
    min_y = ys[0]
    max_y = ys[0]
    
    for i in range(1, len(xs)):
        if xs[i] < min_x:
            min_x = xs[i]
        if xs[i] > max_x:
            max_x = xs[i]
        if ys[i] < min_y:
            min_y = ys[i]
        if ys[i] > max_y:
            max_y = ys[i]
    
    width = max_x - min_x
    height = max_y - min_y
    
    return max(width, height)


def get_tree_vertices() -> np.ndarray:
    """Return a copy of the base tree vertices."""
    return TREE_VERTICES_BASE.copy()


class TreeLayout:
    """
    Represents a layout of n trees with positions and rotations.
    
    Maintains cached transformed vertices and AABBs for efficient operations.
    """
    
    def __init__(self, n: int):
        """
        Initialize layout for n trees.
        
        Args:
            n: Number of trees
        """
        self.n = n
        self.positions = np.zeros((n, 2), dtype=np.float64)  # x, y
        self.rotations = np.zeros(n, dtype=np.float64)       # degrees
        
        # Cache transformed vertices for all trees
        self._vertices_cache = np.zeros((n, NUM_VERTICES, 2), dtype=np.float64)
        self._aabbs = np.zeros((n, 4), dtype=np.float64)  # min_x, min_y, max_x, max_y
        self._valid_cache = np.zeros(n, dtype=np.bool_)
        
        self._base_vertices = TREE_VERTICES_BASE.copy()
    
    def set_tree(self, idx: int, x: float, y: float, deg: float):
        """Set position and rotation for a single tree."""
        self.positions[idx, 0] = x
        self.positions[idx, 1] = y
        self.rotations[idx] = deg
        self._valid_cache[idx] = False
    
    def get_tree(self, idx: int) -> Tuple[float, float, float]:
        """Get (x, y, deg) for a single tree."""
        return (self.positions[idx, 0], self.positions[idx, 1], self.rotations[idx])
    
    def update_cache(self, idx: int):
        """Update cached vertices and AABB for a single tree."""
        if not self._valid_cache[idx]:
            x = self.positions[idx, 0]
            y = self.positions[idx, 1]
            deg = self.rotations[idx]
            
            self._vertices_cache[idx] = transform_tree_fast(x, y, deg, self._base_vertices)
            aabb = compute_aabb(self._vertices_cache[idx])
            self._aabbs[idx] = aabb
            self._valid_cache[idx] = True
    
    def update_all_caches(self):
        """Update all cached data."""
        for i in range(self.n):
            self.update_cache(i)
    
    def get_vertices(self, idx: int) -> np.ndarray:
        """Get transformed vertices for a tree (updates cache if needed)."""
        self.update_cache(idx)
        return self._vertices_cache[idx]
    
    def get_aabb(self, idx: int) -> Tuple[float, float, float, float]:
        """Get AABB for a tree (updates cache if needed)."""
        self.update_cache(idx)
        return tuple(self._aabbs[idx])
    
    def get_all_vertices_flat(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all vertices as flat arrays for fast bounding square computation.
        
        Returns:
            (xs, ys) arrays of all vertex coordinates
        """
        self.update_all_caches()
        xs = self._vertices_cache[:, :, 0].flatten()
        ys = self._vertices_cache[:, :, 1].flatten()
        return xs, ys
    
    def compute_score(self) -> float:
        """Compute bounding square side for this layout."""
        xs, ys = self.get_all_vertices_flat()
        return compute_bounding_square_side_fast(xs, ys)
    
    def copy(self) -> 'TreeLayout':
        """Create a deep copy of this layout."""
        new_layout = TreeLayout(self.n)
        new_layout.positions[:] = self.positions
        new_layout.rotations[:] = self.rotations
        new_layout._vertices_cache[:] = self._vertices_cache
        new_layout._aabbs[:] = self._aabbs
        new_layout._valid_cache[:] = self._valid_cache
        return new_layout
    
    def to_list(self) -> list:
        """Convert to list of (x, y, deg) tuples."""
        return [(self.positions[i, 0], self.positions[i, 1], self.rotations[i]) 
                for i in range(self.n)]


# Pre-compute rotated versions of the tree at common angles
ROTATION_CACHE_ANGLES = [0.0, 90.0, 180.0, 270.0]
ROTATION_CACHE = {}

def _init_rotation_cache():
    """Initialize pre-computed rotated vertices."""
    for angle in ROTATION_CACHE_ANGLES:
        ROTATION_CACHE[angle] = rotate_vertices_fast(TREE_VERTICES_BASE, angle)

_init_rotation_cache()


def get_prerotated_vertices(deg: float) -> Optional[np.ndarray]:
    """
    Get pre-computed rotated vertices if available.
    
    Args:
        deg: Rotation angle
        
    Returns:
        Pre-computed vertices or None if not cached
    """
    # Normalize angle
    deg = deg % 360.0
    if deg in ROTATION_CACHE:
        return ROTATION_CACHE[deg].copy()
    return None

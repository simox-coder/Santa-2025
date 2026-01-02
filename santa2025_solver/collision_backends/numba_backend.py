"""
Santa 2025 Solver - Numba Collision Backend

JIT-compiled collision detection using Numba.
This is the primary performance target.
"""

import numpy as np
from typing import Tuple, List, Optional

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Provide dummy decorator for graceful degradation
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range


# ============================================================================
# JIT-COMPILED FUNCTIONS
# ============================================================================

@njit(cache=True)
def project_polygon_onto_axis_numba(vertices: np.ndarray, axis_x: float, axis_y: float) -> Tuple[float, float]:
    """Project polygon vertices onto an axis and return min/max projections."""
    n = len(vertices)
    proj = vertices[0, 0] * axis_x + vertices[0, 1] * axis_y
    min_proj = proj
    max_proj = proj
    
    for i in range(1, n):
        proj = vertices[i, 0] * axis_x + vertices[i, 1] * axis_y
        if proj < min_proj:
            min_proj = proj
        if proj > max_proj:
            max_proj = proj
    
    return min_proj, max_proj


@njit(cache=True)
def check_convex_overlap_numba(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """Check if two convex polygons overlap using SAT."""
    n1 = len(vertices1)
    n2 = len(vertices2)
    
    # Check axes from polygon 1
    for i in range(n1):
        # Edge from vertex i to vertex (i+1)
        next_i = (i + 1) % n1
        edge_x = vertices1[next_i, 0] - vertices1[i, 0]
        edge_y = vertices1[next_i, 1] - vertices1[i, 1]
        
        # Perpendicular axis (rotate 90 degrees)
        axis_x = -edge_y
        axis_y = edge_x
        
        # Normalize
        length = np.sqrt(axis_x * axis_x + axis_y * axis_y)
        if length > 1e-10:
            axis_x /= length
            axis_y /= length
        
        # Project both polygons
        min1, max1 = project_polygon_onto_axis_numba(vertices1, axis_x, axis_y)
        min2, max2 = project_polygon_onto_axis_numba(vertices2, axis_x, axis_y)
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return False
    
    # Check axes from polygon 2
    for i in range(n2):
        next_i = (i + 1) % n2
        edge_x = vertices2[next_i, 0] - vertices2[i, 0]
        edge_y = vertices2[next_i, 1] - vertices2[i, 1]
        
        axis_x = -edge_y
        axis_y = edge_x
        
        length = np.sqrt(axis_x * axis_x + axis_y * axis_y)
        if length > 1e-10:
            axis_x /= length
            axis_y /= length
        
        min1, max1 = project_polygon_onto_axis_numba(vertices1, axis_x, axis_y)
        min2, max2 = project_polygon_onto_axis_numba(vertices2, axis_x, axis_y)
        
        if max1 <= min2 or max2 <= min1:
            return False
    
    return True


@njit(cache=True)
def check_tree_overlap_numba(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """
    Check if two tree polygons overlap.
    
    Tree vertices layout:
    0: tip (0, 0.5)
    1: left foliage (-0.25, 0)
    2: trunk top left (-0.0625, 0)
    3: trunk bottom left (-0.0625, -0.125)
    4: trunk bottom right (0.0625, -0.125)
    5: trunk top right (0.0625, 0)
    6: right foliage (0.25, 0)
    """
    # Extract triangle (foliage) for tree 1
    triangle1 = np.empty((3, 2), dtype=np.float64)
    triangle1[0] = vertices1[0]  # tip
    triangle1[1] = vertices1[1]  # left foliage
    triangle1[2] = vertices1[6]  # right foliage
    
    # Extract trunk (rectangle) for tree 1
    trunk1 = np.empty((4, 2), dtype=np.float64)
    trunk1[0] = vertices1[2]  # trunk top left
    trunk1[1] = vertices1[3]  # trunk bottom left
    trunk1[2] = vertices1[4]  # trunk bottom right
    trunk1[3] = vertices1[5]  # trunk top right
    
    # Extract triangle for tree 2
    triangle2 = np.empty((3, 2), dtype=np.float64)
    triangle2[0] = vertices2[0]
    triangle2[1] = vertices2[1]
    triangle2[2] = vertices2[6]
    
    # Extract trunk for tree 2
    trunk2 = np.empty((4, 2), dtype=np.float64)
    trunk2[0] = vertices2[2]
    trunk2[1] = vertices2[3]
    trunk2[2] = vertices2[4]
    trunk2[3] = vertices2[5]
    
    # Check all 4 pairs of convex parts
    if check_convex_overlap_numba(triangle1, triangle2):
        return True
    if check_convex_overlap_numba(triangle1, trunk2):
        return True
    if check_convex_overlap_numba(trunk1, triangle2):
        return True
    if check_convex_overlap_numba(trunk1, trunk2):
        return True
    
    return False


@njit(cache=True)
def check_aabb_overlap_numba(aabb1: np.ndarray, aabb2: np.ndarray) -> bool:
    """Check if two AABBs overlap."""
    # aabb = [min_x, min_y, max_x, max_y]
    if aabb1[2] <= aabb2[0] or aabb2[2] <= aabb1[0]:
        return False
    if aabb1[3] <= aabb2[1] or aabb2[3] <= aabb1[1]:
        return False
    return True


@njit(cache=True, parallel=True)
def count_all_collisions_numba(vertices: np.ndarray, aabbs: np.ndarray) -> int:
    """Count all colliding pairs (parallelized)."""
    n = len(vertices)
    count = 0
    
    for i in prange(n):
        for j in range(i + 1, n):
            # Quick AABB check
            if check_aabb_overlap_numba(aabbs[i], aabbs[j]):
                # Detailed polygon check
                if check_tree_overlap_numba(vertices[i], vertices[j]):
                    count += 1
    
    return count


@njit(cache=True)
def has_any_collision_numba(vertices: np.ndarray, aabbs: np.ndarray) -> bool:
    """Check if any trees collide (early exit)."""
    n = len(vertices)
    
    for i in range(n):
        for j in range(i + 1, n):
            if check_aabb_overlap_numba(aabbs[i], aabbs[j]):
                if check_tree_overlap_numba(vertices[i], vertices[j]):
                    return True
    
    return False


@njit(cache=True)
def check_tree_collisions_numba(tree_idx: int, vertices: np.ndarray, aabbs: np.ndarray) -> np.ndarray:
    """Get indices of trees that collide with the given tree."""
    n = len(vertices)
    collisions = np.zeros(n, dtype=np.int32)
    count = 0
    
    for i in range(n):
        if i == tree_idx:
            continue
        if check_aabb_overlap_numba(aabbs[tree_idx], aabbs[i]):
            if check_tree_overlap_numba(vertices[tree_idx], vertices[i]):
                collisions[count] = i
                count += 1
    
    return collisions[:count]


def check_polygon_overlap_numba(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """Public interface for polygon overlap check."""
    return check_tree_overlap_numba(vertices1, vertices2)


class NumbaCollisionBackend:
    """
    Numba JIT-compiled collision detection backend.
    """
    
    def __init__(self, n_trees: int, cell_size: float = 0.5):
        self.n_trees = n_trees
        self.vertices: Optional[np.ndarray] = None  # Shape (n, 7, 2)
        self.aabbs: Optional[np.ndarray] = None  # Shape (n, 4)
        
        # Warm up JIT compilation
        self._warmup()
    
    def _warmup(self):
        """Warm up JIT compilation."""
        if not NUMBA_AVAILABLE:
            return
        
        # Create dummy data for warmup
        dummy_vertices = np.zeros((2, 7, 2), dtype=np.float64)
        dummy_aabbs = np.zeros((2, 4), dtype=np.float64)
        
        # Trigger compilation
        try:
            check_tree_overlap_numba(dummy_vertices[0], dummy_vertices[1])
            has_any_collision_numba(dummy_vertices, dummy_aabbs)
        except:
            pass  # Ignore errors during warmup
    
    def initialize(self, vertices: np.ndarray, aabbs: np.ndarray):
        """
        Initialize backend with tree data.
        
        Args:
            vertices: Shape (n, 7, 2) array of tree vertices
            aabbs: Shape (n, 4) array of AABBs
        """
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.aabbs = np.ascontiguousarray(aabbs, dtype=np.float64)
    
    def update_tree(self, idx: int, vertices: np.ndarray, aabb: np.ndarray):
        """Update a single tree's data."""
        self.vertices[idx] = vertices
        self.aabbs[idx] = aabb
    
    def check_overlap(self, idx1: int, idx2: int) -> bool:
        """Check if two trees overlap."""
        if not check_aabb_overlap_numba(self.aabbs[idx1], self.aabbs[idx2]):
            return False
        return check_tree_overlap_numba(self.vertices[idx1], self.vertices[idx2])
    
    def check_tree_collisions(self, idx: int) -> List[int]:
        """Get list of trees that collide with the given tree."""
        result = check_tree_collisions_numba(idx, self.vertices, self.aabbs)
        return list(result)
    
    def has_any_collision(self) -> bool:
        """Check if any trees collide."""
        return has_any_collision_numba(self.vertices, self.aabbs)
    
    def count_collisions(self) -> int:
        """Count total number of colliding pairs."""
        return count_all_collisions_numba(self.vertices, self.aabbs)
    
    def get_all_collisions(self) -> List[Tuple[int, int]]:
        """Get all colliding pairs."""
        collisions = []
        n = len(self.vertices)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.check_overlap(i, j):
                    collisions.append((i, j))
        
        return collisions

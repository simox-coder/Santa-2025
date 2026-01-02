"""
Santa 2025 Solver - C++ Collision Backend

Optional high-performance C++ backend using pybind11.
Falls back gracefully if not available.
"""

import numpy as np
from typing import Tuple, List, Optional
import os
import sys

# Try to import the C++ module
CPP_MODULE_AVAILABLE = False
cpp_collision = None

try:
    # Try to find and import the C++ module
    module_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'build')
    if os.path.exists(module_path):
        sys.path.insert(0, module_path)
    
    import santa2025_collision_cpp as cpp_collision
    CPP_MODULE_AVAILABLE = True
except ImportError:
    pass


def check_polygon_overlap_cpp(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """
    Check polygon overlap using C++ backend.
    
    Falls back to Python if C++ is not available.
    """
    if not CPP_MODULE_AVAILABLE:
        from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python
        return check_tree_overlap_python(vertices1, vertices2)
    
    return cpp_collision.check_tree_overlap(
        np.ascontiguousarray(vertices1, dtype=np.float64),
        np.ascontiguousarray(vertices2, dtype=np.float64)
    )


class CppCollisionBackend:
    """
    C++ collision detection backend.
    
    This is a placeholder that falls back to Python backend if C++ is not built.
    """
    
    def __init__(self, n_trees: int, cell_size: float = 0.5):
        self.n_trees = n_trees
        self.vertices: Optional[np.ndarray] = None
        self.aabbs: Optional[np.ndarray] = None
        
        if not CPP_MODULE_AVAILABLE:
            # Fall back to Python backend
            from santa2025_solver.collision_backends.python_backend import PythonCollisionBackend
            self._fallback = PythonCollisionBackend(n_trees, cell_size)
        else:
            self._fallback = None
    
    def initialize(self, vertices: np.ndarray, aabbs: np.ndarray):
        """Initialize backend with tree data."""
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.aabbs = np.ascontiguousarray(aabbs, dtype=np.float64)
        
        if self._fallback:
            self._fallback.initialize(vertices, aabbs)
    
    def update_tree(self, idx: int, vertices: np.ndarray, aabb: np.ndarray):
        """Update a single tree's data."""
        self.vertices[idx] = vertices
        self.aabbs[idx] = aabb
        
        if self._fallback:
            self._fallback.update_tree(idx, vertices, aabb)
    
    def check_overlap(self, idx1: int, idx2: int) -> bool:
        """Check if two trees overlap."""
        if self._fallback:
            return self._fallback.check_overlap(idx1, idx2)
        
        # AABB check first
        aabb1, aabb2 = self.aabbs[idx1], self.aabbs[idx2]
        if aabb1[2] <= aabb2[0] or aabb2[2] <= aabb1[0]:
            return False
        if aabb1[3] <= aabb2[1] or aabb2[3] <= aabb1[1]:
            return False
        
        return cpp_collision.check_tree_overlap(self.vertices[idx1], self.vertices[idx2])
    
    def check_tree_collisions(self, idx: int) -> List[int]:
        """Get list of trees that collide with the given tree."""
        if self._fallback:
            return self._fallback.check_tree_collisions(idx)
        
        collisions = []
        for i in range(self.n_trees):
            if i != idx and self.check_overlap(idx, i):
                collisions.append(i)
        return collisions
    
    def has_any_collision(self) -> bool:
        """Check if any trees collide."""
        if self._fallback:
            return self._fallback.has_any_collision()
        
        for i in range(self.n_trees):
            for j in range(i + 1, self.n_trees):
                if self.check_overlap(i, j):
                    return True
        return False
    
    def count_collisions(self) -> int:
        """Count total number of colliding pairs."""
        if self._fallback:
            return self._fallback.count_collisions()
        
        count = 0
        for i in range(self.n_trees):
            for j in range(i + 1, self.n_trees):
                if self.check_overlap(i, j):
                    count += 1
        return count
    
    def get_all_collisions(self) -> List[Tuple[int, int]]:
        """Get all colliding pairs."""
        if self._fallback:
            return self._fallback.get_all_collisions()
        
        collisions = []
        for i in range(self.n_trees):
            for j in range(i + 1, self.n_trees):
                if self.check_overlap(i, j):
                    collisions.append((i, j))
        return collisions

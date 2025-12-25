"""
Numba-accelerated collision detection backend.
Falls back to pure Python if Numba is not available.
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

@jit(nopython=True, cache=True)
def _project_polygon_numba(vertices: np.ndarray, axis_x: float, axis_y: float) -> Tuple[float, float]:
    """Project polygon onto axis."""
    n = len(vertices)
    min_proj = vertices[0, 0] * axis_x + vertices[0, 1] * axis_y
    max_proj = min_proj
    for i in range(1, n):
        proj = vertices[i, 0] * axis_x + vertices[i, 1] * axis_y
        if proj < min_proj:
            min_proj = proj
        if proj > max_proj:
            max_proj = proj
    return min_proj, max_proj

@jit(nopython=True, cache=True)
def _sat_collision_numba(poly_a: np.ndarray, poly_b: np.ndarray, eps: float = 1e-9) -> bool:
    """SAT collision check with Numba JIT."""
    n_a = len(poly_a)
    n_b = len(poly_b)
    
    # Test axes from polygon A
    for i in range(n_a):
        j = (i + 1) % n_a
        edge_x = poly_a[j, 0] - poly_a[i, 0]
        edge_y = poly_a[j, 1] - poly_a[i, 1]
        
        # Normal (perpendicular)
        normal_x = -edge_y
        normal_y = edge_x
        length = np.sqrt(normal_x * normal_x + normal_y * normal_y)
        if length > 1e-12:
            normal_x /= length
            normal_y /= length
        
        a_min, a_max = _project_polygon_numba(poly_a, normal_x, normal_y)
        b_min, b_max = _project_polygon_numba(poly_b, normal_x, normal_y)
        
        if a_max + eps < b_min or b_max + eps < a_min:
            return False
    
    # Test axes from polygon B
    for i in range(n_b):
        j = (i + 1) % n_b
        edge_x = poly_b[j, 0] - poly_b[i, 0]
        edge_y = poly_b[j, 1] - poly_b[i, 1]
        
        normal_x = -edge_y
        normal_y = edge_x
        length = np.sqrt(normal_x * normal_x + normal_y * normal_y)
        if length > 1e-12:
            normal_x /= length
            normal_y /= length
        
        a_min, a_max = _project_polygon_numba(poly_a, normal_x, normal_y)
        b_min, b_max = _project_polygon_numba(poly_b, normal_x, normal_y)
        
        if a_max + eps < b_min or b_max + eps < a_min:
            return False
    
    return True

def sat_collision(poly_a: np.ndarray, poly_b: np.ndarray, eps: float = 1e-9) -> bool:
    """Check if two convex polygons collide using SAT."""
    return _sat_collision_numba(poly_a, poly_b, eps)

@jit(nopython=True, cache=True)
def _check_pairs_numba(polys: np.ndarray, n_trees: int, n_verts: int, 
                       aabbs: np.ndarray, eps: float) -> List[Tuple[int, int]]:
    """Check all pairs with Numba."""
    collisions = []
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            # AABB check
            if (aabbs[i, 1] + eps < aabbs[j, 0] or 
                aabbs[j, 1] + eps < aabbs[i, 0] or
                aabbs[i, 3] + eps < aabbs[j, 2] or 
                aabbs[j, 3] + eps < aabbs[i, 2]):
                continue
            
            # SAT check
            poly_i = polys[i * n_verts:(i + 1) * n_verts]
            poly_j = polys[j * n_verts:(j + 1) * n_verts]
            if _sat_collision_numba(poly_i, poly_j, eps):
                collisions.append((i, j))
    return collisions

def check_all_pairs(polygons: List[np.ndarray], eps: float = 1e-9) -> List[Tuple[int, int]]:
    """Check all pairs of polygons for collisions."""
    n = len(polygons)
    if n < 2:
        return []
    
    n_verts = len(polygons[0])
    
    # Stack all polygons
    polys = np.vstack(polygons)
    
    # Compute AABBs
    aabbs = np.empty((n, 4), dtype=np.float64)
    for i, poly in enumerate(polygons):
        aabbs[i, 0] = poly[:, 0].min()
        aabbs[i, 1] = poly[:, 0].max()
        aabbs[i, 2] = poly[:, 1].min()
        aabbs[i, 3] = poly[:, 1].max()
    
    collisions = []
    for i in range(n):
        for j in range(i + 1, n):
            # AABB check
            if (aabbs[i, 1] + eps < aabbs[j, 0] or 
                aabbs[j, 1] + eps < aabbs[i, 0] or
                aabbs[i, 3] + eps < aabbs[j, 2] or 
                aabbs[j, 3] + eps < aabbs[i, 2]):
                continue
            
            if _sat_collision_numba(polygons[i], polygons[j], eps):
                collisions.append((i, j))
    
    return collisions

def has_any_collision(polygons: List[np.ndarray], eps: float = 1e-9) -> bool:
    """Check if any pair collides."""
    n = len(polygons)
    if n < 2:
        return False
    
    # Compute AABBs
    aabbs = np.empty((n, 4), dtype=np.float64)
    for i, poly in enumerate(polygons):
        aabbs[i, 0] = poly[:, 0].min()
        aabbs[i, 1] = poly[:, 0].max()
        aabbs[i, 2] = poly[:, 1].min()
        aabbs[i, 3] = poly[:, 1].max()
    
    for i in range(n):
        for j in range(i + 1, n):
            if (aabbs[i, 1] + eps < aabbs[j, 0] or 
                aabbs[j, 1] + eps < aabbs[i, 0] or
                aabbs[i, 3] + eps < aabbs[j, 2] or 
                aabbs[j, 3] + eps < aabbs[i, 2]):
                continue
            
            if _sat_collision_numba(polygons[i], polygons[j], eps):
                return True
    return False

def sat_collision_mtv(poly_a: np.ndarray, poly_b: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Get MTV if colliding."""
    n_a = len(poly_a)
    n_b = len(poly_b)
    
    min_overlap = float('inf')
    mtv_axis = None
    
    # Collect all normals
    normals = []
    for i in range(n_a):
        j = (i + 1) % n_a
        edge_x = poly_a[j, 0] - poly_a[i, 0]
        edge_y = poly_a[j, 1] - poly_a[i, 1]
        length = np.sqrt(edge_x ** 2 + edge_y ** 2)
        if length > 1e-12:
            normals.append(np.array([-edge_y / length, edge_x / length]))
    
    for i in range(n_b):
        j = (i + 1) % n_b
        edge_x = poly_b[j, 0] - poly_b[i, 0]
        edge_y = poly_b[j, 1] - poly_b[i, 1]
        length = np.sqrt(edge_x ** 2 + edge_y ** 2)
        if length > 1e-12:
            normals.append(np.array([-edge_y / length, edge_x / length]))
    
    for normal in normals:
        a_min, a_max = _project_polygon_numba(poly_a, normal[0], normal[1])
        b_min, b_max = _project_polygon_numba(poly_b, normal[0], normal[1])
        
        if a_max < b_min or b_max < a_min:
            return None
        
        overlap = min(a_max - b_min, b_max - a_min)
        if overlap < min_overlap:
            min_overlap = overlap
            mtv_axis = normal.copy()
            center_a = poly_a.mean(axis=0)
            center_b = poly_b.mean(axis=0)
            direction = center_b - center_a
            if np.dot(direction, mtv_axis) < 0:
                mtv_axis = -mtv_axis
    
    return (mtv_axis, min_overlap) if mtv_axis is not None else None

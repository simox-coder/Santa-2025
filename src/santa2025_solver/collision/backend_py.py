"""
Pure Python collision detection backend using SAT (Separating Axis Theorem).
"""

import numpy as np
from typing import List, Tuple, Optional

def get_edges(vertices: np.ndarray) -> np.ndarray:
    """Get edge vectors of polygon."""
    n = len(vertices)
    edges = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        edges[i, 0] = vertices[j, 0] - vertices[i, 0]
        edges[i, 1] = vertices[j, 1] - vertices[i, 1]
    return edges

def get_normals(edges: np.ndarray) -> np.ndarray:
    """Get perpendicular normals to edges."""
    normals = np.empty_like(edges)
    normals[:, 0] = -edges[:, 1]
    normals[:, 1] = edges[:, 0]
    # Normalize
    lengths = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2)
    lengths[lengths < 1e-12] = 1.0
    normals[:, 0] /= lengths
    normals[:, 1] /= lengths
    return normals

def project_polygon(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Project polygon onto axis, return (min, max)."""
    dots = vertices[:, 0] * axis[0] + vertices[:, 1] * axis[1]
    return dots.min(), dots.max()

def intervals_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    """Check if two intervals overlap."""
    return not (a_max < b_min or b_max < a_min)

def sat_collision(poly_a: np.ndarray, poly_b: np.ndarray, eps: float = 1e-9) -> bool:
    """
    Check if two convex polygons collide using SAT.
    
    Args:
        poly_a: First polygon vertices (Nx2)
        poly_b: Second polygon vertices (Mx2)
        eps: Epsilon for numerical stability (negative = gap required)
    
    Returns:
        True if polygons overlap or touch
    """
    # Get all edges from both polygons
    edges_a = get_edges(poly_a)
    edges_b = get_edges(poly_b)
    
    # Get perpendicular axes
    normals_a = get_normals(edges_a)
    normals_b = get_normals(edges_b)
    
    # Test all axes from polygon A
    for normal in normals_a:
        a_min, a_max = project_polygon(poly_a, normal)
        b_min, b_max = project_polygon(poly_b, normal)
        if a_max + eps < b_min or b_max + eps < a_min:
            return False  # Separating axis found
    
    # Test all axes from polygon B
    for normal in normals_b:
        a_min, a_max = project_polygon(poly_a, normal)
        b_min, b_max = project_polygon(poly_b, normal)
        if a_max + eps < b_min or b_max + eps < a_min:
            return False  # Separating axis found
    
    return True  # No separating axis found, polygons overlap

def sat_collision_mtv(poly_a: np.ndarray, poly_b: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """
    Check collision and return Minimum Translation Vector (MTV) if colliding.
    
    Returns:
        (direction, magnitude) tuple or None if not colliding
    """
    edges_a = get_edges(poly_a)
    edges_b = get_edges(poly_b)
    normals_a = get_normals(edges_a)
    normals_b = get_normals(edges_b)
    
    min_overlap = float('inf')
    mtv_axis = None
    
    for normal in np.vstack([normals_a, normals_b]):
        a_min, a_max = project_polygon(poly_a, normal)
        b_min, b_max = project_polygon(poly_b, normal)
        
        # Check overlap
        if a_max < b_min or b_max < a_min:
            return None  # No collision
        
        # Calculate overlap
        overlap = min(a_max - b_min, b_max - a_min)
        if overlap < min_overlap:
            min_overlap = overlap
            mtv_axis = normal.copy()
            # Ensure MTV points from A to B
            center_a = poly_a.mean(axis=0)
            center_b = poly_b.mean(axis=0)
            direction = center_b - center_a
            if np.dot(direction, mtv_axis) < 0:
                mtv_axis = -mtv_axis
    
    return (mtv_axis, min_overlap) if mtv_axis is not None else None

def check_all_pairs(polygons: List[np.ndarray], eps: float = 1e-9) -> List[Tuple[int, int]]:
    """
    Check all pairs of polygons for collisions.
    
    Returns:
        List of (i, j) pairs that are colliding
    """
    n = len(polygons)
    collisions = []
    
    # Compute AABBs for broad-phase
    aabbs = []
    for poly in polygons:
        xmin, xmax = poly[:, 0].min(), poly[:, 0].max()
        ymin, ymax = poly[:, 1].min(), poly[:, 1].max()
        aabbs.append((xmin, xmax, ymin, ymax))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Broad-phase: AABB check
            a = aabbs[i]
            b = aabbs[j]
            if a[1] + eps < b[0] or b[1] + eps < a[0] or a[3] + eps < b[2] or b[3] + eps < a[2]:
                continue
            
            # Narrow-phase: SAT check
            if sat_collision(polygons[i], polygons[j], eps):
                collisions.append((i, j))
    
    return collisions

def has_any_collision(polygons: List[np.ndarray], eps: float = 1e-9) -> bool:
    """Check if any pair of polygons collides."""
    n = len(polygons)
    
    # Compute AABBs
    aabbs = []
    for poly in polygons:
        xmin, xmax = poly[:, 0].min(), poly[:, 0].max()
        ymin, ymax = poly[:, 1].min(), poly[:, 1].max()
        aabbs.append((xmin, xmax, ymin, ymax))
    
    for i in range(n):
        for j in range(i + 1, n):
            a = aabbs[i]
            b = aabbs[j]
            if a[1] + eps < b[0] or b[1] + eps < a[0] or a[3] + eps < b[2] or b[3] + eps < a[2]:
                continue
            if sat_collision(polygons[i], polygons[j], eps):
                return True
    return False

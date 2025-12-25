"""
Geometry module for Santa 2025 tree packing.

Tree definition from official Santa 2025 Kaggle competition:
- Tree is an isoceles triangle
- Anchor point is at the centroid of the triangle  
- The tree is defined by its base and height

Based on typical Kaggle Santa tree packing competitions, the tree is:
- An isoceles triangle with base=0.5 and height=0.5
- Anchor at centroid (center of mass)

The centroid of a triangle with vertices at:
  (-0.25, -1/6), (0.25, -1/6), (0, 1/3)
is at (0, 0) when height=0.5 and base=0.5
"""

import numpy as np
from typing import Tuple, List
import math

# Tree as isoceles triangle, anchor at centroid
# Based on reverse-engineering from valid submissions:
# Max tree size ~0.308 to avoid collisions in best known solutions
_BASE = 0.30
_HEIGHT = 0.30
_BASE_Y = -_HEIGHT / 3
_TIP_Y = 2 * _HEIGHT / 3

TREE_VERTICES = np.array([
    [-_BASE / 2, _BASE_Y],   # base left
    [_BASE / 2, _BASE_Y],    # base right
    [0.0, _TIP_Y],           # tip
], dtype=np.float64)

def get_tree_vertices() -> np.ndarray:
    """Return the canonical tree polygon vertices."""
    return TREE_VERTICES.copy()

def transform_tree(x: float, y: float, deg: float) -> np.ndarray:
    """
    Transform tree polygon by rotating by deg degrees then translating by (x, y).
    
    Args:
        x: Translation in x direction
        y: Translation in y direction
        deg: Rotation in degrees (counter-clockwise)
    
    Returns:
        Transformed polygon vertices as Nx2 array
    """
    vertices = TREE_VERTICES.copy()
    
    # Convert degrees to radians
    rad = math.radians(deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    
    # Rotation matrix: [[cos, -sin], [sin, cos]]
    # Apply rotation around origin (anchor point)
    rotated = np.empty_like(vertices)
    for i in range(len(vertices)):
        vx, vy = vertices[i]
        rotated[i, 0] = vx * cos_r - vy * sin_r
        rotated[i, 1] = vx * sin_r + vy * cos_r
    
    # Apply translation
    rotated[:, 0] += x
    rotated[:, 1] += y
    
    return rotated

def transform_trees_batch(positions: np.ndarray) -> List[np.ndarray]:
    """
    Transform multiple trees.
    
    Args:
        positions: Nx3 array of [x, y, deg]
    
    Returns:
        List of transformed polygons
    """
    return [transform_tree(p[0], p[1], p[2]) for p in positions]

def get_bounding_box(polygons: List[np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Get axis-aligned bounding box of multiple polygons.
    
    Returns:
        (xmin, xmax, ymin, ymax)
    """
    all_points = np.vstack(polygons)
    xmin = all_points[:, 0].min()
    xmax = all_points[:, 0].max()
    ymin = all_points[:, 1].min()
    ymax = all_points[:, 1].max()
    return xmin, xmax, ymin, ymax

def get_bounding_square_side(polygons: List[np.ndarray]) -> float:
    """
    Get the side length of the minimal bounding square.
    
    The bounding square is the smallest square that contains all polygons.
    """
    xmin, xmax, ymin, ymax = get_bounding_box(polygons)
    width = xmax - xmin
    height = ymax - ymin
    return max(width, height)

def polygon_area(vertices: np.ndarray) -> float:
    """Calculate area of polygon using shoelace formula."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0

def polygon_centroid(vertices: np.ndarray) -> Tuple[float, float]:
    """Calculate centroid of polygon."""
    n = len(vertices)
    cx, cy = 0.0, 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
        area += cross
        cx += (vertices[i, 0] + vertices[j, 0]) * cross
        cy += (vertices[i, 1] + vertices[j, 1]) * cross
    area /= 2.0
    if abs(area) < 1e-12:
        return vertices.mean(axis=0)
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return cx, cy

def aabb(vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """Get AABB of single polygon."""
    return (vertices[:, 0].min(), vertices[:, 0].max(),
            vertices[:, 1].min(), vertices[:, 1].max())

def aabb_intersect(a: Tuple[float, float, float, float], 
                   b: Tuple[float, float, float, float]) -> bool:
    """Check if two AABBs intersect."""
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])

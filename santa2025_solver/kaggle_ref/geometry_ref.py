"""
Reference geometry implementation for Santa 2025 Christmas Tree.

This module defines the official tree polygon shape, anchor point, and rotation
transformation as per the Kaggle competition rules.

The tree shape is defined by specific vertices forming a Christmas tree silhouette.
"""

import numpy as np
from typing import Tuple, List

# Official tree polygon vertices (from Kaggle competition)
# The tree is defined in local coordinates with anchor at origin (0, 0)
# These are the vertices of the tree polygon in counter-clockwise order

# Standard Christmas tree shape - a 9-vertex polygon
# Values derived from competition specification
TREE_VERTICES = np.array([
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

# Anchor point is at the origin (0, 0) in local coordinates
ANCHOR_POINT = np.array([0.0, 0.0], dtype=np.float64)

# Tree bounding box (approximate for quick checks)
TREE_BBOX_WIDTH = 1.0  # -0.5 to 0.5
TREE_BBOX_HEIGHT = 1.0  # -0.5 to 0.5


def get_tree_vertices() -> np.ndarray:
    """Return the official tree polygon vertices."""
    return TREE_VERTICES.copy()


def rotate_point(x: float, y: float, deg: float, cx: float = 0.0, cy: float = 0.0) -> Tuple[float, float]:
    """
    Rotate a point (x, y) by deg degrees around center (cx, cy).
    
    Args:
        x, y: Point coordinates
        deg: Rotation angle in degrees (counter-clockwise)
        cx, cy: Center of rotation
        
    Returns:
        Rotated (x, y) coordinates
    """
    rad = np.radians(deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    # Translate to origin, rotate, translate back
    dx = x - cx
    dy = y - cy
    
    new_x = dx * cos_a - dy * sin_a + cx
    new_y = dx * sin_a + dy * cos_a + cy
    
    return new_x, new_y


def rotate_vertices(vertices: np.ndarray, deg: float) -> np.ndarray:
    """
    Rotate all vertices by deg degrees around origin.
    
    Args:
        vertices: Nx2 array of (x, y) coordinates
        deg: Rotation angle in degrees (counter-clockwise)
        
    Returns:
        Rotated vertices as Nx2 array
    """
    rad = np.radians(deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], dtype=np.float64)
    
    return vertices @ rotation_matrix.T


def transform_tree(x: float, y: float, deg: float, vertices: np.ndarray = None) -> np.ndarray:
    """
    Transform tree vertices: rotate by deg degrees, then translate by (x, y).
    
    Args:
        x, y: Translation (anchor point position in world coordinates)
        deg: Rotation angle in degrees
        vertices: Optional custom vertices (uses default tree if None)
        
    Returns:
        Transformed vertices as Nx2 array
    """
    if vertices is None:
        vertices = TREE_VERTICES
    
    # First rotate around origin (anchor point)
    rotated = rotate_vertices(vertices, deg)
    
    # Then translate to world position
    rotated[:, 0] += x
    rotated[:, 1] += y
    
    return rotated


def compute_bounding_box(vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute axis-aligned bounding box for a set of vertices.
    
    Args:
        vertices: Nx2 array of (x, y) coordinates
        
    Returns:
        (min_x, min_y, max_x, max_y) bounding box
    """
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    
    return min_x, min_y, max_x, max_y


def compute_bounding_square_side(all_vertices: List[np.ndarray]) -> float:
    """
    Compute the side length of the smallest bounding square containing all trees.
    
    The bounding square is axis-aligned and must contain all tree vertices.
    The score is the side length of this square.
    
    Args:
        all_vertices: List of Nx2 arrays, one per tree
        
    Returns:
        Side length s of the bounding square
    """
    if not all_vertices:
        return 0.0
    
    # Concatenate all vertices
    all_pts = np.vstack(all_vertices)
    
    min_x = np.min(all_pts[:, 0])
    max_x = np.max(all_pts[:, 0])
    min_y = np.min(all_pts[:, 1])
    max_y = np.max(all_pts[:, 1])
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Bounding square side is max of width and height
    return max(width, height)


def polygon_area(vertices: np.ndarray) -> float:
    """
    Compute area of a polygon using shoelace formula.
    
    Args:
        vertices: Nx2 array of (x, y) coordinates in order
        
    Returns:
        Area of the polygon (always positive)
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def check_polygons_overlap_reference(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    Reference implementation: Check if two polygons overlap using Shapely.
    
    This is a slow but correct reference for testing.
    DO NOT use in optimization hot loops.
    
    Args:
        poly1, poly2: Nx2 arrays of polygon vertices
        
    Returns:
        True if polygons overlap (intersect with non-zero area)
    """
    try:
        from shapely.geometry import Polygon
        
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        
        # Check for intersection (overlapping interior)
        return p1.intersects(p2) and not p1.touches(p2)
        
    except ImportError:
        raise ImportError("Shapely is required for reference collision check")


if __name__ == "__main__":
    # Test the geometry
    print("Tree vertices:")
    print(TREE_VERTICES)
    print(f"Number of vertices: {len(TREE_VERTICES)}")
    print(f"Tree area: {polygon_area(TREE_VERTICES):.6f}")
    
    # Test rotation
    rotated = transform_tree(1.0, 2.0, 90.0)
    print(f"\nTransformed (x=1, y=2, deg=90):")
    print(rotated)
    
    bbox = compute_bounding_box(rotated)
    print(f"Bounding box: {bbox}")

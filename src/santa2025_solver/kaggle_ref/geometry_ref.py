"""
Reference geometry for Santa 2025 Christmas Tree Packing.

The Christmas tree is defined as a polygon with specific vertices.
Based on the Kaggle Santa 2025 competition specification.
"""

import numpy as np
from typing import Tuple, List

# Christmas tree polygon vertices (anchor at trunk-top center, i.e., (0, 0))
# The tree shape is:
#   - A triangular canopy on top
#   - A rectangular trunk at the bottom
# These are the standard coordinates from the competition:

TREE_VERTICES = np.array([
    [0.0, 0.5],        # Top of tree
    [-0.25, 0.0],      # Left side of canopy base
    [-0.125, 0.0],     # Left edge of trunk top
    [-0.125, -0.25],   # Left bottom of trunk
    [0.125, -0.25],    # Right bottom of trunk
    [0.125, 0.0],      # Right edge of trunk top
    [0.25, 0.0],       # Right side of canopy base
], dtype=np.float64)


def get_tree_vertices() -> np.ndarray:
    """Return the canonical tree polygon vertices.
    
    Returns:
        np.ndarray: Array of shape (7, 2) with tree vertices.
    """
    return TREE_VERTICES.copy()


def rotate_point(x: float, y: float, angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> Tuple[float, float]:
    """Rotate a point around a center.
    
    Args:
        x, y: Point coordinates
        angle_deg: Rotation angle in degrees (counterclockwise)
        cx, cy: Center of rotation
        
    Returns:
        Tuple of rotated (x, y) coordinates
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Translate to origin
    dx = x - cx
    dy = y - cy
    
    # Rotate
    new_x = dx * cos_a - dy * sin_a
    new_y = dx * sin_a + dy * cos_a
    
    # Translate back
    return new_x + cx, new_y + cy


def transform_tree(x: float, y: float, deg: float) -> np.ndarray:
    """Transform tree vertices by rotation and translation.
    
    Args:
        x: X translation
        y: Y translation  
        deg: Rotation angle in degrees
        
    Returns:
        np.ndarray: Transformed vertices of shape (7, 2)
    """
    vertices = get_tree_vertices()
    
    # Rotation matrix
    angle_rad = np.deg2rad(deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Apply rotation then translation
    rotated = vertices @ rotation_matrix.T
    transformed = rotated + np.array([x, y])
    
    return transformed


def get_tree_aabb(x: float, y: float, deg: float) -> Tuple[float, float, float, float]:
    """Get axis-aligned bounding box for a transformed tree.
    
    Args:
        x, y: Translation
        deg: Rotation angle in degrees
        
    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    vertices = transform_tree(x, y, deg)
    min_x = vertices[:, 0].min()
    max_x = vertices[:, 0].max()
    min_y = vertices[:, 1].min()
    max_y = vertices[:, 1].max()
    return min_x, min_y, max_x, max_y


def aabb_overlap(aabb1: Tuple[float, float, float, float], 
                 aabb2: Tuple[float, float, float, float]) -> bool:
    """Check if two AABBs overlap.
    
    Args:
        aabb1, aabb2: Tuples of (min_x, min_y, max_x, max_y)
        
    Returns:
        True if AABBs overlap
    """
    return not (aabb1[2] < aabb2[0] or  # aabb1 left of aabb2
                aabb1[0] > aabb2[2] or  # aabb1 right of aabb2
                aabb1[3] < aabb2[1] or  # aabb1 below aabb2
                aabb1[1] > aabb2[3])    # aabb1 above aabb2


def get_bounding_circle_radius(positions: List[Tuple[float, float, float]]) -> float:
    """Calculate the radius of the minimum bounding circle for all trees.
    
    For the official metric, we compute the maximum distance from origin
    to any vertex of any tree.
    
    Args:
        positions: List of (x, y, deg) for each tree
        
    Returns:
        Maximum distance from origin to any tree vertex
    """
    if not positions:
        return 0.0
    
    max_dist_sq = 0.0
    for x, y, deg in positions:
        vertices = transform_tree(x, y, deg)
        # Distance squared from origin for each vertex
        dist_sq = (vertices[:, 0] ** 2 + vertices[:, 1] ** 2).max()
        max_dist_sq = max(max_dist_sq, dist_sq)
    
    return np.sqrt(max_dist_sq)


if __name__ == "__main__":
    # Test the geometry
    print("Tree vertices:")
    print(get_tree_vertices())
    
    print("\nTransformed tree (x=1, y=0.5, deg=45):")
    print(transform_tree(1.0, 0.5, 45.0))
    
    print("\nAABB for (0, 0, 0):")
    print(get_tree_aabb(0, 0, 0))
    
    print("\nAABB for (0, 0, 90):")
    print(get_tree_aabb(0, 0, 90))

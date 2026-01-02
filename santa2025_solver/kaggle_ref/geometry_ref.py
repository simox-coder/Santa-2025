"""
Santa 2025 Solver - Geometry Reference

Reference tree geometry from official Kaggle getting-started notebook.
"""

import numpy as np
from typing import Optional


# ============================================================================
# OFFICIAL TREE DEFINITION
# ============================================================================
# This matches the official Kaggle competition definition.
# The tree is a heptagon (7-sided polygon) consisting of:
# - A triangular foliage (tip, left corner, right corner)
# - A rectangular trunk
# The anchor point is at the center of the top of the trunk (origin).

OFFICIAL_TREE_VERTICES = np.array([
    [0.0, 0.5],           # 0: tip
    [-0.25, 0.0],         # 1: left foliage corner
    [-0.0625, 0.0],       # 2: trunk top left
    [-0.0625, -0.125],    # 3: trunk bottom left
    [0.0625, -0.125],     # 4: trunk bottom right
    [0.0625, 0.0],        # 5: trunk top right
    [0.25, 0.0],          # 6: right foliage corner
], dtype=np.float64)

# Anchor point
ANCHOR_POINT = np.array([0.0, 0.0], dtype=np.float64)

# Tree dimensions
TREE_WIDTH = 0.5    # Max width (foliage)
TREE_HEIGHT = 0.625  # Total height (tip to trunk bottom: 0.5 + 0.125)
TRUNK_WIDTH = 0.125  # Trunk width (2 * 0.0625)
TRUNK_HEIGHT = 0.125  # Trunk height


def get_official_rotation_matrix(deg: float) -> np.ndarray:
    """
    Get 2D rotation matrix for given angle in degrees.
    
    Matches official implementation.
    """
    rad = np.radians(deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    return np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], dtype=np.float64)


def get_official_tree_vertices(x: float, y: float, deg: float) -> np.ndarray:
    """
    Get tree polygon vertices at position (x, y) with rotation deg.
    
    This is the reference implementation that matches the official definition.
    
    Args:
        x: X coordinate of anchor point
        y: Y coordinate of anchor point
        deg: Rotation angle in degrees
    
    Returns:
        np.ndarray: Shape (7, 2) array of vertex coordinates
    """
    # Rotate around anchor point (origin)
    R = get_official_rotation_matrix(deg)
    rotated = OFFICIAL_TREE_VERTICES @ R.T
    
    # Translate to position
    translated = rotated + np.array([x, y])
    
    return translated


def get_bounding_square_from_vertices(all_vertices: np.ndarray) -> float:
    """
    Compute bounding square side from all tree vertices.
    
    The bounding square is centered at origin and contains all vertices.
    
    Args:
        all_vertices: Shape (n, 7, 2) array of all tree vertices
    
    Returns:
        Side length s of the bounding square
    """
    if len(all_vertices) == 0:
        return 0.0
    
    # Find maximum absolute coordinate
    max_abs = np.abs(all_vertices).max()
    
    # Bounding square side is 2 * max_abs
    return 2.0 * max_abs


def verify_our_geometry_matches_official() -> bool:
    """
    Verify that our geometry implementation matches the official one.
    
    Returns:
        True if all tests pass
    """
    from santa2025_solver.geometry_fast import get_tree_vertices as our_get_vertices
    
    test_cases = [
        (0, 0, 0),
        (0, 0, 90),
        (0, 0, 180),
        (0, 0, 270),
        (1.5, -0.5, 90),
        (-2.0, 3.0, 180),
    ]
    
    for x, y, deg in test_cases:
        official = get_official_tree_vertices(x, y, deg)
        ours = our_get_vertices(x, y, deg)
        
        if not np.allclose(official, ours, atol=1e-10):
            print(f"Mismatch at ({x}, {y}, {deg})")
            print(f"  Official:\n{official}")
            print(f"  Ours:\n{ours}")
            return False
    
    return True


if __name__ == "__main__":
    # Run verification
    if verify_our_geometry_matches_official():
        print("Geometry verification PASSED")
    else:
        print("Geometry verification FAILED")

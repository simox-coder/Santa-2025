"""
Santa 2025 Solver - Fast Geometry Engine

Tree representation and transformation utilities.
Based on the official Kaggle competition definition.
"""

import numpy as np
from typing import Tuple, Optional
import math


# ============================================================================
# TREE DEFINITION - FROM OFFICIAL KAGGLE COMPETITION
# ============================================================================
# The Christmas tree is defined as a polygon with the following vertices
# (in local coordinates, before any rotation or translation).
# The anchor point (center of top of trunk) is at the origin.

# Tree base vertices (official definition)
# Top of tree (tip)
TREE_TIP = (0.0, 0.5)

# Tree foliage (triangular shape)
TREE_LEFT = (-0.25, 0.0)
TREE_RIGHT = (0.25, 0.0)

# Trunk (rectangular)
TRUNK_TOP_LEFT = (-0.0625, 0.0)
TRUNK_TOP_RIGHT = (0.0625, 0.0)
TRUNK_BOTTOM_LEFT = (-0.0625, -0.125)
TRUNK_BOTTOM_RIGHT = (0.0625, -0.125)

# Full tree polygon vertices (counterclockwise)
BASE_TREE_VERTICES = np.array([
    [0.0, 0.5],           # tip
    [-0.25, 0.0],         # left foliage
    [-0.0625, 0.0],       # trunk top left
    [-0.0625, -0.125],    # trunk bottom left
    [0.0625, -0.125],     # trunk bottom right
    [0.0625, 0.0],        # trunk top right
    [0.25, 0.0],          # right foliage
], dtype=np.float64)

# Anchor point is at origin (0, 0) which is center of top of trunk
ANCHOR_POINT = np.array([0.0, 0.0], dtype=np.float64)


def get_rotation_matrix(deg: float) -> np.ndarray:
    """Get 2D rotation matrix for given angle in degrees."""
    rad = math.radians(deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], dtype=np.float64)


def get_tree_vertices(x: float, y: float, deg: float) -> np.ndarray:
    """
    Get tree polygon vertices for a tree at position (x, y) with rotation deg.
    
    Args:
        x: X coordinate of anchor point
        y: Y coordinate of anchor point
        deg: Rotation angle in degrees (0, 90, 180, 270 typically)
    
    Returns:
        np.ndarray: Shape (7, 2) array of vertex coordinates
    """
    # Rotate around origin (anchor point)
    R = get_rotation_matrix(deg)
    rotated = BASE_TREE_VERTICES @ R.T  # Transpose for row vectors
    
    # Translate to position
    translated = rotated + np.array([x, y])
    return translated


def get_tree_aabb(x: float, y: float, deg: float) -> Tuple[float, float, float, float]:
    """
    Get axis-aligned bounding box for a tree.
    
    Returns:
        (min_x, min_y, max_x, max_y)
    """
    vertices = get_tree_vertices(x, y, deg)
    min_x = vertices[:, 0].min()
    max_x = vertices[:, 0].max()
    min_y = vertices[:, 1].min()
    max_y = vertices[:, 1].max()
    return (min_x, min_y, max_x, max_y)


def aabb_overlap(aabb1: Tuple[float, float, float, float], 
                  aabb2: Tuple[float, float, float, float]) -> bool:
    """Check if two AABBs overlap."""
    min_x1, min_y1, max_x1, max_y1 = aabb1
    min_x2, min_y2, max_x2, max_y2 = aabb2
    
    # Check for non-overlap
    if max_x1 <= min_x2 or max_x2 <= min_x1:
        return False
    if max_y1 <= min_y2 or max_y2 <= min_y1:
        return False
    return True


class TreeGeometry:
    """
    Fast tree geometry representation with caching.
    """
    
    def __init__(self, x: float = 0.0, y: float = 0.0, deg: float = 0.0):
        self.x = x
        self.y = y
        self.deg = deg
        self._vertices: Optional[np.ndarray] = None
        self._aabb: Optional[Tuple[float, float, float, float]] = None
    
    @property
    def vertices(self) -> np.ndarray:
        """Get tree vertices (cached)."""
        if self._vertices is None:
            self._vertices = get_tree_vertices(self.x, self.y, self.deg)
        return self._vertices
    
    @property
    def aabb(self) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box (cached)."""
        if self._aabb is None:
            self._aabb = get_tree_aabb(self.x, self.y, self.deg)
        return self._aabb
    
    def update(self, x: float = None, y: float = None, deg: float = None):
        """Update position/rotation and invalidate cache."""
        changed = False
        if x is not None and x != self.x:
            self.x = x
            changed = True
        if y is not None and y != self.y:
            self.y = y
            changed = True
        if deg is not None and deg != self.deg:
            self.deg = deg
            changed = True
        if changed:
            self._vertices = None
            self._aabb = None
    
    def overlaps_aabb(self, other: 'TreeGeometry') -> bool:
        """Quick AABB overlap check."""
        return aabb_overlap(self.aabb, other.aabb)
    
    def copy(self) -> 'TreeGeometry':
        """Create a copy of this tree."""
        return TreeGeometry(self.x, self.y, self.deg)


# ============================================================================
# PRECOMPUTED ROTATIONS FOR COMMON ANGLES
# ============================================================================

# Precompute rotation matrices for 0, 90, 180, 270 degrees
ROTATION_MATRICES = {
    0: get_rotation_matrix(0),
    90: get_rotation_matrix(90),
    180: get_rotation_matrix(180),
    270: get_rotation_matrix(270),
}

# Precompute base vertices rotated by common angles
ROTATED_VERTICES = {
    deg: BASE_TREE_VERTICES @ R.T
    for deg, R in ROTATION_MATRICES.items()
}


def get_tree_vertices_fast(x: float, y: float, deg: int) -> np.ndarray:
    """
    Fast path for common rotation angles (0, 90, 180, 270).
    Uses precomputed rotated vertices.
    """
    if deg in ROTATED_VERTICES:
        return ROTATED_VERTICES[deg] + np.array([x, y])
    else:
        return get_tree_vertices(x, y, deg)


# ============================================================================
# BATCH OPERATIONS FOR MULTIPLE TREES
# ============================================================================

def get_all_tree_vertices(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """
    Get vertices for multiple trees.
    
    Args:
        positions: Shape (n, 2) array of (x, y) positions
        rotations: Shape (n,) array of rotation angles in degrees
    
    Returns:
        Shape (n, 7, 2) array of vertices
    """
    n = len(positions)
    result = np.zeros((n, 7, 2), dtype=np.float64)
    
    for i in range(n):
        result[i] = get_tree_vertices_fast(
            positions[i, 0], positions[i, 1], int(rotations[i])
        )
    
    return result


def get_all_aabbs(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """
    Get AABBs for multiple trees.
    
    Args:
        positions: Shape (n, 2) array of (x, y) positions
        rotations: Shape (n,) array of rotation angles in degrees
    
    Returns:
        Shape (n, 4) array of (min_x, min_y, max_x, max_y)
    """
    all_vertices = get_all_tree_vertices(positions, rotations)
    min_coords = all_vertices.min(axis=1)
    max_coords = all_vertices.max(axis=1)
    return np.hstack([min_coords, max_coords])


# ============================================================================
# BOUNDING SQUARE COMPUTATION
# ============================================================================

def compute_bounding_square_side(positions: np.ndarray, rotations: np.ndarray) -> float:
    """
    Compute the side length of the minimum bounding square for all trees.
    
    The bounding square is centered at the origin and contains all tree vertices.
    This is the metric we're trying to minimize.
    
    Returns:
        s: Side length of bounding square (2 * max(|x|, |y|) for all vertices)
    """
    if len(positions) == 0:
        return 0.0
    
    all_vertices = get_all_tree_vertices(positions, rotations)
    
    # Find maximum absolute coordinate
    max_abs_x = np.abs(all_vertices[:, :, 0]).max()
    max_abs_y = np.abs(all_vertices[:, :, 1]).max()
    
    # Bounding square side is 2 * max(|x|, |y|)
    s = 2.0 * max(max_abs_x, max_abs_y)
    
    return s


def compute_score_for_n(positions: np.ndarray, rotations: np.ndarray, n: int) -> float:
    """
    Compute the score contribution for a single n-tree layout.
    
    The official score is the sum of s^2 for all n from 1 to 200.
    This function computes s^2 for a single n.
    """
    s = compute_bounding_square_side(positions, rotations)
    return s * s


# ============================================================================
# TREE LAYOUT CLASS
# ============================================================================

class Layout:
    """
    Represents a layout of n trees.
    """
    
    def __init__(self, n: int):
        self.n = n
        self.positions = np.zeros((n, 2), dtype=np.float64)
        self.rotations = np.zeros(n, dtype=np.float64)
        self._vertices: Optional[np.ndarray] = None
        self._aabbs: Optional[np.ndarray] = None
        self._bounding_side: Optional[float] = None
    
    def invalidate_cache(self, indices: Optional[np.ndarray] = None):
        """Invalidate cached computations."""
        self._vertices = None
        self._aabbs = None
        self._bounding_side = None
    
    @property
    def vertices(self) -> np.ndarray:
        """Get all tree vertices (n, 7, 2)."""
        if self._vertices is None:
            self._vertices = get_all_tree_vertices(self.positions, self.rotations)
        return self._vertices
    
    @property
    def aabbs(self) -> np.ndarray:
        """Get all AABBs (n, 4)."""
        if self._aabbs is None:
            self._aabbs = get_all_aabbs(self.positions, self.rotations)
        return self._aabbs
    
    @property
    def bounding_side(self) -> float:
        """Get bounding square side length."""
        if self._bounding_side is None:
            self._bounding_side = compute_bounding_square_side(
                self.positions, self.rotations
            )
        return self._bounding_side
    
    @property
    def score(self) -> float:
        """Get s^2 for this layout."""
        return self.bounding_side ** 2
    
    def set_tree(self, idx: int, x: float, y: float, deg: float):
        """Set position and rotation for a single tree."""
        self.positions[idx, 0] = x
        self.positions[idx, 1] = y
        self.rotations[idx] = deg
        self.invalidate_cache()
    
    def get_tree(self, idx: int) -> Tuple[float, float, float]:
        """Get (x, y, deg) for a single tree."""
        return (self.positions[idx, 0], self.positions[idx, 1], self.rotations[idx])
    
    def copy(self) -> 'Layout':
        """Create a deep copy of this layout."""
        new_layout = Layout(self.n)
        new_layout.positions = self.positions.copy()
        new_layout.rotations = self.rotations.copy()
        return new_layout
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'n': self.n,
            'positions': self.positions.tolist(),
            'rotations': self.rotations.tolist(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Layout':
        """Create from dictionary."""
        layout = cls(d['n'])
        layout.positions = np.array(d['positions'], dtype=np.float64)
        layout.rotations = np.array(d['rotations'], dtype=np.float64)
        return layout

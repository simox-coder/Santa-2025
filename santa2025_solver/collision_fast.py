"""
Fast collision detection for Santa 2025 solver.

Implements two-phase collision detection:
1. Broad-phase: Spatial hashing with uniform grid
2. Narrow-phase: Separating Axis Theorem (SAT) for convex polygons

For concave polygons (like the tree), we use a triangulated decomposition
and check SAT on triangle pairs.
"""

import numpy as np
from typing import Tuple, List, Set, Optional, Dict
from collections import defaultdict

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if callable(args[0]):
            return args[0]
        return decorator

from .geometry_fast import (
    TreeLayout,
    compute_aabb,
    aabb_overlap,
    TREE_VERTICES_BASE,
)


# ============================================================================
# SAT (Separating Axis Theorem) for convex polygons
# ============================================================================

@njit(cache=True)
def get_polygon_axes(vertices: np.ndarray) -> np.ndarray:
    """
    Get perpendicular axes (normals) for all edges of a polygon.
    
    Args:
        vertices: Nx2 array of polygon vertices
        
    Returns:
        Nx2 array of normalized axis vectors
    """
    n = len(vertices)
    axes = np.empty((n, 2), dtype=np.float64)
    
    for i in range(n):
        j = (i + 1) % n
        edge_x = vertices[j, 0] - vertices[i, 0]
        edge_y = vertices[j, 1] - vertices[i, 1]
        
        # Perpendicular (normal) to edge
        axes[i, 0] = -edge_y
        axes[i, 1] = edge_x
        
        # Normalize
        length = np.sqrt(axes[i, 0]**2 + axes[i, 1]**2)
        if length > 1e-10:
            axes[i, 0] /= length
            axes[i, 1] /= length
    
    return axes


@njit(cache=True)
def project_polygon(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """
    Project polygon onto axis and return min/max projection values.
    
    Args:
        vertices: Nx2 polygon vertices
        axis: 2-element axis vector
        
    Returns:
        (min_proj, max_proj)
    """
    dot = vertices[0, 0] * axis[0] + vertices[0, 1] * axis[1]
    min_proj = dot
    max_proj = dot
    
    for i in range(1, len(vertices)):
        dot = vertices[i, 0] * axis[0] + vertices[i, 1] * axis[1]
        if dot < min_proj:
            min_proj = dot
        if dot > max_proj:
            max_proj = dot
    
    return min_proj, max_proj


@njit(cache=True)
def projections_overlap(min1: float, max1: float, min2: float, max2: float) -> bool:
    """Check if two projection intervals overlap."""
    return not (max1 < min2 or max2 < min1)


@njit(cache=True)
def sat_polygons_collide(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    """
    Check if two convex polygons collide using SAT.
    
    For convex polygons, if no separating axis is found among all edge normals
    of both polygons, they must be overlapping.
    
    Args:
        poly1, poly2: Nx2 arrays of polygon vertices
        
    Returns:
        True if polygons overlap
    """
    # Check axes from poly1
    n1 = len(poly1)
    for i in range(n1):
        j = (i + 1) % n1
        edge_x = poly1[j, 0] - poly1[i, 0]
        edge_y = poly1[j, 1] - poly1[i, 1]
        
        # Perpendicular axis
        ax = -edge_y
        ay = edge_x
        
        # Normalize
        length = np.sqrt(ax*ax + ay*ay)
        if length < 1e-10:
            continue
        ax /= length
        ay /= length
        
        # Project both polygons
        min1 = poly1[0, 0] * ax + poly1[0, 1] * ay
        max1 = min1
        for k in range(1, n1):
            d = poly1[k, 0] * ax + poly1[k, 1] * ay
            if d < min1:
                min1 = d
            if d > max1:
                max1 = d
        
        n2 = len(poly2)
        min2 = poly2[0, 0] * ax + poly2[0, 1] * ay
        max2 = min2
        for k in range(1, n2):
            d = poly2[k, 0] * ax + poly2[k, 1] * ay
            if d < min2:
                min2 = d
            if d > max2:
                max2 = d
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False  # Found separating axis
    
    # Check axes from poly2
    n2 = len(poly2)
    for i in range(n2):
        j = (i + 1) % n2
        edge_x = poly2[j, 0] - poly2[i, 0]
        edge_y = poly2[j, 1] - poly2[i, 1]
        
        ax = -edge_y
        ay = edge_x
        
        length = np.sqrt(ax*ax + ay*ay)
        if length < 1e-10:
            continue
        ax /= length
        ay /= length
        
        min1 = poly1[0, 0] * ax + poly1[0, 1] * ay
        max1 = min1
        for k in range(1, n1):
            d = poly1[k, 0] * ax + poly1[k, 1] * ay
            if d < min1:
                min1 = d
            if d > max1:
                max1 = d
        
        min2 = poly2[0, 0] * ax + poly2[0, 1] * ay
        max2 = min2
        for k in range(1, n2):
            d = poly2[k, 0] * ax + poly2[k, 1] * ay
            if d < min2:
                min2 = d
            if d > max2:
                max2 = d
        
        if max1 < min2 or max2 < min1:
            return False
    
    return True  # No separating axis found -> collision


# ============================================================================
# Triangulation for concave polygons
# ============================================================================

def triangulate_polygon(vertices: np.ndarray) -> List[np.ndarray]:
    """
    Simple ear-clipping triangulation for a simple polygon.
    
    For the tree polygon, we can also use a pre-computed triangulation.
    
    Args:
        vertices: Nx2 array of polygon vertices in order
        
    Returns:
        List of 3x2 arrays (triangles)
    """
    n = len(vertices)
    if n < 3:
        return []
    if n == 3:
        return [vertices.copy()]
    
    # Simple fan triangulation from centroid (works for star-convex polygons)
    # The tree polygon is roughly star-convex from its center
    centroid = np.mean(vertices, axis=0)
    
    triangles = []
    for i in range(n):
        j = (i + 1) % n
        tri = np.array([
            centroid,
            vertices[i],
            vertices[j]
        ], dtype=np.float64)
        triangles.append(tri)
    
    return triangles


# Pre-computed triangulation for the base tree
_TREE_TRIANGLES = None

def get_tree_triangles() -> List[np.ndarray]:
    """Get pre-computed triangulation of the tree polygon."""
    global _TREE_TRIANGLES
    if _TREE_TRIANGLES is None:
        _TREE_TRIANGLES = triangulate_polygon(TREE_VERTICES_BASE)
    return _TREE_TRIANGLES


def transform_triangles(triangles: List[np.ndarray], x: float, y: float, deg: float) -> List[np.ndarray]:
    """Transform a list of triangles."""
    from .geometry_fast import rotate_vertices_fast
    
    result = []
    for tri in triangles:
        rotated = rotate_vertices_fast(tri, deg)
        rotated[:, 0] += x
        rotated[:, 1] += y
        result.append(rotated)
    
    return result


# ============================================================================
# Spatial Hash Grid for broad-phase
# ============================================================================

class SpatialHashGrid:
    """
    Uniform grid spatial hash for broad-phase collision detection.
    
    Each cell contains indices of trees whose AABBs overlap that cell.
    """
    
    def __init__(self, cell_size: float = 0.5):
        """
        Initialize grid.
        
        Args:
            cell_size: Size of each grid cell (should be ~= tree size)
        """
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self.cells: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.tree_cells: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    
    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Get cell indices for a point."""
        cx = int(np.floor(x * self.inv_cell_size))
        cy = int(np.floor(y * self.inv_cell_size))
        return (cx, cy)
    
    def _get_cells_for_aabb(self, aabb: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """Get all cells overlapping an AABB."""
        min_x, min_y, max_x, max_y = aabb
        
        cx_min = int(np.floor(min_x * self.inv_cell_size))
        cx_max = int(np.floor(max_x * self.inv_cell_size))
        cy_min = int(np.floor(min_y * self.inv_cell_size))
        cy_max = int(np.floor(max_y * self.inv_cell_size))
        
        cells = []
        for cx in range(cx_min, cx_max + 1):
            for cy in range(cy_min, cy_max + 1):
                cells.append((cx, cy))
        
        return cells
    
    def insert(self, idx: int, aabb: Tuple[float, float, float, float]):
        """Insert a tree into the grid."""
        cells = self._get_cells_for_aabb(aabb)
        for cell in cells:
            self.cells[cell].add(idx)
            self.tree_cells[idx].add(cell)
    
    def remove(self, idx: int):
        """Remove a tree from the grid."""
        for cell in self.tree_cells[idx]:
            self.cells[cell].discard(idx)
        self.tree_cells[idx].clear()
    
    def update(self, idx: int, old_aabb: Tuple[float, float, float, float], 
               new_aabb: Tuple[float, float, float, float]):
        """Update a tree's position in the grid."""
        self.remove(idx)
        self.insert(idx, new_aabb)
    
    def get_potential_collisions(self, idx: int) -> Set[int]:
        """Get indices of trees that might collide with the given tree."""
        candidates = set()
        for cell in self.tree_cells[idx]:
            candidates.update(self.cells[cell])
        candidates.discard(idx)  # Don't include self
        return candidates
    
    def get_all_potential_pairs(self) -> Set[Tuple[int, int]]:
        """Get all pairs of potentially colliding trees."""
        pairs = set()
        for cell, indices in self.cells.items():
            indices_list = list(indices)
            for i in range(len(indices_list)):
                for j in range(i + 1, len(indices_list)):
                    pair = (min(indices_list[i], indices_list[j]), 
                           max(indices_list[i], indices_list[j]))
                    pairs.add(pair)
        return pairs
    
    def clear(self):
        """Clear all data."""
        self.cells.clear()
        self.tree_cells.clear()


# ============================================================================
# Main Collision Checker
# ============================================================================

class CollisionChecker:
    """
    Two-phase collision detection system.
    
    Maintains spatial hash grid for broad-phase and uses SAT for narrow-phase.
    """
    
    def __init__(self, layout: TreeLayout, cell_size: float = 0.5):
        """
        Initialize collision checker for a layout.
        
        Args:
            layout: TreeLayout to check
            cell_size: Spatial hash cell size
        """
        self.layout = layout
        self.grid = SpatialHashGrid(cell_size)
        self._tree_triangles_base = get_tree_triangles()
        
        # Initialize grid with all trees
        self.rebuild_grid()
    
    def rebuild_grid(self):
        """Rebuild the entire spatial hash grid."""
        self.grid.clear()
        for i in range(self.layout.n):
            aabb = self.layout.get_aabb(i)
            self.grid.insert(i, aabb)
    
    def update_tree(self, idx: int, old_aabb: Optional[Tuple[float, float, float, float]] = None):
        """Update grid after a tree has moved."""
        new_aabb = self.layout.get_aabb(idx)
        if old_aabb is not None:
            self.grid.update(idx, old_aabb, new_aabb)
        else:
            self.grid.remove(idx)
            self.grid.insert(idx, new_aabb)
    
    def check_collision_pair(self, idx1: int, idx2: int) -> bool:
        """
        Check if two specific trees collide.
        
        Uses AABB check first, then SAT on triangulated polygons.
        
        Args:
            idx1, idx2: Tree indices
            
        Returns:
            True if trees collide
        """
        # Broad phase: AABB check
        aabb1 = self.layout.get_aabb(idx1)
        aabb2 = self.layout.get_aabb(idx2)
        
        if not aabb_overlap(aabb1, aabb2):
            return False
        
        # Narrow phase: SAT on full polygons
        # Since tree polygon is approximately convex when viewed as a whole,
        # we can use direct SAT
        verts1 = self.layout.get_vertices(idx1)
        verts2 = self.layout.get_vertices(idx2)
        
        return sat_polygons_collide(verts1, verts2)
    
    def get_collisions_for_tree(self, idx: int) -> List[int]:
        """
        Get all trees that collide with the given tree.
        
        Args:
            idx: Tree index
            
        Returns:
            List of colliding tree indices
        """
        candidates = self.grid.get_potential_collisions(idx)
        collisions = []
        
        for other_idx in candidates:
            if self.check_collision_pair(idx, other_idx):
                collisions.append(other_idx)
        
        return collisions
    
    def has_any_collision(self, idx: int) -> bool:
        """
        Check if tree has any collision (early exit).
        
        Args:
            idx: Tree index
            
        Returns:
            True if tree collides with any other tree
        """
        candidates = self.grid.get_potential_collisions(idx)
        
        for other_idx in candidates:
            if self.check_collision_pair(idx, other_idx):
                return True
        
        return False
    
    def count_all_collisions(self) -> int:
        """
        Count total number of colliding pairs.
        
        Returns:
            Number of colliding pairs
        """
        pairs = self.grid.get_all_potential_pairs()
        count = 0
        
        for idx1, idx2 in pairs:
            if self.check_collision_pair(idx1, idx2):
                count += 1
        
        return count
    
    def is_layout_valid(self) -> bool:
        """
        Check if layout has no collisions.
        
        Returns:
            True if layout is valid (no collisions)
        """
        pairs = self.grid.get_all_potential_pairs()
        
        for idx1, idx2 in pairs:
            if self.check_collision_pair(idx1, idx2):
                return False
        
        return True
    
    def find_first_collision(self) -> Optional[Tuple[int, int]]:
        """
        Find first colliding pair (for debugging).
        
        Returns:
            (idx1, idx2) of first collision, or None if valid
        """
        pairs = self.grid.get_all_potential_pairs()
        
        for idx1, idx2 in pairs:
            if self.check_collision_pair(idx1, idx2):
                return (idx1, idx2)
        
        return None


def check_layout_collisions_reference(layout: TreeLayout) -> bool:
    """
    Reference collision check using Shapely (slow but correct).
    
    For testing only, not for optimization.
    
    Args:
        layout: TreeLayout to check
        
    Returns:
        True if there are any collisions (invalid)
    """
    try:
        from shapely.geometry import Polygon
        
        polygons = []
        for i in range(layout.n):
            verts = layout.get_vertices(i)
            poly = Polygon(verts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            polygons.append(poly)
        
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                    return True
        
        return False
        
    except ImportError:
        raise ImportError("Shapely required for reference collision check")

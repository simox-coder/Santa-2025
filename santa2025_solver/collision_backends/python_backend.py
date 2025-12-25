"""
Santa 2025 Solver - Python Collision Backend

Pure Python/NumPy implementation of collision detection.
This serves as the correctness baseline for other backends.
"""

import numpy as np
from typing import Tuple, List, Optional


def project_polygon_onto_axis(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """
    Project polygon vertices onto an axis and return min/max projections.
    
    Args:
        vertices: Shape (n, 2) array of vertex coordinates
        axis: Shape (2,) normalized axis vector
    
    Returns:
        (min_proj, max_proj)
    """
    projections = vertices @ axis
    return projections.min(), projections.max()


def get_polygon_axes(vertices: np.ndarray) -> np.ndarray:
    """
    Get perpendicular axes for SAT (Separating Axis Theorem).
    
    For a convex polygon, we need to check axes perpendicular to each edge.
    
    Args:
        vertices: Shape (n, 2) array of vertex coordinates
    
    Returns:
        Shape (n, 2) array of normalized axis vectors
    """
    n = len(vertices)
    axes = np.zeros((n, 2), dtype=np.float64)
    
    for i in range(n):
        # Edge from vertex i to vertex (i+1)
        edge = vertices[(i + 1) % n] - vertices[i]
        # Perpendicular axis (rotate 90 degrees)
        axes[i, 0] = -edge[1]
        axes[i, 1] = edge[0]
        # Normalize
        length = np.sqrt(axes[i, 0]**2 + axes[i, 1]**2)
        if length > 1e-10:
            axes[i] /= length
    
    return axes


def check_polygon_overlap_sat(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """
    Check if two convex polygons overlap using SAT.
    
    Args:
        vertices1: Shape (n1, 2) array of first polygon vertices
        vertices2: Shape (n2, 2) array of second polygon vertices
    
    Returns:
        True if polygons overlap, False otherwise
    """
    # Get axes from both polygons
    axes1 = get_polygon_axes(vertices1)
    axes2 = get_polygon_axes(vertices2)
    
    # Check all axes from polygon 1
    for axis in axes1:
        min1, max1 = project_polygon_onto_axis(vertices1, axis)
        min2, max2 = project_polygon_onto_axis(vertices2, axis)
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return False  # Separation found, no overlap
    
    # Check all axes from polygon 2
    for axis in axes2:
        min1, max1 = project_polygon_onto_axis(vertices1, axis)
        min2, max2 = project_polygon_onto_axis(vertices2, axis)
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return False  # Separation found, no overlap
    
    # No separating axis found, polygons overlap
    return True


def decompose_tree_to_convex(vertices: np.ndarray) -> List[np.ndarray]:
    """
    Decompose tree polygon into convex components.
    
    The tree shape is non-convex (concave), so we split it into:
    1. Upper triangle (foliage)
    2. Lower rectangle (trunk)
    
    Args:
        vertices: Shape (7, 2) array of tree vertices
    
    Returns:
        List of convex polygon vertex arrays
    """
    # Tree vertices layout:
    # 0: tip (0, 0.5)
    # 1: left foliage (-0.25, 0)
    # 2: trunk top left (-0.0625, 0)
    # 3: trunk bottom left (-0.0625, -0.125)
    # 4: trunk bottom right (0.0625, -0.125)
    # 5: trunk top right (0.0625, 0)
    # 6: right foliage (0.25, 0)
    
    # Upper triangle (foliage): tip, left foliage, right foliage
    triangle = vertices[[0, 1, 6], :]
    
    # Lower rectangle (trunk): 4 corners
    trunk = vertices[[2, 3, 4, 5], :]
    
    return [triangle, trunk]


def check_tree_overlap_python(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """
    Check if two tree polygons overlap.
    
    Args:
        vertices1: Shape (7, 2) array of first tree vertices
        vertices2: Shape (7, 2) array of second tree vertices
    
    Returns:
        True if trees overlap, False otherwise
    """
    # Decompose both trees into convex parts
    parts1 = decompose_tree_to_convex(vertices1)
    parts2 = decompose_tree_to_convex(vertices2)
    
    # Check all pairs of convex parts
    for p1 in parts1:
        for p2 in parts2:
            if check_polygon_overlap_sat(p1, p2):
                return True
    
    return False


def check_polygon_overlap_python(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """
    Public interface for polygon overlap check.
    
    Alias for check_tree_overlap_python.
    """
    return check_tree_overlap_python(vertices1, vertices2)


class SpatialHashGrid:
    """
    Spatial hash grid for broad-phase collision detection.
    """
    
    def __init__(self, cell_size: float = 0.5):
        self.cell_size = cell_size
        self.grid: dict = {}  # cell -> set of tree indices
        self.tree_cells: dict = {}  # tree index -> set of cells
    
    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Get grid cell for a point."""
        return (int(x // self.cell_size), int(y // self.cell_size))
    
    def _get_cells_for_aabb(self, aabb: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """Get all grid cells covered by an AABB."""
        min_x, min_y, max_x, max_y = aabb
        
        min_cell_x = int(min_x // self.cell_size)
        max_cell_x = int(max_x // self.cell_size)
        min_cell_y = int(min_y // self.cell_size)
        max_cell_y = int(max_y // self.cell_size)
        
        cells = []
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cells.append((cx, cy))
        
        return cells
    
    def insert(self, tree_idx: int, aabb: Tuple[float, float, float, float]):
        """Insert a tree into the grid."""
        cells = self._get_cells_for_aabb(aabb)
        self.tree_cells[tree_idx] = set(cells)
        
        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = set()
            self.grid[cell].add(tree_idx)
    
    def remove(self, tree_idx: int):
        """Remove a tree from the grid."""
        if tree_idx not in self.tree_cells:
            return
        
        for cell in self.tree_cells[tree_idx]:
            if cell in self.grid:
                self.grid[cell].discard(tree_idx)
                if not self.grid[cell]:
                    del self.grid[cell]
        
        del self.tree_cells[tree_idx]
    
    def update(self, tree_idx: int, aabb: Tuple[float, float, float, float]):
        """Update a tree's position in the grid."""
        self.remove(tree_idx)
        self.insert(tree_idx, aabb)
    
    def get_potential_collisions(self, tree_idx: int) -> List[int]:
        """Get indices of trees that might collide with the given tree."""
        if tree_idx not in self.tree_cells:
            return []
        
        potential = set()
        for cell in self.tree_cells[tree_idx]:
            if cell in self.grid:
                potential.update(self.grid[cell])
        
        # Remove self
        potential.discard(tree_idx)
        
        return list(potential)
    
    def get_all_potential_pairs(self) -> List[Tuple[int, int]]:
        """Get all pairs of trees that might collide."""
        pairs = set()
        
        for cell, tree_indices in self.grid.items():
            indices = list(tree_indices)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = min(indices[i], indices[j]), max(indices[i], indices[j])
                    pairs.add((idx1, idx2))
        
        return list(pairs)
    
    def clear(self):
        """Clear the grid."""
        self.grid.clear()
        self.tree_cells.clear()


class PythonCollisionBackend:
    """
    Pure Python collision detection backend.
    """
    
    def __init__(self, n_trees: int, cell_size: float = 0.5):
        self.n_trees = n_trees
        self.spatial_grid = SpatialHashGrid(cell_size)
        self.vertices: Optional[np.ndarray] = None  # Shape (n, 7, 2)
        self.aabbs: Optional[np.ndarray] = None  # Shape (n, 4)
    
    def initialize(self, vertices: np.ndarray, aabbs: np.ndarray):
        """
        Initialize backend with tree data.
        
        Args:
            vertices: Shape (n, 7, 2) array of tree vertices
            aabbs: Shape (n, 4) array of AABBs
        """
        self.vertices = vertices.copy()
        self.aabbs = aabbs.copy()
        
        # Build spatial hash
        self.spatial_grid.clear()
        for i in range(self.n_trees):
            self.spatial_grid.insert(i, tuple(self.aabbs[i]))
    
    def update_tree(self, idx: int, vertices: np.ndarray, aabb: np.ndarray):
        """Update a single tree's data."""
        self.vertices[idx] = vertices
        self.aabbs[idx] = aabb
        self.spatial_grid.update(idx, tuple(aabb))
    
    def check_overlap(self, idx1: int, idx2: int) -> bool:
        """Check if two trees overlap."""
        # Quick AABB check first
        aabb1 = self.aabbs[idx1]
        aabb2 = self.aabbs[idx2]
        
        if aabb1[2] <= aabb2[0] or aabb2[2] <= aabb1[0]:
            return False
        if aabb1[3] <= aabb2[1] or aabb2[3] <= aabb1[1]:
            return False
        
        # Detailed polygon check
        return check_tree_overlap_python(self.vertices[idx1], self.vertices[idx2])
    
    def check_tree_collisions(self, idx: int) -> List[int]:
        """Get list of trees that collide with the given tree."""
        potential = self.spatial_grid.get_potential_collisions(idx)
        collisions = []
        
        for other_idx in potential:
            if self.check_overlap(idx, other_idx):
                collisions.append(other_idx)
        
        return collisions
    
    def has_any_collision(self) -> bool:
        """Check if any trees collide."""
        pairs = self.spatial_grid.get_all_potential_pairs()
        
        for idx1, idx2 in pairs:
            if self.check_overlap(idx1, idx2):
                return True
        
        return False
    
    def count_collisions(self) -> int:
        """Count total number of colliding pairs."""
        pairs = self.spatial_grid.get_all_potential_pairs()
        count = 0
        
        for idx1, idx2 in pairs:
            if self.check_overlap(idx1, idx2):
                count += 1
        
        return count
    
    def get_all_collisions(self) -> List[Tuple[int, int]]:
        """Get all colliding pairs."""
        pairs = self.spatial_grid.get_all_potential_pairs()
        collisions = []
        
        for idx1, idx2 in pairs:
            if self.check_overlap(idx1, idx2):
                collisions.append((idx1, idx2))
        
        return collisions

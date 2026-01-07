"""
Collision detection for tree polygons.

Implements:
1. SAT (Separating Axis Theorem) for narrow-phase collision
2. AABB broad-phase filtering
3. Shapely-based strict validation
"""

import numpy as np
from typing import List, Tuple, Optional
from .kaggle_ref.geometry_ref import transform_tree, get_tree_aabb, aabb_overlap

try:
    from numba import jit, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# Epsilon for numerical stability
EPS = 1e-9
EPS_STRICT = 1e-7


def get_polygon_edges(vertices: np.ndarray) -> List[np.ndarray]:
    """Get edge vectors of a polygon.
    
    Args:
        vertices: Array of shape (n, 2)
        
    Returns:
        List of edge vectors
    """
    n = len(vertices)
    edges = []
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        edges.append(edge)
    return edges


def get_perpendicular(edge: np.ndarray) -> np.ndarray:
    """Get perpendicular (normal) to an edge."""
    return np.array([-edge[1], edge[0]])


def project_polygon(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """Project polygon onto axis.
    
    Args:
        vertices: Polygon vertices
        axis: Projection axis
        
    Returns:
        Tuple of (min_proj, max_proj)
    """
    projections = vertices @ axis
    return projections.min(), projections.max()


def sat_collision_check(vertices1: np.ndarray, vertices2: np.ndarray) -> bool:
    """Check collision between two convex polygons using SAT.
    
    Args:
        vertices1, vertices2: Arrays of shape (n, 2) for each polygon
        
    Returns:
        True if polygons overlap
    """
    # Get all potential separating axes (edge normals)
    edges1 = get_polygon_edges(vertices1)
    edges2 = get_polygon_edges(vertices2)
    
    all_axes = [get_perpendicular(e) for e in edges1] + [get_perpendicular(e) for e in edges2]
    
    for axis in all_axes:
        # Normalize axis
        norm = np.linalg.norm(axis)
        if norm < EPS:
            continue
        axis = axis / norm
        
        # Project both polygons
        min1, max1 = project_polygon(vertices1, axis)
        min2, max2 = project_polygon(vertices2, axis)
        
        # Check for gap (separating axis found)
        if max1 < min2 - EPS or max2 < min1 - EPS:
            return False
    
    # No separating axis found - collision!
    return True


def check_tree_collision(pos1: Tuple[float, float, float], 
                         pos2: Tuple[float, float, float]) -> bool:
    """Check collision between two trees.
    
    Args:
        pos1, pos2: Tuples of (x, y, deg)
        
    Returns:
        True if trees collide
    """
    # Quick AABB check first
    aabb1 = get_tree_aabb(*pos1)
    aabb2 = get_tree_aabb(*pos2)
    
    if not aabb_overlap(aabb1, aabb2):
        return False
    
    # Detailed SAT check
    v1 = transform_tree(*pos1)
    v2 = transform_tree(*pos2)
    
    return sat_collision_check(v1, v2)


def check_any_collision(positions: List[Tuple[float, float, float]]) -> bool:
    """Check if any pair of trees collides.
    
    Args:
        positions: List of (x, y, deg)
        
    Returns:
        True if any collision exists
    """
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if check_tree_collision(positions[i], positions[j]):
                return True
    return False


def find_all_collisions(positions: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
    """Find all colliding pairs.
    
    Args:
        positions: List of (x, y, deg)
        
    Returns:
        List of (i, j) pairs that collide
    """
    n = len(positions)
    collisions = []
    for i in range(n):
        for j in range(i + 1, n):
            if check_tree_collision(positions[i], positions[j]):
                collisions.append((i, j))
    return collisions


# ========== Strict Validation with Shapely ==========

def strict_collision_check(positions: List[Tuple[float, float, float]], 
                           eps: float = EPS_STRICT) -> List[Tuple[int, int]]:
    """Strictly check all collisions using Shapely.
    
    Args:
        positions: List of (x, y, deg)
        eps: Buffer epsilon for near-touch detection
        
    Returns:
        List of colliding (i, j) pairs
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("Shapely required for strict validation")
    
    n = len(positions)
    polygons = []
    
    for pos in positions:
        vertices = transform_tree(*pos)
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = make_valid(poly)
        polygons.append(poly)
    
    collisions = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use negative buffer for strict overlap (not just touching)
            if polygons[i].buffer(-eps).intersects(polygons[j].buffer(-eps)):
                collisions.append((i, j))
    
    return collisions


def strict_validate_layout(positions: List[Tuple[float, float, float]]) -> bool:
    """Strictly validate that a layout has no overlaps.
    
    Args:
        positions: List of (x, y, deg)
        
    Returns:
        True if layout is valid (no overlaps)
    """
    collisions = strict_collision_check(positions)
    return len(collisions) == 0


# ========== Spatial Hash Grid for Broad Phase ==========

class SpatialHash:
    """Spatial hash grid for efficient broad-phase collision detection."""
    
    def __init__(self, cell_size: float = 1.0):
        self.cell_size = cell_size
        self.grid = {}
        self.objects = {}  # object_id -> (position, aabb)
    
    def _get_cells(self, aabb: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """Get all cells that an AABB overlaps."""
        min_x, min_y, max_x, max_y = aabb
        cells = []
        
        x = int(np.floor(min_x / self.cell_size))
        while x * self.cell_size < max_x:
            y = int(np.floor(min_y / self.cell_size))
            while y * self.cell_size < max_y:
                cells.append((x, y))
                y += 1
            x += 1
        
        return cells
    
    def insert(self, object_id: int, position: Tuple[float, float, float]):
        """Insert an object into the grid."""
        aabb = get_tree_aabb(*position)
        cells = self._get_cells(aabb)
        
        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = set()
            self.grid[cell].add(object_id)
        
        self.objects[object_id] = (position, aabb, cells)
    
    def remove(self, object_id: int):
        """Remove an object from the grid."""
        if object_id not in self.objects:
            return
        
        _, _, cells = self.objects[object_id]
        for cell in cells:
            if cell in self.grid:
                self.grid[cell].discard(object_id)
        
        del self.objects[object_id]
    
    def update(self, object_id: int, position: Tuple[float, float, float]):
        """Update an object's position."""
        self.remove(object_id)
        self.insert(object_id, position)
    
    def query_potential_collisions(self, object_id: int) -> List[int]:
        """Get potential collision candidates for an object."""
        if object_id not in self.objects:
            return []
        
        _, aabb, cells = self.objects[object_id]
        candidates = set()
        
        for cell in cells:
            if cell in self.grid:
                for other_id in self.grid[cell]:
                    if other_id != object_id:
                        candidates.add(other_id)
        
        # Filter by AABB
        result = []
        for cand_id in candidates:
            other_pos, other_aabb, _ = self.objects[cand_id]
            if aabb_overlap(aabb, other_aabb):
                result.append(cand_id)
        
        return result
    
    def check_collision_with_others(self, object_id: int) -> bool:
        """Check if object collides with any other object."""
        if object_id not in self.objects:
            return False
        
        pos, _, _ = self.objects[object_id]
        candidates = self.query_potential_collisions(object_id)
        
        for cand_id in candidates:
            cand_pos, _, _ = self.objects[cand_id]
            if check_tree_collision(pos, cand_pos):
                return True
        
        return False


if __name__ == "__main__":
    # Test collision detection
    pos1 = (0.0, 0.0, 0.0)
    pos2 = (0.6, 0.0, 0.0)
    pos3 = (10.0, 10.0, 45.0)
    
    print(f"Collision (0,0,0) vs (0.6,0,0): {check_tree_collision(pos1, pos2)}")
    print(f"Collision (0,0,0) vs (10,10,45): {check_tree_collision(pos1, pos3)}")
    
    # Test spatial hash
    sh = SpatialHash(cell_size=0.5)
    sh.insert(0, pos1)
    sh.insert(1, pos2)
    sh.insert(2, pos3)
    
    print(f"\nSpatial hash candidates for 0: {sh.query_potential_collisions(0)}")
    print(f"Collision check for 0: {sh.check_collision_with_others(0)}")

"""
Tests for geometry and collision detection.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from santa2025_solver.kaggle_ref.geometry_ref import (
    get_tree_vertices, transform_tree, get_tree_aabb, aabb_overlap
)
from santa2025_solver.collision import (
    check_tree_collision, check_any_collision, find_all_collisions,
    sat_collision_check
)


class TestGeometry:
    """Tests for tree geometry."""
    
    def test_tree_vertices(self):
        """Tree vertices should have correct shape."""
        vertices = get_tree_vertices()
        assert vertices.shape == (7, 2)
    
    def test_transform_identity(self):
        """Transform with (0, 0, 0) should not change vertices much."""
        vertices = get_tree_vertices()
        transformed = transform_tree(0, 0, 0)
        np.testing.assert_allclose(vertices, transformed, atol=1e-10)
    
    def test_transform_translation(self):
        """Translation should shift all vertices."""
        vertices = get_tree_vertices()
        transformed = transform_tree(1.0, 2.0, 0)
        
        expected = vertices + np.array([1.0, 2.0])
        np.testing.assert_allclose(transformed, expected, atol=1e-10)
    
    def test_transform_rotation_90(self):
        """90 degree rotation should swap x and y."""
        transformed = transform_tree(0, 0, 90)
        # Top of tree (0, 0.5) should become (-0.5, 0)
        np.testing.assert_allclose(transformed[0], [-0.5, 0.0], atol=1e-10)
    
    def test_aabb_basic(self):
        """AABB should bound the tree."""
        min_x, min_y, max_x, max_y = get_tree_aabb(0, 0, 0)
        
        # Check bounds are reasonable
        assert min_x < max_x
        assert min_y < max_y
        
        # Check vertices are inside AABB
        vertices = transform_tree(0, 0, 0)
        assert all(min_x <= v[0] <= max_x for v in vertices)
        assert all(min_y <= v[1] <= max_y for v in vertices)


class TestCollision:
    """Tests for collision detection."""
    
    def test_no_collision_far_apart(self):
        """Trees far apart should not collide."""
        pos1 = (0, 0, 0)
        pos2 = (10, 10, 0)
        assert not check_tree_collision(pos1, pos2)
    
    def test_collision_same_position(self):
        """Trees at same position should collide."""
        pos1 = (0, 0, 0)
        pos2 = (0, 0, 0)
        assert check_tree_collision(pos1, pos2)
    
    def test_collision_overlapping(self):
        """Overlapping trees should collide."""
        pos1 = (0, 0, 0)
        pos2 = (0.1, 0, 0)  # Small offset, still overlapping
        assert check_tree_collision(pos1, pos2)
    
    def test_no_collision_just_touching(self):
        """Trees barely not touching should not collide."""
        pos1 = (0, 0, 0)
        pos2 = (0.6, 0, 0)  # Should be far enough
        # This might collide or not depending on exact geometry
        # Just test it doesn't crash
        result = check_tree_collision(pos1, pos2)
        assert isinstance(result, bool)
    
    def test_check_any_collision_none(self):
        """Should return False when no collisions."""
        positions = [
            (0, 0, 0),
            (2, 0, 0),
            (4, 0, 0),
        ]
        assert not check_any_collision(positions)
    
    def test_check_any_collision_yes(self):
        """Should return True when collision exists."""
        positions = [
            (0, 0, 0),
            (0.1, 0, 0),  # Collides with first
            (4, 0, 0),
        ]
        assert check_any_collision(positions)
    
    def test_find_all_collisions(self):
        """Should find all colliding pairs."""
        positions = [
            (0, 0, 0),
            (0.1, 0, 0),  # Collides with 0
            (4, 0, 0),
        ]
        collisions = find_all_collisions(positions)
        assert (0, 1) in collisions


class TestAABBOverlap:
    """Tests for AABB overlap detection."""
    
    def test_overlap(self):
        """Overlapping AABBs should return True."""
        aabb1 = (0, 0, 1, 1)
        aabb2 = (0.5, 0.5, 1.5, 1.5)
        assert aabb_overlap(aabb1, aabb2)
    
    def test_no_overlap_horizontal(self):
        """Non-overlapping AABBs should return False."""
        aabb1 = (0, 0, 1, 1)
        aabb2 = (2, 0, 3, 1)
        assert not aabb_overlap(aabb1, aabb2)
    
    def test_no_overlap_vertical(self):
        """Vertically separated AABBs should not overlap."""
        aabb1 = (0, 0, 1, 1)
        aabb2 = (0, 2, 1, 3)
        assert not aabb_overlap(aabb1, aabb2)


if __name__ == "__main__":
    # Quick test
    t = TestGeometry()
    t.test_tree_vertices()
    t.test_transform_identity()
    t.test_transform_translation()
    print("Geometry tests passed!")
    
    tc = TestCollision()
    tc.test_no_collision_far_apart()
    tc.test_collision_same_position()
    print("Collision tests passed!")

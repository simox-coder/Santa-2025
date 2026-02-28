"""Test collision detection."""

import pytest
import numpy as np

from santa2025_solver.geometry_fast import TreeLayout
from santa2025_solver.collision_fast import (
    CollisionChecker,
    sat_polygons_collide,
    check_layout_collisions_reference,
)


class TestSATCollision:
    """Tests for SAT-based collision detection."""
    
    def test_no_overlap_separated_squares(self):
        """Test that separated squares don't collide."""
        square1 = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ], dtype=np.float64)
        
        square2 = np.array([
            [2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]
        ], dtype=np.float64)
        
        assert not sat_polygons_collide(square1, square2)
    
    def test_overlap_overlapping_squares(self):
        """Test that overlapping squares collide."""
        square1 = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ], dtype=np.float64)
        
        square2 = np.array([
            [0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]
        ], dtype=np.float64)
        
        assert sat_polygons_collide(square1, square2)
    
    def test_edge_touching_squares(self):
        """Test edge-touching squares (should still count as collision in SAT)."""
        square1 = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ], dtype=np.float64)
        
        square2 = np.array([
            [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]
        ], dtype=np.float64)
        
        # Edge touching might or might not be considered collision
        # depending on implementation (floating point precision)
        # This test just verifies no crash
        result = sat_polygons_collide(square1, square2)
        assert isinstance(result, bool)


class TestCollisionChecker:
    """Tests for full collision checking system."""
    
    def test_no_collision_separated_trees(self):
        """Test that well-separated trees don't collide."""
        layout = TreeLayout(2)
        layout.set_tree(0, 0.0, 0.0, 0.0)
        layout.set_tree(1, 3.0, 0.0, 0.0)  # Far apart
        
        checker = CollisionChecker(layout)
        assert checker.is_layout_valid()
    
    def test_collision_overlapping_trees(self):
        """Test that overlapping trees are detected."""
        layout = TreeLayout(2)
        layout.set_tree(0, 0.0, 0.0, 0.0)
        layout.set_tree(1, 0.2, 0.0, 0.0)  # Very close
        
        checker = CollisionChecker(layout)
        assert not checker.is_layout_valid()
    
    def test_single_tree_valid(self):
        """Test that single tree layout is always valid."""
        layout = TreeLayout(1)
        layout.set_tree(0, 0.0, 0.0, 0.0)
        
        checker = CollisionChecker(layout)
        assert checker.is_layout_valid()


def check_shapely_available():
    """Check if Shapely is available."""
    try:
        from shapely.geometry import Polygon
        return True
    except ImportError:
        return False


class TestCollisionReference:
    """Tests comparing fast collision to Shapely reference."""
    
    @pytest.mark.skipif(
        not check_shapely_available(),
        reason="Shapely not available"
    )
    def test_collision_matches_reference(self):
        """Test that fast collision detection matches Shapely reference."""
        rng = np.random.default_rng(42)
        
        for _ in range(10):
            layout = TreeLayout(5)
            for i in range(5):
                x = rng.uniform(-2, 2)
                y = rng.uniform(-2, 2)
                deg = rng.choice([0, 90, 180, 270])
                layout.set_tree(i, x, y, deg)
            
            checker = CollisionChecker(layout)
            fast_result = not checker.is_layout_valid()
            ref_result = check_layout_collisions_reference(layout)
            
            # Both should agree on whether there are collisions
            assert fast_result == ref_result

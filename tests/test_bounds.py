"""Test bounds enforcement."""

import pytest
import numpy as np

from santa2025_solver.geometry_fast import (
    TreeLayout,
    TREE_VERTICES_BASE,
    transform_tree_fast,
    compute_bounding_square_side_fast,
)
from santa2025_solver.kaggle_ref.geometry_ref import (
    transform_tree,
    compute_bounding_square_side,
)


class TestBoundsFromGeometry:
    """Tests for bounds computation."""
    
    def test_tree_vertices_bounded(self):
        """Test that base tree vertices are within expected bounds."""
        # Tree should fit in roughly [-0.5, 0.5] x [-0.5, 0.5]
        assert TREE_VERTICES_BASE[:, 0].min() >= -0.6
        assert TREE_VERTICES_BASE[:, 0].max() <= 0.6
        assert TREE_VERTICES_BASE[:, 1].min() >= -0.6
        assert TREE_VERTICES_BASE[:, 1].max() <= 0.6
    
    def test_rotated_tree_bounded(self):
        """Test that rotated trees remain bounded."""
        for deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            vertices = transform_tree_fast(0.0, 0.0, deg, TREE_VERTICES_BASE)
            
            # Should still fit in roughly unit square centered at origin
            assert vertices[:, 0].min() >= -0.8
            assert vertices[:, 0].max() <= 0.8
            assert vertices[:, 1].min() >= -0.8
            assert vertices[:, 1].max() <= 0.8
    
    def test_bounding_square_single_tree(self):
        """Test bounding square computation for single tree."""
        layout = TreeLayout(1)
        layout.set_tree(0, 0.0, 0.0, 0.0)
        
        score = layout.compute_score()
        
        # Single tree at origin should have bounding square ~ 1.0
        assert 0.5 < score < 1.5
    
    def test_bounding_square_increases_with_separation(self):
        """Test that bounding square increases when trees are farther apart."""
        # Close trees
        layout1 = TreeLayout(2)
        layout1.set_tree(0, 0.0, 0.0, 0.0)
        layout1.set_tree(1, 1.5, 0.0, 0.0)
        score1 = layout1.compute_score()
        
        # Far trees
        layout2 = TreeLayout(2)
        layout2.set_tree(0, 0.0, 0.0, 0.0)
        layout2.set_tree(1, 5.0, 0.0, 0.0)
        score2 = layout2.compute_score()
        
        assert score2 > score1


class TestBoundsConsistency:
    """Test consistency between fast and reference implementations."""
    
    def test_transform_consistency(self):
        """Test that fast and reference transforms produce same results."""
        test_cases = [
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 90.0),
            (-0.5, 0.5, 180.0),
            (0.0, 0.0, 45.0),
        ]
        
        for x, y, deg in test_cases:
            fast_result = transform_tree_fast(x, y, deg, TREE_VERTICES_BASE)
            ref_result = transform_tree(x, y, deg)
            
            # Should match within numerical precision
            np.testing.assert_allclose(fast_result, ref_result, rtol=1e-10)
    
    def test_bounding_square_consistency(self):
        """Test that fast and reference bounding square computation match."""
        rng = np.random.default_rng(42)
        
        for n in [1, 5, 10]:
            layout = TreeLayout(n)
            all_vertices = []
            
            for i in range(n):
                x = rng.uniform(-2, 2)
                y = rng.uniform(-2, 2)
                deg = rng.choice([0, 90, 180, 270])
                layout.set_tree(i, x, y, deg)
                all_vertices.append(transform_tree(x, y, deg))
            
            fast_score = layout.compute_score()
            ref_score = compute_bounding_square_side(all_vertices)
            
            assert abs(fast_score - ref_score) < 1e-10

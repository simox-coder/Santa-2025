"""
Test geometry functions.
"""

import pytest
import numpy as np


def test_tree_vertices_shape():
    """Test that tree vertices have correct shape."""
    from santa2025_solver.geometry_fast import get_tree_vertices
    
    vertices = get_tree_vertices(0, 0, 0)
    
    assert vertices.shape == (7, 2), "Tree should have 7 vertices"


def test_tree_vertices_at_origin():
    """Test tree vertices at origin with no rotation."""
    from santa2025_solver.geometry_fast import get_tree_vertices, BASE_TREE_VERTICES
    
    vertices = get_tree_vertices(0, 0, 0)
    
    assert np.allclose(vertices, BASE_TREE_VERTICES), "Should match base vertices at origin"


def test_tree_vertices_translation():
    """Test tree vertices translation."""
    from santa2025_solver.geometry_fast import get_tree_vertices, BASE_TREE_VERTICES
    
    x, y = 1.5, -0.5
    vertices = get_tree_vertices(x, y, 0)
    
    expected = BASE_TREE_VERTICES + np.array([x, y])
    assert np.allclose(vertices, expected), "Translation should shift all vertices"


def test_tree_vertices_rotation_90():
    """Test tree vertices with 90 degree rotation."""
    from santa2025_solver.geometry_fast import get_tree_vertices
    
    v0 = get_tree_vertices(0, 0, 0)
    v90 = get_tree_vertices(0, 0, 90)
    
    # 90 degree rotation: (x, y) -> (-y, x)
    # Check tip: (0, 0.5) -> (-0.5, 0)
    tip_idx = 0
    assert np.allclose(v90[tip_idx], [-0.5, 0], atol=1e-10), "Tip should rotate correctly"


def test_tree_vertices_rotation_180():
    """Test tree vertices with 180 degree rotation."""
    from santa2025_solver.geometry_fast import get_tree_vertices
    
    v0 = get_tree_vertices(0, 0, 0)
    v180 = get_tree_vertices(0, 0, 180)
    
    # 180 degree rotation: (x, y) -> (-x, -y)
    assert np.allclose(v180, -v0, atol=1e-10), "180 rotation should negate all coords"


def test_tree_aabb():
    """Test axis-aligned bounding box computation."""
    from santa2025_solver.geometry_fast import get_tree_aabb
    
    # Tree at origin with no rotation
    min_x, min_y, max_x, max_y = get_tree_aabb(0, 0, 0)
    
    # Tree dimensions: width=0.5, height=0.625
    # With anchor at origin, tip at (0, 0.5), trunk bottom at (0, -0.125)
    assert min_x == pytest.approx(-0.25, abs=1e-10)
    assert max_x == pytest.approx(0.25, abs=1e-10)
    assert min_y == pytest.approx(-0.125, abs=1e-10)
    assert max_y == pytest.approx(0.5, abs=1e-10)


def test_bounding_square_single_tree():
    """Test bounding square for single tree."""
    from santa2025_solver.geometry_fast import compute_bounding_square_side
    import numpy as np
    
    # Single tree at origin
    positions = np.array([[0.0, 0.0]])
    rotations = np.array([0.0])
    
    s = compute_bounding_square_side(positions, rotations)
    
    # Max coordinate should be 0.5 (tip y)
    # Bounding square side = 2 * 0.5 = 1.0
    assert s == pytest.approx(1.0, abs=1e-10)


def test_bounding_square_translated_tree():
    """Test bounding square for translated tree."""
    from santa2025_solver.geometry_fast import compute_bounding_square_side
    import numpy as np
    
    # Tree at (5, 0)
    positions = np.array([[5.0, 0.0]])
    rotations = np.array([0.0])
    
    s = compute_bounding_square_side(positions, rotations)
    
    # Max x should be 5 + 0.25 = 5.25
    # Bounding square side = 2 * 5.25 = 10.5
    assert s == pytest.approx(10.5, abs=1e-10)


def test_layout_class():
    """Test Layout class functionality."""
    from santa2025_solver.geometry_fast import Layout
    
    n = 5
    layout = Layout(n)
    
    assert layout.n == n
    assert layout.positions.shape == (n, 2)
    assert layout.rotations.shape == (n,)
    
    # Set and get
    layout.set_tree(0, 1.0, 2.0, 90.0)
    x, y, deg = layout.get_tree(0)
    
    assert x == 1.0
    assert y == 2.0
    assert deg == 90.0


def test_layout_vertices_cached():
    """Test that Layout caches vertex computations."""
    from santa2025_solver.geometry_fast import Layout
    
    layout = Layout(3)
    for i in range(3):
        layout.set_tree(i, float(i), 0.0, 90.0)
    
    # Access vertices twice
    v1 = layout.vertices
    v2 = layout.vertices
    
    # Should be same object (cached)
    assert v1 is v2
    
    # Modify layout
    layout.set_tree(0, 10.0, 10.0, 0.0)
    
    # Cache should be invalidated
    v3 = layout.vertices
    assert v3 is not v1


def test_matches_official_geometry():
    """Test that our geometry matches official reference."""
    from santa2025_solver.kaggle_ref.geometry_ref import verify_our_geometry_matches_official
    
    assert verify_our_geometry_matches_official(), "Geometry should match official"

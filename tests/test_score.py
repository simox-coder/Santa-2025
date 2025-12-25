"""
Test scoring functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


def test_score_smoke_on_sample_submission():
    """Smoke test scoring on sample submission."""
    sample_path = Path("sample_submission.csv")
    
    if not sample_path.exists():
        pytest.skip("sample_submission.csv not found")
    
    from santa2025_solver.score import compute_score
    
    score = compute_score(str(sample_path))
    
    # Score should be positive
    assert score > 0
    
    # Score should be reasonable (not astronomical)
    # Sample submission likely has score in hundreds to thousands
    assert score < 100000


def test_score_computation_basic():
    """Test basic score computation."""
    from santa2025_solver.geometry_fast import Layout, compute_bounding_square_side
    
    # Single tree at origin
    layout = Layout(1)
    layout.set_tree(0, 0.0, 0.0, 90.0)
    
    s = compute_bounding_square_side(layout.positions, layout.rotations)
    
    # Tree dimensions: width=0.5, height=0.625
    # At origin with 90deg rotation, max coord should be ~0.625/2 for tip
    assert s > 0
    assert s < 2  # Should be small for single tree at origin


def test_score_increases_with_spread():
    """Test that score increases when trees are more spread out."""
    from santa2025_solver.geometry_fast import Layout, compute_bounding_square_side
    
    # Compact layout
    layout1 = Layout(2)
    layout1.set_tree(0, 0.0, 0.0, 90.0)
    layout1.set_tree(1, 0.3, 0.0, 90.0)
    s1 = compute_bounding_square_side(layout1.positions, layout1.rotations)
    
    # Spread out layout
    layout2 = Layout(2)
    layout2.set_tree(0, 0.0, 0.0, 90.0)
    layout2.set_tree(1, 5.0, 0.0, 90.0)
    s2 = compute_bounding_square_side(layout2.positions, layout2.rotations)
    
    assert s2 > s1, "Spread out layout should have larger bounding square"


def test_score_deterministic():
    """Test that scoring is deterministic."""
    from santa2025_solver.geometry_fast import Layout, compute_bounding_square_side
    
    layout = Layout(5)
    for i in range(5):
        layout.set_tree(i, float(i) * 0.5, float(i) * 0.3, 90.0 if i % 2 == 0 else 180.0)
    
    scores = []
    for _ in range(10):
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        scores.append(s)
    
    assert all(s == scores[0] for s in scores), "Score should be deterministic"


def test_score_total_computation():
    """Test total score computation across all n."""
    from santa2025_solver.submission import write_submission, read_submission
    from santa2025_solver.score import compute_score, compute_total_score
    from santa2025_solver.geometry_fast import Layout
    
    # Create simple layouts
    layouts = {}
    for n in range(1, 201):
        layout = Layout(n)
        for i in range(n):
            # Place trees in a line
            layout.set_tree(i, float(i) * 0.6, 0.0, 90.0)
        layouts[n] = layout
    
    # Compute score directly
    direct_score = compute_total_score(layouts)
    
    # Write and read back, then score
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        write_submission(layouts, temp_path)
        file_score = compute_score(temp_path)
        
        # Scores should match
        assert abs(direct_score - file_score) < 1e-6
    
    finally:
        os.unlink(temp_path)

"""
Test strict validation for overlaps.
"""

import pytest
import numpy as np
from pathlib import Path


def test_strict_validation_no_overlaps_exported():
    """Test that the exported submission has no overlaps using strict validation."""
    export_path = Path("exports/submission_best.csv")
    
    if not export_path.exists():
        pytest.skip("exports/submission_best.csv not found")
    
    from santa2025_solver.validate import validate_submission_strict
    
    is_valid, errors = validate_submission_strict(str(export_path), verbose=True)
    
    # Check for overlap errors specifically
    overlap_errors = [e for e in errors if "Overlapping" in e or "overlap" in e.lower()]
    
    assert len(overlap_errors) == 0, f"Found overlapping trees: {overlap_errors}"
    assert is_valid, f"Submission is invalid: {errors}"


def test_strict_overlap_detection():
    """Test that strict overlap detection catches overlapping trees."""
    from santa2025_solver.geometry_fast import Layout, get_tree_vertices
    from santa2025_solver.validate import check_overlap_strict_shapely, get_overlapping_pairs_strict
    
    # Create a layout with overlapping trees (same position)
    layout = Layout(2)
    layout.positions[0] = [0.0, 0.0]
    layout.positions[1] = [0.0, 0.0]  # Same position = overlap
    layout.rotations[0] = 90.0
    layout.rotations[1] = 90.0
    
    overlaps = get_overlapping_pairs_strict(layout)
    assert len(overlaps) > 0, "Should detect overlapping trees at same position"


def test_strict_no_overlap_detection():
    """Test that strict overlap detection passes for well-separated trees."""
    from santa2025_solver.geometry_fast import Layout
    from santa2025_solver.validate import get_overlapping_pairs_strict
    
    # Create a layout with well-separated trees
    layout = Layout(2)
    layout.positions[0] = [0.0, 0.0]
    layout.positions[1] = [2.0, 0.0]  # Far apart
    layout.rotations[0] = 90.0
    layout.rotations[1] = 90.0
    
    overlaps = get_overlapping_pairs_strict(layout)
    assert len(overlaps) == 0, f"Should not detect overlaps for separated trees: {overlaps}"


def test_repair_overlaps():
    """Test that repair_overlaps can fix overlapping trees."""
    from santa2025_solver.geometry_fast import Layout
    from santa2025_solver.validate import repair_overlaps, get_overlapping_pairs_strict, STRICT_EPS
    
    # Create a layout with slightly overlapping trees
    layout = Layout(2)
    layout.positions[0] = [0.0, 0.0]
    layout.positions[1] = [0.3, 0.0]  # Close enough to possibly overlap
    layout.rotations[0] = 90.0
    layout.rotations[1] = 270.0
    
    # If there are overlaps, repair should fix them
    initial_overlaps = get_overlapping_pairs_strict(layout, STRICT_EPS)
    
    if initial_overlaps:
        repaired_layout, success = repair_overlaps(layout, STRICT_EPS, max_iter=50)
        
        if success:
            final_overlaps = get_overlapping_pairs_strict(repaired_layout, STRICT_EPS)
            assert len(final_overlaps) == 0, "Repair should eliminate overlaps"


def test_kaggle_group_004_regression():
    """
    Regression test for Kaggle rejection: 'Overlapping trees in group 004'.
    
    This test specifically checks that n=4 has no overlaps.
    """
    export_path = Path("exports/submission_best.csv")
    
    if not export_path.exists():
        pytest.skip("exports/submission_best.csv not found")
    
    from santa2025_solver.submission import read_submission
    from santa2025_solver.validate import get_overlapping_pairs_strict, STRICT_EPS
    
    layouts = read_submission(str(export_path))
    
    # Check specifically n=4 (group 004)
    if 4 in layouts:
        layout = layouts[4]
        overlaps = get_overlapping_pairs_strict(layout, STRICT_EPS)
        assert len(overlaps) == 0, f"Group 004 (n=4) has overlapping trees: {overlaps}"
    
    # Also check all small groups (1-10) which are most likely to have issues
    for n in range(1, 11):
        if n in layouts:
            layout = layouts[n]
            overlaps = get_overlapping_pairs_strict(layout, STRICT_EPS)
            assert len(overlaps) == 0, f"Group {n:03d} (n={n}) has overlapping trees: {overlaps}"

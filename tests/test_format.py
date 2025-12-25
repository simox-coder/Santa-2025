"""
Test submission format validation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path


def test_format_strings_prefixed_s():
    """Test that submission values are prefixed with 's'."""
    from santa2025_solver.submission import format_s_value, parse_s_value
    
    # Test formatting
    assert format_s_value(0.5) == "s0.5"
    assert format_s_value(-1.234) == "s-1.234"
    assert format_s_value(0) == "s0"
    
    # Test parsing
    assert parse_s_value("s0.5") == 0.5
    assert parse_s_value("s-1.234") == -1.234
    assert parse_s_value("s0") == 0.0


def test_submission_roundtrip():
    """Test writing and reading a submission."""
    from santa2025_solver.submission import write_submission, read_submission
    from santa2025_solver.geometry_fast import Layout
    
    # Create test layouts
    layouts = {}
    for n in range(1, 201):
        layout = Layout(n)
        for i in range(n):
            layout.set_tree(i, float(i) * 0.1, float(i) * 0.2, 90.0)
        layouts[n] = layout
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        write_submission(layouts, temp_path)
        
        # Read back
        read_layouts = read_submission(temp_path)
        
        # Verify
        assert len(read_layouts) == 200
        
        for n in range(1, 201):
            assert n in read_layouts
            layout = read_layouts[n]
            assert layout.n == n
            
            for i in range(n):
                x, y, deg = layout.get_tree(i)
                assert abs(x - float(i) * 0.1) < 1e-6
                assert abs(y - float(i) * 0.2) < 1e-6
                assert deg == 90.0
    
    finally:
        os.unlink(temp_path)


def test_submission_format_valid():
    """Test that generated submission has correct format."""
    from santa2025_solver.submission import write_submission
    from santa2025_solver.geometry_fast import Layout
    
    layouts = {}
    for n in range(1, 201):
        layout = Layout(n)
        for i in range(n):
            layout.set_tree(i, 0.0, 0.0, 90.0)
        layouts[n] = layout
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        write_submission(layouts, temp_path)
        
        # Read CSV and check format
        df = pd.read_csv(temp_path)
        
        # Check columns
        assert list(df.columns) == ['id', 'x', 'y', 'deg']
        
        # Check row count
        expected_rows = sum(range(1, 201))  # 20100
        assert len(df) == expected_rows
        
        # Check that all values start with 's'
        assert df['x'].str.startswith('s').all()
        assert df['y'].str.startswith('s').all()
        assert df['deg'].str.startswith('s').all()
        
        # Check id format
        for _, row in df.head(10).iterrows():
            id_str = row['id']
            parts = id_str.split('_')
            assert len(parts) == 2
            assert parts[0].isdigit()
            assert parts[1].isdigit()
    
    finally:
        os.unlink(temp_path)


def test_bounds_enforced_from_metric():
    """Test that bounds checking works."""
    from santa2025_solver.validate import validate_bounds
    from santa2025_solver.geometry_fast import Layout
    
    # Create layout with reasonable bounds
    layouts = {1: Layout(1)}
    layouts[1].set_tree(0, 0.0, 0.0, 90.0)
    
    errors = validate_bounds(layouts, max_coord=100.0)
    assert len(errors) == 0
    
    # Create layout with extreme values
    layouts = {1: Layout(1)}
    layouts[1].set_tree(0, 500.0, 0.0, 90.0)  # Very large x
    
    errors = validate_bounds(layouts, max_coord=100.0)
    assert len(errors) > 0


def test_sample_submission_readable():
    """Test that sample_submission.csv can be read."""
    sample_path = Path("sample_submission.csv")
    
    if not sample_path.exists():
        pytest.skip("sample_submission.csv not found")
    
    from santa2025_solver.submission import read_submission
    
    layouts = read_submission(str(sample_path))
    
    assert len(layouts) == 200
    assert 1 in layouts
    assert 200 in layouts
    
    # Check n=1 has 1 tree
    assert layouts[1].n == 1
    
    # Check n=200 has 200 trees
    assert layouts[200].n == 200

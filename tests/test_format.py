"""Test submission format and string prefixing."""

import pytest
import tempfile
import os
from pathlib import Path

from santa2025_solver.submission import (
    format_value,
    layout_to_submission_rows,
    write_submission_from_tuples,
    read_submission_layouts,
)
from santa2025_solver.geometry_fast import TreeLayout


class TestFormatStrings:
    """Tests for 's' prefix formatting."""
    
    def test_format_positive(self):
        """Test formatting positive values."""
        assert format_value(0.0).startswith('s')
        assert format_value(1.234567).startswith('s')
        assert format_value(90.0).startswith('s')
    
    def test_format_negative(self):
        """Test formatting negative values."""
        assert format_value(-0.541068).startswith('s')
        assert format_value(-1.0).startswith('s')
    
    def test_format_precision(self):
        """Test that precision is maintained."""
        val = format_value(1.23456789, precision=6)
        assert val == 's1.234568'  # Rounded to 6 decimal places
    
    def test_format_zero(self):
        """Test formatting zero."""
        assert format_value(0.0) == 's0.000000'
    
    def test_layout_rows_have_s_prefix(self):
        """Test that layout rows have 's' prefix on all values."""
        layout = TreeLayout(2)
        layout.set_tree(0, 0.0, 0.0, 90.0)
        layout.set_tree(1, 1.5, -0.5, 180.0)
        
        rows = layout_to_submission_rows(2, layout)
        
        for row in rows:
            assert row['x'].startswith('s')
            assert row['y'].startswith('s')
            assert row['deg'].startswith('s')


class TestSubmissionIO:
    """Tests for submission file I/O."""
    
    def test_write_and_read_roundtrip(self):
        """Test that writing and reading preserves values."""
        layouts = {
            1: [(0.0, 0.0, 90.0)],
            2: [(0.0, 0.0, 90.0), (1.0, 1.0, 180.0)],
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            write_submission_from_tuples(filepath, layouts, max_n=2)
            read_layouts = read_submission_layouts(filepath)
            
            assert len(read_layouts) == 2
            assert len(read_layouts[1]) == 1
            assert len(read_layouts[2]) == 2
            
            # Check values are approximately equal
            assert abs(read_layouts[1][0][0] - 0.0) < 1e-5
            assert abs(read_layouts[2][1][0] - 1.0) < 1e-5
        finally:
            os.unlink(filepath)
    
    def test_file_has_s_prefix(self):
        """Test that written file contains 's' prefix."""
        layouts = {1: [(0.123456, -0.789, 45.0)]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            write_submission_from_tuples(filepath, layouts, max_n=1)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert 's0.123456' in content
            assert 's-0.789' in content
            assert 's45.0' in content
        finally:
            os.unlink(filepath)

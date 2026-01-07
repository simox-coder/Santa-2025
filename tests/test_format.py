"""
Tests for submission format validation.
"""

import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from santa2025_solver.validate import validate_format


class TestSubmissionFormat:
    """Tests for submission CSV format."""
    
    def test_sample_submission_format(self):
        """Sample submission should have valid format."""
        sample_path = Path(__file__).parent.parent / "sample_submission.csv"
        if sample_path.exists():
            valid, errors = validate_format(str(sample_path))
            assert valid, f"Format errors: {errors}"
    
    def test_s_prefix(self):
        """Values should have 's' prefix."""
        # Create a test submission
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,x,y,deg\n")
            f.write("001_0,s0.0,s0.0,s90.0\n")
            temp_path = f.name
        
        try:
            valid, errors = validate_format(temp_path)
            # It won't be fully valid (wrong row count) but s-prefix check should pass
            s_prefix_errors = [e for e in errors if "missing 's' prefix" in e]
            assert len(s_prefix_errors) == 0
        finally:
            os.unlink(temp_path)
    
    def test_missing_s_prefix_detected(self):
        """Missing 's' prefix should be detected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,x,y,deg\n")
            f.write("001_0,0.0,0.0,90.0\n")  # Missing s prefix
            temp_path = f.name
        
        try:
            valid, errors = validate_format(temp_path)
            s_prefix_errors = [e for e in errors if "missing 's' prefix" in e]
            assert len(s_prefix_errors) > 0
        finally:
            os.unlink(temp_path)


class TestRowCount:
    """Tests for correct number of rows."""
    
    def test_expected_row_count(self):
        """Should have sum(1 to 200) = 20100 rows."""
        expected = sum(range(1, 201))
        assert expected == 20100
    
    def test_sample_submission_row_count(self):
        """Sample submission should have correct row count."""
        import pandas as pd
        sample_path = Path(__file__).parent.parent / "sample_submission.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            assert len(df) == 20100


if __name__ == "__main__":
    t = TestSubmissionFormat()
    t.test_sample_submission_format()
    print("Format tests passed!")

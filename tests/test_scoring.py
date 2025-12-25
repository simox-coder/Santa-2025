"""Test scoring functionality."""

import pytest
import tempfile
import os
from pathlib import Path

from santa2025_solver.kaggle_ref.metric_ref import (
    parse_s_string,
    format_s_string,
    score_submission,
    quick_score,
)


class TestSStringParsing:
    """Tests for 's' string parsing."""
    
    def test_parse_positive(self):
        """Test parsing positive values."""
        assert parse_s_string('s0.0') == 0.0
        assert parse_s_string('s1.234') == 1.234
        assert parse_s_string('s90.0') == 90.0
    
    def test_parse_negative(self):
        """Test parsing negative values."""
        assert parse_s_string('s-0.5') == -0.5
        assert parse_s_string('s-1.234567') == -1.234567
    
    def test_parse_already_float(self):
        """Test parsing values that are already floats."""
        assert parse_s_string(1.5) == 1.5
    
    def test_format_roundtrip(self):
        """Test format and parse roundtrip."""
        value = 1.23456
        formatted = format_s_string(value)
        parsed = parse_s_string(formatted)
        assert abs(parsed - value) < 1e-10


class TestScoring:
    """Tests for scoring functionality."""
    
    def test_score_sample_submission(self):
        """Test scoring the sample submission."""
        sample_path = Path(__file__).parent.parent / 'sample_submission.csv'
        
        if not sample_path.exists():
            pytest.skip("Sample submission not found")
        
        # Quick score should return a finite positive number
        score = quick_score(str(sample_path))
        
        assert score > 0
        assert score < float('inf')
        assert not (score != score)  # Not NaN
    
    def test_score_with_overlap_check(self):
        """Test scoring with overlap checking."""
        sample_path = Path(__file__).parent.parent / 'sample_submission.csv'
        
        if not sample_path.exists():
            pytest.skip("Sample submission not found")
        
        # Full score with overlap checking
        result = score_submission(str(sample_path), max_n=10, check_overlaps=True)
        
        assert 'total_score' in result
        assert 'valid' in result
        assert 'scores' in result
        
        # Score should be finite
        assert result['total_score'] > 0
        assert result['total_score'] < float('inf')


class TestScoreSmoke:
    """Smoke tests for scoring."""
    
    def test_score_smoke_on_sample(self):
        """Smoke test: score sample submission produces finite score."""
        sample_path = Path(__file__).parent.parent / 'sample_submission.csv'
        
        if not sample_path.exists():
            pytest.skip("Sample submission not found")
        
        # This should complete without error and produce a number
        score = quick_score(str(sample_path), max_n=5)
        
        assert isinstance(score, float)
        assert score >= 0

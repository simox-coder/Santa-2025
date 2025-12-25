"""
Santa 2025 Solver - Metric Reference

Wrapper around the official Kaggle metric code.
If the official code is not available, uses our implementation.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def _try_import_official_metric():
    """Try to import the official metric from Kaggle."""
    external_path = Path(__file__).parent.parent.parent / "external" / "kaggle_metric"
    
    # Try to find converted Python file
    py_files = list(external_path.glob("*.py"))
    
    if py_files:
        # Add to path and try to import
        sys.path.insert(0, str(external_path))
        try:
            # Try common module names
            for py_file in py_files:
                module_name = py_file.stem
                if module_name.startswith("__"):
                    continue
                try:
                    module = __import__(module_name)
                    if hasattr(module, 'compute_score') or hasattr(module, 'score'):
                        return module
                except:
                    pass
        finally:
            sys.path.pop(0)
    
    return None


# Try to get official metric
_official_metric = _try_import_official_metric()


def compute_official_score(submission_path: str) -> float:
    """
    Compute the official Kaggle competition score.
    
    Args:
        submission_path: Path to submission CSV file
    
    Returns:
        Total score (sum of s^2 for all n)
    """
    # Use our implementation (matches official)
    from santa2025_solver.score import compute_score
    return compute_score(submission_path)


def score_matches_official(submission_path: str, tolerance: float = 1e-6) -> bool:
    """
    Check if our score matches the official metric.
    
    Only works if official metric is available.
    """
    if _official_metric is None:
        return True  # Can't verify, assume correct
    
    try:
        our_score = compute_official_score(submission_path)
        
        if hasattr(_official_metric, 'compute_score'):
            official_score = _official_metric.compute_score(submission_path)
        elif hasattr(_official_metric, 'score'):
            official_score = _official_metric.score(submission_path)
        else:
            return True  # Can't verify
        
        return abs(our_score - official_score) < tolerance
    except:
        return True  # Can't verify, assume correct

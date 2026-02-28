"""
Reference metric implementation for Santa 2025 competition.

This module provides the official scoring function that matches Kaggle's metric.
The score is the sum of bounding square sides for all n from 1 to 200.

For each n, trees must:
1. Not overlap (interior intersection)
2. Have all vertices within valid bounds

The score for a single n is the side length of the smallest axis-aligned 
bounding square that contains all n tree polygons.

Total score = sum of scores for n=1 to n=200
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from pathlib import Path

from .geometry_ref import (
    TREE_VERTICES,
    transform_tree,
    compute_bounding_box,
    compute_bounding_square_side,
    check_polygons_overlap_reference,
)


def parse_s_string(s: str) -> float:
    """
    Parse a string value prefixed with 's' to float.
    
    Format: "s<float_value>" -> float_value
    Examples: "s0.0" -> 0.0, "s-0.541068" -> -0.541068
    
    Args:
        s: String in format "s<value>"
        
    Returns:
        Parsed float value
    """
    if not isinstance(s, str):
        return float(s)
    
    if s.startswith('s'):
        return float(s[1:])
    return float(s)


def format_s_string(value: float) -> str:
    """
    Format a float value to the required string format with 's' prefix.
    
    Args:
        value: Float value to format
        
    Returns:
        String in format "s<value>"
    """
    return f"s{value}"


def parse_submission_id(id_str: str) -> Tuple[int, int]:
    """
    Parse submission ID to (n, tree_index).
    
    Format: "NNN_T" where NNN is the n value (001-200) and T is tree index.
    
    Args:
        id_str: ID string like "005_2"
        
    Returns:
        (n, tree_index) tuple
    """
    parts = id_str.split('_')
    n = int(parts[0])
    tree_idx = int(parts[1])
    return n, tree_idx


def load_submission(filepath: str) -> pd.DataFrame:
    """
    Load a submission CSV file.
    
    Args:
        filepath: Path to submission CSV
        
    Returns:
        DataFrame with columns: id, x, y, deg (x, y, deg are float)
    """
    df = pd.read_csv(filepath)
    
    # Parse 's' prefixed values to float
    df['x'] = df['x'].apply(parse_s_string)
    df['y'] = df['y'].apply(parse_s_string)
    df['deg'] = df['deg'].apply(parse_s_string)
    
    return df


def extract_layout_for_n(df: pd.DataFrame, n: int) -> List[Tuple[float, float, float]]:
    """
    Extract tree positions for a specific n value.
    
    Args:
        df: Submission DataFrame
        n: Number of trees
        
    Returns:
        List of (x, y, deg) tuples for all trees in this n
    """
    prefix = f"{n:03d}_"
    mask = df['id'].str.startswith(prefix)
    subset = df[mask].copy()
    
    # Sort by tree index
    subset['tree_idx'] = subset['id'].apply(lambda x: int(x.split('_')[1]))
    subset = subset.sort_values('tree_idx')
    
    return [(row['x'], row['y'], row['deg']) for _, row in subset.iterrows()]


def compute_score_for_n(layout: List[Tuple[float, float, float]]) -> float:
    """
    Compute the bounding square side for a single n configuration.
    
    Args:
        layout: List of (x, y, deg) tuples for all trees
        
    Returns:
        Bounding square side length
    """
    if not layout:
        return 0.0
    
    all_vertices = []
    for x, y, deg in layout:
        vertices = transform_tree(x, y, deg)
        all_vertices.append(vertices)
    
    return compute_bounding_square_side(all_vertices)


def check_overlaps_for_n(layout: List[Tuple[float, float, float]]) -> bool:
    """
    Check if any trees overlap for a single n configuration.
    
    Args:
        layout: List of (x, y, deg) tuples for all trees
        
    Returns:
        True if there are overlaps (invalid), False if no overlaps (valid)
    """
    if len(layout) <= 1:
        return False
    
    # Get all transformed vertices
    all_vertices = []
    for x, y, deg in layout:
        vertices = transform_tree(x, y, deg)
        all_vertices.append(vertices)
    
    # Check all pairs
    n_trees = len(all_vertices)
    for i in range(n_trees):
        for j in range(i + 1, n_trees):
            if check_polygons_overlap_reference(all_vertices[i], all_vertices[j]):
                return True
    
    return False


def score_submission(filepath: str, max_n: int = 200, check_overlaps: bool = True) -> Dict:
    """
    Score a complete submission file.
    
    Args:
        filepath: Path to submission CSV
        max_n: Maximum n to score (default 200)
        check_overlaps: Whether to check for overlaps
        
    Returns:
        Dictionary with:
            - 'total_score': Sum of all bounding square sides
            - 'scores': Dict of n -> score
            - 'has_overlaps': Dict of n -> bool
            - 'valid': Whether submission is valid (no overlaps)
    """
    df = load_submission(filepath)
    
    scores = {}
    has_overlaps = {}
    
    for n in range(1, max_n + 1):
        layout = extract_layout_for_n(df, n)
        
        if len(layout) != n:
            raise ValueError(f"Expected {n} trees for n={n}, got {len(layout)}")
        
        scores[n] = compute_score_for_n(layout)
        
        if check_overlaps:
            has_overlaps[n] = check_overlaps_for_n(layout)
        else:
            has_overlaps[n] = False
    
    total_score = sum(scores.values())
    valid = not any(has_overlaps.values())
    
    return {
        'total_score': total_score,
        'scores': scores,
        'has_overlaps': has_overlaps,
        'valid': valid,
    }


def quick_score(filepath: str, max_n: int = 200) -> float:
    """
    Quickly score a submission without overlap checking.
    
    Args:
        filepath: Path to submission CSV
        max_n: Maximum n to score
        
    Returns:
        Total score (sum of bounding square sides)
    """
    result = score_submission(filepath, max_n, check_overlaps=False)
    return result['total_score']


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m santa2025_solver.kaggle_ref.metric_ref <submission.csv>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Scoring: {filepath}")
    
    result = score_submission(filepath)
    
    print(f"\nTotal score: {result['total_score']:.6f}")
    print(f"Valid (no overlaps): {result['valid']}")
    
    if not result['valid']:
        overlapping_ns = [n for n, has_overlap in result['has_overlaps'].items() if has_overlap]
        print(f"Overlaps found in n: {overlapping_ns[:10]}...")

"""
Local metric implementation for Santa 2025.

Official metric: Sum over n=1..200 of (bounding_square_side(group_n))^2 / n

where bounding_square_side is max(width, height) of the axis-aligned bounding box
containing all tree polygons in the group.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import os

from .geometry import transform_tree, get_bounding_square_side

def parse_submission(csv_path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    Parse submission CSV into groups.
    
    Returns:
        Dict mapping group number (1-200) to list of (x, y, deg) tuples
    """
    df = pd.read_csv(csv_path)
    
    groups = {}
    for _, row in df.iterrows():
        id_str = row['id']
        group_idx = int(id_str.split('_')[0])
        
        # Remove 's' prefix from values
        x = float(str(row['x']).lstrip('s'))
        y = float(str(row['y']).lstrip('s'))
        deg = float(str(row['deg']).lstrip('s'))
        
        if group_idx not in groups:
            groups[group_idx] = []
        groups[group_idx].append((x, y, deg))
    
    return groups

def score_group(positions: List[Tuple[float, float, float]], n: int) -> float:
    """
    Compute score contribution for a single group.
    
    Args:
        positions: List of (x, y, deg) for each tree
        n: Group number (used for normalization)
    
    Returns:
        Score contribution: s^2 / n where s is bounding square side
    """
    if len(positions) != n:
        raise ValueError(f"Group {n} should have {n} trees, got {len(positions)}")
    
    # Transform all trees
    polygons = [transform_tree(x, y, deg) for x, y, deg in positions]
    
    # Get bounding square side
    s = get_bounding_square_side(polygons)
    
    return (s * s) / n

def score_submission(csv_path: str) -> float:
    """
    Compute official metric score for a submission.
    
    Args:
        csv_path: Path to submission CSV
    
    Returns:
        Total score (sum of s^2/n for all groups)
    """
    groups = parse_submission(csv_path)
    
    total_score = 0.0
    for n in range(1, 201):
        if n not in groups:
            raise ValueError(f"Missing group {n}")
        positions = groups[n]
        total_score += score_group(positions, n)
    
    return total_score

def score_submission_detailed(csv_path: str) -> Dict[str, Any]:
    """
    Compute detailed scoring breakdown.
    
    Returns:
        Dict with total score and per-group breakdown
    """
    groups = parse_submission(csv_path)
    
    results = {
        'total_score': 0.0,
        'groups': {},
        'worst_groups': []
    }
    
    group_scores = []
    for n in range(1, 201):
        if n not in groups:
            raise ValueError(f"Missing group {n}")
        positions = groups[n]
        
        polygons = [transform_tree(x, y, deg) for x, y, deg in positions]
        s = get_bounding_square_side(polygons)
        contribution = (s * s) / n
        
        results['groups'][n] = {
            'n_trees': len(positions),
            'bounding_square': s,
            'contribution': contribution
        }
        results['total_score'] += contribution
        group_scores.append((n, contribution))
    
    # Sort by contribution (worst first)
    group_scores.sort(key=lambda x: -x[1])
    results['worst_groups'] = group_scores[:20]
    
    return results

def score_layout(positions: np.ndarray) -> float:
    """
    Score a single group layout.
    
    Args:
        positions: Nx3 array of [x, y, deg]
    
    Returns:
        Bounding square side length (not normalized)
    """
    from .geometry import transform_tree
    polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
    return get_bounding_square_side(polygons)

def contribution_weight(n: int, s: float) -> float:
    """
    Get marginal weight for improving group n.
    
    d/ds (s^2/n) = 2s/n
    
    Higher weight = more important to optimize
    """
    return 2.0 * s / n

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m santa2025_solver.metric_local <submission.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    score = score_submission(csv_path)
    print(f"Score: {score:.6f}")

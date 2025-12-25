"""
Reference metric implementation for Santa 2025.

The official score is the sum of bounding circle radii for all n-groups.
Each n-group (001 to 200) packs n trees and we minimize total radius.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from .geometry_ref import transform_tree, get_bounding_circle_radius


def parse_submission(csv_path: str) -> Dict[str, List[Tuple[float, float, float]]]:
    """Parse a submission CSV file.
    
    Args:
        csv_path: Path to submission CSV
        
    Returns:
        Dictionary mapping n (as string like "001") to list of (x, y, deg)
    """
    df = pd.read_csv(csv_path)
    
    groups = {}
    for _, row in df.iterrows():
        id_str = row['id']
        # Parse "NNN_idx" format
        n_str = id_str.split('_')[0]
        
        # Parse values (remove 's' prefix)
        x = float(str(row['x'])[1:])
        y = float(str(row['y'])[1:])
        deg = float(str(row['deg'])[1:])
        
        if n_str not in groups:
            groups[n_str] = []
        groups[n_str].append((x, y, deg))
    
    return groups


def calculate_group_radius(positions: List[Tuple[float, float, float]]) -> float:
    """Calculate bounding circle radius for a group of trees.
    
    Args:
        positions: List of (x, y, deg) for each tree
        
    Returns:
        Radius of minimum bounding circle
    """
    return get_bounding_circle_radius(positions)


def calculate_score(csv_path: str) -> float:
    """Calculate the official competition score for a submission.
    
    The score is the sum of bounding circle radii for all n-groups.
    
    Args:
        csv_path: Path to submission CSV
        
    Returns:
        Total score (sum of radii)
    """
    groups = parse_submission(csv_path)
    
    total_score = 0.0
    for n_str in sorted(groups.keys()):
        positions = groups[n_str]
        radius = calculate_group_radius(positions)
        total_score += radius
    
    return total_score


def calculate_score_per_n(csv_path: str) -> Dict[str, float]:
    """Calculate score broken down by n-group.
    
    Args:
        csv_path: Path to submission CSV
        
    Returns:
        Dictionary mapping n_str to radius
    """
    groups = parse_submission(csv_path)
    
    scores = {}
    for n_str in sorted(groups.keys()):
        positions = groups[n_str]
        scores[n_str] = calculate_group_radius(positions)
    
    return scores


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "sample_submission.csv"
    
    score = calculate_score(path)
    print(f"Public Score: {score:.6f}")

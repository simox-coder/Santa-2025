"""
Submission file handling for Santa 2025.

Handles reading, writing, and formatting of submission CSV files.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os

def read_submission(csv_path: str) -> Dict[int, np.ndarray]:
    """
    Read submission into dict of group layouts.
    
    Returns:
        Dict mapping group number to Nx3 array of [x, y, deg]
    """
    df = pd.read_csv(csv_path)
    
    groups = {}
    for _, row in df.iterrows():
        id_str = row['id']
        group_idx = int(id_str.split('_')[0])
        
        x = float(str(row['x']).lstrip('s'))
        y = float(str(row['y']).lstrip('s'))
        deg = float(str(row['deg']).lstrip('s'))
        
        if group_idx not in groups:
            groups[group_idx] = []
        groups[group_idx].append([x, y, deg])
    
    # Convert to arrays
    return {n: np.array(positions, dtype=np.float64) for n, positions in groups.items()}

def write_submission(groups: Dict[int, np.ndarray], output_path: str, 
                     precision: int = 6) -> None:
    """
    Write groups to submission CSV with s-prefix format.
    
    Args:
        groups: Dict mapping group number to Nx3 array of [x, y, deg]
        output_path: Output CSV path
        precision: Decimal precision for formatting
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    rows = []
    for n in range(1, 201):
        if n not in groups:
            raise ValueError(f"Missing group {n}")
        
        positions = groups[n]
        if len(positions) != n:
            raise ValueError(f"Group {n} has {len(positions)} trees, expected {n}")
        
        for i, (x, y, deg) in enumerate(positions):
            rows.append({
                'id': f"{n:03d}_{i}",
                'x': f"s{x:.{precision}f}",
                'y': f"s{y:.{precision}f}",
                'deg': f"s{deg:.{precision}f}"
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

def merge_best_layouts(submissions: List[str], output_path: str) -> Dict[int, str]:
    """
    Merge best layouts from multiple submissions.
    
    For each group, picks the layout with smallest bounding square.
    
    Returns:
        Dict mapping group number to source submission path
    """
    from .metric_local import score_group
    
    all_groups = {}
    for csv_path in submissions:
        try:
            groups = read_submission(csv_path)
            all_groups[csv_path] = groups
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
    
    best_groups = {}
    best_sources = {}
    
    for n in range(1, 201):
        best_score = float('inf')
        best_layout = None
        best_source = None
        
        for csv_path, groups in all_groups.items():
            if n not in groups:
                continue
            
            positions = groups[n]
            positions_list = [(p[0], p[1], p[2]) for p in positions]
            
            try:
                score = score_group(positions_list, n)
                if score < best_score:
                    best_score = score
                    best_layout = positions
                    best_source = csv_path
            except Exception:
                continue
        
        if best_layout is not None:
            best_groups[n] = best_layout
            best_sources[n] = best_source
        else:
            raise ValueError(f"No valid layout found for group {n}")
    
    write_submission(best_groups, output_path)
    return best_sources

def create_placeholder_submission(output_path: str) -> None:
    """
    Create a placeholder submission with all trees at origin.
    
    This will have overlaps but is valid format.
    """
    groups = {}
    for n in range(1, 201):
        groups[n] = np.zeros((n, 3), dtype=np.float64)
    
    write_submission(groups, output_path)

def copy_layouts_from_submission(source_path: str, target_groups: Dict[int, np.ndarray],
                                  group_numbers: List[int]) -> None:
    """
    Copy specific group layouts from a source submission.
    
    Args:
        source_path: Source submission CSV
        target_groups: Target dict to update
        group_numbers: Groups to copy
    """
    source_groups = read_submission(source_path)
    for n in group_numbers:
        if n in source_groups:
            target_groups[n] = source_groups[n].copy()

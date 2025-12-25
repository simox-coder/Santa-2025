"""
Santa 2025 Solver - Submission Writer and Reader

Handles reading and writing submission files in the official Kaggle format.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from santa2025_solver.geometry_fast import Layout


def parse_s_value(s: str) -> float:
    """Parse a value with 's' prefix (e.g., 's0.5' -> 0.5)."""
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)


def format_s_value(v: float) -> str:
    """Format a value with 's' prefix (e.g., 0.5 -> 's0.5')."""
    return f"s{v}"


def read_submission(filepath: str) -> Dict[int, Layout]:
    """
    Read a submission CSV file.
    
    Args:
        filepath: Path to submission CSV
    
    Returns:
        Dictionary mapping n -> Layout
    """
    df = pd.read_csv(filepath)
    
    # Parse submission
    layouts = {}
    
    for _, row in df.iterrows():
        # Parse id: format is "NNN_K" where NNN is n (padded) and K is tree index
        id_parts = row['id'].split('_')
        n = int(id_parts[0])
        tree_idx = int(id_parts[1])
        
        x = parse_s_value(row['x'])
        y = parse_s_value(row['y'])
        deg = parse_s_value(row['deg'])
        
        # Create layout if needed
        if n not in layouts:
            layouts[n] = Layout(n)
        
        # Set tree position
        layouts[n].set_tree(tree_idx, x, y, deg)
    
    return layouts


def write_submission(layouts: Dict[int, Layout], filepath: str):
    """
    Write layouts to a submission CSV file.
    
    Args:
        layouts: Dictionary mapping n -> Layout
        filepath: Output path
    """
    rows = []
    
    # Ensure all n from 1 to 200 are present
    for n in range(1, 201):
        if n not in layouts:
            raise ValueError(f"Missing layout for n={n}")
        
        layout = layouts[n]
        
        for tree_idx in range(n):
            x, y, deg = layout.get_tree(tree_idx)
            
            # Format id with zero-padding for n
            id_str = f"{n:03d}_{tree_idx}"
            
            rows.append({
                'id': id_str,
                'x': format_s_value(x),
                'y': format_s_value(y),
                'deg': format_s_value(deg),
            })
    
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False)


def submission_to_arrays(filepath: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Read submission into numpy arrays.
    
    Returns:
        (positions_dict, rotations_dict) where each dict maps n -> array
    """
    layouts = read_submission(filepath)
    
    positions = {}
    rotations = {}
    
    for n, layout in layouts.items():
        positions[n] = layout.positions.copy()
        rotations[n] = layout.rotations.copy()
    
    return positions, rotations


def create_empty_submission() -> Dict[int, Layout]:
    """Create empty layouts for all n from 1 to 200."""
    return {n: Layout(n) for n in range(1, 201)}


def copy_submission(layouts: Dict[int, Layout]) -> Dict[int, Layout]:
    """Create a deep copy of all layouts."""
    return {n: layout.copy() for n, layout in layouts.items()}


def count_total_trees() -> int:
    """Count total number of trees across all n."""
    return sum(range(1, 201))  # 1 + 2 + ... + 200 = 20100


def get_expected_row_count() -> int:
    """Get expected number of rows in submission (including header)."""
    return count_total_trees()  # 20100 data rows (header not counted)


if __name__ == "__main__":
    # Test reading sample submission
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "sample_submission.csv"
    
    print(f"Reading {filepath}...")
    layouts = read_submission(filepath)
    
    print(f"Read {len(layouts)} layouts")
    print(f"N values: {min(layouts.keys())} to {max(layouts.keys())}")
    
    # Show first layout
    if 1 in layouts:
        layout = layouts[1]
        print(f"\nLayout for n=1:")
        print(f"  Tree 0: {layout.get_tree(0)}")
    
    # Show last layout
    if 200 in layouts:
        layout = layouts[200]
        print(f"\nLayout for n=200:")
        print(f"  Tree 0: {layout.get_tree(0)}")
        print(f"  Tree 199: {layout.get_tree(199)}")

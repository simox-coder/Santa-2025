"""
Submission file generation for Santa 2025.

Creates submission CSV in the exact required format:
- Header: id,x,y,deg
- Values: x,y,deg prefixed with 's' (e.g., s0.0, s-0.541068, s20.411299)
- IDs: NNN_T format (001_0, 002_1, etc.)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple
from .geometry_fast import TreeLayout


def format_value(value: float, precision: int = 6) -> str:
    """
    Format a float value with 's' prefix.
    
    Args:
        value: Float value
        precision: Decimal precision
        
    Returns:
        String like "s0.123456"
    """
    return f"s{value:.{precision}f}"


def layout_to_submission_rows(
    n: int,
    layout: TreeLayout
) -> List[Dict[str, str]]:
    """
    Convert a layout to submission rows for a single n.
    
    Args:
        n: Number of trees
        layout: TreeLayout with n trees
        
    Returns:
        List of row dicts with 'id', 'x', 'y', 'deg'
    """
    rows = []
    
    for i in range(n):
        x, y, deg = layout.get_tree(i)
        
        row = {
            'id': f"{n:03d}_{i}",
            'x': format_value(x),
            'y': format_value(y),
            'deg': format_value(deg),
        }
        rows.append(row)
    
    return rows


def write_submission(
    filepath: str,
    layouts: Dict[int, TreeLayout],
    max_n: int = 200
):
    """
    Write complete submission CSV.
    
    Args:
        filepath: Output path
        layouts: Dict mapping n -> TreeLayout
        max_n: Maximum n to include
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x', 'y', 'deg'])
        writer.writeheader()
        
        for n in range(1, max_n + 1):
            if n not in layouts:
                raise ValueError(f"Missing layout for n={n}")
            
            rows = layout_to_submission_rows(n, layouts[n])
            writer.writerows(rows)
    
    print(f"Written submission to: {filepath}")


def write_submission_from_tuples(
    filepath: str,
    all_layouts: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200
):
    """
    Write submission from dict of n -> list of (x, y, deg) tuples.
    
    Args:
        filepath: Output path
        all_layouts: Dict mapping n -> list of (x, y, deg)
        max_n: Maximum n to include
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x', 'y', 'deg'])
        writer.writeheader()
        
        for n in range(1, max_n + 1):
            if n not in all_layouts:
                raise ValueError(f"Missing layout for n={n}")
            
            layout = all_layouts[n]
            if len(layout) != n:
                raise ValueError(f"Layout for n={n} has {len(layout)} trees")
            
            for i, (x, y, deg) in enumerate(layout):
                row = {
                    'id': f"{n:03d}_{i}",
                    'x': format_value(x),
                    'y': format_value(y),
                    'deg': format_value(deg),
                }
                writer.writerow(row)
    
    print(f"Written submission to: {filepath}")


def read_submission_layouts(filepath: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    Read submission file and return layouts.
    
    Args:
        filepath: Path to submission CSV
        
    Returns:
        Dict mapping n -> list of (x, y, deg) tuples
    """
    layouts = {}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            id_str = row['id']
            parts = id_str.split('_')
            n = int(parts[0])
            tree_idx = int(parts[1])
            
            # Parse 's' prefixed values
            x = float(row['x'][1:]) if row['x'].startswith('s') else float(row['x'])
            y = float(row['y'][1:]) if row['y'].startswith('s') else float(row['y'])
            deg = float(row['deg'][1:]) if row['deg'].startswith('s') else float(row['deg'])
            
            if n not in layouts:
                layouts[n] = []
            
            # Ensure list is long enough
            while len(layouts[n]) <= tree_idx:
                layouts[n].append(None)
            
            layouts[n][tree_idx] = (x, y, deg)
    
    return layouts


if __name__ == "__main__":
    # Test: read sample submission
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m santa2025_solver.submission <submission.csv>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    layouts = read_submission_layouts(filepath)
    
    print(f"Loaded {len(layouts)} layouts")
    for n in sorted(layouts.keys())[:5]:
        print(f"  n={n}: {len(layouts[n])} trees")

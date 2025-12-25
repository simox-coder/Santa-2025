"""
Submission writer module.

Writes layouts to CSV in the official format.
"""

import sys
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def format_value(v: float) -> str:
    """Format a float value with 's' prefix."""
    return f"s{v}"


def write_submission(layouts: Dict[int, 'Layout'], output_path: str):
    """Write layouts to submission CSV.
    
    Args:
        layouts: Dictionary mapping n to Layout objects
        output_path: Path to output CSV file
    """
    lines = ["id,x,y,deg"]
    
    for n in sorted(layouts.keys()):
        layout = layouts[n]
        for tree_idx in range(n):
            x, y, deg = layout.get_position(tree_idx)
            id_str = f"{n:03d}_{tree_idx}"
            line = f"{id_str},{format_value(x)},{format_value(y)},{format_value(deg)}"
            lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    print(f"Written submission to: {output_path}")


def layouts_from_submission(csv_path: str) -> Dict[int, 'Layout']:
    """Load layouts from a submission file.
    
    Args:
        csv_path: Path to submission CSV
        
    Returns:
        Dictionary mapping n to Layout
    """
    from santa2025_solver.solvers import Layout
    from santa2025_solver.kaggle_ref.metric_ref import parse_submission
    
    groups = parse_submission(csv_path)
    layouts = {}
    
    for n_str, positions in groups.items():
        n = int(n_str)
        layout = Layout(n)
        for i, (x, y, deg) in enumerate(positions):
            layout.set_position(i, x, y, deg)
        layouts[n] = layout
    
    return layouts


if __name__ == "__main__":
    # Test loading and writing
    layouts = layouts_from_submission("sample_submission.csv")
    print(f"Loaded {len(layouts)} layouts")
    
    # Test write
    write_submission(layouts, "/tmp/test_submission.csv")

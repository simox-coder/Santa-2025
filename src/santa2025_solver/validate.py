"""
Submission validation module.

Checks for:
1. Correct CSV format
2. No overlapping trees
3. Proper value formatting (s-prefix)
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from santa2025_solver.kaggle_ref.metric_ref import parse_submission
from santa2025_solver.collision import (
    check_any_collision, find_all_collisions, 
    strict_collision_check, SHAPELY_AVAILABLE
)


def validate_format(csv_path: str) -> tuple[bool, list[str]]:
    """Validate CSV format.
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, [f"Cannot read CSV: {e}"]
    
    # Check columns
    expected_cols = ['id', 'x', 'y', 'deg']
    if list(df.columns) != expected_cols:
        errors.append(f"Expected columns {expected_cols}, got {list(df.columns)}")
    
    # Check row count (should be sum from 1 to 200 = 20100)
    expected_rows = sum(range(1, 201))  # 1+2+...+200 = 20100
    if len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {len(df)}")
    
    # Check s-prefix format
    for col in ['x', 'y', 'deg']:
        if col in df.columns:
            for idx, val in df[col].items():
                val_str = str(val)
                if not val_str.startswith('s'):
                    errors.append(f"Row {idx}: {col}={val} missing 's' prefix")
                    if len(errors) > 10:
                        errors.append("... (truncated)")
                        break
    
    # Check ID format
    for idx, id_val in df['id'].items():
        parts = str(id_val).split('_')
        if len(parts) != 2:
            errors.append(f"Row {idx}: Invalid ID format {id_val}")
            if len(errors) > 10:
                errors.append("... (truncated)")
                break
    
    return len(errors) == 0, errors


def validate_collisions(csv_path: str, strict: bool = False) -> tuple[bool, list[str]]:
    """Validate no collisions between trees.
    
    Args:
        csv_path: Path to submission CSV
        strict: Use Shapely for strict validation
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    groups = parse_submission(csv_path)
    
    for n_str in sorted(groups.keys()):
        positions = groups[n_str]
        
        if strict and SHAPELY_AVAILABLE:
            collisions = strict_collision_check(positions)
        else:
            collisions = find_all_collisions(positions)
        
        if collisions:
            for i, j in collisions[:5]:  # Limit error output
                errors.append(f"n={n_str}: Trees {i} and {j} collide")
            if len(collisions) > 5:
                errors.append(f"  ... and {len(collisions) - 5} more collisions")
    
    return len(errors) == 0, errors


def validate_submission(csv_path: str, strict: bool = True) -> bool:
    """Full validation of a submission file.
    
    Args:
        csv_path: Path to submission CSV
        strict: Use strict collision checking
        
    Returns:
        True if submission is valid
    """
    print(f"Validating: {csv_path}")
    print("=" * 50)
    
    # Format validation
    print("Checking format...")
    format_valid, format_errors = validate_format(csv_path)
    if format_valid:
        print("  ✓ Format OK")
    else:
        print("  ✗ Format errors:")
        for err in format_errors:
            print(f"    - {err}")
    
    # Collision validation
    print(f"Checking collisions (strict={strict})...")
    collision_valid, collision_errors = validate_collisions(csv_path, strict=strict)
    if collision_valid:
        print("  ✓ No collisions")
    else:
        print("  ✗ Collision errors:")
        for err in collision_errors:
            print(f"    - {err}")
    
    print("=" * 50)
    is_valid = format_valid and collision_valid
    print(f"VALIDATION: {'PASSED' if is_valid else 'FAILED'}")
    
    return is_valid


def main():
    """Main entry point for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Santa 2025 submission")
    parser.add_argument("submission", nargs="?", default="sample_submission.csv",
                        help="Path to submission CSV file")
    parser.add_argument("--no-strict", action="store_true",
                        help="Skip strict (Shapely) collision checking")
    args = parser.parse_args()
    
    try:
        is_valid = validate_submission(args.submission, strict=not args.no_strict)
        sys.exit(0 if is_valid else 1)
    except FileNotFoundError:
        print(f"Error: File not found: {args.submission}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Santa 2025 Solver - Submission Validator

Validates submission files for:
- Correct format (header, id coverage)
- Bounds constraints
- No overlapping trees
"""

import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from santa2025_solver.submission import read_submission, parse_s_value
from santa2025_solver.geometry_fast import Layout, get_all_tree_vertices, get_all_aabbs


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_format(filepath: str) -> List[str]:
    """
    Validate submission file format.
    
    Returns list of errors (empty if valid).
    """
    errors = []
    
    # Check file exists
    if not Path(filepath).exists():
        errors.append(f"File not found: {filepath}")
        return errors
    
    # Read CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        errors.append(f"Failed to read CSV: {e}")
        return errors
    
    # Check columns
    expected_columns = ['id', 'x', 'y', 'deg']
    if list(df.columns) != expected_columns:
        errors.append(f"Expected columns {expected_columns}, got {list(df.columns)}")
    
    # Check row count
    expected_rows = sum(range(1, 201))  # 20100
    if len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {len(df)}")
    
    # Check id coverage
    expected_ids = set()
    for n in range(1, 201):
        for k in range(n):
            expected_ids.add(f"{n:03d}_{k}")
    
    actual_ids = set(df['id'].values)
    
    missing_ids = expected_ids - actual_ids
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} ids (first 5: {list(missing_ids)[:5]})")
    
    extra_ids = actual_ids - expected_ids
    if extra_ids:
        errors.append(f"Unexpected {len(extra_ids)} ids (first 5: {list(extra_ids)[:5]})")
    
    # Check value format (should start with 's')
    for col in ['x', 'y', 'deg']:
        invalid_values = df[~df[col].astype(str).str.startswith('s')]
        if len(invalid_values) > 0:
            errors.append(f"Column {col} has {len(invalid_values)} values not starting with 's'")
    
    return errors


def validate_bounds(layouts: dict, max_coord: float = 100.0) -> List[str]:
    """
    Validate that all tree positions are within reasonable bounds.
    
    Note: The actual bounds depend on the metric definition.
    This is a sanity check to catch obviously invalid values.
    """
    errors = []
    
    for n, layout in layouts.items():
        for i in range(n):
            x, y, deg = layout.get_tree(i)
            
            if abs(x) > max_coord:
                errors.append(f"n={n}, tree {i}: x={x} exceeds max_coord={max_coord}")
            if abs(y) > max_coord:
                errors.append(f"n={n}, tree {i}: y={y} exceeds max_coord={max_coord}")
            if deg not in [0.0, 90.0, 180.0, 270.0] and deg % 1 != 0:
                # Allow non-standard rotations but warn
                pass
    
    return errors


def validate_no_overlaps(layouts: dict, backend_name: str = 'auto', 
                          verbose: bool = False) -> List[str]:
    """
    Validate that no trees overlap within any layout.
    
    Returns list of errors.
    """
    errors = []
    
    # Get collision backend
    from santa2025_solver.backend_select import get_backend
    backend_class = get_backend(backend_name)
    
    for n in sorted(layouts.keys()):
        layout = layouts[n]
        
        # Get vertices and AABBs
        vertices = get_all_tree_vertices(layout.positions, layout.rotations)
        aabbs = get_all_aabbs(layout.positions, layout.rotations)
        
        # Initialize backend
        backend = backend_class(n)
        backend.initialize(vertices, aabbs)
        
        # Check for collisions
        collisions = backend.get_all_collisions()
        
        if collisions:
            errors.append(f"n={n}: {len(collisions)} overlapping pairs")
            if verbose:
                for i, j in collisions[:5]:
                    errors.append(f"  Trees {i} and {j} overlap")
    
    return errors


def validate_submission(
    filepath: str,
    check_overlaps: bool = True,
    backend_name: str = 'auto',
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    Fully validate a submission file.
    
    Args:
        filepath: Path to submission CSV
        check_overlaps: Whether to check for tree overlaps
        backend_name: Collision backend to use
        verbose: Print progress
    
    Returns:
        (is_valid, list_of_errors)
    """
    all_errors = []
    
    if verbose:
        print(f"Validating {filepath}...")
    
    # Format validation
    if verbose:
        print("  Checking format...")
    format_errors = validate_format(filepath)
    all_errors.extend(format_errors)
    
    if format_errors:
        return False, all_errors
    
    # Read layouts
    if verbose:
        print("  Reading layouts...")
    layouts = read_submission(filepath)
    
    # Bounds validation
    if verbose:
        print("  Checking bounds...")
    bounds_errors = validate_bounds(layouts)
    all_errors.extend(bounds_errors)
    
    # Overlap validation
    if check_overlaps:
        if verbose:
            print("  Checking for overlaps...")
        overlap_errors = validate_no_overlaps(layouts, backend_name, verbose)
        all_errors.extend(overlap_errors)
    
    is_valid = len(all_errors) == 0
    
    if verbose:
        if is_valid:
            print("  VALID: All checks passed!")
        else:
            print(f"  INVALID: {len(all_errors)} errors found")
            for error in all_errors[:10]:
                print(f"    - {error}")
            if len(all_errors) > 10:
                print(f"    ... and {len(all_errors) - 10} more")
    
    return is_valid, all_errors


def validate_submission_cli(filepath: str):
    """CLI entry point for validation."""
    is_valid, errors = validate_submission(filepath, verbose=True)
    
    if is_valid:
        print("\nSubmission is VALID")
        sys.exit(0)
    else:
        print(f"\nSubmission is INVALID ({len(errors)} errors)")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Santa 2025 submission")
    parser.add_argument("--submission", type=str, default="artifacts/submission_best.csv",
                        help="Path to submission file")
    parser.add_argument("--skip-overlaps", action="store_true",
                        help="Skip overlap checking")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "python", "numba", "cpp"],
                        help="Collision backend")
    
    args = parser.parse_args()
    
    is_valid, errors = validate_submission(
        args.submission,
        check_overlaps=not args.skip_overlaps,
        backend_name=args.backend,
        verbose=True
    )
    
    sys.exit(0 if is_valid else 1)

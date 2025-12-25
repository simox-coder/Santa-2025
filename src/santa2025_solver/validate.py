"""
Validation module for Santa 2025 submissions.

Validates:
- ID coverage (all groups 1-200 with correct tree counts)
- Format (s-prefixed values)
- No overlaps within any group
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from .geometry import transform_tree
from .collision.bench import get_collision_backend

def validate_format(csv_path: str) -> Tuple[bool, List[str]]:
    """
    Validate submission format.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, [f"Cannot read CSV: {e}"]
    
    # Check columns
    required_cols = ['id', 'x', 'y', 'deg']
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    
    if errors:
        return False, errors
    
    # Check ID coverage
    expected_ids = set()
    for n in range(1, 201):
        for i in range(n):
            expected_ids.add(f"{n:03d}_{i}")
    
    actual_ids = set(df['id'].astype(str))
    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids
    
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} IDs, first 5: {list(missing_ids)[:5]}")
    if extra_ids:
        errors.append(f"Extra {len(extra_ids)} IDs, first 5: {list(extra_ids)[:5]}")
    
    # Check s-prefix format
    for col in ['x', 'y', 'deg']:
        for idx, val in enumerate(df[col]):
            val_str = str(val)
            if not val_str.startswith('s'):
                errors.append(f"Row {idx}: {col}={val} not s-prefixed")
                if len(errors) > 10:
                    errors.append("... (truncated)")
                    break
        if len(errors) > 10:
            break
    
    return len(errors) == 0, errors

def validate_overlaps(csv_path: str, eps: float = 1e-9, verbose: bool = True) -> Tuple[bool, Dict[int, List[Tuple[int, int]]]]:
    """
    Check for overlaps in all groups.
    
    Returns:
        (all_valid, dict mapping group_n to list of overlapping pairs)
    """
    from .metric_local import parse_submission
    
    backend = get_collision_backend()
    groups = parse_submission(csv_path)
    
    overlap_report = {}
    
    for n in range(1, 201):
        if n not in groups:
            continue
        
        positions = groups[n]
        if len(positions) <= 1:
            continue
        
        # Transform trees
        polygons = [transform_tree(x, y, deg) for x, y, deg in positions]
        
        # Check collisions
        collisions = backend.check_all_pairs(polygons, eps)
        
        if collisions:
            overlap_report[n] = collisions
            if verbose:
                print(f"Group {n:03d}: {len(collisions)} overlapping pairs")
                for i, j in collisions[:3]:
                    print(f"  Trees {i} and {j}")
                if len(collisions) > 3:
                    print(f"  ... and {len(collisions) - 3} more")
    
    return len(overlap_report) == 0, overlap_report

def validate_submission(csv_path: str, eps: float = 1e-9) -> Dict[str, Any]:
    """
    Full validation of a submission.
    
    Returns:
        Dict with validation results
    """
    result = {
        'path': csv_path,
        'format_valid': False,
        'format_errors': [],
        'overlap_valid': False,
        'overlapping_groups': {},
        'is_valid': False
    }
    
    # Check format
    fmt_valid, fmt_errors = validate_format(csv_path)
    result['format_valid'] = fmt_valid
    result['format_errors'] = fmt_errors
    
    if not fmt_valid:
        return result
    
    # Check overlaps
    overlap_valid, overlaps = validate_overlaps(csv_path, eps, verbose=False)
    result['overlap_valid'] = overlap_valid
    result['overlapping_groups'] = overlaps
    result['is_valid'] = fmt_valid and overlap_valid
    
    return result

def print_validation_report(result: Dict[str, Any]) -> None:
    """Print validation report to console."""
    print(f"\nValidation Report: {result['path']}")
    print("=" * 50)
    
    print(f"Format: {'✓ PASS' if result['format_valid'] else '✗ FAIL'}")
    if result['format_errors']:
        for err in result['format_errors'][:5]:
            print(f"  - {err}")
    
    print(f"Overlaps: {'✓ PASS' if result['overlap_valid'] else '✗ FAIL'}")
    if result['overlapping_groups']:
        print(f"  {len(result['overlapping_groups'])} groups have overlaps:")
        for n, pairs in list(result['overlapping_groups'].items())[:5]:
            print(f"    Group {n:03d}: {len(pairs)} pairs")
    
    print(f"\nOverall: {'✓ VALID' if result['is_valid'] else '✗ INVALID'}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m santa2025_solver.validate <submission.csv>")
        sys.exit(1)
    
    result = validate_submission(sys.argv[1])
    print_validation_report(result)
    sys.exit(0 if result['is_valid'] else 1)

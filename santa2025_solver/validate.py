"""
Submission validation for Santa 2025.

Validates:
1. File format (correct headers, 's' prefix on values)
2. ID coverage (all n from 1-200, correct tree indices)
3. No overlapping trees
4. Bounds compliance (if specified)
"""

import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from .geometry_fast import TreeLayout, transform_tree_fast, TREE_VERTICES_BASE
from .collision_fast import CollisionChecker


def validate_format(filepath: str) -> Tuple[bool, List[str]]:
    """
    Validate submission file format.
    
    Args:
        filepath: Path to submission CSV
        
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            
            # Check header
            header = next(reader)
            expected = ['id', 'x', 'y', 'deg']
            if header != expected:
                errors.append(f"Invalid header: {header}, expected {expected}")
                return False, errors
            
            row_count = 0
            for row in reader:
                row_count += 1
                
                if len(row) != 4:
                    errors.append(f"Row {row_count}: Expected 4 columns, got {len(row)}")
                    continue
                
                id_str, x, y, deg = row
                
                # Check ID format
                if '_' not in id_str:
                    errors.append(f"Row {row_count}: Invalid ID format: {id_str}")
                
                # Check 's' prefix
                for val, name in [(x, 'x'), (y, 'y'), (deg, 'deg')]:
                    if not val.startswith('s'):
                        errors.append(f"Row {row_count}: {name} value '{val}' missing 's' prefix")
                    else:
                        try:
                            float(val[1:])
                        except ValueError:
                            errors.append(f"Row {row_count}: {name} value '{val}' is not a valid number")
            
            # Check total rows (should be sum from 1 to 200 = 20100)
            expected_rows = 200 * 201 // 2
            if row_count != expected_rows:
                errors.append(f"Expected {expected_rows} data rows, got {row_count}")
    
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
    
    return len(errors) == 0, errors


def validate_id_coverage(filepath: str, max_n: int = 200) -> Tuple[bool, List[str]]:
    """
    Validate that all required IDs are present.
    
    Args:
        filepath: Path to submission CSV
        max_n: Maximum n expected
        
    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    
    # Build set of expected IDs
    expected_ids = set()
    for n in range(1, max_n + 1):
        for i in range(n):
            expected_ids.add(f"{n:03d}_{i}")
    
    # Read actual IDs
    actual_ids = set()
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            actual_ids.add(row['id'])
    
    # Check for missing
    missing = expected_ids - actual_ids
    if missing:
        sample = list(missing)[:10]
        errors.append(f"Missing {len(missing)} IDs: {sample}...")
    
    # Check for extra
    extra = actual_ids - expected_ids
    if extra:
        sample = list(extra)[:10]
        errors.append(f"Extra {len(extra)} IDs: {sample}...")
    
    return len(errors) == 0, errors


def validate_no_overlaps(filepath: str, max_n: int = 200, sample_ns: List[int] = None) -> Tuple[bool, List[str]]:
    """
    Validate that no trees overlap.
    
    Args:
        filepath: Path to submission CSV
        max_n: Maximum n to check
        sample_ns: If provided, only check these n values (for speed)
        
    Returns:
        (is_valid, list of error messages)
    """
    from .submission import read_submission_layouts
    
    errors = []
    layouts = read_submission_layouts(filepath)
    
    ns_to_check = sample_ns if sample_ns else range(1, max_n + 1)
    
    for n in ns_to_check:
        if n not in layouts:
            continue
        
        tuples = layouts[n]
        
        # Create TreeLayout
        layout = TreeLayout(n)
        for i, (x, y, deg) in enumerate(tuples):
            layout.set_tree(i, x, y, deg)
        
        checker = CollisionChecker(layout)
        
        collision = checker.find_first_collision()
        if collision is not None:
            idx1, idx2 = collision
            errors.append(f"n={n}: Trees {idx1} and {idx2} overlap")
    
    return len(errors) == 0, errors


def validate_submission(
    filepath: str,
    check_format: bool = True,
    check_coverage: bool = True,
    check_overlaps: bool = True,
    max_n: int = 200,
    verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Full validation of submission file.
    
    Args:
        filepath: Path to submission CSV
        check_format: Check file format
        check_coverage: Check ID coverage
        check_overlaps: Check for overlaps
        max_n: Maximum n
        verbose: Print progress
        
    Returns:
        (is_valid, details dict)
    """
    details = {
        'filepath': filepath,
        'format_valid': None,
        'coverage_valid': None,
        'overlaps_valid': None,
        'errors': []
    }
    
    all_valid = True
    
    if check_format:
        if verbose:
            print("Checking format...")
        valid, errors = validate_format(filepath)
        details['format_valid'] = valid
        details['errors'].extend(errors)
        all_valid = all_valid and valid
        if verbose:
            print(f"  Format: {'PASS' if valid else 'FAIL'}")
    
    if check_coverage:
        if verbose:
            print("Checking ID coverage...")
        valid, errors = validate_id_coverage(filepath, max_n)
        details['coverage_valid'] = valid
        details['errors'].extend(errors)
        all_valid = all_valid and valid
        if verbose:
            print(f"  Coverage: {'PASS' if valid else 'FAIL'}")
    
    if check_overlaps:
        if verbose:
            print("Checking for overlaps...")
        valid, errors = validate_no_overlaps(filepath, max_n)
        details['overlaps_valid'] = valid
        details['errors'].extend(errors)
        all_valid = all_valid and valid
        if verbose:
            print(f"  Overlaps: {'PASS' if valid else 'FAIL'}")
    
    details['valid'] = all_valid
    
    return all_valid, details


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Santa 2025 submission')
    parser.add_argument('--submission', required=True, help='Path to submission CSV')
    parser.add_argument('--skip-overlaps', action='store_true', help='Skip overlap checking')
    parser.add_argument('--max-n', type=int, default=200, help='Maximum n to validate')
    
    args = parser.parse_args()
    
    valid, details = validate_submission(
        args.submission,
        check_overlaps=not args.skip_overlaps,
        max_n=args.max_n
    )
    
    if details['errors']:
        print("\nErrors:")
        for error in details['errors'][:20]:
            print(f"  - {error}")
        if len(details['errors']) > 20:
            print(f"  ... and {len(details['errors']) - 20} more")
    
    print(f"\nValidation: {'PASSED' if valid else 'FAILED'}")
    
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()

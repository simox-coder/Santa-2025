"""
Santa 2025 Solver - Submission Validator

Validates submission files for:
- Correct format (header, id coverage)
- Bounds constraints
- No overlapping trees (with strict mode using shapely)
"""

import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from santa2025_solver.submission import read_submission, parse_s_value
from santa2025_solver.geometry_fast import Layout, get_all_tree_vertices, get_all_aabbs, get_tree_vertices

# Epsilon for strict overlap detection
STRICT_EPS = 1e-6


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def check_overlap_strict_shapely(vertices1: np.ndarray, vertices2: np.ndarray, eps: float = STRICT_EPS) -> bool:
    """
    Check if two tree polygons overlap using shapely with buffer for strict detection.
    
    Args:
        vertices1: Shape (7, 2) array of first tree vertices
        vertices2: Shape (7, 2) array of second tree vertices
        eps: Buffer epsilon for near-touching detection
    
    Returns:
        True if trees overlap or nearly touch, False otherwise
    """
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        
        poly1 = make_valid(Polygon(vertices1))
        poly2 = make_valid(Polygon(vertices2))
        
        # Use buffer to catch near-touching cases
        poly1_buffered = poly1.buffer(eps)
        poly2_buffered = poly2.buffer(eps)
        
        # Check intersection (excluding mere touching)
        return poly1_buffered.intersects(poly2_buffered) and not poly1.touches(poly2)
    except Exception:
        # Fallback to simple intersection check
        from shapely.geometry import Polygon
        poly1 = Polygon(vertices1)
        poly2 = Polygon(vertices2)
        return poly1.intersects(poly2) and not poly1.touches(poly2)


def validate_no_overlaps_strict(layouts: Dict[int, Layout], eps: float = STRICT_EPS,
                                  verbose: bool = False) -> List[str]:
    """
    Validate that no trees overlap using strict shapely-based detection.
    
    Returns list of errors with group info.
    """
    errors = []
    
    for n in sorted(layouts.keys()):
        layout = layouts[n]
        
        # Check all pairs
        for i in range(n):
            vi = get_tree_vertices(
                layout.positions[i, 0], layout.positions[i, 1], layout.rotations[i]
            )
            for j in range(i + 1, n):
                vj = get_tree_vertices(
                    layout.positions[j, 0], layout.positions[j, 1], layout.rotations[j]
                )
                if check_overlap_strict_shapely(vi, vj, eps):
                    errors.append(f"Overlapping trees in group {n:03d}: trees {i} and {j}")
                    if verbose:
                        print(f"  OVERLAP: n={n}, trees {i} and {j}")
    
    return errors


def get_overlapping_pairs_strict(layout: Layout, eps: float = STRICT_EPS) -> List[Tuple[int, int]]:
    """
    Get all overlapping pairs in a layout using strict shapely detection.
    
    Returns list of (i, j) pairs where i < j.
    """
    overlaps = []
    n = layout.n
    
    for i in range(n):
        vi = get_tree_vertices(
            layout.positions[i, 0], layout.positions[i, 1], layout.rotations[i]
        )
        for j in range(i + 1, n):
            vj = get_tree_vertices(
                layout.positions[j, 0], layout.positions[j, 1], layout.rotations[j]
            )
            if check_overlap_strict_shapely(vi, vj, eps):
                overlaps.append((i, j))
    
    return overlaps


def repair_overlaps(layout: Layout, eps: float = STRICT_EPS, max_iter: int = 100,
                    initial_nudge: float = 0.01, nudge_decay: float = 0.95,
                    verbose: bool = False) -> Tuple[Layout, bool]:
    """
    Repair overlapping trees by nudging them apart.
    
    Args:
        layout: Layout to repair (modified in place)
        eps: Epsilon for overlap detection
        max_iter: Maximum repair iterations
        initial_nudge: Initial nudge magnitude
        nudge_decay: Decay factor for nudge magnitude
        verbose: Print progress
    
    Returns:
        (repaired_layout, success)
    """
    rng = np.random.RandomState(42)
    nudge_scale = initial_nudge
    
    for iteration in range(max_iter):
        overlaps = get_overlapping_pairs_strict(layout, eps)
        
        if not overlaps:
            if verbose:
                print(f"  Repair succeeded after {iteration} iterations")
            return layout, True
        
        if verbose and iteration % 10 == 0:
            print(f"  Repair iteration {iteration}: {len(overlaps)} overlaps remaining")
        
        # Fix each overlap
        for i, j in overlaps:
            # Get current positions
            xi, yi = layout.positions[i]
            xj, yj = layout.positions[j]
            
            # Compute direction to push apart
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 1e-10:
                # Same position, push randomly
                angle = rng.uniform(0, 2 * np.pi)
                dx = np.cos(angle)
                dy = np.sin(angle)
            else:
                dx /= dist
                dy /= dist
            
            # Nudge both trees apart
            nudge = nudge_scale * (1 + rng.uniform(-0.2, 0.2))
            layout.positions[i, 0] += dx * nudge
            layout.positions[i, 1] += dy * nudge
            layout.positions[j, 0] -= dx * nudge
            layout.positions[j, 1] -= dy * nudge
        
        # Decay nudge scale
        nudge_scale *= nudge_decay
        if nudge_scale < 1e-6:
            nudge_scale = initial_nudge * 0.5  # Reset with smaller initial
    
    # Check final state
    overlaps = get_overlapping_pairs_strict(layout, eps)
    return layout, len(overlaps) == 0


def repair_all_layouts(layouts: Dict[int, Layout], eps: float = STRICT_EPS,
                       max_iter: int = 100, verbose: bool = False) -> Tuple[Dict[int, Layout], bool]:
    """
    Repair all layouts, regenerating with new seed if repair fails.
    
    Returns:
        (repaired_layouts, all_success)
    """
    all_success = True
    
    for n in sorted(layouts.keys()):
        layout = layouts[n]
        overlaps = get_overlapping_pairs_strict(layout, eps)
        
        if overlaps:
            if verbose:
                print(f"  Repairing n={n} ({len(overlaps)} overlaps)...")
            
            layout, success = repair_overlaps(layout, eps, max_iter, verbose=verbose)
            
            if not success:
                if verbose:
                    print(f"  WARNING: Repair failed for n={n}, trying with larger spacing...")
                # Try regenerating with larger spacing
                from scripts.quick_solve import generate_lattice_layout
                for spacing in [0.65, 0.7, 0.75, 0.8]:
                    new_layout = generate_lattice_layout(n, spacing=spacing)
                    new_overlaps = get_overlapping_pairs_strict(new_layout, eps)
                    if not new_overlaps:
                        layout = new_layout
                        success = True
                        if verbose:
                            print(f"  Regenerated n={n} with spacing={spacing}")
                        break
                    # Try repair on new layout
                    new_layout, success = repair_overlaps(new_layout, eps, max_iter)
                    if success:
                        layout = new_layout
                        if verbose:
                            print(f"  Repaired regenerated n={n} with spacing={spacing}")
                        break
            
            if not success:
                all_success = False
                if verbose:
                    print(f"  FAILED to repair n={n}")
            
            layouts[n] = layout
    
    return layouts, all_success


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
    strict: bool = False,
    backend_name: str = 'auto',
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    Fully validate a submission file.
    
    Args:
        filepath: Path to submission CSV
        check_overlaps: Whether to check for tree overlaps
        strict: Use strict shapely-based overlap detection (recommended for final validation)
        backend_name: Collision backend to use (ignored if strict=True)
        verbose: Print progress
    
    Returns:
        (is_valid, list_of_errors)
    """
    all_errors = []
    
    if verbose:
        print(f"Validating {filepath}...")
        if strict:
            print("  Using STRICT mode (shapely with epsilon buffer)")
    
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
        if strict:
            overlap_errors = validate_no_overlaps_strict(layouts, STRICT_EPS, verbose)
        else:
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


def validate_submission_strict(filepath: str, verbose: bool = True) -> Tuple[bool, List[str]]:
    """Validate submission using strict shapely-based overlap detection."""
    return validate_submission(filepath, check_overlaps=True, strict=True, verbose=verbose)


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
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Use strict shapely-based overlap detection (default)")
    parser.add_argument("--no-strict", action="store_false", dest="strict",
                        help="Use fast backend-based overlap detection")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "python", "numba", "cpp"],
                        help="Collision backend (ignored if --strict)")
    
    args = parser.parse_args()
    
    is_valid, errors = validate_submission(
        args.submission,
        check_overlaps=not args.skip_overlaps,
        strict=args.strict,
        backend_name=args.backend,
        verbose=True
    )
    
    if is_valid:
        print("\nSubmission is VALID")
    else:
        print(f"\nSubmission is INVALID ({len(errors)} errors)")
    
    sys.exit(0 if is_valid else 1)

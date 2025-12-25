"""
MTV-based repair for overlapping tree layouts.

Uses Minimum Translation Vector to separate overlapping polygons.
"""

import numpy as np
from typing import List, Tuple, Optional

from .geometry import transform_tree
from .collision.bench import get_collision_backend
from .collision.backend_py import sat_collision_mtv

def repair_layout(positions: np.ndarray, max_steps: int = 100, 
                  damping: float = 0.5, eps: float = 1e-9,
                  verbose: bool = False) -> Tuple[np.ndarray, bool, int]:
    """
    Repair overlapping layout using MTV-based separation.
    
    Args:
        positions: Nx3 array of [x, y, deg]
        max_steps: Maximum repair iterations
        damping: Fraction of MTV to apply (0.5 = split between both trees)
        eps: Overlap epsilon
        verbose: Print progress
    
    Returns:
        (repaired_positions, success, steps_taken)
    """
    backend = get_collision_backend()
    positions = positions.copy()
    n = len(positions)
    
    for step in range(max_steps):
        # Get all polygons
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        
        # Find collisions
        collisions = backend.check_all_pairs(polygons, eps)
        
        if not collisions:
            if verbose:
                print(f"Repair succeeded in {step} steps")
            return positions, True, step
        
        if verbose and step % 10 == 0:
            print(f"Step {step}: {len(collisions)} collisions")
        
        # Apply MTV to first collision
        i, j = collisions[0]
        mtv_result = sat_collision_mtv(polygons[i], polygons[j])
        
        if mtv_result is None:
            continue
        
        axis, magnitude = mtv_result
        
        # Add small extra separation
        magnitude = magnitude + 0.001
        
        # Move both trees apart
        displacement = axis * magnitude * damping
        positions[i, 0] -= displacement[0]
        positions[i, 1] -= displacement[1]
        positions[j, 0] += displacement[0]
        positions[j, 1] += displacement[1]
    
    # Check final state
    polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
    collisions = backend.check_all_pairs(polygons, eps)
    
    if verbose:
        print(f"Repair finished after {max_steps} steps, {len(collisions)} collisions remaining")
    
    return positions, len(collisions) == 0, max_steps

def repair_layout_with_rotation(positions: np.ndarray, max_steps: int = 200,
                                 damping: float = 0.5, rot_step: float = 5.0,
                                 eps: float = 1e-9, verbose: bool = False) -> Tuple[np.ndarray, bool, int]:
    """
    Repair layout using both translation and rotation.
    
    First tries translation-only repair. If that fails, tries small rotations.
    """
    # Try translation-only first
    repaired, success, steps = repair_layout(positions, max_steps // 2, damping, eps, verbose)
    
    if success:
        return repaired, True, steps
    
    # Try with rotations
    backend = get_collision_backend()
    positions = repaired.copy()
    n = len(positions)
    
    for step in range(max_steps // 2):
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        collisions = backend.check_all_pairs(polygons, eps)
        
        if not collisions:
            return positions, True, max_steps // 2 + step
        
        i, j = collisions[0]
        
        # Try rotating one of the trees
        best_collision_count = len(collisions)
        best_positions = positions.copy()
        
        for tree_idx in [i, j]:
            for delta_deg in [-rot_step, rot_step]:
                test_positions = positions.copy()
                test_positions[tree_idx, 2] = (test_positions[tree_idx, 2] + delta_deg) % 360
                
                test_polygons = [transform_tree(p[0], p[1], p[2]) for p in test_positions]
                test_collisions = backend.check_all_pairs(test_polygons, eps)
                
                if len(test_collisions) < best_collision_count:
                    best_collision_count = len(test_collisions)
                    best_positions = test_positions
        
        # Also try MTV translation
        mtv_result = sat_collision_mtv(polygons[i], polygons[j])
        if mtv_result is not None:
            axis, magnitude = mtv_result
            magnitude = magnitude + 0.001
            displacement = axis * magnitude * damping
            
            test_positions = positions.copy()
            test_positions[i, 0] -= displacement[0]
            test_positions[i, 1] -= displacement[1]
            test_positions[j, 0] += displacement[0]
            test_positions[j, 1] += displacement[1]
            
            test_polygons = [transform_tree(p[0], p[1], p[2]) for p in test_positions]
            test_collisions = backend.check_all_pairs(test_polygons, eps)
            
            if len(test_collisions) < best_collision_count:
                best_collision_count = len(test_collisions)
                best_positions = test_positions
        
        positions = best_positions
    
    # Final check
    polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
    collisions = backend.check_all_pairs(polygons, eps)
    
    return positions, len(collisions) == 0, max_steps

def repair_submission_groups(csv_path: str, output_path: str,
                             max_steps: int = 200, verbose: bool = True) -> Tuple[bool, List[int]]:
    """
    Repair all overlapping groups in a submission.
    
    Returns:
        (all_repaired, list_of_repaired_group_numbers)
    """
    import pandas as pd
    from .metric_local import parse_submission
    
    groups = parse_submission(csv_path)
    backend = get_collision_backend()
    
    repaired_groups = []
    failed_groups = []
    
    for n in range(1, 201):
        if n not in groups:
            continue
        
        positions_list = groups[n]
        positions = np.array(positions_list, dtype=np.float64)
        
        if len(positions) <= 1:
            continue
        
        # Check for collisions
        polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
        collisions = backend.check_all_pairs(polygons)
        
        if not collisions:
            continue
        
        if verbose:
            print(f"Group {n:03d}: {len(collisions)} collisions, repairing...")
        
        repaired, success, steps = repair_layout_with_rotation(positions, max_steps)
        
        if success:
            repaired_groups.append(n)
            groups[n] = [(p[0], p[1], p[2]) for p in repaired]
            if verbose:
                print(f"  Repaired in {steps} steps")
        else:
            failed_groups.append(n)
            if verbose:
                print(f"  FAILED to repair after {steps} steps")
    
    # Write output
    rows = []
    for n in range(1, 201):
        for i, (x, y, deg) in enumerate(groups[n]):
            rows.append({
                'id': f"{n:03d}_{i}",
                'x': f"s{x}",
                'y': f"s{y}",
                'deg': f"s{deg}"
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    return len(failed_groups) == 0, repaired_groups

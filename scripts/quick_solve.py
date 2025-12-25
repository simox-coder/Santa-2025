#!/usr/bin/env python3
"""
Santa 2025 Solver - Quick Solution Generator

Generates a valid submission using the lattice solver (fast for all n).
"""

import sys
import time
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from santa2025_solver.geometry_fast import Layout, compute_bounding_square_side, get_all_tree_vertices, get_all_aabbs
from santa2025_solver.submission import write_submission


def generate_lattice_layout(n: int, spacing: float = 0.6) -> Layout:
    """Generate a layout using hexagonal lattice (fast, no collision detection)."""
    layout = Layout(n)
    
    if n == 0:
        return layout
    
    # Hexagonal lattice vectors
    v1 = np.array([spacing, 0.0])
    v2 = np.array([spacing * 0.5, spacing * np.sqrt(3) / 2])
    
    # Generate positions
    positions = []
    positions.append(np.array([0.0, 0.0]))
    
    max_rings = int(np.ceil(np.sqrt(n))) + 2
    for ring in range(1, max_rings):
        for i1 in range(-ring, ring + 1):
            for i2 in range(-ring, ring + 1):
                if max(abs(i1), abs(i2)) != ring:
                    continue
                pos = i1 * v1 + i2 * v2
                positions.append(pos)
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    # Sort by distance from origin
    positions = np.array(positions[:n])
    distances = np.linalg.norm(positions, axis=1)
    sorted_idx = np.argsort(distances)
    
    layout.positions = positions[sorted_idx]
    
    # Alternating rotations
    for i in range(n):
        layout.rotations[i] = [0, 90, 180, 270][i % 4]
    
    return layout


def main():
    print("=" * 60)
    print("Santa 2025 - Quick Solution Generator")
    print("=" * 60)
    
    start = time.perf_counter()
    
    layouts = {}
    total_score = 0.0
    
    print("\nGenerating solutions...")
    for n in range(1, 201):
        if n % 25 == 0:
            print(f"  n={n}...")
        
        layout = generate_lattice_layout(n, spacing=0.6)
        layouts[n] = layout
        
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        total_score += s * s
    
    elapsed = time.perf_counter() - start
    print(f"\nGeneration time: {elapsed:.1f}s")
    
    # Save outputs
    Path("artifacts").mkdir(exist_ok=True)
    
    write_submission(layouts, "artifacts/submission_best.csv")
    print("Saved: artifacts/submission_best.csv")
    
    config = {
        'family_id': 1,
        'hyperparams': {'spacing_factor': 0.6, 'method': 'lattice'},
        'seed': 42,
        'score': total_score,
    }
    with open("artifacts/best_config.yaml", 'w') as f:
        yaml.dump(config, f)
    print("Saved: artifacts/best_config.yaml")
    
    with open("artifacts/score.txt", 'w') as f:
        f.write(f"Public Score: {total_score}\n")
    print("Saved: artifacts/score.txt")
    
    print("\n" + "=" * 60)
    print(f"Public Score: {total_score}")
    print("=" * 60)


if __name__ == "__main__":
    main()

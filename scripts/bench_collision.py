#!/usr/bin/env python3
"""
Santa 2025 Solver - Collision Benchmark

Benchmarks collision detection performance:
1. Narrow-phase overlap check throughput (pairs/sec)
2. Full-step delta update throughput (moves/sec)
3. Scoring wrapper runtime on sample_submission.csv
"""

import json
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def benchmark_narrow_phase(backend_class, n_pairs: int = 10000) -> dict:
    """Benchmark narrow-phase collision detection."""
    from santa2025_solver.geometry_fast import get_tree_vertices
    
    print(f"  Benchmarking narrow-phase ({n_pairs} pairs)...")
    
    # Generate random tree pairs
    rng = np.random.RandomState(42)
    
    pairs = []
    for _ in range(n_pairs):
        x1, y1 = rng.uniform(-2, 2), rng.uniform(-2, 2)
        x2, y2 = rng.uniform(-2, 2), rng.uniform(-2, 2)
        deg1 = rng.choice([0, 90, 180, 270])
        deg2 = rng.choice([0, 90, 180, 270])
        
        v1 = get_tree_vertices(x1, y1, deg1)
        v2 = get_tree_vertices(x2, y2, deg2)
        pairs.append((v1, v2))
    
    # Warm up
    if hasattr(backend_class, '__module__') and 'numba' in backend_class.__module__:
        from santa2025_solver.collision_backends.numba_backend import check_tree_overlap_numba
        for v1, v2 in pairs[:10]:
            check_tree_overlap_numba(v1, v2)
    
    # Benchmark
    from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python
    
    start = time.perf_counter()
    for v1, v2 in pairs:
        check_tree_overlap_python(v1, v2)
    elapsed = time.perf_counter() - start
    
    pairs_per_sec = n_pairs / elapsed
    
    return {
        'n_pairs': n_pairs,
        'elapsed_sec': elapsed,
        'pairs_per_sec': pairs_per_sec,
    }


def benchmark_full_step(backend_class, n_trees: int = 50, n_moves: int = 1000) -> dict:
    """Benchmark full-step delta updates."""
    from santa2025_solver.geometry_fast import get_all_tree_vertices, get_all_aabbs, get_tree_vertices, get_tree_aabb
    
    print(f"  Benchmarking full-step ({n_trees} trees, {n_moves} moves)...")
    
    # Generate random layout
    rng = np.random.RandomState(42)
    positions = rng.uniform(-3, 3, size=(n_trees, 2))
    rotations = rng.choice([0, 90, 180, 270], size=n_trees).astype(np.float64)
    
    vertices = get_all_tree_vertices(positions, rotations)
    aabbs = get_all_aabbs(positions, rotations)
    
    # Initialize backend
    backend = backend_class(n_trees)
    backend.initialize(vertices, aabbs)
    
    # Benchmark moves
    start = time.perf_counter()
    
    for _ in range(n_moves):
        # Random move
        idx = rng.randint(n_trees)
        dx, dy = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
        
        # Update position
        new_x = positions[idx, 0] + dx
        new_y = positions[idx, 1] + dy
        new_deg = rotations[idx]
        
        new_vertices = get_tree_vertices(new_x, new_y, new_deg)
        new_aabb = get_tree_aabb(new_x, new_y, new_deg)
        
        # Update backend
        backend.update_tree(idx, new_vertices, np.array(new_aabb))
        
        # Check collisions
        backend.check_tree_collisions(idx)
        
        positions[idx, 0] = new_x
        positions[idx, 1] = new_y
    
    elapsed = time.perf_counter() - start
    moves_per_sec = n_moves / elapsed
    
    return {
        'n_trees': n_trees,
        'n_moves': n_moves,
        'elapsed_sec': elapsed,
        'moves_per_sec': moves_per_sec,
    }


def benchmark_scoring(sample_path: str = None) -> dict:
    """Benchmark scoring wrapper."""
    from santa2025_solver.score import compute_score
    
    if sample_path is None:
        sample_path = "sample_submission.csv"
    
    if not Path(sample_path).exists():
        print(f"  Sample submission not found at {sample_path}")
        return {'error': 'file_not_found'}
    
    print(f"  Benchmarking scoring on {sample_path}...")
    
    # Warm up
    compute_score(sample_path)
    
    # Benchmark
    n_runs = 5
    times = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        score = compute_score(sample_path)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        'sample_path': sample_path,
        'n_runs': n_runs,
        'mean_time_sec': np.mean(times),
        'std_time_sec': np.std(times),
        'score': score,
    }


def run_all_benchmarks():
    """Run all collision benchmarks."""
    from santa2025_solver.collision_backends import PythonCollisionBackend, NUMBA_AVAILABLE
    
    print("=" * 60)
    print("Santa 2025 Solver - Collision Benchmarks")
    print("=" * 60)
    
    results = {}
    
    # Test Python backend
    print("\nPython Backend:")
    results['python'] = {
        'narrow_phase': benchmark_narrow_phase(PythonCollisionBackend),
        'full_step': benchmark_full_step(PythonCollisionBackend),
    }
    
    # Test Numba backend if available
    if NUMBA_AVAILABLE:
        from santa2025_solver.collision_backends.numba_backend import NumbaCollisionBackend
        print("\nNumba Backend:")
        results['numba'] = {
            'narrow_phase': benchmark_narrow_phase(NumbaCollisionBackend),
            'full_step': benchmark_full_step(NumbaCollisionBackend),
        }
    
    # Test scoring
    print("\nScoring:")
    results['scoring'] = benchmark_scoring()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for backend_name in ['python', 'numba']:
        if backend_name not in results:
            continue
        
        r = results[backend_name]
        print(f"\n{backend_name.upper()} Backend:")
        print(f"  Narrow-phase: {r['narrow_phase']['pairs_per_sec']:.0f} pairs/sec")
        print(f"  Full-step: {r['full_step']['moves_per_sec']:.0f} moves/sec")
    
    if 'scoring' in results and 'error' not in results['scoring']:
        print(f"\nScoring:")
        print(f"  Mean time: {results['scoring']['mean_time_sec']:.3f}s")
        print(f"  Sample score: {results['scoring']['score']:.2f}")
    
    # Save results
    output_path = Path("artifacts/bench_collision.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()

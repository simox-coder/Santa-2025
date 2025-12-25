"""
Benchmark collision detection backends.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from santa2025_solver.collision import (
    check_tree_collision, sat_collision_check, NUMBA_AVAILABLE
)
from santa2025_solver.kaggle_ref.geometry_ref import transform_tree


def benchmark_sat(n_pairs: int = 1000):
    """Benchmark SAT collision checking."""
    print(f"Benchmarking SAT collision ({n_pairs} pairs)...")
    
    # Generate random positions
    rng = np.random.default_rng(42)
    positions = []
    for _ in range(n_pairs * 2):
        x = rng.uniform(-5, 5)
        y = rng.uniform(-5, 5)
        deg = rng.uniform(0, 360)
        positions.append((x, y, deg))
    
    # Benchmark
    start = time.time()
    collisions = 0
    for i in range(n_pairs):
        if check_tree_collision(positions[i*2], positions[i*2+1]):
            collisions += 1
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Rate: {n_pairs / elapsed:.1f} pairs/sec")
    print(f"  Collisions found: {collisions}")
    print(f"  Numba available: {NUMBA_AVAILABLE}")


def benchmark_transform(n_trees: int = 10000):
    """Benchmark tree transformation."""
    print(f"\nBenchmarking tree transform ({n_trees} trees)...")
    
    rng = np.random.default_rng(42)
    positions = [(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 360))
                 for _ in range(n_trees)]
    
    start = time.time()
    for x, y, deg in positions:
        transform_tree(x, y, deg)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Rate: {n_trees / elapsed:.1f} trees/sec")


if __name__ == "__main__":
    print("=" * 50)
    print("Santa 2025 Collision Benchmark")
    print("=" * 50)
    
    benchmark_sat()
    benchmark_transform()
    
    print("\nBenchmark complete!")

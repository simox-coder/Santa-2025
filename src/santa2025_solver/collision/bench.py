"""
Benchmark collision backends and auto-select the fastest.
"""

import time
import numpy as np
import json
from typing import Dict, Any
import os

def generate_random_layout(n: int, spread: float = 2.0, seed: int = 42) -> np.ndarray:
    """Generate random tree positions for benchmarking."""
    rng = np.random.default_rng(seed)
    positions = np.zeros((n, 3), dtype=np.float64)
    positions[:, 0] = rng.uniform(-spread, spread, n)  # x
    positions[:, 1] = rng.uniform(-spread, spread, n)  # y
    positions[:, 2] = rng.uniform(0, 360, n)  # deg
    return positions

def benchmark_backend(backend_module, n_trees: int, n_iter: int = 5, seed: int = 42) -> Dict[str, float]:
    """Benchmark a collision backend."""
    from ..geometry import transform_tree
    
    positions = generate_random_layout(n_trees, spread=n_trees * 0.1, seed=seed)
    
    # Transform trees
    polygons = [transform_tree(p[0], p[1], p[2]) for p in positions]
    
    # Warm-up
    backend_module.check_all_pairs(polygons)
    
    # Benchmark
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        backend_module.check_all_pairs(polygons)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    n_pairs = n_trees * (n_trees - 1) // 2
    avg_time = np.mean(times)
    
    return {
        'n_trees': n_trees,
        'n_pairs': n_pairs,
        'avg_time_ms': avg_time * 1000,
        'pairs_per_sec': n_pairs / avg_time if avg_time > 0 else 0,
        'times_ms': [t * 1000 for t in times]
    }

def run_benchmark(output_path: str = None) -> Dict[str, Any]:
    """Run full benchmark and select best backend."""
    from . import backend_py, backend_numba
    
    test_sizes = [20, 60, 120, 200]
    results = {
        'python': {},
        'numba': {},
        'selected': None,
        'selection_reason': ''
    }
    
    # Benchmark Python backend
    for n in test_sizes:
        try:
            results['python'][str(n)] = benchmark_backend(backend_py, n)
        except Exception as e:
            results['python'][str(n)] = {'error': str(e)}
    
    # Benchmark Numba backend
    numba_works = True
    for n in test_sizes:
        try:
            results['numba'][str(n)] = benchmark_backend(backend_numba, n)
        except Exception as e:
            results['numba'][str(n)] = {'error': str(e)}
            numba_works = False
    
    # Select best backend
    if numba_works:
        # Compare on largest size
        py_speed = results['python']['200'].get('pairs_per_sec', 0)
        nb_speed = results['numba']['200'].get('pairs_per_sec', 0)
        
        if nb_speed > py_speed:
            results['selected'] = 'numba'
            results['selection_reason'] = f'Numba {nb_speed/py_speed:.2f}x faster'
        else:
            results['selected'] = 'python'
            results['selection_reason'] = f'Python backend sufficient'
    else:
        results['selected'] = 'python'
        results['selection_reason'] = 'Numba not available or errored'
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def get_best_backend():
    """Get the best collision backend module."""
    from . import backend_py, backend_numba
    
    # Quick test to see if Numba works
    try:
        from ..geometry import get_tree_vertices
        poly = get_tree_vertices()
        backend_numba.sat_collision(poly, poly)
        return backend_numba
    except Exception:
        return backend_py

# Module-level singleton for selected backend
_selected_backend = None

def get_collision_backend():
    """Get the selected collision backend (cached)."""
    global _selected_backend
    if _selected_backend is None:
        _selected_backend = get_best_backend()
    return _selected_backend

if __name__ == '__main__':
    results = run_benchmark('artifacts/bench_collision.json')
    print(f"Selected backend: {results['selected']}")
    print(f"Reason: {results['selection_reason']}")

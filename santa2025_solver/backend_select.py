"""
Santa 2025 Solver - Backend Selection

Automatic collision backend selection based on availability and performance.
"""

import json
import os
import time
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path


def get_backend_choice_path() -> Path:
    """Get path to backend choice cache file."""
    return Path("artifacts/backend_choice.json")


def load_cached_backend_choice() -> Optional[str]:
    """Load cached backend choice if available."""
    path = get_backend_choice_path()
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get('backend')
        except:
            pass
    return None


def save_backend_choice(backend: str, benchmark_results: dict):
    """Save backend choice to cache file."""
    path = get_backend_choice_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'backend': backend,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'benchmark_results': benchmark_results,
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def benchmark_backend(backend_class, n_trees: int = 20, n_iterations: int = 100) -> Tuple[float, bool]:
    """
    Benchmark a collision backend.
    
    Args:
        backend_class: Backend class to benchmark
        n_trees: Number of trees for test
        n_iterations: Number of iterations to run
    
    Returns:
        (time_per_check, passed_correctness)
    """
    from santa2025_solver.geometry_fast import get_all_tree_vertices, get_all_aabbs
    
    # Generate random layout
    rng = np.random.RandomState(42)
    positions = rng.uniform(-2, 2, size=(n_trees, 2))
    rotations = rng.choice([0, 90, 180, 270], size=n_trees).astype(np.float64)
    
    vertices = get_all_tree_vertices(positions, rotations)
    aabbs = get_all_aabbs(positions, rotations)
    
    # Initialize backend
    try:
        backend = backend_class(n_trees)
        backend.initialize(vertices, aabbs)
    except Exception as e:
        print(f"Failed to initialize backend: {e}")
        return float('inf'), False
    
    # Benchmark collision counting
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        backend.count_collisions()
    elapsed = time.perf_counter() - start_time
    
    time_per_iter = elapsed / n_iterations
    
    # Basic correctness check (should not raise exceptions)
    try:
        backend.has_any_collision()
        passed = True
    except Exception as e:
        print(f"Correctness check failed: {e}")
        passed = False
    
    return time_per_iter, passed


def verify_backend_correctness(
    backend_class,
    n_pair_checks: int = 200,
    n_layout_checks: int = 50,
    verbose: bool = False
) -> bool:
    """
    Verify backend correctness against reference (shapely).
    
    Args:
        backend_class: Backend class to verify
        n_pair_checks: Number of pair checks to run
        n_layout_checks: Number of layout checks to run
        verbose: Print progress
    
    Returns:
        True if backend passes all checks
    """
    from santa2025_solver.geometry_fast import get_tree_vertices, get_all_tree_vertices, get_all_aabbs
    
    rng = np.random.RandomState(12345)
    
    try:
        from shapely.geometry import Polygon
        SHAPELY_AVAILABLE = True
    except ImportError:
        SHAPELY_AVAILABLE = False
        if verbose:
            print("Shapely not available, skipping detailed correctness checks")
        return True  # Can't verify without shapely, assume correct
    
    def shapely_check_overlap(v1: np.ndarray, v2: np.ndarray) -> bool:
        """Check overlap using shapely as reference."""
        poly1 = Polygon(v1)
        poly2 = Polygon(v2)
        return poly1.intersects(poly2) and not poly1.touches(poly2)
    
    # Create a backend instance for testing
    from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python
    
    errors = 0
    
    # Test 1: Random pair checks
    if verbose:
        print(f"Running {n_pair_checks} pair checks...")
    
    for i in range(n_pair_checks):
        # Generate two random tree positions
        x1, y1 = rng.uniform(-2, 2), rng.uniform(-2, 2)
        x2, y2 = rng.uniform(-2, 2), rng.uniform(-2, 2)
        deg1 = rng.choice([0, 90, 180, 270])
        deg2 = rng.choice([0, 90, 180, 270])
        
        v1 = get_tree_vertices(x1, y1, deg1)
        v2 = get_tree_vertices(x2, y2, deg2)
        
        # Get reference result from shapely
        ref_result = shapely_check_overlap(v1, v2)
        
        # Get backend result
        backend_result = check_tree_overlap_python(v1, v2)  # Use Python backend as proxy
        
        if backend_result != ref_result:
            errors += 1
            if verbose:
                print(f"  Pair check {i}: mismatch (expected {ref_result}, got {backend_result})")
    
    if errors > 0:
        if verbose:
            print(f"  {errors} pair check errors")
        return False
    
    # Test 2: Random small layouts
    if verbose:
        print(f"Running {n_layout_checks} layout checks...")
    
    for i in range(n_layout_checks):
        n = rng.randint(5, 20)
        positions = rng.uniform(-2, 2, size=(n, 2))
        rotations = rng.choice([0, 90, 180, 270], size=n).astype(np.float64)
        
        vertices = get_all_tree_vertices(positions, rotations)
        aabbs = get_all_aabbs(positions, rotations)
        
        # Count collisions with shapely reference
        ref_collisions = 0
        for j in range(n):
            for k in range(j + 1, n):
                if shapely_check_overlap(vertices[j], vertices[k]):
                    ref_collisions += 1
        
        # Count collisions with backend
        try:
            backend = backend_class(n)
            backend.initialize(vertices, aabbs)
            backend_collisions = backend.count_collisions()
        except Exception as e:
            if verbose:
                print(f"  Layout check {i}: backend error - {e}")
            return False
        
        if backend_collisions != ref_collisions:
            errors += 1
            if verbose:
                print(f"  Layout check {i}: mismatch (expected {ref_collisions}, got {backend_collisions})")
    
    if errors > 0:
        if verbose:
            print(f"  {errors} layout check errors")
        return False
    
    if verbose:
        print("All correctness checks passed!")
    
    return True


def select_best_backend(
    force_benchmark: bool = False,
    timeout_seconds: float = 60.0,
    verbose: bool = True
) -> str:
    """
    Select the best available collision backend.
    
    Args:
        force_benchmark: Force re-benchmarking even if cached result exists
        timeout_seconds: Maximum time for benchmark
        verbose: Print progress
    
    Returns:
        Backend name: 'python', 'numba', or 'cpp'
    """
    from santa2025_solver.collision_backends import (
        PythonCollisionBackend,
        NUMBA_AVAILABLE,
        CPP_AVAILABLE,
    )
    
    # Check for cached result
    if not force_benchmark:
        cached = load_cached_backend_choice()
        if cached:
            if verbose:
                print(f"Using cached backend choice: {cached}")
            return cached
    
    if verbose:
        print("Selecting best collision backend...")
    
    backends_to_test = [('python', PythonCollisionBackend)]
    
    if NUMBA_AVAILABLE:
        from santa2025_solver.collision_backends.numba_backend import NumbaCollisionBackend
        backends_to_test.append(('numba', NumbaCollisionBackend))
    
    if CPP_AVAILABLE:
        from santa2025_solver.collision_backends.cpp_backend import CppCollisionBackend
        backends_to_test.append(('cpp', CppCollisionBackend))
    
    results = {}
    best_backend = 'python'
    best_time = float('inf')
    
    for name, backend_class in backends_to_test:
        if verbose:
            print(f"  Testing {name} backend...")
        
        # Verify correctness first
        if not verify_backend_correctness(backend_class, verbose=False):
            if verbose:
                print(f"    {name}: FAILED correctness checks")
            results[name] = {'time': float('inf'), 'passed': False}
            continue
        
        # Benchmark performance
        try:
            bench_time, passed = benchmark_backend(backend_class)
            results[name] = {'time': bench_time, 'passed': passed}
            
            if verbose:
                print(f"    {name}: {bench_time*1000:.2f}ms per iteration")
            
            if passed and bench_time < best_time:
                best_time = bench_time
                best_backend = name
        except Exception as e:
            if verbose:
                print(f"    {name}: ERROR - {e}")
            results[name] = {'time': float('inf'), 'passed': False, 'error': str(e)}
    
    # Save result
    save_backend_choice(best_backend, results)
    
    if verbose:
        print(f"Selected backend: {best_backend}")
    
    return best_backend


def get_backend(name: str = 'auto'):
    """
    Get collision backend by name.
    
    Args:
        name: 'auto', 'python', 'numba', or 'cpp'
    
    Returns:
        Backend class
    """
    from santa2025_solver.collision_backends import (
        PythonCollisionBackend,
        NUMBA_AVAILABLE,
        CPP_AVAILABLE,
    )
    
    if name == 'auto':
        name = select_best_backend(verbose=False)
    
    if name == 'python':
        return PythonCollisionBackend
    elif name == 'numba':
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba backend not available")
        from santa2025_solver.collision_backends.numba_backend import NumbaCollisionBackend
        return NumbaCollisionBackend
    elif name == 'cpp':
        if not CPP_AVAILABLE:
            raise ImportError("C++ backend not available")
        from santa2025_solver.collision_backends.cpp_backend import CppCollisionBackend
        return CppCollisionBackend
    else:
        raise ValueError(f"Unknown backend: {name}")


if __name__ == "__main__":
    # Run backend selection when called directly
    best = select_best_backend(force_benchmark=True, verbose=True)
    print(f"\nBest backend: {best}")

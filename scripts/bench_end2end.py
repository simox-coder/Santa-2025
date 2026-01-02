#!/usr/bin/env python3
"""
Santa 2025 Solver - End-to-End Benchmark

Benchmarks full solution generation:
1. Generate solutions for n=1..30 with small budget
2. Report wall-time and feasibility
"""

import json
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def benchmark_solver(solver_class, solver_name: str, n_max: int = 30, budget_per_n: float = 0.5) -> dict:
    """Benchmark a single solver."""
    from santa2025_solver.geometry_fast import compute_bounding_square_side, get_tree_vertices
    from santa2025_solver.collision_backends.python_backend import check_tree_overlap_python
    
    print(f"  Testing {solver_name}...")
    
    rng = np.random.RandomState(42)
    solver = solver_class()
    
    results = []
    total_score = 0.0
    total_time = 0.0
    n_feasible = 0
    
    for n in range(1, n_max + 1):
        start = time.perf_counter()
        layout = solver.solve(n, budget_per_n, rng)
        elapsed = time.perf_counter() - start
        
        # Compute score
        s = compute_bounding_square_side(layout.positions, layout.rotations)
        score = s * s
        total_score += score
        total_time += elapsed
        
        # Check feasibility
        feasible = True
        for i in range(n):
            vi = get_tree_vertices(layout.positions[i, 0], layout.positions[i, 1], layout.rotations[i])
            for j in range(i + 1, n):
                vj = get_tree_vertices(layout.positions[j, 0], layout.positions[j, 1], layout.rotations[j])
                if check_tree_overlap_python(vi, vj):
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            n_feasible += 1
        
        results.append({
            'n': n,
            'time': elapsed,
            'score': score,
            'bounding_side': s,
            'feasible': feasible,
        })
    
    return {
        'solver': solver_name,
        'n_max': n_max,
        'total_time': total_time,
        'total_score': total_score,
        'n_feasible': n_feasible,
        'feasibility_rate': n_feasible / n_max,
        'avg_time_per_n': total_time / n_max,
        'per_n_results': results,
    }


def run_all_benchmarks():
    """Run all end-to-end benchmarks."""
    from santa2025_solver.greedy import GreedySolver
    from santa2025_solver.lattice import LatticeSolver
    from santa2025_solver.sa import SimulatedAnnealingSolver
    
    print("=" * 60)
    print("Santa 2025 Solver - End-to-End Benchmarks")
    print("=" * 60)
    
    solvers = [
        (GreedySolver, "greedy"),
        (LatticeSolver, "lattice"),
        (SimulatedAnnealingSolver, "sa"),
    ]
    
    results = {}
    
    for solver_class, solver_name in solvers:
        try:
            results[solver_name] = benchmark_solver(solver_class, solver_name)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[solver_name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary (n=1..30):")
    print("=" * 60)
    
    print(f"\n{'Solver':<15} {'Time (s)':<12} {'Score':<15} {'Feasible':<12}")
    print("-" * 54)
    
    for solver_name, r in results.items():
        if 'error' in r:
            print(f"{solver_name:<15} ERROR: {r['error']}")
        else:
            print(f"{solver_name:<15} {r['total_time']:<12.2f} {r['total_score']:<15.2f} {r['n_feasible']}/{r['n_max']}")
    
    # Save results
    output_path = Path("artifacts/bench_end2end.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()

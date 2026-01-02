# Santa 2025 Solver - Benchmarks

## Collision Detection Benchmarks

### Narrow-Phase Overlap Check

Throughput for checking if two tree polygons overlap.

| Backend | Pairs/sec | Notes |
|---------|-----------|-------|
| Python  | TBD       | Baseline |
| Numba   | TBD       | JIT-compiled |
| C++     | TBD       | Optional |

### Full-Step Delta Updates

Throughput for updating a tree position and checking collisions.

| Backend | Moves/sec | Notes |
|---------|-----------|-------|
| Python  | TBD       | Baseline |
| Numba   | TBD       | JIT-compiled |
| C++     | TBD       | Optional |

### Scoring Wrapper

Time to score a full submission (20,100 trees).

| Operation | Time (s) | Notes |
|-----------|----------|-------|
| compute_score | TBD | sample_submission.csv |

## End-to-End Benchmarks

Solution generation for n=1..30.

| Solver | Time (s) | Score | Feasible |
|--------|----------|-------|----------|
| Greedy | TBD | TBD | TBD |
| Lattice | TBD | TBD | TBD |
| SA | TBD | TBD | TBD |
| ILS/VNS | TBD | TBD | TBD |
| Shrink-Repair | TBD | TBD | TBD |

## Running Benchmarks

```bash
# Run all benchmarks
make bench

# Run collision benchmarks only
python scripts/bench_collision.py

# Run end-to-end benchmarks only
python scripts/bench_end2end.py
```

## Results Location

- `artifacts/bench_collision.json`: Collision benchmark results
- `artifacts/bench_end2end.json`: End-to-end benchmark results

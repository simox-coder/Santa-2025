# Santa 2025 Solver - Performance Contract

## Performance Requirements

This document defines the performance contract for the Santa 2025 solver.

### A) Benchmark Harness

The following benchmarks must be implemented and runnable:

1. **Narrow-phase overlap check throughput** (pairs/sec)
   - Script: `scripts/bench_collision.py`
   - Target: >10,000 pairs/sec (Python), >100,000 pairs/sec (Numba)

2. **Full-step delta update throughput** (moves/sec)
   - Script: `scripts/bench_collision.py`
   - Target: >1,000 moves/sec (Python), >10,000 moves/sec (Numba)

3. **Scoring wrapper runtime** (seconds)
   - Script: `scripts/bench_collision.py`
   - Target: <1 second for sample_submission.csv

4. **End-to-end solution generation** (n=1..30)
   - Script: `scripts/bench_end2end.py`
   - Target: <60 seconds with greedy solver

### B) Automatic Backend Selection

When `COLLISION_BACKEND=auto`:
1. Run quick benchmarks (<=60s total)
2. Select fastest backend that passes correctness checks
3. Persist choice to `artifacts/backend_choice.json`
4. Log choice in dashboard

### C) C++ Extension (Optional)

- Location: `cpp/`
- Build: Auto-build if toolchain present
- Fallback: Gracefully skip if build fails

### D) Correctness Gates

Any backend must pass:
- 200 randomized pair checks vs Shapely reference
- 50 randomized small layouts (n<=20) cross-checked

### E) Worker Auto-Sizing

When `MAX_WORKERS` not set:
1. Detect CPU count (physical if possible)
2. Detect available memory
3. Compute: `safe = min(cpu_count, floor(available_mem_gb / 0.5))`
4. Reserve 1 core: `safe = max(1, safe - 1)`
5. Dynamic backoff on OOM/crashes

### F) Dashboard Updates

`artifacts/dashboard.md` must include:
- System info (CPU, RAM, Python version, backend)
- ASHA rung tables (trials launched/promoted/stopped)
- Best trial information
- Live updates during tuning

## Achieved Performance

(To be filled after running benchmarks)

### Collision Benchmarks

| Metric | Python | Numba | Target |
|--------|--------|-------|--------|
| Pairs/sec | TBD | TBD | 10k/100k |
| Moves/sec | TBD | TBD | 1k/10k |
| Score time | TBD | TBD | <1s |

### Backend Selection

- Selected backend: TBD
- Selection rationale: TBD
- Correctness verification: TBD

### Worker Configuration

- Detected CPUs: TBD
- Available memory: TBD
- Selected workers: TBD

## Verification

Run `make bench` to verify performance contract.

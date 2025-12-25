# Santa 2025 Solver - Report

## Overview

This repository contains a comprehensive optimization framework for the Kaggle Santa 2025 - Christmas Tree Packing Challenge.

## Competition Goal

Pack n Christmas trees (n=1..200) into the smallest possible bounding square while avoiding overlaps. The score is the sum of s² for all n, where s is the side length of the minimum bounding square.

## Solution Architecture

### Solver Families

1. **Greedy (F0)**: Constructive algorithm that places trees one at a time
2. **Lattice (F1)**: Template-based initialization using hexagonal/square lattice
3. **Simulated Annealing (F2)**: Local search with multiple move types
4. **ILS/VNS (F3)**: Iterated Local Search with Variable Neighborhood Search
5. **Shrink-Repair (F4)**: Iteratively shrinks bounding box and repairs collisions

### ASHA Hyperparameter Tuning

Uses Asynchronous Successive Halving Algorithm with 4 stages:
- Stage 0: n=1..30 (quick evaluation)
- Stage 1: n=1..60
- Stage 2: n=1..120
- Stage 3: n=1..200 (full)

### Collision Detection

Three backends available:
- **Python**: Pure NumPy implementation (baseline)
- **Numba**: JIT-compiled (primary target)
- **C++**: Optional pybind11 extension

## Best Configuration

(To be filled after running `make tune`)

```yaml
family_id: TBD
hyperparams: TBD
seed: TBD
score: TBD
```

## Local Score

After running `make tune`:

```
Public Score: TBD
```

## Usage

```bash
# Setup
make setup

# Download data
make download

# Run benchmarks
make bench

# Run tests
make test

# Run tuning
make tune

# Validate submission
make validate

# Compute score
make score
```

## Environment Variables

- `TOTAL_BUDGET_MIN`: Total tuning budget in minutes (default: 180)
- `MAX_WORKERS`: Number of parallel workers (default: auto)
- `CLEAN_RUN`: Set to 1 to reset previous runs (default: 0)
- `COLLISION_BACKEND`: Backend choice (auto/python/numba/cpp, default: auto)

## Files

- `artifacts/submission_best.csv`: Best submission found
- `artifacts/best_config.yaml`: Best hyperparameters
- `artifacts/score.txt`: Final score
- `artifacts/dashboard.md`: Live progress dashboard
- `runs/asha_results.jsonl`: Trial history

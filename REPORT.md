# Santa 2025 Solver Report

## Summary

This repository implements a multi-strategy solver for the Kaggle Santa 2025 Christmas Tree Packing Challenge.

## Problem Description

The challenge is to pack `n` Christmas trees (for n = 1 to 200) into the smallest possible axis-aligned bounding square. Trees cannot overlap and can be placed at any (x, y) position with any rotation in degrees.

## Solution Approach

### Solver Families Implemented

1. **F0 - Greedy Placement**: Places trees one by one, choosing the position that minimizes the bounding square from a set of candidates.

2. **F1 - Lattice Initialization**: Arranges trees on a parametric 2D lattice, then refines locally.

3. **F2 - Simulated Annealing**: Uses SA with multiple move types:
   - Translation
   - Rotation
   - Swap
   - Boundary push

4. **F3 - ILS/VNS**: Iterated Local Search with perturbation ("kick") and variable neighborhood search.

5. **F4 - Shrink-and-Repair**: Progressively shrinks the bounding box while maintaining feasibility through repair operations.

### Collision Detection

- **Broad phase**: Spatial hash grid for AABB filtering
- **Narrow phase**: Separating Axis Theorem (SAT) for convex polygon collision

### ASHA Scheduling

The orchestrator uses Asynchronous Successive Halving Algorithm (ASHA) for hyperparameter tuning:
- Stage 0: n in [1..30]
- Stage 1: n in [1..60]  
- Stage 2: n in [1..120]
- Stage 3: n in [1..200]

Only top 1/3 of trials are promoted between stages.

## Best Configuration

The best configuration found during tuning is saved to `artifacts/best_config.yaml`.

Example configuration:
```yaml
family_id: F2
hyperparams:
  T0: 1.0
  T_end: 0.001
  alpha: 0.995
  dx0: 0.2
  dy0: 0.2
  iterations: 10000
seed: 42
```

## Results

### Achieved Score

Run `make tune` followed by `make score` to see the achieved score.

The score is computed as the sum of bounding square sides for all n from 1 to 200.

### Validation

All submissions pass:
- Format validation (correct CSV format with 's' prefix)
- ID coverage (all required tree placements present)
- Overlap checking (no trees intersect)

## Usage

```bash
# Setup
make setup

# Run full tuning (default 180 minutes)
make tune

# Or with custom budget
TOTAL_BUDGET_MIN=60 MAX_WORKERS=8 make tune

# Score the submission
make score

# Validate the submission
make validate

# Run tests
make test
```

## File Structure

```
santa2025_solver/
├── geometry_fast.py     # Fast geometry operations
├── collision_fast.py    # Collision detection
├── greedy.py            # F0: Greedy solver
├── lattice.py           # F1: Lattice solver
├── sa.py                # F2: Simulated annealing
├── ils_vns.py           # F3: ILS/VNS solver
├── shrink_repair.py     # F4: Shrink-repair solver
├── orchestrator.py      # Main orchestrator
├── asha.py              # ASHA scheduler
├── submission.py        # CSV generation
├── validate.py          # Validation
└── score.py             # Scoring
```

## Dependencies

- Python 3.10+
- NumPy, SciPy, Pandas
- Numba (optional, for acceleration)
- Shapely (for reference collision checks)

## Notes

- The tree polygon is a 15-vertex Christmas tree shape
- Rotations are in degrees (counter-clockwise)
- All coordinates use float64 for precision
- Submission format requires 's' prefix on numeric values

## Public Score

Final score is printed in format:
```
Public Score: <float>
```

Run `make tune && make score` to see the actual score.

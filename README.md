# Santa 2025 - Christmas Tree Packing Solver

A comprehensive solver for the Kaggle Santa 2025 Christmas Tree Packing Challenge.

## Quick Start

```bash
# Install dependencies
make setup

# Download competition data (requires Kaggle credentials)
make download

# Pull official metric notebook
make pull_kaggle_metric

# Run full tuning (default 180 minutes)
make tune

# Or with custom budget
TOTAL_BUDGET_MIN=60 MAX_WORKERS=8 make tune

# Score submission
make score
```

## Project Structure

```
santa2025_solver/          # Main solver package
├── kaggle_ref/            # Extracted Kaggle reference code
│   ├── metric_ref.py      # Official scoring implementation
│   └── geometry_ref.py    # Tree geometry definitions
├── geometry_fast.py       # Optimized geometry operations
├── collision_fast.py      # Fast collision detection
├── greedy.py              # F0: Greedy placement solver
├── lattice.py             # F1: Lattice-based solver
├── sa.py                  # F2: Simulated annealing solver
├── ils_vns.py             # F3: ILS/VNS solver
├── shrink_repair.py       # F4: Shrink-and-repair solver
├── orchestrator.py        # Multi-solver orchestrator
├── asha.py                # ASHA scheduler
├── submission.py          # Submission file generation
├── validate.py            # Submission validation
└── score.py               # Official scoring

data/raw/                  # Downloaded competition data
external/                  # Kaggle notebooks
artifacts/                 # Output files
runs/                      # Experiment logs
tests/                     # Unit tests
```

## Requirements

- Python 3.10+
- Kaggle API credentials (see [Kaggle API docs](https://www.kaggle.com/docs/api))

## Submission Format

CSV file with columns: `id,x,y,deg`
- `id`: Format `NNN_T` where NNN is the n value (001-200) and T is the tree index (0 to n-1)
- `x,y,deg`: String values prefixed with "s" (e.g., `s0.0`, `s-0.541068`, `s20.411299`)

## License

MIT

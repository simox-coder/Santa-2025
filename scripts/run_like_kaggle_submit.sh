#!/bin/bash
# Run like a Kaggle submission

set -e

echo "=== Santa 2025 - Kaggle-like Run ==="
echo ""

# Change to repo root
cd "$(dirname "$0")/.."

# Step 1: Download data if missing
if [ ! -f "data/raw/sample_submission.csv" ] && [ ! -f "sample_submission.csv" ]; then
    echo "Step 1: Downloading data..."
    make download || echo "Data download skipped (no credentials)"
else
    echo "Step 1: Data already present"
fi

# Step 2: Pull notebooks
echo ""
echo "Step 2: Pulling notebooks..."
bash scripts/pull_notebooks.sh all || echo "Notebook pull skipped"

# Step 3: Run tuning (short version)
echo ""
echo "Step 3: Running optimization..."

# Use short budget for testing
export TOTAL_BUDGET_MIN=${TOTAL_BUDGET_MIN:-5}
export MAX_WORKERS=${MAX_WORKERS:-auto}
export COLLISION_BACKEND=${COLLISION_BACKEND:-auto}

python -m santa2025_solver tune

# Step 4: Validate
echo ""
echo "Step 4: Validating submission..."
python -m santa2025_solver.validate --submission artifacts/submission_best.csv || echo "Validation skipped"

# Step 5: Score
echo ""
echo "Step 5: Computing final score..."
python -m santa2025_solver.score --submission artifacts/submission_best.csv

echo ""
echo "=== Run Complete ==="

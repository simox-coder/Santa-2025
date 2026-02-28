#!/bin/bash
# Run end-to-end like a Kaggle submission
# This script:
# 1. Downloads data if missing
# 2. Runs tuning (or uses existing best config)
# 3. Generates submission
# 4. Validates submission
# 5. Scores submission
# 6. Prints Public Score

set -e

cd "$(dirname "$0")/.."

# Configuration
QUICK_RUN=${QUICK_RUN:-0}
TOTAL_BUDGET_MIN=${TOTAL_BUDGET_MIN:-5}  # Short default for testing
MAX_WORKERS=${MAX_WORKERS:-1}

echo "============================================"
echo "Santa 2025 - End-to-End Submission Pipeline"
echo "============================================"
echo "TOTAL_BUDGET_MIN: $TOTAL_BUDGET_MIN"
echo "MAX_WORKERS: $MAX_WORKERS"
echo ""

# Step 1: Check data
if [ ! -f "sample_submission.csv" ]; then
    echo "ERROR: sample_submission.csv not found"
    exit 1
fi

# Step 2: Install dependencies if needed
if ! python -c "import numpy" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Step 3: Run tuning or use existing config
if [ -f "artifacts/best_config.yaml" ] && [ "$QUICK_RUN" = "1" ]; then
    echo "Using existing best config..."
else
    echo "Running tuning..."
    TOTAL_BUDGET_MIN=$TOTAL_BUDGET_MIN MAX_WORKERS=$MAX_WORKERS python -m santa2025_solver.orchestrator
fi

# Step 4: Generate submission (uses best config)
echo ""
echo "Generating submission..."
python -m santa2025_solver.solve

# Step 5: Validate submission
echo ""
echo "Validating submission..."
python -m santa2025_solver.validate --submission artifacts/submission_best.csv

# Step 6: Score submission
echo ""
echo "Scoring submission..."
python -m santa2025_solver.score --submission artifacts/submission_best.csv

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"

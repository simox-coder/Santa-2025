#!/bin/bash
# Pull Kaggle notebooks
set -e

cd "$(dirname "$0")/.."

echo "Pulling Kaggle metric notebook..."
python -m santa2025_solver.pull_notebooks metric || echo "Warning: Could not pull metric notebook"

echo "Pulling Kaggle getting started notebook..."
python -m santa2025_solver.pull_notebooks getting_started || echo "Warning: Could not pull getting started notebook"

echo "Done!"

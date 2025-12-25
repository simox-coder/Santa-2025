#!/bin/bash
# Download competition data from Kaggle
set -e

cd "$(dirname "$0")/.."

echo "Downloading Santa 2025 competition data..."
python -m santa2025_solver.download

echo "Done!"

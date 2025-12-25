#!/bin/bash
# Pull official Kaggle notebooks for Santa 2025

set -e

NOTEBOOK_TYPE=${1:-"all"}

echo "=== Santa 2025 Notebook Pull ==="

# Check for Kaggle credentials
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "Found Kaggle credentials in ~/.kaggle/kaggle.json"
elif [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "Found Kaggle credentials in environment variables"
    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
else
    echo "ERROR: Kaggle credentials not found!"
    echo "Please set up credentials (see scripts/download.sh for instructions)"
    exit 1
fi

# Create directories
mkdir -p external/kaggle_metric
mkdir -p external/kaggle_getting_started

pull_metric() {
    echo "Pulling official metric notebook..."
    
    # Try different possible slugs
    SLUGS=(
        "metric/santa-2025-metric"
        "kaggle/santa-2025-metric"
        "santa/santa-2025-metric"
    )
    
    PULLED=0
    for SLUG in "${SLUGS[@]}"; do
        echo "  Trying: $SLUG"
        if kaggle kernels pull "$SLUG" -p external/kaggle_metric --metadata 2>/dev/null; then
            PULLED=1
            break
        fi
    done
    
    if [ $PULLED -eq 0 ]; then
        echo "  Could not pull metric notebook automatically."
        echo "  You may need to download it manually from Kaggle."
        echo "  Creating placeholder..."
        
        # Create a placeholder metric implementation
        cat > external/kaggle_metric/metric.py << 'EOF'
"""
Santa 2025 - Official Metric (Placeholder)

This is a placeholder implementation based on the competition description.
For the official metric, download from Kaggle.

The score is the sum of s^2 for all n from 1 to 200,
where s is the side length of the minimum bounding square
containing all tree vertices.
"""

import pandas as pd
import numpy as np

def parse_s_value(s):
    """Parse value with 's' prefix."""
    if isinstance(s, str) and s.startswith('s'):
        return float(s[1:])
    return float(s)

def compute_score(submission_path):
    """Compute official score for a submission."""
    df = pd.read_csv(submission_path)
    
    total_score = 0.0
    
    for n in range(1, 201):
        # Get rows for this n
        prefix = f"{n:03d}_"
        rows = df[df['id'].str.startswith(prefix)]
        
        if len(rows) != n:
            raise ValueError(f"Expected {n} trees for n={n}, got {len(rows)}")
        
        # Parse coordinates
        xs = rows['x'].apply(parse_s_value).values
        ys = rows['y'].apply(parse_s_value).values
        degs = rows['deg'].apply(parse_s_value).values
        
        # Compute tree vertices and bounding box
        max_coord = 0.0
        for i in range(n):
            # Get tree vertices (simplified - ignores rotation for placeholder)
            # Real implementation should use rotated tree geometry
            tree_max = max(abs(xs[i]) + 0.3, abs(ys[i]) + 0.7)
            max_coord = max(max_coord, tree_max)
        
        s = 2 * max_coord
        total_score += s * s
    
    return total_score

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        score = compute_score(sys.argv[1])
        print(f"Public Score: {score}")
EOF
    fi
    
    # Convert notebook to Python if it exists
    if ls external/kaggle_metric/*.ipynb 1>/dev/null 2>&1; then
        echo "  Converting notebook to Python..."
        jupyter nbconvert --to script external/kaggle_metric/*.ipynb 2>/dev/null || true
    fi
}

pull_getting_started() {
    echo "Pulling getting-started notebook..."
    
    SLUGS=(
        "inversion/santa-2025-getting-started"
        "kaggle/santa-2025-getting-started"
        "santa/santa-2025-getting-started"
    )
    
    PULLED=0
    for SLUG in "${SLUGS[@]}"; do
        echo "  Trying: $SLUG"
        if kaggle kernels pull "$SLUG" -p external/kaggle_getting_started --metadata 2>/dev/null; then
            PULLED=1
            break
        fi
    done
    
    if [ $PULLED -eq 0 ]; then
        echo "  Could not pull getting-started notebook automatically."
        echo "  Creating placeholder..."
        
        cat > external/kaggle_getting_started/geometry.py << 'EOF'
"""
Santa 2025 - Tree Geometry (Placeholder)

This is a placeholder based on the competition description.
For the official definition, download from Kaggle.
"""

import numpy as np

# Tree polygon vertices (anchor at origin)
# The anchor point is the center of the top of the trunk
TREE_VERTICES = np.array([
    [0.0, 0.5],           # tip
    [-0.25, 0.0],         # left foliage
    [-0.0625, 0.0],       # trunk top left
    [-0.0625, -0.125],    # trunk bottom left
    [0.0625, -0.125],     # trunk bottom right
    [0.0625, 0.0],        # trunk top right
    [0.25, 0.0],          # right foliage
], dtype=np.float64)

def rotate_vertices(vertices, deg):
    """Rotate vertices by deg degrees around origin."""
    rad = np.radians(deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return vertices @ R.T

def translate_vertices(vertices, x, y):
    """Translate vertices by (x, y)."""
    return vertices + np.array([x, y])

def get_tree_vertices(x, y, deg):
    """Get tree vertices at position (x, y) with rotation deg."""
    rotated = rotate_vertices(TREE_VERTICES, deg)
    return translate_vertices(rotated, x, y)
EOF
    fi
    
    # Convert notebook to Python if it exists
    if ls external/kaggle_getting_started/*.ipynb 1>/dev/null 2>&1; then
        echo "  Converting notebook to Python..."
        jupyter nbconvert --to script external/kaggle_getting_started/*.ipynb 2>/dev/null || true
    fi
}

case $NOTEBOOK_TYPE in
    "metric")
        pull_metric
        ;;
    "getting_started")
        pull_getting_started
        ;;
    "all")
        pull_metric
        pull_getting_started
        ;;
    *)
        echo "Usage: $0 [metric|getting_started|all]"
        exit 1
        ;;
esac

echo "Notebook pull complete!"

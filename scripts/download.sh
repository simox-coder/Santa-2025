#!/bin/bash
# Download Santa 2025 competition data from Kaggle

set -e

echo "=== Santa 2025 Data Download ==="

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
    echo ""
    echo "Please set up Kaggle credentials using one of these methods:"
    echo ""
    echo "1. Create ~/.kaggle/kaggle.json with content:"
    echo '   {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}'
    echo ""
    echo "2. Set environment variables:"
    echo "   export KAGGLE_USERNAME=YOUR_USERNAME"
    echo "   export KAGGLE_KEY=YOUR_API_KEY"
    echo ""
    echo "Get your API key from: https://www.kaggle.com/settings/account"
    exit 1
fi

# Create data directory
mkdir -p data/raw

# Download competition data
echo "Downloading competition data..."
kaggle competitions download -c santa-2025 -p data/raw

# Unzip if needed
if [ -f data/raw/santa-2025.zip ]; then
    echo "Extracting data..."
    cd data/raw
    unzip -o santa-2025.zip
    cd ../..
fi

echo "Data download complete!"
echo "Files in data/raw:"
ls -la data/raw/

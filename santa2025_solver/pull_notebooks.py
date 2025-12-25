"""Pull and convert Kaggle notebooks for reference implementations."""

import os
import sys
import subprocess
from pathlib import Path


def check_kaggle_credentials():
    """Check if Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        return True
    
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    
    return False


def pull_notebook(slug: str, output_dir: Path) -> bool:
    """Pull a notebook from Kaggle and convert to Python script."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not check_kaggle_credentials():
        print("ERROR: Kaggle credentials not found!")
        print()
        print("Please set up Kaggle credentials. See README.md for instructions.")
        return False
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print(f"Pulling notebook: {slug} to {output_dir}")
        api.kernels_pull(slug, path=str(output_dir), metadata=True)
        
        # Convert ipynb to py
        ipynb_files = list(output_dir.glob("*.ipynb"))
        for ipynb_file in ipynb_files:
            print(f"Converting {ipynb_file} to Python script...")
            try:
                subprocess.run(
                    ["jupyter", "nbconvert", "--to", "script", str(ipynb_file)],
                    check=True,
                    capture_output=True
                )
                print(f"Converted: {ipynb_file.stem}.py")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not convert {ipynb_file}: {e}")
                # Try alternative: just read the notebook and extract code
                extract_code_from_ipynb(ipynb_file, output_dir / "converted.py")
        
        return True
        
    except Exception as e:
        print(f"Error pulling notebook: {e}")
        return False


def extract_code_from_ipynb(ipynb_path: Path, output_path: Path):
    """Extract Python code from Jupyter notebook without nbconvert."""
    import json
    
    with open(ipynb_path, 'r') as f:
        notebook = json.load(f)
    
    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code)
    
    with open(output_path, 'w') as f:
        f.write("# Extracted from Kaggle notebook\n\n")
        f.write("\n\n".join(code_cells))
    
    print(f"Extracted code to: {output_path}")


# Known notebook slugs for Santa 2025
METRIC_SLUGS = [
    "metric/santa-2025-metric",
    "kaggle/santa-2025-metric",
    "kaggle/evaluation-metric-santa-2025",
]

GETTING_STARTED_SLUGS = [
    "inversion/santa-2025-getting-started",
    "kaggle/santa-2025-getting-started",
]


def pull_metric_notebook():
    """Pull the official metric notebook."""
    base_dir = Path(__file__).parent.parent / "external" / "kaggle_metric"
    
    for slug in METRIC_SLUGS:
        try:
            if pull_notebook(slug, base_dir):
                return True
        except Exception as e:
            print(f"Could not pull {slug}: {e}")
            continue
    
    print("Warning: Could not pull official metric notebook.")
    print("Creating stub implementation based on competition rules...")
    return False


def pull_getting_started_notebook():
    """Pull the getting started notebook."""
    base_dir = Path(__file__).parent.parent / "external" / "kaggle_getting_started"
    
    for slug in GETTING_STARTED_SLUGS:
        try:
            if pull_notebook(slug, base_dir):
                return True
        except Exception as e:
            print(f"Could not pull {slug}: {e}")
            continue
    
    print("Warning: Could not pull getting started notebook.")
    return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m santa2025_solver.pull_notebooks [metric|getting_started|all]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "metric":
        pull_metric_notebook()
    elif cmd == "getting_started":
        pull_getting_started_notebook()
    elif cmd == "all":
        pull_metric_notebook()
        pull_getting_started_notebook()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()

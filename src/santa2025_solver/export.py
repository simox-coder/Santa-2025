"""
Export module.

Copies artifacts to exports directory for commit.
"""

import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def export_artifacts(artifacts_dir: str = "artifacts", 
                     exports_dir: str = "exports"):
    """Export artifacts to exports directory.
    
    Args:
        artifacts_dir: Source directory
        exports_dir: Destination directory
    """
    artifacts_path = Path(artifacts_dir)
    exports_path = Path(exports_dir)
    
    exports_path.mkdir(exist_ok=True)
    
    files_to_export = [
        "submission_best.csv",
        "score.txt",
        "best_config.yaml",
        "dashboard.md"
    ]
    
    for filename in files_to_export:
        src = artifacts_path / filename
        if src.exists():
            dst = exports_path / filename
            shutil.copy(src, dst)
            print(f"Exported: {filename}")
        else:
            print(f"Warning: {filename} not found in artifacts")
    
    print(f"\nExports complete: {exports_dir}/")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export artifacts")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts",
                        help="Source artifacts directory")
    parser.add_argument("--exports-dir", type=str, default="exports",
                        help="Destination exports directory")
    args = parser.parse_args()
    
    export_artifacts(args.artifacts_dir, args.exports_dir)


if __name__ == "__main__":
    main()

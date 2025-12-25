"""
Main solver module.

Runs the optimization and produces the submission file.
"""

import os
import sys
import time
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from santa2025_solver.solvers import (
    solve_all_n, solve_n, Layout, SolverConfig, repair_overlaps
)
from santa2025_solver.submission import write_submission
from santa2025_solver.kaggle_ref.metric_ref import calculate_score
from santa2025_solver.collision import strict_validate_layout


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def solve_main(config_path: str = None, output_dir: str = "artifacts",
               export_dir: str = "exports", max_n: int = 200):
    """Main solve function.
    
    Args:
        config_path: Path to config YAML (optional)
        output_dir: Directory for artifacts
        max_n: Maximum n to solve
    """
    print("=" * 60)
    print("Santa 2025 Solver")
    print("=" * 60)
    
    # Load config
    config_dict = load_config(config_path)
    
    solver_config = SolverConfig(
        seed=config_dict.get('seed', 42),
        max_iterations=config_dict.get('max_iterations', 5000),
        temperature_init=config_dict.get('temperature_init', 1.0),
        cooling_rate=config_dict.get('cooling_rate', 0.995),
        move_step=config_dict.get('move_step', 0.05),
        rotation_step=config_dict.get('rotation_step', 15.0),
        num_candidates=config_dict.get('num_candidates', 36),
        verbose=True
    )
    
    # Create output directories
    Path(output_dir).mkdir(exist_ok=True)
    Path(export_dir).mkdir(exist_ok=True)
    
    start_time = time.time()
    
    # Solve all n
    print(f"\nSolving for n=1 to {max_n}...")
    solutions = solve_all_n(max_n, solver_config, verbose=True)
    
    # Validate and repair
    print("\nValidating and repairing solutions...")
    for n, layout in solutions.items():
        if not strict_validate_layout(layout.positions):
            print(f"  Repairing n={n}...")
            solutions[n] = repair_overlaps(layout, solver_config)
    
    # Calculate total score
    total_radius = sum(layout.get_radius() for layout in solutions.values())
    
    elapsed = time.time() - start_time
    print(f"\nSolved in {elapsed:.1f}s")
    
    # Write submission
    submission_path = Path(output_dir) / "submission_best.csv"
    write_submission(solutions, str(submission_path))
    
    # Calculate official score
    score = calculate_score(str(submission_path))
    print(f"\nPublic Score: {score:.6f}")
    
    # Save score
    score_path = Path(output_dir) / "score.txt"
    with open(score_path, 'w') as f:
        f.write(f"{score:.6f}\n")
    
    # Save config
    config_out = {
        'seed': solver_config.seed,
        'max_iterations': solver_config.max_iterations,
        'temperature_init': solver_config.temperature_init,
        'cooling_rate': solver_config.cooling_rate,
        'move_step': solver_config.move_step,
        'rotation_step': solver_config.rotation_step,
        'num_candidates': solver_config.num_candidates,
        'score': score,
        'elapsed_seconds': elapsed
    }
    config_out_path = Path(output_dir) / "best_config.yaml"
    with open(config_out_path, 'w') as f:
        yaml.dump(config_out, f)
    
    # Copy to exports
    import shutil
    shutil.copy(submission_path, Path(export_dir) / "submission_best.csv")
    shutil.copy(score_path, Path(export_dir) / "score.txt")
    shutil.copy(config_out_path, Path(export_dir) / "best_config.yaml")
    
    # Generate dashboard
    generate_dashboard(solutions, score, elapsed, output_dir, export_dir)
    
    print(f"\nExported to: {export_dir}/")
    print(f"  - submission_best.csv")
    print(f"  - score.txt")
    print(f"  - best_config.yaml")
    print(f"  - dashboard.md")
    
    return score


def generate_dashboard(solutions: dict, score: float, elapsed: float,
                       output_dir: str, export_dir: str):
    """Generate markdown dashboard."""
    
    lines = [
        "# Santa 2025 Solver Dashboard",
        "",
        f"## Summary",
        f"- **Public Score**: {score:.6f}",
        f"- **Elapsed Time**: {elapsed:.1f}s",
        f"- **Groups Solved**: {len(solutions)}",
        "",
        "## Per-Group Statistics",
        "",
        "| n | Radius | Trees |",
        "|---|--------|-------|",
    ]
    
    for n in sorted(solutions.keys())[:20]:
        layout = solutions[n]
        lines.append(f"| {n} | {layout.get_radius():.4f} | {n} |")
    
    lines.append("| ... | ... | ... |")
    
    for n in [50, 100, 150, 200]:
        if n in solutions:
            layout = solutions[n]
            lines.append(f"| {n} | {layout.get_radius():.4f} | {n} |")
    
    lines.extend([
        "",
        "## Configuration",
        "```yaml",
    ])
    
    config_path = Path(output_dir) / "best_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            lines.append(f.read().strip())
    
    lines.extend([
        "```",
        "",
        "---",
        "Generated by Santa 2025 Solver"
    ])
    
    dashboard = '\n'.join(lines)
    
    # Save to both locations
    for dir_path in [output_dir, export_dir]:
        with open(Path(dir_path) / "dashboard.md", 'w') as f:
            f.write(dashboard)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Santa 2025 solver")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="artifacts",
                        help="Output directory for artifacts")
    parser.add_argument("--export-dir", type=str, default="exports",
                        help="Export directory (will be committed)")
    parser.add_argument("--max-n", type=int, default=200,
                        help="Maximum n to solve")
    args = parser.parse_args()
    
    solve_main(
        config_path=args.config,
        output_dir=args.output_dir,
        export_dir=args.export_dir,
        max_n=args.max_n
    )


if __name__ == "__main__":
    main()

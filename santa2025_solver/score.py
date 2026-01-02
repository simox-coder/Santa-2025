"""
Santa 2025 Solver - Official Scoring

Computes the official Kaggle competition score.
Score = sum of (bounding_square_side)^2 for all n from 1 to 200.
"""

import sys
import argparse
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from santa2025_solver.submission import read_submission
from santa2025_solver.geometry_fast import Layout, compute_bounding_square_side


def compute_layout_score(layout: Layout) -> float:
    """
    Compute score (s^2) for a single layout.
    
    The bounding square is the smallest axis-aligned square centered at origin
    that contains all tree vertices.
    """
    s = compute_bounding_square_side(layout.positions, layout.rotations)
    return s * s


def compute_total_score(layouts: Dict[int, Layout]) -> float:
    """
    Compute total score for all layouts.
    
    Score = sum of s^2 for all n from 1 to 200.
    """
    total = 0.0
    
    for n in range(1, 201):
        if n not in layouts:
            raise ValueError(f"Missing layout for n={n}")
        
        layout_score = compute_layout_score(layouts[n])
        total += layout_score
    
    return total


def compute_score(filepath: str) -> float:
    """
    Compute score for a submission file.
    
    Args:
        filepath: Path to submission CSV
    
    Returns:
        Total score (sum of s^2 for all n)
    """
    layouts = read_submission(filepath)
    return compute_total_score(layouts)


def compute_score_breakdown(filepath: str) -> Dict[int, float]:
    """
    Compute score breakdown by n.
    
    Returns:
        Dictionary mapping n -> s^2
    """
    layouts = read_submission(filepath)
    
    breakdown = {}
    for n in range(1, 201):
        if n in layouts:
            breakdown[n] = compute_layout_score(layouts[n])
    
    return breakdown


def format_score_report(filepath: str, verbose: bool = True) -> str:
    """
    Generate a score report.
    
    Returns formatted report string.
    """
    layouts = read_submission(filepath)
    total_score = compute_total_score(layouts)
    
    lines = []
    lines.append(f"Score Report for {filepath}")
    lines.append("=" * 50)
    lines.append("")
    
    if verbose:
        # Show per-n breakdown for a few key values
        lines.append("Sample scores by n:")
        sample_ns = [1, 5, 10, 20, 50, 100, 150, 200]
        for n in sample_ns:
            if n in layouts:
                s = compute_bounding_square_side(layouts[n].positions, layouts[n].rotations)
                s2 = s * s
                lines.append(f"  n={n:3d}: s={s:.6f}, s^2={s2:.6f}")
        
        lines.append("")
    
    lines.append(f"Total Score: {total_score:.6f}")
    lines.append("")
    lines.append(f"Public Score: {total_score}")
    
    return "\n".join(lines)


def score_submission_cli(filepath: str):
    """CLI entry point for scoring."""
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    report = format_score_report(filepath)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Santa 2025 submission")
    parser.add_argument("--submission", type=str, default="artifacts/submission_best.csv",
                        help="Path to submission file")
    parser.add_argument("--brief", action="store_true",
                        help="Only print the score line")
    
    args = parser.parse_args()
    
    if not Path(args.submission).exists():
        print(f"Error: File not found: {args.submission}")
        sys.exit(1)
    
    if args.brief:
        score = compute_score(args.submission)
        print(f"Public Score: {score}")
    else:
        report = format_score_report(args.submission)
        print(report)

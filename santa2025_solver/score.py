"""
Official scoring for Santa 2025 submission.

Uses the same scoring logic as Kaggle's official metric.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

from .kaggle_ref.metric_ref import score_submission, quick_score


def score_file(
    filepath: str,
    max_n: int = 200,
    check_overlaps: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Score a submission file.
    
    Args:
        filepath: Path to submission CSV
        max_n: Maximum n to score
        check_overlaps: Whether to check for overlaps
        verbose: Print progress
        
    Returns:
        Dict with scoring results
    """
    if verbose:
        print(f"Scoring: {filepath}")
        print(f"  Max n: {max_n}")
        print(f"  Check overlaps: {check_overlaps}")
        print()
    
    result = score_submission(filepath, max_n, check_overlaps)
    
    if verbose:
        print(f"Total Score: {result['total_score']:.6f}")
        print(f"Valid (no overlaps): {result['valid']}")
        
        if not result['valid']:
            overlapping = [n for n, has in result['has_overlaps'].items() if has]
            print(f"Overlaps found in {len(overlapping)} configurations")
        
        # Print breakdown
        print("\nScore breakdown (first 10):")
        for n in range(1, min(11, max_n + 1)):
            overlap_str = " [OVERLAP]" if result['has_overlaps'].get(n, False) else ""
            print(f"  n={n}: {result['scores'][n]:.6f}{overlap_str}")
        
        print(f"\n  ...")
        print(f"  n={max_n}: {result['scores'][max_n]:.6f}")
    
    return result


def print_public_score(score: float):
    """Print the public score in the exact required format."""
    print(f"Public Score: {score}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Score Santa 2025 submission')
    parser.add_argument('--submission', required=True, help='Path to submission CSV')
    parser.add_argument('--max-n', type=int, default=200, help='Maximum n to score')
    parser.add_argument('--skip-overlaps', action='store_true', help='Skip overlap checking')
    parser.add_argument('--quiet', action='store_true', help='Only print final score')
    
    args = parser.parse_args()
    
    if args.quiet:
        score = quick_score(args.submission, args.max_n)
        print_public_score(score)
    else:
        result = score_file(
            args.submission,
            max_n=args.max_n,
            check_overlaps=not args.skip_overlaps,
            verbose=True
        )
        
        print()
        print_public_score(result['total_score'])
        
        if not result['valid']:
            print("WARNING: Submission has overlapping trees!")
            sys.exit(1)


if __name__ == "__main__":
    main()

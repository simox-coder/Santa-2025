"""
Score calculation module.

Computes the official competition score for a submission file.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from santa2025_solver.kaggle_ref.metric_ref import calculate_score, calculate_score_per_n


def main():
    """Main entry point for scoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Santa 2025 submission score")
    parser.add_argument("submission", nargs="?", default="sample_submission.csv",
                        help="Path to submission CSV file")
    parser.add_argument("--per-n", action="store_true", 
                        help="Show score breakdown by n")
    args = parser.parse_args()
    
    try:
        score = calculate_score(args.submission)
        print(f"Public Score: {score:.6f}")
        
        if args.per_n:
            scores = calculate_score_per_n(args.submission)
            print("\nScore by n:")
            total = 0.0
            for n_str in sorted(scores.keys()):
                r = scores[n_str]
                total += r
                print(f"  n={n_str}: {r:.4f}")
            print(f"\nTotal: {total:.6f}")
    except FileNotFoundError:
        print(f"Error: File not found: {args.submission}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

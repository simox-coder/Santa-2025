"""
CLI entry point for scoring a submission.
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Score a Santa 2025 submission')
    parser.add_argument('--submission', '-s', required=True, help='Path to submission CSV')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed breakdown')
    args = parser.parse_args()
    
    from .metric_local import score_submission, score_submission_detailed
    
    if args.detailed:
        result = score_submission_detailed(args.submission)
        print(f"Total Score: {result['total_score']:.6f}")
        print(f"\nTop 20 worst groups:")
        for n, contrib in result['worst_groups'][:20]:
            print(f"  Group {n:03d}: {contrib:.4f}")
    else:
        score = score_submission(args.submission)
        print(f"Score: {score:.6f}")

if __name__ == '__main__':
    main()

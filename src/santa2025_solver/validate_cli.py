"""
CLI entry point for validating a submission.
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Validate a Santa 2025 submission')
    parser.add_argument('--submission', '-s', required=True, help='Path to submission CSV')
    parser.add_argument('--eps', type=float, default=1e-9, help='Epsilon for collision detection')
    args = parser.parse_args()
    
    from .validate import validate_submission, print_validation_report
    
    result = validate_submission(args.submission, args.eps)
    print_validation_report(result)
    
    sys.exit(0 if result['is_valid'] else 1)

if __name__ == '__main__':
    main()

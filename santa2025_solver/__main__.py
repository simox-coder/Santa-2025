"""
Santa 2025 Solver - Command Line Interface

Usage:
    python -m santa2025_solver tune     - Run ASHA hyperparameter tuning
    python -m santa2025_solver solve    - Generate submission from best config
    python -m santa2025_solver validate - Validate submission
    python -m santa2025_solver score    - Compute official score
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Santa 2025 - Christmas Tree Packing Solver"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run ASHA hyperparameter tuning")
    tune_parser.add_argument(
        "--budget",
        type=int,
        default=int(os.environ.get("TOTAL_BUDGET_MIN", "180")),
        help="Total budget in minutes (default: 180)",
    )
    tune_parser.add_argument(
        "--workers",
        type=str,
        default=os.environ.get("MAX_WORKERS", "auto"),
        help="Number of workers (default: auto)",
    )
    tune_parser.add_argument(
        "--backend",
        type=str,
        default=os.environ.get("COLLISION_BACKEND", "auto"),
        choices=["auto", "python", "numba", "cpp"],
        help="Collision detection backend (default: auto)",
    )

    # Solve command
    solve_parser = subparsers.add_parser(
        "solve", help="Generate submission from best config"
    )
    solve_parser.add_argument(
        "--config",
        type=str,
        default="artifacts/best_config.yaml",
        help="Path to config file",
    )
    solve_parser.add_argument(
        "--output",
        type=str,
        default="artifacts/submission_best.csv",
        help="Output submission path",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate submission file"
    )
    validate_parser.add_argument(
        "--submission",
        type=str,
        default="artifacts/submission_best.csv",
        help="Submission file to validate",
    )

    # Score command
    score_parser = subparsers.add_parser("score", help="Compute official score")
    score_parser.add_argument(
        "--submission",
        type=str,
        default="artifacts/submission_best.csv",
        help="Submission file to score",
    )

    args = parser.parse_args()

    if args.command == "tune":
        from santa2025_solver.orchestrator import run_tuning

        run_tuning(
            budget_minutes=args.budget,
            max_workers=args.workers,
            collision_backend=args.backend,
        )

    elif args.command == "solve":
        from santa2025_solver.orchestrator import generate_from_config

        generate_from_config(config_path=args.config, output_path=args.output)

    elif args.command == "validate":
        from santa2025_solver.validate import validate_submission_cli

        validate_submission_cli(args.submission)

    elif args.command == "score":
        from santa2025_solver.score import score_submission_cli

        score_submission_cli(args.submission)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

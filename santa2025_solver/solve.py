"""
Simple solver for Santa 2025.

Uses best config or default to generate a submission.
"""

import sys
import yaml
from pathlib import Path

from .trial_runner import run_trial
from .submission import write_submission
from .score import print_public_score


def main():
    """CLI entry point."""
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load best config if available
    config_path = artifacts_dir / 'best_config.yaml'
    
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print("No best config found, using default")
        config = {
            'family_id': 'F0',
            'hyperparams': {'num_candidates': 300},
            'seed': 42
        }
    
    print(f"Solver family: {config['family_id']}")
    print(f"Seed: {config['seed']}")
    print()
    
    # Run solver
    print("Solving for n=1..200...")
    output = run_trial(
        config['family_id'],
        config['hyperparams'],
        config['seed'],
        list(range(1, 201)),
        time_budget=None
    )
    
    print(f"Solved {len(output.layouts)} configurations")
    print(f"Total score: {output.total_score:.6f}")
    print(f"Valid: {output.is_valid}")
    
    # Save submission
    submission_path = artifacts_dir / 'submission_best.csv'
    write_submission(str(submission_path), output.layouts)
    
    # Save score
    score_path = artifacts_dir / 'score.txt'
    with open(score_path, 'w') as f:
        f.write(f"{output.total_score}\n")
    
    print()
    print_public_score(output.total_score)


if __name__ == "__main__":
    main()

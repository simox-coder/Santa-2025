"""
CLI entry point for tuning (HPO across arenas).
"""

import os
import sys

def main():
    from .orchestrator import run_orchestrator
    
    # Get parameters from environment
    budget = float(os.environ.get('TOTAL_BUDGET_MIN', 180))
    
    # Find seed submissions
    seeds = []
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for filename in ['sample_submission.csv', 'submission_best_165.csv']:
        path = os.path.join(repo_root, filename)
        if os.path.exists(path):
            seeds.append(path)
    
    print(f"Starting tuning with budget={budget} min, seeds={seeds}")
    score = run_orchestrator(seeds, budget)
    print(f"Public Score: {score}")

if __name__ == '__main__':
    main()

"""
CLI entry point for solving (single run with best known config).
"""

import os
import sys

def main():
    from .orchestrator import Orchestrator
    
    # Quick solve with moderate budget
    budget = float(os.environ.get('TOTAL_BUDGET_MIN', 30))
    
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    orch = Orchestrator(seed=42, total_budget_min=budget)
    
    # Load seeds
    for filename in ['sample_submission.csv', 'submission_best_165.csv']:
        path = os.path.join(repo_root, filename)
        if os.path.exists(path):
            orch.load_seed_submission(path)
    
    score = orch.run_optimization(n_trials_per_arena=3)
    print(f"Public Score: {score}")

if __name__ == '__main__':
    main()

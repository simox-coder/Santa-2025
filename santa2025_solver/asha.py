"""
Santa 2025 Solver - ASHA Scheduler

Asynchronous Successive Halving Algorithm for hyperparameter tuning.
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class Trial:
    """Represents a single trial in the ASHA scheduler."""
    trial_id: str
    family_id: int
    hyperparams: Dict[str, Any]
    seed: int
    stage: int = 0
    proxy_scores: List[float] = field(default_factory=list)
    runtimes: List[float] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, stopped
    feasible: bool = True
    best_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Trial':
        return cls(**d)


class ASHAScheduler:
    """
    ASHA (Asynchronous Successive Halving) scheduler.
    
    Stages:
    - Stage 0: n=1..30 (quick evaluation)
    - Stage 1: n=1..60
    - Stage 2: n=1..120
    - Stage 3: n=1..200 (full)
    
    At each stage, keeps top 1/eta fraction of trials.
    """
    
    STAGES = [
        {"name": "Stage0", "n_range": (1, 30), "budget_factor": 0.25},
        {"name": "Stage1", "n_range": (1, 60), "budget_factor": 0.5},
        {"name": "Stage2", "n_range": (1, 120), "budget_factor": 0.75},
        {"name": "Stage3", "n_range": (1, 200), "budget_factor": 1.0},
    ]
    
    def __init__(
        self,
        eta: int = 3,
        results_path: str = "runs/asha_results.jsonl",
        dashboard_path: str = "artifacts/dashboard.md"
    ):
        """
        Args:
            eta: Reduction factor (keep 1/eta at each stage)
            results_path: Path to save results
            dashboard_path: Path to dashboard markdown
        """
        self.eta = eta
        self.results_path = Path(results_path)
        self.dashboard_path = Path(dashboard_path)
        
        self.trials: Dict[str, Trial] = {}
        self.completed_stages: Dict[int, List[str]] = {i: [] for i in range(4)}
        
        # Load existing results
        self._load_results()
    
    def _load_results(self):
        """Load existing results from file."""
        if self.results_path.exists():
            with open(self.results_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            trial = Trial.from_dict(data)
                            self.trials[trial.trial_id] = trial
                            if trial.status == "completed":
                                self.completed_stages[trial.stage].append(trial.trial_id)
                        except:
                            pass
    
    def _save_trial(self, trial: Trial):
        """Append trial to results file."""
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, 'a') as f:
            f.write(json.dumps(trial.to_dict()) + '\n')
    
    def add_trial(
        self,
        family_id: int,
        hyperparams: Dict[str, Any],
        seed: int
    ) -> str:
        """
        Add a new trial to the scheduler.
        
        Returns:
            trial_id
        """
        trial_id = f"trial_{len(self.trials):05d}"
        trial = Trial(
            trial_id=trial_id,
            family_id=family_id,
            hyperparams=hyperparams,
            seed=seed
        )
        self.trials[trial_id] = trial
        return trial_id
    
    def get_pending_trials(self, stage: int) -> List[str]:
        """Get trials pending for a specific stage."""
        pending = []
        
        for trial_id, trial in self.trials.items():
            if trial.stage == stage and trial.status == "pending":
                pending.append(trial_id)
        
        return pending
    
    def get_promotable_trials(self, from_stage: int) -> List[str]:
        """
        Get trials that can be promoted from one stage to the next.
        
        Uses successive halving: keep top 1/eta trials by proxy score.
        """
        if from_stage >= 3:
            return []
        
        # Get completed trials at this stage
        completed = []
        for trial_id in self.completed_stages[from_stage]:
            trial = self.trials[trial_id]
            if trial.feasible and trial.proxy_scores:
                completed.append((trial_id, trial.proxy_scores[-1]))
        
        if not completed:
            return []
        
        # Sort by score (lower is better)
        completed.sort(key=lambda x: x[1])
        
        # Keep top 1/eta
        n_keep = max(1, len(completed) // self.eta)
        promotable = [trial_id for trial_id, _ in completed[:n_keep]]
        
        return promotable
    
    def promote_trial(self, trial_id: str):
        """Promote a trial to the next stage."""
        trial = self.trials[trial_id]
        trial.stage += 1
        trial.status = "pending"
    
    def report_result(
        self,
        trial_id: str,
        proxy_score: float,
        runtime: float,
        feasible: bool = True
    ):
        """Report result for a trial at its current stage."""
        trial = self.trials[trial_id]
        trial.proxy_scores.append(proxy_score)
        trial.runtimes.append(runtime)
        trial.feasible = feasible
        trial.status = "completed"
        
        if feasible:
            if trial.best_score is None or proxy_score < trial.best_score:
                trial.best_score = proxy_score
        
        self.completed_stages[trial.stage].append(trial_id)
        self._save_trial(trial)
    
    def get_best_trial(self) -> Optional[Tuple[str, Trial]]:
        """Get the best trial overall."""
        best_id = None
        best_score = float('inf')
        
        for trial_id, trial in self.trials.items():
            if trial.feasible and trial.best_score is not None:
                if trial.best_score < best_score:
                    best_score = trial.best_score
                    best_id = trial_id
        
        if best_id:
            return best_id, self.trials[best_id]
        return None
    
    def get_stage_stats(self, stage: int) -> Dict[str, Any]:
        """Get statistics for a stage."""
        completed = len(self.completed_stages[stage])
        pending = len(self.get_pending_trials(stage))
        
        # Get top trials
        trials_with_scores = []
        for trial_id in self.completed_stages[stage]:
            trial = self.trials[trial_id]
            if trial.proxy_scores:
                trials_with_scores.append((trial_id, trial.proxy_scores[-1], trial))
        
        trials_with_scores.sort(key=lambda x: x[1])
        
        return {
            "completed": completed,
            "pending": pending,
            "top_trials": trials_with_scores[:5]
        }
    
    def update_dashboard(self, additional_info: Dict[str, Any] = None):
        """Update the dashboard markdown file."""
        lines = []
        lines.append("# Santa 2025 Solver Dashboard")
        lines.append("")
        lines.append(f"*Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        # System info
        if additional_info and 'system_info' in additional_info:
            lines.append("## System Information")
            lines.append("")
            for key, value in additional_info['system_info'].items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # Backend
        if additional_info and 'backend' in additional_info:
            lines.append(f"**Collision Backend**: {additional_info['backend']}")
            lines.append("")
        
        # Best trial
        best = self.get_best_trial()
        if best:
            trial_id, trial = best
            lines.append("## Best Trial So Far")
            lines.append("")
            lines.append(f"- **Trial ID**: {trial_id}")
            lines.append(f"- **Family**: {trial.family_id}")
            lines.append(f"- **Stage**: {trial.stage}")
            lines.append(f"- **Best Score**: {trial.best_score:.6f}")
            lines.append(f"- **Seed**: {trial.seed}")
            lines.append("")
        
        # Stage tables
        for stage_idx in range(4):
            stage_info = self.STAGES[stage_idx]
            stats = self.get_stage_stats(stage_idx)
            
            lines.append(f"## {stage_info['name']} (n={stage_info['n_range'][0]}..{stage_info['n_range'][1]})")
            lines.append("")
            lines.append(f"- Completed: {stats['completed']}")
            lines.append(f"- Pending: {stats['pending']}")
            lines.append("")
            
            if stats['top_trials']:
                lines.append("| Rank | Trial ID | Score | Family | Runtime |")
                lines.append("|------|----------|-------|--------|---------|")
                
                for rank, (trial_id, score, trial) in enumerate(stats['top_trials'], 1):
                    runtime = sum(trial.runtimes) if trial.runtimes else 0
                    lines.append(f"| {rank} | {trial_id} | {score:.4f} | {trial.family_id} | {runtime:.1f}s |")
                
                lines.append("")
        
        # Workers
        if additional_info and 'workers' in additional_info:
            lines.append("## Worker Information")
            lines.append("")
            lines.append(f"- **Active Workers**: {additional_info['workers']}")
            lines.append("")
        
        # Write to file
        self.dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dashboard_path, 'w') as f:
            f.write('\n'.join(lines))


def create_default_scheduler() -> ASHAScheduler:
    """Create a scheduler with default settings."""
    return ASHAScheduler()

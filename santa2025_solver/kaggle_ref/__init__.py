"""
Santa 2025 Solver - Kaggle Reference Package

Contains wrappers around official Kaggle metric and geometry code.
"""

from santa2025_solver.kaggle_ref.metric_ref import compute_official_score
from santa2025_solver.kaggle_ref.geometry_ref import get_official_tree_vertices

__all__ = ['compute_official_score', 'get_official_tree_vertices']

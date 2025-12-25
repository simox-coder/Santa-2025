"""
Santa 2025 - Christmas Tree Packing Solver

A comprehensive optimization framework for the Kaggle Santa 2025 competition.
"""

__version__ = "1.0.0"
__author__ = "Santa 2025 Solver Team"

from santa2025_solver.geometry_fast import TreeGeometry, get_tree_vertices
from santa2025_solver.submission import write_submission, read_submission
from santa2025_solver.validate import validate_submission
from santa2025_solver.score import compute_score

__all__ = [
    "TreeGeometry",
    "get_tree_vertices",
    "write_submission",
    "read_submission",
    "validate_submission",
    "compute_score",
]

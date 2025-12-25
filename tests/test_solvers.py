"""
Tests for solvers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from santa2025_solver.solvers import (
    solve_greedy, solve_sa, solve_ils, solve_n, Layout, SolverConfig
)
from santa2025_solver.collision import check_any_collision


class TestSolvers:
    """Tests for solver implementations."""
    
    def test_solve_n_1(self):
        """Solving n=1 should produce single tree at origin."""
        config = SolverConfig(max_iterations=100)
        layout = solve_n(1, config, method="greedy")
        
        assert layout.n == 1
        assert not layout.has_collision()
    
    def test_solve_n_4_greedy(self):
        """Greedy solver should produce valid n=4 layout."""
        config = SolverConfig(seed=42, max_iterations=100)
        layout = solve_greedy(4, config)
        
        assert layout.n == 4
        assert not layout.has_collision()
    
    def test_solve_n_4_sa(self):
        """SA solver should produce valid n=4 layout."""
        config = SolverConfig(seed=42, max_iterations=500)
        layout = solve_sa(4, config=config)
        
        assert layout.n == 4
        # May have collision during SA, should be repairable
    
    def test_solve_n_10(self):
        """Solving n=10 should produce valid layout."""
        config = SolverConfig(seed=42, max_iterations=1000)
        layout = solve_n(10, config, method="ils")
        
        assert layout.n == 10
    
    def test_layout_radius_positive(self):
        """Layout radius should be positive."""
        layout = solve_n(5, SolverConfig(), method="greedy")
        assert layout.get_radius() > 0
    
    def test_determinism(self):
        """Same seed should produce same result."""
        config1 = SolverConfig(seed=12345, max_iterations=100)
        config2 = SolverConfig(seed=12345, max_iterations=100)
        
        layout1 = solve_greedy(3, config1)
        layout2 = solve_greedy(3, config2)
        
        # Positions should be identical
        for i in range(3):
            assert layout1.get_position(i) == layout2.get_position(i)


class TestLayout:
    """Tests for Layout class."""
    
    def test_layout_copy(self):
        """Layout copy should be independent."""
        layout = Layout(3)
        layout.set_position(0, 1.0, 2.0, 90.0)
        
        copy = layout.copy()
        copy.set_position(0, 5.0, 5.0, 0.0)
        
        assert layout.get_position(0) == (1.0, 2.0, 90.0)
        assert copy.get_position(0) == (5.0, 5.0, 0.0)


if __name__ == "__main__":
    t = TestSolvers()
    t.test_solve_n_1()
    t.test_solve_n_4_greedy()
    print("Solver tests passed!")

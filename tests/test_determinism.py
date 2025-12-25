"""Test deterministic seeding."""

import pytest
import numpy as np

from santa2025_solver.geometry_fast import TreeLayout
from santa2025_solver.greedy import greedy_solve
from santa2025_solver.trial_runner import run_trial


class TestDeterministicSeeding:
    """Tests for deterministic behavior with same seed."""
    
    def test_same_seed_same_layout(self):
        """Test that same seed produces same greedy layout."""
        # Run twice with same seed
        rng1 = np.random.default_rng(12345)
        layout1 = greedy_solve(5, rng1)
        
        rng2 = np.random.default_rng(12345)
        layout2 = greedy_solve(5, rng2)
        
        # Should produce identical layouts
        assert layout1 is not None
        assert layout2 is not None
        
        for i in range(5):
            x1, y1, deg1 = layout1.get_tree(i)
            x2, y2, deg2 = layout2.get_tree(i)
            
            assert abs(x1 - x2) < 1e-10
            assert abs(y1 - y2) < 1e-10
            assert abs(deg1 - deg2) < 1e-10
    
    def test_different_seeds_potentially_different_layouts(self):
        """Test that RNG is properly seeded (different seeds = different RNG state)."""
        # This test verifies that seeds properly initialize the RNG
        # The actual layouts may or may not differ since greedy is somewhat deterministic
        
        rng1 = np.random.default_rng(11111)
        rng2 = np.random.default_rng(22222)
        
        # RNG should produce different sequences
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        
        # At least some values should differ
        differences = sum(1 for v1, v2 in zip(vals1, vals2) if abs(v1 - v2) > 1e-10)
        assert differences > 0, "Different seeds should give different random sequences"
    
    def test_seeded_run_small_determinism(self):
        """Test determinism on small n with trial runner."""
        # Run trial twice with same parameters
        output1 = run_trial(
            family_id='F0',
            hyperparams={'num_candidates': 100},
            seed=42,
            ns=[1, 2, 3, 4, 5],
            time_budget=60
        )
        
        output2 = run_trial(
            family_id='F0',
            hyperparams={'num_candidates': 100},
            seed=42,
            ns=[1, 2, 3, 4, 5],
            time_budget=60
        )
        
        # Scores should be identical
        assert abs(output1.total_score - output2.total_score) < 1e-10
        
        # Individual n scores should match
        for n in [1, 2, 3, 4, 5]:
            assert abs(output1.scores[n] - output2.scores[n]) < 1e-10

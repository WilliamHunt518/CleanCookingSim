from __future__ import annotations

from functools import lru_cache

import prism_export as pe

# build_chain enumerates ~1M states (~10s) -- cache per persona so the tests below (which all
# need it) only pay that cost once each, not once per test.
_build_chain_cached = lru_cache(maxsize=None)(pe.build_chain)


def test_compute_exact_properties_returns_valid_probabilities():
    transitions, _all_states = _build_chain_cached("household")
    result = pe.compute_exact_properties(transitions)

    assert 0.0 <= result["prob_incomplete_day"] <= 1.0
    assert 0.0 <= result["expected_wood_meals"] <= 3.0


def test_exact_and_monte_carlo_agree_closely_for_household():
    """The headline check: the exact PRISM-chain computation and the real 5-minute-block
    simulator, run under the same simplifications, should land close together -- this is what
    monte_carlo_comparison's docstring promises, and what the block-rate rescale in _fire_prob
    (BLOCKS_PER_COARSE) fixed. Before that fix these two were off by close to an order of
    magnitude (expected_wood_meals 0.57 vs 1.17, prob_incomplete_day 87% vs 5%)."""
    transitions, _all_states = _build_chain_cached("household")
    exact = pe.compute_exact_properties(transitions)
    mc = pe.monte_carlo_comparison("household", n_agents=300, R=30, seed=0)

    assert abs(exact["expected_wood_meals"] - mc["expected_wood_meals"]) < 0.15
    assert abs(exact["prob_incomplete_day"] - mc["prob_incomplete_day"]) < 0.05


def test_exact_and_monte_carlo_agree_for_school_near_total_incompleteness():
    """school's breakfast/dinner lam=-6 means it should almost never complete all 3 meals in a
    day (it's essentially a lunch-only persona) -- both computations should agree it's close to
    100% incomplete, not just close to each other in the abstract."""
    transitions, _all_states = _build_chain_cached("school")
    exact = pe.compute_exact_properties(transitions)
    mc = pe.monte_carlo_comparison("school", n_agents=300, R=30, seed=0)

    assert exact["prob_incomplete_day"] > 0.95
    assert mc["prob_incomplete_day"] > 0.95
    assert abs(exact["prob_incomplete_day"] - mc["prob_incomplete_day"]) < 0.05


def test_monte_carlo_comparison_restores_config_timing_after_running():
    orig_noise = pe.config.TIMING.sigma_logit_noise
    orig_repeat = pe.config.TIMING.repeat_meal_prob

    pe.monte_carlo_comparison("household", n_agents=20, R=2, seed=0)

    assert pe.config.TIMING.sigma_logit_noise == orig_noise
    assert pe.config.TIMING.repeat_meal_prob == orig_repeat

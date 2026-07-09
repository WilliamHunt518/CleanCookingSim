from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grid_energy import pricing
from grid_energy.quartz_forecast import BLOCK_MINUTES, ForecastResult
from grid_energy.soc import SOCResult
from sim import ga


@pytest.fixture
def fake_week():
    """A small, deterministic week-length forecast + SOC pair -- no network, fast, and varied
    enough (a real diurnal PV cycle, a battery that both fills and drains) to exercise every
    strategy's non-trivial branch."""
    n = 672
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    hours = idx.hour + idx.minute / 60.0
    pv_vals = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) * 40.0
    power_kw = pd.Series(pv_vals, index=idx)
    forecast = ForecastResult(power_kw=power_kw, energy_wh=power_kw * 1000.0 * (BLOCK_MINUTES / 60.0))

    usage_vals = np.full(n, 10.0)
    zeros = np.zeros(n)
    soc = SOCResult(
        t_hr=np.arange(n) * 0.25, pv_kw=pv_vals, usage_kw=usage_vals, net_kw=pv_vals - usage_vals,
        energy_potential_kwh=zeros, socs_pct=zeros, energy_actual_kwh=zeros,
        actual_soc_pct=np.full(n, 50.0), surplus_kwh=np.maximum(pv_vals - usage_vals, 0) * 0.25,
        deficit_kwh=np.maximum(usage_vals - pv_vals, 0) * 0.25,
    )
    return forecast, soc


def _day_slice(arr, day=0, n=96):
    return arr[day * n:(day + 1) * n]


# --------------------------------------------------------------------------- price_curve_kes_day

def test_price_curve_all_zero_weights_is_flat(fake_week):
    from grid_energy import config as grid_config
    forecast, soc = fake_week
    theta = np.array([0., 0., 0., 0., 0., 1.0, 90.0])
    price = ga.price_curve_kes_day(theta, forecast, soc)
    assert np.all(price == grid_config.PRICING.P_FLAT)


def test_price_curve_pure_strategy_matches_pricing_function(fake_week):
    """weight=1 on exactly one strategy should reproduce that strategy's own price curve exactly
    -- this is what makes the GA's search space literally contain every heuristic (section 7.5)."""
    forecast, soc = fake_week
    theta = np.array([0., 1., 0., 0., 0., 1.5, 90.0])  # pure pv_following_real, gamma=1.5
    price = ga.price_curve_kes_day(theta, forecast, soc)
    expected = pricing.pv_following_real(forecast, gamma=1.5)
    np.testing.assert_allclose(price, expected)


def test_price_curve_blend_is_between_component_curves(fake_week):
    forecast, soc = fake_week
    theta_blend = np.array([0.5, 0.5, 0., 0., 0., 1.0, 90.0])
    price_blend = ga.price_curve_kes_day(theta_blend, forecast, soc)
    price_a = pricing.green_light(soc)
    price_b = pricing.pv_following_real(forecast, gamma=1.0)
    assert np.all(price_blend >= np.minimum(price_a, price_b) - 1e-9)
    assert np.all(price_blend <= np.maximum(price_a, price_b) + 1e-9)


def test_describe_chromosome_reports_dominant_strategy_first():
    theta = np.array([0., 0.9, 0., 0.1, 0., 1.2, 85.0])
    desc = ga.describe_chromosome(theta)
    assert "pv_following_real" in desc
    assert "residual_load" in desc
    assert desc.index("pv_following_real") < desc.index("residual_load")


def test_describe_chromosome_flat_when_all_weights_zero():
    theta = np.array([0., 0., 0., 0., 0., 1.0, 90.0])
    assert "flat" in ga.describe_chromosome(theta)


# --------------------------------------------------------------------------- evaluate / run_ga

def test_evaluate_returns_finite_breakdown(fake_week):
    forecast, soc = fake_week
    from sim import config, population as population_mod
    population = population_mod.build_population(np.random.default_rng(0), n_agents=15)
    seeds = np.random.SeedSequence(0).spawn(3)
    from grid_energy import GridEnergyComponent
    theta = np.array([0., 1., 0., 0., 0., 1.0, 90.0])
    breakdown = ga.evaluate(
        theta, population=population, seeds=seeds,
        scenario=config.SCENARIOS["reference"], forecast_day=_slice(forecast), soc_day=_slice(soc),
        grid_component=GridEnergyComponent(), forecast_full=forecast)
    assert np.isfinite(breakdown.fitness)
    assert breakdown.total_deficit_kwh >= 0.0
    assert breakdown.total_surplus_kwh >= 0.0
    assert 0.0 <= breakdown.wood_share <= 1.0


def _slice(x):
    from sim import tariffs
    from grid_energy.quartz_forecast import ForecastResult
    from grid_energy.soc import SOCResult
    if isinstance(x, ForecastResult):
        return tariffs._slice_forecast_day(x, 0)
    return tariffs._slice_soc_day(x, 0)


def test_run_ga_best_fitness_never_gets_worse_across_generations(fake_week):
    """Elitism guarantee: the best-seen fitness is non-increasing generation to generation
    (minimising) -- a core GA correctness property, not just an empirical nicety."""
    forecast, soc = fake_week
    result = ga.run_ga(pop_size=8, n_generations=5, R=3, n_agents=15, seed=0,
                        forecast=forecast, reference_soc=soc, patience=100)
    bests = [h["best"] for h in result.history]
    assert all(b2 <= b1 + 1e-9 for b1, b2 in zip(bests, bests[1:]))


def test_run_ga_seeded_population_includes_pure_strategies(fake_week):
    forecast, soc = fake_week
    result = ga.run_ga(pop_size=8, n_generations=1, R=2, n_agents=10, seed=0,
                        forecast=forecast, reference_soc=soc, patience=100)
    genomes = [ind.theta for ind in result.final_population]
    seed_thetas = list(ga.SEED_CHROMOSOMES.values())
    for seed_theta in seed_thetas[:8]:  # pop_size=8 truncates SEED_CHROMOSOMES.values() to first 8
        assert any(np.allclose(g, seed_theta) for g in genomes)


def test_run_ga_respects_hard_generation_cap(fake_week):
    forecast, soc = fake_week
    result = ga.run_ga(pop_size=6, n_generations=3, R=2, n_agents=10, seed=0,
                        forecast=forecast, reference_soc=soc, patience=1000)
    assert result.n_generations_run == 3
    assert len(result.history) == 3


def test_run_ga_convergence_stops_early_with_tight_patience(fake_week):
    forecast, soc = fake_week
    result = ga.run_ga(pop_size=6, n_generations=40, R=2, n_agents=10, seed=0,
                        forecast=forecast, reference_soc=soc, patience=1, convergence_epsilon=1e6)
    # convergence_epsilon absurdly large -> "no improvement" is immediate -> stops well before 40
    assert result.n_generations_run < 40


def test_run_ga_is_deterministic_given_the_same_seed(fake_week):
    forecast, soc = fake_week
    r1 = ga.run_ga(pop_size=8, n_generations=3, R=2, n_agents=10, seed=42,
                    forecast=forecast, reference_soc=soc, patience=100)
    r2 = ga.run_ga(pop_size=8, n_generations=3, R=2, n_agents=10, seed=42,
                    forecast=forecast, reference_soc=soc, patience=100)
    assert r1.best_fitness == r2.best_fitness
    np.testing.assert_array_equal(r1.best_theta, r2.best_theta)


def test_run_ga_generation_callback_fires_once_per_generation(fake_week):
    forecast, soc = fake_week
    calls = []
    ga.run_ga(pop_size=6, n_generations=4, R=2, n_agents=10, seed=0, forecast=forecast,
              reference_soc=soc, patience=100, generation_callback=lambda info: calls.append(info))
    assert len(calls) == 4
    assert [c["generation"] for c in calls] == [0, 1, 2, 3]

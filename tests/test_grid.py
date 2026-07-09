from __future__ import annotations

import numpy as np
import pandas as pd

from grid_energy.component import GridEnergyComponent
from grid_energy.quartz_forecast import BLOCK_MINUTES, ForecastResult
from sim import grid
from sim.run import CookEvent, TariffRunResult


def _fake_week_forecast(power_kw_value: float, n_blocks: int = 672) -> ForecastResult:
    idx = pd.date_range("2026-01-01", periods=n_blocks, freq="15min")
    power_kw = pd.Series(np.full(n_blocks, power_kw_value), index=idx)
    energy_wh = power_kw * 1000.0 * (BLOCK_MINUTES / 60.0)
    return ForecastResult(power_kw=power_kw, energy_wh=energy_wh)


def _make_tariff_result(tariff_name: str, demand_curves: list[np.ndarray]) -> TariffRunResult:
    return TariffRunResult(tariff_name=tariff_name, price=np.zeros(288), events_all_runs=[],
                            demand_curves=demand_curves, trace_rows=[])


def test_demand_kw_for_tariff_averages_across_runs():
    curves = [np.full(288, 1.0), np.full(288, 3.0)]
    result = _make_tariff_result("flat", curves)
    np.testing.assert_allclose(grid.demand_kw_for_tariff(result), 2.0)


def test_demand_kw_for_tariff_handles_no_events():
    result = _make_tariff_result("extreme_test", [])
    out = grid.demand_kw_for_tariff(result)
    assert np.all(out == 0.0)


def test_battery_never_dips_when_pv_always_exceeds_usage():
    """Abundant PV, tiny usage -> the battery should stay full (or close to it) all week, and
    demand should always be fully met."""
    forecast = _fake_week_forecast(power_kw_value=50.0)
    result = _make_tariff_result("flat", [np.full(288, 1.0)])
    component = GridEnergyComponent(capacity_kwh=100.0, soc_init_pct=100.0)

    fitness = grid.run_grid_for_tariff_result(result, component=component, forecast=forecast)

    assert fitness.total_deficit_kwh == 0.0
    assert fitness.demand_met == 1.0
    assert fitness.battery_preserved == 1.0
    assert fitness.fitness == 1.0


def test_battery_drains_and_fitness_drops_when_usage_swamps_pv():
    """Near-zero PV, heavy usage, battery starts empty -> deficits every block, fitness near 0."""
    forecast = _fake_week_forecast(power_kw_value=0.0)
    result = _make_tariff_result("evening_peak", [np.full(288, 20.0)])
    component = GridEnergyComponent(capacity_kwh=10.0, soc_init_pct=0.0)

    fitness = grid.run_grid_for_tariff_result(result, component=component, forecast=forecast)

    assert fitness.total_deficit_kwh > 0.0
    assert fitness.demand_met < 0.2
    assert fitness.min_actual_soc_pct == 0.0
    assert fitness.battery_preserved == 0.0
    assert fitness.fitness < 0.2


def test_battery_weight_and_demand_weight_are_applied():
    forecast = _fake_week_forecast(power_kw_value=0.0)
    result = _make_tariff_result("evening_peak", [np.full(288, 20.0)])
    component = GridEnergyComponent(capacity_kwh=10.0, soc_init_pct=0.0)

    all_battery = grid.run_grid_for_tariff_result(result, component=component, forecast=forecast,
                                                    battery_weight=1.0, demand_weight=0.0)
    all_demand = grid.run_grid_for_tariff_result(result, component=component, forecast=forecast,
                                                   battery_weight=0.0, demand_weight=1.0)

    assert all_battery.fitness == all_battery.battery_preserved
    assert all_demand.fitness == all_demand.demand_met


def test_surplus_is_not_penalised_in_fitness():
    """Huge PV surplus should score a perfect fitness, not a penalised one -- surplus_kwh is
    diagnostic only (see grid_energy's README: it could power extra load directly), not part of
    the battery_preserved/demand_met fitness terms."""
    forecast = _fake_week_forecast(power_kw_value=1000.0)
    result = _make_tariff_result("flat", [np.full(288, 1.0)])
    component = GridEnergyComponent(capacity_kwh=10.0, soc_init_pct=0.0)

    fitness = grid.run_grid_for_tariff_result(result, component=component, forecast=forecast)

    assert fitness.total_surplus_kwh > 0.0
    assert fitness.fitness == 1.0


def test_evaluate_tariff_runs_its_own_sweep_and_returns_both_results():
    forecast = _fake_week_forecast(power_kw_value=10.0)
    fitness, result = grid.evaluate_tariff("flat", R=3, n_agents=20, seed=0, forecast=forecast)

    assert fitness.tariff_name == "flat"
    assert result.tariff_name == "flat"
    assert len(result.demand_curves) == 3

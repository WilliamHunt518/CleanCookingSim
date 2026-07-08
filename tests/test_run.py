import numpy as np

from sim import config, meals, run
from sim.population import build_population


def test_demand_assembly_conserves_energy():
    rng_pop = np.random.default_rng(1)
    population = build_population(rng_pop, n_agents=30)
    price = np.full(config.STATE.T, config.TARIFF.p_bar)
    scenario = config.SCENARIOS["reference"]
    rng = np.random.default_rng(2)

    day = run.simulate_day(population, price, scenario, rng)

    energy_from_curve = day.demand_kw.sum() * meals.BLOCK_HOURS
    energy_from_events = sum(e.e_kwh for e in day.events)
    assert np.isclose(energy_from_curve, energy_from_events)


def test_seeded_sweep_is_reproducible():
    results1, pop1 = run.run_sweep(["flat"], seed=42, R=5, n_agents=20)
    results2, pop2 = run.run_sweep(["flat"], seed=42, R=5, n_agents=20)

    np.testing.assert_array_equal(pop1.gamma, pop2.gamma)
    np.testing.assert_array_equal(pop1.gamma_cost, pop2.gamma_cost)

    d1 = np.stack(results1["flat"].demand_curves)
    d2 = np.stack(results2["flat"].demand_curves)
    np.testing.assert_array_equal(d1, d2)

    events1 = [(e.agent_idx, e.meal_idx0, e.start_block) for e in results1["flat"].events_all_runs]
    events2 = [(e.agent_idx, e.meal_idx0, e.start_block) for e in results2["flat"].events_all_runs]
    assert events1 == events2


def test_different_seeds_give_different_results():
    results1, _ = run.run_sweep(["flat"], seed=1, R=5, n_agents=20)
    results2, _ = run.run_sweep(["flat"], seed=2, R=5, n_agents=20)
    d1 = np.stack(results1["flat"].demand_curves)
    d2 = np.stack(results2["flat"].demand_curves)
    assert not np.array_equal(d1, d2)

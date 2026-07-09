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


def test_school_and_kiosk_events_scale_by_meals_per_cook():
    """A school/kiosk agent's one cook event represents an institutional kitchen serving many
    people at once (config.PERSONAS.meals_per_cook), not one household -- its e_kwh (and
    therefore its demand-curve contribution) should be scaled up accordingly, not identical to a
    household's for the same meal."""
    from sim.population import PERSONA_NAMES

    rng_pop = np.random.default_rng(1)
    population = build_population(rng_pop, n_agents=100)
    price = np.full(config.STATE.T, config.TARIFF.p_bar)
    scenario = config.SCENARIOS["reference"]
    day = run.simulate_day(population, price, scenario, np.random.default_rng(2))

    assert len(day.events) > 0
    for e in day.events:
        persona_name = PERSONA_NAMES[e.persona_idx]
        expected_scale = config.PERSONAS.meals_per_cook[persona_name]
        base_e_kwh = meals.E_KWH[e.meal_idx0]
        if base_e_kwh > 0:
            assert np.isclose(e.e_kwh, base_e_kwh * expected_scale)

    school_events = [e for e in day.events if PERSONA_NAMES[e.persona_idx] == "school"]
    household_events = [e for e in day.events if PERSONA_NAMES[e.persona_idx] == "household"]
    if school_events and household_events and config.PERSONAS.meals_per_cook["school"] > 1:
        # Same underlying per-meal e_kwh table, but school events should individually draw more
        # energy on average, since each one stands in for many meals at once.
        assert config.PERSONAS.meals_per_cook["school"] > config.PERSONAS.meals_per_cook["household"]


def test_extreme_test_tariff_saturates_toward_almost_no_cooking():
    """Sanity check that price response saturates at an extreme input (sim/tariffs.py's
    extreme_test candidate, 5x p_bar) rather than silently no-op'ing: Stage 1 doesn't know which
    fuel an agent would pick until *after* it fires, so an extreme enough price should crush
    firing altogether, not just push it toward wood -- meals/day should collapse, not settle at
    a normal-looking number."""
    results, _ = run.run_sweep(["flat", "extreme_test"], seed=0, R=30, n_agents=40)
    n_flat = len(results["flat"].events_all_runs)
    n_extreme = len(results["extreme_test"].events_all_runs)
    assert n_flat > 0
    assert n_extreme < 0.1 * n_flat

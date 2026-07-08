import numpy as np
import pytest

from sim import agent, config, meals
from sim.population import ATTR_ORDER, Population, persona_gamma_vector


def make_population(n=1, gamma=None, gamma_cost=1.2, lam=None):
    if gamma is None:
        gamma = np.tile(persona_gamma_vector("household"), (n, 1))
    if lam is None:
        lam = np.zeros((n, 3))
    return Population(
        persona_idx=np.zeros(n, dtype=int),
        gamma=np.asarray(gamma, dtype=float).reshape(n, len(ATTR_ORDER)),
        gamma_cost=np.full(n, gamma_cost, dtype=float),
        lam=lam,
    )


class DummyScenario:
    eta_t_hr_offsets = {}
    eta_k = {}


def test_softmax_probs_sum_to_one():
    pop = make_population(n=5)
    state = agent.init_state(5)
    rng = np.random.default_rng(0)
    hunger = agent.hunger_of(state.h, state.tau, 12.5)
    fired = np.ones(5, dtype=bool)
    result = agent.which(state, pop, DummyScenario(), price_t=0.25, fired=fired,
                          hunger=hunger, rng=rng)
    sums = result.probs.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0)


def test_raising_price_shifts_choice_toward_wood_away_from_big_ecook():
    """nyama_choma_oven (ELEC) vs. nyama_choma_open_fire (FIRE, e_kwh=0) is the same
    dish via two cooking methods -- the cleanest apples-to-apples price-sensitivity check."""
    pop = make_population(n=1)
    state = agent.init_state(1)
    rng = np.random.default_rng(0)
    hunger = agent.hunger_of(state.h, state.tau, 12.5)
    fired = np.ones(1, dtype=bool)

    low = agent.which(state, pop, DummyScenario(), price_t=0.0, fired=fired,
                       hunger=hunger, rng=rng)
    high = agent.which(state, pop, DummyScenario(), price_t=5.0, fired=fired,
                        hunger=hunger, rng=rng)

    big_ecook = meals.IDX_BY_NAME["nyama_choma_oven_kachumbari"]
    big_wood = meals.IDX_BY_NAME["nyama_choma_open_fire"]
    wood_prob_low = low.probs[0, big_wood]
    wood_prob_high = high.probs[0, big_wood]

    assert wood_prob_high > wood_prob_low
    assert high.probs[0, big_ecook] < low.probs[0, big_ecook]


def test_hunger_grows_when_meal_skipped_and_resets_on_eating():
    h = np.zeros((1, 3), dtype=int)
    tau0 = np.array([0.0])
    tau_later = np.array([10.0])

    hunger_fresh = agent.hunger_of(h, tau0, t_hr=12.0)
    hunger_hungry = agent.hunger_of(h, tau_later, t_hr=12.0)
    assert hunger_hungry > hunger_fresh

    h_ate_lunch = h.copy()
    h_ate_lunch[0, 1] = meals.IDX_BY_NAME["ugali_ndengu_stew"] + 1
    hunger_after_eating = agent.hunger_of(h_ate_lunch, np.array([0.0]), t_hr=12.0)
    assert hunger_after_eating < hunger_hungry


def test_stage_windows_respected_no_dinner_at_8am():
    assert agent.STAGE_ORDER[agent.active_stage(8.0)] == "breakfast"
    assert agent.STAGE_ORDER[agent.active_stage(20.0)] == "dinner"

    pop = make_population(n=1)
    rng = np.random.default_rng(0)
    scenario = DummyScenario()

    state = agent.init_state(1)
    fr_morning = agent.fire(state, t_hr=8.0, population=pop, scenario=scenario, rng=rng)
    assert fr_morning.stage_idx == 0
    assert fr_morning.eligible[0]

    fr_evening = agent.fire(state, t_hr=20.0, population=pop, scenario=scenario, rng=rng)
    assert fr_evening.stage_idx == 2

    state_dinner_done = agent.init_state(1)
    state_dinner_done.h[0, 2] = meals.IDX_BY_NAME["ugali_ndengu_stew"] + 1
    fr_after = agent.fire(state_dinner_done, t_hr=20.0, population=pop, scenario=scenario, rng=rng)
    assert not fr_after.eligible[0]
    assert not fr_after.fired[0]


def test_overnight_has_no_active_stage():
    assert agent.active_stage(2.0) == -1

"""Stretch goal: export one persona's coarse-resolution chain as a PRISM DTMC -- and validate it
against the real Monte Carlo simulator (see compute_exact_properties / monte_carlo_comparison).

30-minute blocks (T'=48/day), tau capped at 12 hours. This is a separate,
simplified export reusing the same equations as sim.agent -- no individual
noise (uses the persona-mean gamma/gamma_cost/lam), a flat reference tariff
(price = p_bar), and the reference scenario (eta = 0). It demonstrates the
model is portable to a formal-verification tool; it is not a re-derivation
of the full 5-minute Monte Carlo simulator's exact transition probabilities.
_fire_prob's per-block firing probability q IS rescaled (BLOCKS_PER_COARSE)
to keep this export's firing *rate* consistent with sim.agent.fire's, despite
the coarser block -- an earlier version used sim.agent's 5-minute-calibrated
q directly as this export's 30-minute q, silently giving every agent 1/6th as
many effective chances to fire per day and understating completion so badly
it flipped complete-day from the modal outcome to the rare one. See
monte_carlo_comparison's docstring for the numbers.

Also omitted, for the same "exact DTMC" reason: sim.agent.fire's per-block
sigma_logit_noise (a DTMC needs a fixed transition probability per state, not
one redrawn per realisation -- marginalising a sigmoid over Gaussian noise
has no closed form) and repeat_meal_prob (tractable in principle, just not
implemented here -- would need the reward/eligibility bookkeeping reworked
to allow a stage to fire more than once).

Because PRISM's modelling language has no exp()/sigmoid/softmax, every
transition probability is computed exactly in Python (reusing sim.agent's
hazard/utility formulas) and emitted as an explicit numeric DTMC.

compute_exact_properties solves that same chain exactly (forward propagation
of the state distribution -- the same linear algebra PRISM's own model
checker performs), so the "cross-check against Monte Carlo" this module has
always claimed to enable can actually be run from Python, without PRISM
itself installed. monte_carlo_comparison runs the real simulator under this
export's own simplifications (mean-only population, its noise terms zeroed)
so the two numbers are answering the same question and should now agree up
to genuine remaining differences (block size, tau cap).
"""
from __future__ import annotations

import os

import numpy as np

from sim import agent, config, meals
from sim import population as population_mod
from sim import run as run_mod
from sim.population import persona_gamma_vector, persona_gamma_cost, persona_lam_vector
from sim.utils import softmax

BLOCK_MIN = 30
T_COARSE = 24 * 60 // BLOCK_MIN                      # 48 half-hour blocks/day
TAU_CAP_HR = 12.0
TAU_TICKS_MAX = int(TAU_CAP_HR / (BLOCK_MIN / 60.0))  # 24 half-hour ticks
PRICE_FLAT = config.TARIFF.p_bar                      # flat reference tariff
BLOCKS_PER_COARSE = round(BLOCK_MIN / config.STATE.block_minutes)  # fine (sim.agent-native) blocks per coarse block


def _hunger(h: list[int], tau_hr: float, t_hr: float) -> float:
    n = sum(1 for x in h if x > 0)
    nb = config.nbar(t_hr)
    return max(0, nb - n) + config.HUNGER.kappa * tau_hr


def _fire_prob(h: list[int], tau_hr: float, t_hr: float, lam_vec: np.ndarray,
                gamma_cost: float) -> tuple[int, float, float]:
    hunger = _hunger(h, tau_hr, t_hr)
    # Which stage is "now" -- whichever stage's own bump is highest at this instant, regardless of
    # eaten status, mirroring sim.agent.fire's argmax mechanism (no hard stage_windows_hr clock
    # gate). Only *then* check whether that stage is already eaten -- an agent who just finished
    # lunch doesn't immediately start "thinking about dinner" just because it's their only
    # remaining meal; they wait until dinner's bump has actually overtaken lunch's.
    bump = agent.stage_bump(t_hr)
    stage_idx = int(np.argmax(bump))
    if h[stage_idx] != 0:
        # Already ate today's most-relevant stage. The real simulator has a tiny repeat_meal_prob
        # chance of firing anyway; this export doesn't implement that (see module docstring), so
        # it's a hard 0 here instead.
        return stage_idx, 0.0, hunger
    w = bump[stage_idx]
    # PRICE_FLAT == p_bar (this export always uses the flat reference tariff), so this term is
    # identically 0 here -- see sim.agent.fire's docstring comment for why it's centered on p_bar
    # and clipped to never reward a below-average price, only penalise an above-average one.
    price_term = -config.TIMING.kappa_price_time * gamma_cost * max(PRICE_FLAT - config.TARIFF.p_bar, 0.0)
    logit = w + lam_vec[stage_idx] + config.HUNGER.alpha0 * hunger + price_term
    # DELTA is calibrated for sim.agent.fire's native 5-minute block (config.STATE.block_minutes) --
    # q_fine is "the probability of firing in one such 5-minute block." Using q_fine directly as
    # this 30-minute block's firing probability would silently give an agent only 1/BLOCKS_PER_COARSE
    # as many effective chances to fire per day as the real simulator (48 coarse blocks/day here vs.
    # 288 fine ones there), making this export systematically under-fire relative to sim.run -- badly
    # enough to flip which is more likely, complete vs. incomplete day (see monte_carlo_comparison).
    # Converting via "probability of at least one fire in BLOCKS_PER_COARSE independent fine blocks"
    # (assuming the hazard logit is ~constant across those 6 sub-blocks, a reasonable coarse-graining
    # approximation) keeps this export's firing rate consistent with the real, finer-grained model.
    q_fine = 1.0 / (1.0 + np.exp(-logit)) * config.TIMING.DELTA
    q = float(1.0 - (1.0 - q_fine) ** BLOCKS_PER_COARSE)
    return stage_idx, q, hunger


def _choice_probs(gamma: np.ndarray, gamma_cost: float, hunger: float) -> np.ndarray:
    appeal = gamma @ meals.Z.T
    eta_k = np.zeros(meals.K)  # reference scenario
    cost_term = -gamma_cost * PRICE_FLAT * meals.E_KWH
    hunger_term = meals.ALPHA_K * hunger
    u = appeal + eta_k + cost_term + hunger_term
    return softmax(u[None, :], axis=1)[0]


def build_chain(persona: str = "household"):
    gamma = persona_gamma_vector(persona)
    gamma_cost = persona_gamma_cost(persona)
    lam_vec = persona_lam_vector(persona)

    init = (0, 0, 0, 0, 0)
    layer = {init}
    all_states = {init}
    transitions: dict[tuple, list[tuple[float, tuple, int]]] = {}

    for t in range(T_COARSE):
        next_layer = set()
        t_hr = t * (BLOCK_MIN / 60.0)
        for (tt, hb, hl, hd, tau_ticks) in layer:
            h = [hb, hl, hd]
            tau_hr = tau_ticks * (BLOCK_MIN / 60.0)
            stage_idx, q, hunger = _fire_prob(h, tau_hr, t_hr, lam_vec, gamma_cost)
            tau_next_no_fire = min(tau_ticks + 1, TAU_TICKS_MAX)
            branches: list[tuple[float, tuple, int]] = []

            if q > 0.0:
                probs = _choice_probs(gamma, gamma_cost, hunger)
                for k in range(meals.K):
                    p = q * float(probs[k])
                    if p <= 0.0:
                        continue
                    new_h = [hb, hl, hd]
                    new_h[stage_idx] = k + 1
                    branches.append((p, (t + 1, new_h[0], new_h[1], new_h[2], 0), k + 1))
            p_no_fire = 1.0 - q
            if p_no_fire > 0.0:
                branches.append((p_no_fire, (t + 1, hb, hl, hd, tau_next_no_fire), 0))

            transitions[(tt, hb, hl, hd, tau_ticks)] = branches
            for _, ns, _ in branches:
                next_layer.add(ns)
                all_states.add(ns)
        layer = next_layer

    for s in layer:  # terminal self-loops at t = T_COARSE
        transitions[s] = [(1.0, s, 0)]

    return transitions, all_states


def compute_exact_properties(transitions: dict, init: tuple = (0, 0, 0, 0, 0)) -> dict[str, float]:
    """The two write_props queries, computed exactly in Python instead of by the external PRISM
    binary -- forward-propagates the exact probability distribution over states one coarse block
    at a time (dist[state] = P(agent is in this state at time t)), which is exactly what PRISM's
    own linear-algebra transient/reachability analysis does over the same chain, just without
    needing PRISM installed to get the same numbers. This is what makes the "Cross-check against
    Monte Carlo" comparison below possible without an external dependency.

    Returns "expected_wood_meals" (R{"wood_meals"}=? [C<=T_COARSE]) and "prob_incomplete_day"
    (P=? [F (t=T_COARSE & (hb=0 | hl=0 | hd=0))]).
    """
    dist: dict[tuple, float] = {init: 1.0}
    expected_wood_meals = 0.0
    for _ in range(T_COARSE):
        next_dist: dict[tuple, float] = {}
        for state, prob_here in dist.items():
            for p, next_state, choice in transitions[state]:
                joint_p = prob_here * p
                if choice != 0 and config.MEALS[choice - 1].fire_only:
                    expected_wood_meals += joint_p
                next_dist[next_state] = next_dist.get(next_state, 0.0) + joint_p
        dist = next_dist
    prob_incomplete = sum(p for (_t, hb, hl, hd, _tau), p in dist.items() if hb == 0 or hl == 0 or hd == 0)
    return {"expected_wood_meals": expected_wood_meals, "prob_incomplete_day": prob_incomplete}


def _mean_only_population(n_agents: int, persona: str) -> population_mod.Population:
    """n_agents identical, noise-free copies of persona's mean gamma/gamma_cost/lam -- the same
    "one representative persona" simplification build_chain uses, but as a real sim.run Population
    so the actual 5-minute-block simulator can be run over it (see monte_carlo_comparison)."""
    gamma = np.tile(persona_gamma_vector(persona), (n_agents, 1))
    gamma_cost = np.full(n_agents, persona_gamma_cost(persona))
    lam = np.tile(persona_lam_vector(persona), (n_agents, 1))
    meals_per_cook = np.full(n_agents, float(config.PERSONAS.meals_per_cook.get(persona, 1)))
    persona_idx = np.full(n_agents, population_mod.PERSONA_NAMES.index(persona), dtype=int)
    return population_mod.Population(persona_idx=persona_idx, gamma=gamma, gamma_cost=gamma_cost, lam=lam,
                                      meals_per_cook=meals_per_cook, bump_center_jitter=np.zeros((n_agents, 3)))


def monte_carlo_comparison(persona: str, n_agents: int = 300, R: int = 30, seed: int = 0) -> dict[str, float]:
    """Runs the REAL simulator (sim.run.simulate_day, full 288 x 5-minute blocks, sim.agent's
    actual fire/which/update) with this export's own simplifications applied on top -- a
    population of n_agents identical noise-free copies of persona's mean parameters
    (_mean_only_population, bump_center_jitter=0), sigma_logit_noise and repeat_meal_prob forced
    to 0 (module docstring: the PRISM chain omits both), flat reference tariff (price=p_bar
    throughout), reference scenario (eta=0) -- so this estimates the *same* quantities
    compute_exact_properties answers exactly, under as close to the same assumptions as the real
    simulator allows. Any remaining gap is attributable only to what's still genuinely different:
    5-minute vs. this export's 30-minute blocks, and tau capped at 24h here vs. TAU_CAP_HR there.

    Mutates config.TIMING for the duration of the run and restores it afterwards -- not
    thread-safe, matches the monkeypatch-style ablation already used in tests/app.py elsewhere in
    this codebase, just without a pytest fixture to do the restoring.
    """
    population = _mean_only_population(n_agents, persona)
    price = np.full(config.STATE.T, PRICE_FLAT)
    scenario = config.SCENARIOS["reference"]

    orig_noise, orig_repeat = config.TIMING.sigma_logit_noise, config.TIMING.repeat_meal_prob
    config.TIMING.sigma_logit_noise = 0.0
    config.TIMING.repeat_meal_prob = 0.0
    try:
        rng = np.random.default_rng(seed)
        total_wood = 0
        stages_by_agent_day = []
        for _ in range(R):
            day = run_mod.simulate_day(population, price, scenario, rng)
            stages_by_agent: dict[int, set] = {}
            for e in day.events:
                stages_by_agent.setdefault(e.agent_idx, set()).add(e.stage_idx)
                if meals.WOOD_MASK[e.meal_idx0]:
                    total_wood += 1
            stages_by_agent_day.append(stages_by_agent)
    finally:
        config.TIMING.sigma_logit_noise, config.TIMING.repeat_meal_prob = orig_noise, orig_repeat

    n_agent_days = n_agents * R
    n_incomplete = sum(sum(1 for a in range(n_agents) if len(stages.get(a, ())) < 3)
                        for stages in stages_by_agent_day)
    return {
        "expected_wood_meals": total_wood / n_agent_days,
        "prob_incomplete_day": n_incomplete / n_agent_days,
        "n_agent_days": n_agent_days,
    }


def write_pm(transitions: dict, path: str, persona: str) -> None:
    lines = [
        f"// Coarse (30-min block) PRISM DTMC export for persona='{persona}'.",
        f"// State: t in [0,{T_COARSE}] (half-hour blocks), hb/hl/hd in [0,{meals.K}] (0 = not eaten,",
        f"// else 1-based meal index), tau in [0,{TAU_TICKS_MAX}] (half-hour ticks since last meal,",
        f"// capped at {TAU_CAP_HR:g}h). last_choice is a transient bookkeeping variable (0 = no meal",
        "// eaten this step, else 1-based meal index) used to attach energy/wood-meal rewards.",
        "// Reference scenario (eta=0), flat tariff (price = p_bar), persona-mean gamma (no individual noise).",
        "dtmc",
        "",
        "module persona",
        f"  t : [0..{T_COARSE}] init 0;",
        f"  hb : [0..{meals.K}] init 0;",
        f"  hl : [0..{meals.K}] init 0;",
        f"  hd : [0..{meals.K}] init 0;",
        f"  tau : [0..{TAU_TICKS_MAX}] init 0;",
        f"  last_choice : [0..{meals.K}] init 0;",
        "",
    ]
    for (tt, hb, hl, hd, tau_ticks), branches in transitions.items():
        guard = f"t={tt} & hb={hb} & hl={hl} & hd={hd} & tau={tau_ticks}"
        parts = []
        for p, (nt, nhb, nhl, nhd, ntau), choice in branches:
            upd = f"(t'={nt})&(hb'={nhb})&(hl'={nhl})&(hd'={nhd})&(tau'={ntau})&(last_choice'={choice})"
            parts.append(f"{p:.10g}:{upd}")
        lines.append(f"  [] {guard} -> " + " + ".join(parts) + ";")
    lines += ["endmodule", ""]

    # A school/kiosk "cook" is an institutional kitchen serving many people at once, not one
    # household (see config.PERSONAS.meals_per_cook) -- scale the energy reward the same way
    # sim.run.simulate_day scales the real demand curve, so this persona's expected daily energy
    # is comparable to what the Monte Carlo simulator would show for the same persona.
    meals_per_cook = config.PERSONAS.meals_per_cook.get(persona, 1)
    lines.append('rewards "energy_kwh"')
    for k, m in enumerate(config.MEALS):
        if m.e_kwh > 0:
            lines.append(f"  (last_choice={k + 1}) : {m.e_kwh * meals_per_cook:g};")
    lines.append("endrewards")
    lines.append("")

    lines.append('rewards "wood_meals"')
    for k, m in enumerate(config.MEALS):
        if m.fire_only:
            lines.append(f"  (last_choice={k + 1}) : 1;")
    lines.append("endrewards")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_props(path: str) -> None:
    lines = [
        "// Expected number of wood meals eaten over the whole day",
        f'R{{"wood_meals"}}=? [ C<={T_COARSE} ]',
        "",
        "// Probability the agent has not completed all 3 meals by day end",
        f"P=? [ F (t={T_COARSE} & (hb=0 | hl=0 | hd=0)) ]",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(persona: str = "household", out_dir: str = "out") -> None:
    transitions, all_states = build_chain(persona)
    os.makedirs(out_dir, exist_ok=True)
    pm_path = os.path.join(out_dir, f"prism_{persona}.pm")
    props_path = os.path.join(out_dir, f"prism_{persona}.props")
    write_pm(transitions, pm_path, persona)
    write_props(props_path)
    print(f"Persona: {persona}")
    print(f"Reachable states: {len(all_states)}")
    print(f"Wrote {pm_path}")
    print(f"Wrote {props_path}")


if __name__ == "__main__":
    main()

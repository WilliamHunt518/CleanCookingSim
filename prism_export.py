"""Stretch goal: export one persona's coarse-resolution chain as a PRISM DTMC.

30-minute blocks (T'=48/day), tau capped at 12 hours. This is a separate,
simplified export reusing the same equations as sim.agent -- no individual
noise (uses the persona-mean gamma/gamma_cost/lam), a flat reference tariff
(price = p_bar), and the reference scenario (eta = 0). It demonstrates the
model is portable to a formal-verification tool; it is not a re-derivation
of the full 5-minute Monte Carlo simulator's exact transition probabilities.

Because PRISM's modelling language has no exp()/sigmoid/softmax, every
transition probability is computed exactly in Python (reusing sim.agent's
hazard/utility formulas) and emitted as an explicit numeric DTMC.
"""
from __future__ import annotations

import os

import numpy as np

from sim import agent, config, meals
from sim.population import persona_gamma_vector, persona_gamma_cost, persona_lam_vector
from sim.utils import softmax

BLOCK_MIN = 30
T_COARSE = 24 * 60 // BLOCK_MIN                      # 48 half-hour blocks/day
TAU_CAP_HR = 12.0
TAU_TICKS_MAX = int(TAU_CAP_HR / (BLOCK_MIN / 60.0))  # 24 half-hour ticks
PRICE_FLAT = config.TARIFF.p_bar                      # flat reference tariff


def _hunger(h: list[int], tau_hr: float, t_hr: float) -> float:
    n = sum(1 for x in h if x > 0)
    nb = config.nbar(t_hr)
    return max(0, nb - n) + config.HUNGER.kappa * tau_hr


def _fire_prob(h: list[int], tau_hr: float, t_hr: float, lam_vec: np.ndarray,
                gamma_cost: float) -> tuple[int, float, float]:
    stage_idx = agent.active_stage(t_hr)
    hunger = _hunger(h, tau_hr, t_hr)
    if stage_idx == -1 or h[stage_idx] != 0:
        return stage_idx, 0.0, hunger
    w = agent.w_of_t(t_hr)
    # PRICE_FLAT == p_bar (this export always uses the flat reference tariff), so this term is
    # identically 0 here -- see sim.agent.fire's docstring comment for why it's centered on p_bar
    # and clipped to never reward a below-average price, only penalise an above-average one.
    price_term = -config.TIMING.kappa_price_time * gamma_cost * max(PRICE_FLAT - config.TARIFF.p_bar, 0.0)
    logit = w + lam_vec[stage_idx] + config.HUNGER.alpha0 * hunger + price_term
    q = float(1.0 / (1.0 + np.exp(-logit)) * config.TIMING.DELTA)
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

    lines.append('rewards "energy_kwh"')
    for k, m in enumerate(config.MEALS):
        if m.e_kwh > 0:
            lines.append(f"  (last_choice={k + 1}) : {m.e_kwh};")
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

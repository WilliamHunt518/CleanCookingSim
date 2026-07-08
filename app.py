"""Interactive Streamlit dashboard: tune parameters, run the sim, watch a
house map animate through the day, and see the exact arithmetic behind any
one decision.

Launch with:  streamlit run app.py

Design:
- Sidebar = "how to run" (scenario, which tariffs to sweep, R, seed, ablation
  switches, the Run button). These are cheap execution settings, not model
  design, so they take effect immediately.
- Parameters tab = "what to simulate" (gamma/lam/DELTA/tariff levels/PI/
  population mix). These live inside an st.form, so nothing changes until
  you explicitly press Save -- no on-the-fly reactivity here.
- A staleness banner compares the live config against a snapshot taken at
  the last Run, so it's always visible when the scoreboard/plots are out of
  date relative to the sliders.
"""
from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.path import Path

from sim import agent, config, meals, plots, run as run_mod, score, tariffs as tariffs_mod
from sim import population as population_mod

st.set_page_config(page_title="Cooking Tariff Simulator", layout="wide")
st.title("Clean-Cooking Mini-Grid Tariff Simulator")


def _snapshot(scenario_name, tariff_names, R, seed, no_cost, no_hunger, no_personas) -> dict:
    """Everything that affects a run's outcome, for staleness comparison."""
    return {
        "n_agents": config.N_AGENTS, "mix": dict(config.PERSONAS.mix),
        "gamma": dict(config.PERSONAS.base_gamma), "gamma_cost": config.PERSONAS.base_gamma_cost,
        "sigma_ind": config.PERSONAS.sigma_ind,
        "school_overrides": {"gamma": dict(config.PERSONAS.school_overrides["gamma"]),
                              "lam": dict(config.PERSONAS.school_overrides["lam"])},
        "DELTA": config.TIMING.DELTA,
        "tariff": (config.TARIFF.p_bar, config.TARIFF.p_lo, config.TARIFF.p_hi, config.TARIFF.cap_kw),
        "PI": config.SCORING.PI,
        "scenario": scenario_name, "tariffs": tuple(tariff_names), "R": R, "seed": seed,
        "no_cost": no_cost, "no_hunger": no_hunger, "no_personas": no_personas,
    }


# ---------------------------------------------------------------------------
# Sidebar: run configuration only (cheap, instant -- not model design)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Run configuration")
    st.caption("These are execution settings, not model design -- they apply immediately. "
               "Persona/tariff/timing values live in the Parameters tab and need an explicit Save.")
    scenario_name = st.selectbox("Scenario", list(config.SCENARIOS.keys()))
    tariff_names = st.multiselect("Tariffs to sweep", list(tariffs_mod.CANDIDATES.keys()),
                                   default=list(tariffs_mod.CANDIDATES.keys()))
    R = st.slider("Monte Carlo runs (R)", 5, 200, 20, 5,
                  help="Lower = faster iteration while tuning; raise for a stable final read.")
    seed = st.number_input("seed", value=config.DEFAULT_SEED, step=1)
    st.caption("Ablation switches")
    no_cost = st.checkbox("--no-cost (zero price sensitivity)")
    no_hunger = st.checkbox("--no-hunger")
    no_personas = st.checkbox("--no-personas (everyone base household)")
    st.divider()
    run_clicked = st.button("Run simulation", type="primary", width="stretch")

if not tariff_names:
    st.error("Select at least one tariff to sweep in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Run (only on button press, a Parameters-tab Save&Run, or first load)
# ---------------------------------------------------------------------------
trigger_run = st.session_state.pop("trigger_run", False)
if run_clicked or trigger_run or "results" not in st.session_state:
    with st.spinner("Simulating..."):
        results, population = run_mod.run_sweep(
            tariff_names, scenario_name=scenario_name, seed=int(seed), R=int(R), n_agents=config.N_AGENTS,
            no_hunger=no_hunger, no_cost=no_cost, no_personas=no_personas,
        )
        board = score.scoreboard(results, population)
        plot_paths = plots.make_all_plots(results, population, out_dir="out")
        st.session_state.update(results=results, population=population, board=board, plot_paths=plot_paths)
        st.session_state["last_run_snapshot"] = _snapshot(scenario_name, tariff_names, R, seed,
                                                            no_cost, no_hunger, no_personas)
        st.session_state.pop("day", None)      # population changed -- invalidate cached day
        st.session_state.pop("day_tariff", None)

if "results" not in st.session_state:
    st.stop()

current_snapshot = _snapshot(scenario_name, tariff_names, R, seed, no_cost, no_hunger, no_personas)
if current_snapshot != st.session_state.get("last_run_snapshot"):
    st.warning("Parameters or run settings have changed since the last run -- the scoreboard, plots and "
               "house map below are stale. Press **Run simulation** (sidebar) to recompute.")

population = st.session_state["population"]
board = st.session_state["board"]

tab_sim, tab_plots, tab_params, tab_explain = st.tabs(
    ["Simulation", "Plots", "Parameters", "Explainability"])

# ---------------------------------------------------------------------------
# Simulation tab: house map + day playback
# ---------------------------------------------------------------------------
with tab_sim:
    st.subheader("Scoreboard")
    st.caption("Lower score wins. score = wood_share + PI * P_exceed")
    st.dataframe(board, width="stretch", hide_index=True)

    st.subheader("House map -- day playback")
    tariff_for_map = st.selectbox("Tariff to visualize", list(tariffs_mod.CANDIDATES.keys()))
    st.caption("Re-simulates a single representative day for the chosen tariff -- cheap, "
               "so this updates immediately without needing Run simulation.")

    if st.session_state.get("day_tariff") != tariff_for_map or "day" not in st.session_state:
        rng_day = np.random.default_rng(int(seed) + 999_983)
        price = tariffs_mod.build_tariff(tariff_for_map)
        scenario_obj = config.SCENARIOS[scenario_name]
        day = run_mod.simulate_day(population, price, scenario_obj, rng_day, no_hunger=no_hunger,
                                    track_agent_power=True)
        st.session_state["day"] = day
        st.session_state["day_tariff"] = tariff_for_map
    day = st.session_state["day"]

    layout_key = f"layout_{population.n_agents}"
    if layout_key not in st.session_state:
        ncols = math.ceil(math.sqrt(population.n_agents))
        rng_layout = np.random.default_rng(0)
        xs, ys = [], []
        for i in range(population.n_agents):
            row, col = divmod(i, ncols)
            jitter = rng_layout.uniform(-0.15, 0.15, size=2)
            xs.append(col + jitter[0])
            ys.append(-row + jitter[1])
        st.session_state[layout_key] = np.column_stack([xs, ys])
    positions = st.session_state[layout_key]

    household_mask = population.persona_idx == 0
    school_mask = ~household_mask
    vmax = max(day.agent_power.max(), 1e-6)
    day_max_kw = max(day.demand_kw.max(), config.TARIFF.cap_kw, 1e-6)
    t_hr_full = np.arange(config.STATE.T) * config.STATE.block_minutes / 60.0

    # A literal house-shaped marker (square base + triangular roof) for households.
    HOUSE_MARKER = Path([(-1, -1), (1, -1), (1, 0.15), (0, 1), (-1, 0.15), (-1, -1)], closed=True)

    def _active_meals(block_idx: int) -> dict[int, int]:
        """agent_idx -> 1-based meal index, for agents mid-cook at this block."""
        active = {}
        for e in day.events:
            if e.start_block <= block_idx < e.start_block + e.duration_blocks:
                active[e.agent_idx] = e.meal_idx0 + 1
        return active

    def _draw_house_map(block_idx: int):
        power = day.agent_power[:, block_idx]
        active = _active_meals(block_idx)
        sizes = 70 + 260 * (power / vmax)

        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.scatter(positions[household_mask, 0], positions[household_mask, 1], c=power[household_mask],
                   cmap="hot_r", vmin=0, vmax=vmax, s=sizes[household_mask], marker=HOUSE_MARKER,
                   edgecolors="tab:blue", linewidths=1.3, label="household")
        ax.scatter(positions[school_mask, 0], positions[school_mask, 1], c=power[school_mask],
                   cmap="hot_r", vmin=0, vmax=vmax, s=sizes[school_mask] * 1.3, marker="s",
                   edgecolors="tab:orange", linewidths=1.3, label="school")
        for agent_idx, meal_idx1 in active.items():
            x, y = positions[agent_idx]
            ax.annotate(f"Cooking {meal_idx1}", (x, y), xytext=(0, -11), textcoords="offset points",
                        ha="center", fontsize=6, color="black",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        ax.set_title("color/size = kW draw", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    def _draw_live_load_curve(block_idx: int):
        fig, ax = plt.subplots(figsize=(6.5, 2.0))
        ax.plot(t_hr_full[: block_idx + 1], day.demand_kw[: block_idx + 1], color="tab:red")
        ax.axhline(config.TARIFF.cap_kw, color="black", linestyle=":", linewidth=1, label="cap")
        ax.axvline(block_idx * config.STATE.block_minutes / 60.0, color="gray", linewidth=0.8)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, day_max_kw * 1.1)
        ax.set_xlabel("hour", fontsize=8)
        ax.set_ylabel("aggregate kW", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        return fig

    speed = st.slider("Playback speed (blocks/sec)", 1, 60, 12)
    block = st.slider("Time of day (block)", 0, config.STATE.T - 1, 0, key="map_block_slider")

    map_col, info_col = st.columns([2, 1])
    map_ph = map_col.empty()
    info_ph = info_col.empty()
    graph_ph = st.empty()

    def _show(block_idx: int) -> None:
        power = day.agent_power[:, block_idx]
        hour = block_idx * config.STATE.block_minutes / 60.0
        hh, mm = divmod(int(round(hour * 60)), 60)

        with info_ph.container():
            st.metric("Time", f"{hh:02d}:{mm:02d}")
            st.metric("Total draw", f"{day.demand_kw[block_idx]:.2f} kW",
                      delta=f"cap {config.TARIFF.cap_kw:g} kW", delta_color="off")
            st.metric("Cooking now", f"{int(np.sum(power > 0))} agents")
            st.metric("Household kW", f"{power[household_mask].sum():.2f}")
            st.metric("School kW", f"{power[school_mask].sum():.2f}")

        fig_map = _draw_house_map(block_idx)
        map_ph.pyplot(fig_map)
        plt.close(fig_map)

        fig_graph = _draw_live_load_curve(block_idx)
        graph_ph.pyplot(fig_graph)
        plt.close(fig_graph)

    _show(block)

    if st.button("Run day -- step through at playback speed"):
        for b in range(0, config.STATE.T):
            _show(b)
            time.sleep(1.0 / speed)

# ---------------------------------------------------------------------------
# Plots tab
# ---------------------------------------------------------------------------
with tab_plots:
    st.subheader("Summary plots")
    cols = st.columns(2)
    for i, path in enumerate(st.session_state["plot_paths"]):
        with cols[i % 2]:
            st.image(path)

# ---------------------------------------------------------------------------
# Parameters tab: deliberate, form-gated editor ("craft a persona")
# ---------------------------------------------------------------------------
with tab_params:
    st.subheader("Model parameters")
    st.caption("Nothing here affects the simulation until you press Save -- edit freely, then "
               "Save (and optionally run) when you're happy with a persona.")

    with st.form("params_form"):
        st.markdown("#### Population")
        c1, c2 = st.columns(2)
        with c1:
            n_agents_in = st.slider("Number of agents", 20, 300, config.N_AGENTS, step=10)
        with c2:
            current_school_pct = 100 * config.PERSONAS.mix["school"] / sum(config.PERSONAS.mix.values())
            pct_school_in = st.slider("School share (%)", 0, 40, int(round(current_school_pct)), step=1)

        st.markdown("#### Household persona -- gamma (Stage 2: which meal)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            taste_in = st.slider("taste", -2.0, 2.0, config.PERSONAS.base_gamma["taste"], 0.05)
        with c2:
            trad_in = st.slider("trad", -2.0, 2.0, config.PERSONAS.base_gamma["trad"], 0.05)
        with c3:
            effort_in = st.slider("effort (keep negative)", -2.0, 2.0, config.PERSONAS.base_gamma["effort"], 0.05)
        with c4:
            fuelcost_in = st.slider("fuelcost (keep negative)", -2.0, 2.0,
                                     config.PERSONAS.base_gamma["fuelcost"], 0.05)
        c1, c2 = st.columns(2)
        with c1:
            gamma_cost_in = st.slider("gamma_cost (price sensitivity -- THE tariff knob)", 0.0, 3.0,
                                       config.PERSONAS.base_gamma_cost, 0.05)
        with c2:
            sigma_ind_in = st.slider("sigma_ind (individual variation)", 0.0, 0.5,
                                      config.PERSONAS.sigma_ind, 0.01)

        st.markdown("#### School persona -- overrides on the household base")
        school_trad_in = st.slider("school trad weight", -1.0, 1.0,
                                    config.PERSONAS.school_overrides["gamma"]["trad"], 0.05)
        lam_now = config.PERSONAS.school_overrides["lam"]
        c1, c2, c3 = st.columns(3)
        with c1:
            lam_b_in = st.slider("lam: breakfast (Stage 1: whether it fires)", -6.0, 3.0,
                                  lam_now.get("breakfast", 0.0), 0.5)
        with c2:
            lam_l_in = st.slider("lam: lunch", -6.0, 3.0, lam_now.get("lunch", 0.0), 0.5)
        with c3:
            lam_d_in = st.slider("lam: dinner", -6.0, 3.0, lam_now.get("dinner", 0.0), 0.5)
        st.caption("More negative lam = that stage effectively switched off (-6 ~ off, matching the "
                   "overnight base logit). This is what makes schools unimodal at lunch.")

        st.markdown("#### Timing")
        delta_in = st.slider("DELTA (hazard scale -- how often anyone cooks at all)", 0.01, 0.5,
                              config.TIMING.DELTA, 0.01)

        st.markdown("#### Tariff")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p_bar_in = st.slider("p_bar", 0.05, 1.0, config.TARIFF.p_bar, 0.01)
        with c2:
            p_lo_in = st.slider("p_lo", 0.0, 1.0, config.TARIFF.p_lo, 0.01)
        with c3:
            p_hi_in = st.slider("p_hi", 0.0, 2.0, config.TARIFF.p_hi, 0.01)
        with c4:
            cap_kw_in = st.slider("cap_kw", 5.0, 100.0, config.TARIFF.cap_kw, 1.0)

        st.markdown("#### Scoring")
        PI_in = st.slider("PI (exceedance penalty weight)", 0.0, 20.0, config.SCORING.PI, 0.5)

        col_save, col_save_run = st.columns(2)
        save_clicked = col_save.form_submit_button("Save parameters", width="stretch")
        save_run_clicked = col_save_run.form_submit_button("Save & run simulation", type="primary",
                                                            width="stretch")

    if save_clicked or save_run_clicked:
        n_school = round(n_agents_in * pct_school_in / 100)
        config.N_AGENTS = n_agents_in
        config.PERSONAS.mix = {"household": n_agents_in - n_school, "school": n_school}
        config.PERSONAS.base_gamma = {"taste": taste_in, "trad": trad_in, "effort": effort_in,
                                       "fuelcost": fuelcost_in}
        config.PERSONAS.base_gamma_cost = gamma_cost_in
        config.PERSONAS.sigma_ind = sigma_ind_in
        config.PERSONAS.school_overrides = {"gamma": {"trad": school_trad_in},
                                             "lam": {"breakfast": lam_b_in, "lunch": lam_l_in,
                                                     "dinner": lam_d_in}}
        config.TIMING.DELTA = delta_in
        config.TARIFF.p_bar, config.TARIFF.p_lo, config.TARIFF.p_hi, config.TARIFF.cap_kw = (
            p_bar_in, p_lo_in, p_hi_in, cap_kw_in)
        config.SCORING.PI = PI_in
        st.success("Parameters saved.")
        if save_run_clicked:
            st.session_state["trigger_run"] = True
        st.rerun()

# ---------------------------------------------------------------------------
# Explainability tab: full glossary + a literal, numeric worked example
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("Parameter glossary")
    for group in config.GROUP_ORDER:
        rows = [p for p in config.all_params() if p.group == group]
        if not rows:
            continue
        with st.expander(f"{group} ({len(rows)} parameters)"):
            df = pd.DataFrame([{"name": r.name, "tbd": r.tbd, "value": repr(r.value), "units": r.units,
                                 "meaning": r.meaning, "effect": r.effect} for r in rows])
            st.dataframe(df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Worked example -- the exact arithmetic for one agent")
    st.caption("Pick a persona, a time, a price, and a hunger state, and see every term substituted "
               "with real numbers -- literally what's multiplied by what.")

    c1, c2, c3 = st.columns(3)
    with c1:
        ex_persona = st.selectbox("Persona", ["household", "school"])
    with c2:
        ex_hour = st.slider("Hour of day", 0.0, 24.0, 19.0, 0.25)
    with c3:
        ex_price = st.slider("Grid price at this hour", 0.0, 2.0, config.TARIFF.p_hi, 0.05)
    c1, c2 = st.columns(2)
    with c1:
        ex_n = st.select_slider("Meals eaten so far today (n)", options=[0, 1, 2, 3], value=1)
    with c2:
        ex_tau = st.slider("Hours since last meal (tau)", 0.0, 24.0, 3.0, 0.5)
    st.caption("Assumes the currently-active stage's slot hasn't been eaten yet (still eligible to fire) -- "
               "this is an illustration of the arithmetic, not a full state replay.")

    gamma_vec = population_mod.persona_gamma_vector(ex_persona)
    gamma_cost_val = population_mod.persona_gamma_cost(ex_persona)
    lam_vec = population_mod.persona_lam_vector(ex_persona)
    scenario_obj = config.SCENARIOS[scenario_name]

    stage_idx = agent.active_stage(ex_hour)
    stage_name = agent.STAGE_ORDER[stage_idx] if stage_idx != -1 else None

    st.markdown("### Stage 1 -- firing hazard (does the agent start cooking at all)")
    if stage_idx == -1:
        st.write(f"No stage window is open at {ex_hour:g}h (overnight gap) -- q = 0 regardless of hunger.")
    else:
        nb = config.nbar(ex_hour)
        hunger = max(0, nb - ex_n) + config.HUNGER.kappa * ex_tau
        w = agent.w_of_t(ex_hour)
        eta_t = agent.eta_t_of_t(ex_hour, scenario_obj)
        lam_val = float(lam_vec[stage_idx])
        logit = w + eta_t + lam_val + config.HUNGER.alpha0 * hunger
        sig = 1.0 / (1.0 + np.exp(-logit))
        q = sig * config.TIMING.DELTA

        st.write(f"Active stage: **{stage_name}**")
        st.latex(r"hunger = \max(0,\ \bar n(t) - n) + \kappa \cdot \tau")
        st.write(f"= max(0, {nb} - {ex_n}) + {config.HUNGER.kappa:g} x {ex_tau:g}"
                 f" = {max(0, nb - ex_n):.3f} + {config.HUNGER.kappa * ex_tau:.3f} = **{hunger:.3f}**")

        st.latex(r"logit = w(t) + \eta_t + \lambda_{persona,stage} + \alpha_0 \cdot hunger")
        st.write(f"= {w:.3f} + {eta_t:.3f} + {lam_val:.3f} + {config.HUNGER.alpha0:g} x {hunger:.3f}"
                 f" = {w:.3f} + {eta_t:.3f} + {lam_val:.3f} + {config.HUNGER.alpha0 * hunger:.3f}"
                 f" = **{logit:.3f}**")

        st.latex(r"q = \mathrm{sigmoid}(logit) \times DELTA")
        st.write(f"= sigmoid({logit:.3f}) x {config.TIMING.DELTA:g} = {sig:.4f} x {config.TIMING.DELTA:g}"
                 f" = **{q:.4f}** (probability this agent starts cooking in this one 5-minute block)")

    st.markdown("### Stage 2 -- which meal (softmax choice)")
    st.caption("Shown regardless of whether Stage 1 fires, to display the full arithmetic.")
    nb2 = config.nbar(ex_hour)
    hunger2 = max(0, nb2 - ex_n) + config.HUNGER.kappa * ex_tau
    eta_k = agent.eta_k_vector(scenario_obj)

    rows = []
    for k, name in enumerate(meals.MEAL_NAMES):
        z = meals.Z[k]
        appeal_terms = gamma_vec * z
        appeal = float(appeal_terms.sum())
        cost_term = -gamma_cost_val * ex_price * meals.E_KWH[k]
        hunger_term = meals.ALPHA_K[k] * hunger2
        u = appeal + float(eta_k[k]) + cost_term + hunger_term
        rows.append({
            "meal": name,
            "taste x g_taste": f"{z[0]:.2f} x {gamma_vec[0]:.2f} = {appeal_terms[0]:.3f}",
            "trad x g_trad": f"{z[1]:.2f} x {gamma_vec[1]:.2f} = {appeal_terms[1]:.3f}",
            "effort x g_effort": f"{z[2]:.2f} x {gamma_vec[2]:.2f} = {appeal_terms[2]:.3f}",
            "fuelcost x g_fuelcost": f"{z[3]:.2f} x {gamma_vec[3]:.2f} = {appeal_terms[3]:.3f}",
            "appeal (sum)": round(appeal, 3),
            "eta_k": round(float(eta_k[k]), 3),
            "cost = -g_cost x price x e_k": f"-{gamma_cost_val:.2f} x {ex_price:.2f} x {meals.E_KWH[k]:.2f}"
                                             f" = {cost_term:.3f}",
            "hunger_term = alpha_k x hunger": f"{meals.ALPHA_K[k]:.2f} x {hunger2:.3f} = {hunger_term:.3f}",
            "u (total)": round(u, 3),
        })
    df_u = pd.DataFrame(rows)
    st.dataframe(df_u, width="stretch", hide_index=True)

    u_values = df_u["u (total)"].to_numpy(dtype=float)
    exp_u = np.exp(u_values - u_values.max())
    probs = exp_u / exp_u.sum()
    st.write("softmax: prob_k = exp(u_k - max(u)) / sum_j exp(u_j - max(u))")
    df_probs = pd.DataFrame({"meal": meals.MEAL_NAMES, "u": u_values,
                              "exp(u - max)": exp_u, "softmax prob": probs})
    st.dataframe(df_probs, width="stretch", hide_index=True)

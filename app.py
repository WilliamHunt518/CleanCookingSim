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
        "persona_gamma_offsets": {p: dict(off) for p, off in config.PERSONAS.persona_gamma_offsets.items()},
        "persona_lam": {p: dict(lam) for p, lam in config.PERSONAS.persona_lam.items()},
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

    persona_masks = {name: (population.persona_idx == idx)
                      for idx, name in enumerate(population_mod.PERSONA_NAMES)}
    vmax = max(day.agent_power.max(), 1e-6)
    day_max_kw = max(day.demand_kw.max(), config.TARIFF.cap_kw, 1e-6)
    t_hr_full = np.arange(config.STATE.T) * config.STATE.block_minutes / 60.0

    # A literal house-shaped marker (square base + triangular roof) for households; school
    # and kiosk get distinct shapes so all three personas are visually unambiguous.
    HOUSE_MARKER = Path([(-1, -1), (1, -1), (1, 0.15), (0, 1), (-1, 0.15), (-1, -1)], closed=True)
    PERSONA_MARKERS = {"household": HOUSE_MARKER, "school": "s", "kiosk": "^"}
    PERSONA_EDGE_COLORS = {"household": "tab:blue", "school": "tab:orange", "kiosk": "tab:green"}
    PERSONA_SIZE_MULT = {"household": 1.0, "school": 1.3, "kiosk": 1.15}

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
        for name, mask in persona_masks.items():
            if not np.any(mask):
                continue
            ax.scatter(positions[mask, 0], positions[mask, 1], c=power[mask],
                       cmap="hot_r", vmin=0, vmax=vmax, s=sizes[mask] * PERSONA_SIZE_MULT[name],
                       marker=PERSONA_MARKERS[name], edgecolors=PERSONA_EDGE_COLORS[name],
                       linewidths=1.3, label=name)
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
            for name, mask in persona_masks.items():
                st.metric(f"{name} kW", f"{power[mask].sum():.2f}")

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
        c1, c2, c3 = st.columns(3)
        mix_now = config.PERSONAS.mix
        mix_total = sum(mix_now.values())
        with c1:
            n_agents_in = st.slider("Number of agents", 20, 300, config.N_AGENTS, step=10)
        with c2:
            pct_school_in = st.slider("School share (%)", 0, 40,
                                       int(round(100 * mix_now["school"] / mix_total)), step=1)
        with c3:
            pct_kiosk_in = st.slider("Kiosk share (%)", 0, 40,
                                      int(round(100 * mix_now["kiosk"] / mix_total)), step=1)
        st.caption("Household gets whatever's left of the 100% after school + kiosk.")

        st.markdown("#### Household persona -- gamma (Stage 2: which meal), the base every "
                    "persona's offsets sit on top of")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            taste_in = st.slider("taste", -2.0, 2.0, config.PERSONAS.base_gamma["taste"], 0.05)
        with c2:
            tradition_in = st.slider("tradition", -2.0, 2.0, config.PERSONAS.base_gamma["tradition"], 0.05)
        with c3:
            kid_in = st.slider("kid (kid-acceptance)", -2.0, 2.0, config.PERSONAS.base_gamma["kid"], 0.05)
        with c4:
            batch_in = st.slider("batch (leftover potential)", -2.0, 2.0, config.PERSONAS.base_gamma["batch"], 0.05)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ing_cost_in = st.slider("ing_cost (keep negative)", -3.0, 1.0,
                                     config.PERSONAS.base_gamma["ing_cost"], 0.05)
        with c2:
            prep_min_in = st.slider("prep_min (keep negative)", -3.0, 1.0,
                                     config.PERSONAS.base_gamma["prep_min"], 0.05)
        with c3:
            kcal_in = st.slider("kcal", -2.0, 2.0, config.PERSONAS.base_gamma["kcal"], 0.05)
        with c4:
            fuelcost_in = st.slider("fuelcost (keep negative)", -3.0, 1.0,
                                     config.PERSONAS.base_gamma["fuelcost"], 0.05)
        c1, c2 = st.columns(2)
        with c1:
            gamma_cost_in = st.slider("gamma_cost (price sensitivity -- THE tariff knob)", 0.0, 3.0,
                                       config.PERSONAS.base_gamma_cost, 0.05)
        with c2:
            sigma_ind_in = st.slider("sigma_ind (individual variation)", 0.0, 0.5,
                                      config.PERSONAS.sigma_ind, 0.01)

        st.markdown("#### School persona -- gamma offsets (additive on the household base) + timing")
        school_off = config.PERSONAS.persona_gamma_offsets.get("school", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            school_taste_off = st.slider("taste offset", -2.0, 2.0, school_off.get("taste", 0.0), 0.05,
                                          key="school_taste_off")
        with c2:
            school_kid_off = st.slider("kid offset", -2.0, 2.0, school_off.get("kid", 0.0), 0.05,
                                        key="school_kid_off")
        with c3:
            school_batch_off = st.slider("batch offset", -2.0, 2.0, school_off.get("batch", 0.0), 0.05,
                                          key="school_batch_off")
        with c4:
            school_ing_cost_off = st.slider("ing_cost offset", -3.0, 3.0, school_off.get("ing_cost", 0.0), 0.05,
                                             key="school_ing_cost_off")
        school_lam_now = config.PERSONAS.persona_lam.get("school", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            school_lam_b = st.slider("lam: breakfast", -6.0, 3.0, school_lam_now.get("breakfast", 0.0), 0.5,
                                      key="school_lam_b")
        with c2:
            school_lam_l = st.slider("lam: lunch", -6.0, 3.0, school_lam_now.get("lunch", 0.0), 0.5,
                                      key="school_lam_l")
        with c3:
            school_lam_d = st.slider("lam: dinner", -6.0, 3.0, school_lam_now.get("dinner", 0.0), 0.5,
                                      key="school_lam_d")
        st.caption("Unlisted features (tradition/prep_min/kcal/fuelcost) get 0 offset -- school inherits "
                   "the household value on those. lam is a per-stage REPLACEMENT (not additive): "
                   "-6 ~ that stage effectively off, matching the overnight base logit -- this is what "
                   "makes schools unimodal at lunch.")

        st.markdown("#### Kiosk persona (mama ntilie / food vendor) -- gamma offsets + timing")
        kiosk_off = config.PERSONAS.persona_gamma_offsets.get("kiosk", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kiosk_taste_off = st.slider("taste offset", -2.0, 2.0, kiosk_off.get("taste", 0.0), 0.05,
                                         key="kiosk_taste_off")
        with c2:
            kiosk_batch_off = st.slider("batch offset", -2.0, 2.0, kiosk_off.get("batch", 0.0), 0.05,
                                         key="kiosk_batch_off")
        with c3:
            kiosk_prep_min_off = st.slider("prep_min offset", -3.0, 3.0, kiosk_off.get("prep_min", 0.0), 0.05,
                                            key="kiosk_prep_min_off")
        with c4:
            kiosk_ing_cost_off = st.slider("ing_cost offset", -3.0, 3.0, kiosk_off.get("ing_cost", 0.0), 0.05,
                                            key="kiosk_ing_cost_off")
        kiosk_lam_now = config.PERSONAS.persona_lam.get("kiosk", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            kiosk_lam_b = st.slider("lam: breakfast", -6.0, 3.0, kiosk_lam_now.get("breakfast", 0.0), 0.5,
                                     key="kiosk_lam_b")
        with c2:
            kiosk_lam_l = st.slider("lam: lunch", -6.0, 3.0, kiosk_lam_now.get("lunch", 0.0), 0.5,
                                     key="kiosk_lam_l")
        with c3:
            kiosk_lam_d = st.slider("lam: dinner", -6.0, 3.0, kiosk_lam_now.get("dinner", 0.0), 0.5,
                                     key="kiosk_lam_d")
        st.caption("master_table_Z.md gives no kiosk operating-hours data -- lam defaults to 0 "
                   "(household-like timing) until you set otherwise.")

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
        n_kiosk = round(n_agents_in * pct_kiosk_in / 100)
        n_household = max(0, n_agents_in - n_school - n_kiosk)
        config.N_AGENTS = n_agents_in
        config.PERSONAS.mix = {"household": n_household, "school": n_school, "kiosk": n_kiosk}
        config.PERSONAS.base_gamma = {"taste": taste_in, "tradition": tradition_in, "kid": kid_in,
                                       "batch": batch_in, "ing_cost": ing_cost_in, "prep_min": prep_min_in,
                                       "kcal": kcal_in, "fuelcost": fuelcost_in}
        config.PERSONAS.base_gamma_cost = gamma_cost_in
        config.PERSONAS.sigma_ind = sigma_ind_in
        config.PERSONAS.persona_gamma_offsets = {
            "school": {"taste": school_taste_off, "kid": school_kid_off, "batch": school_batch_off,
                       "ing_cost": school_ing_cost_off},
            "kiosk": {"taste": kiosk_taste_off, "batch": kiosk_batch_off, "prep_min": kiosk_prep_min_off,
                      "ing_cost": kiosk_ing_cost_off},
        }
        config.PERSONAS.persona_lam = {
            "school": {"breakfast": school_lam_b, "lunch": school_lam_l, "dinner": school_lam_d},
            "kiosk": {"breakfast": kiosk_lam_b, "lunch": kiosk_lam_l, "dinner": kiosk_lam_d},
        }
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

    with st.expander(f"Meal menu ({meals.K} meals, from master_table_Z.md)"):
        st.caption("Full source-table data per meal -- physical/cost and nutrition columns aren't "
                   "gamma-weighted (except kcal), but are shown here for reference.")
        menu_df = pd.DataFrame({
            "meal": meals.MEAL_NAMES, "type": meals.MEAL_TYPE, "fire_only": meals.WOOD_MASK,
            "duration_min": [m.dbar_min for m in config.MEALS], "e_kwh": meals.E_KWH,
            "charcoal_kes": meals.CHARCOAL_KES, "ing_cost_kes": meals.ING_COST_KES,
            "kcal": meals.KCAL, "protein_g": meals.PROTEIN_G, "carb_g": meals.CARB_G, "fat_g": meals.FAT_G,
            "taste": meals.Z[:, population_mod.ATTR_ORDER.index("taste")],
            "tradition": meals.Z[:, population_mod.ATTR_ORDER.index("tradition")],
            "kid": meals.Z[:, population_mod.ATTR_ORDER.index("kid")],
            "batch": meals.Z[:, population_mod.ATTR_ORDER.index("batch")],
        })
        st.dataframe(menu_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Worked example -- the exact arithmetic for one agent")
    st.caption("Pick a persona, a time, a price, and a hunger state, and see every term substituted "
               "with real numbers -- literally what's multiplied by what.")

    c1, c2, c3 = st.columns(3)
    with c1:
        ex_persona = st.selectbox("Persona", population_mod.PERSONA_NAMES)
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

    st.caption("ing_cost/prep_min/kcal/fuelcost are pre-normalised to ~0-1 (see ING_COST_MAX_KES etc. "
               "in the glossary above) before gamma is applied -- the table below shows that "
               "normalised z value x gamma, per attribute.")
    rows = []
    for k, name in enumerate(meals.MEAL_NAMES):
        z = meals.Z[k]
        appeal_terms = gamma_vec * z
        appeal = float(appeal_terms.sum())
        cost_term = -gamma_cost_val * ex_price * meals.E_KWH[k]
        hunger_term = meals.ALPHA_K[k] * hunger2
        u = appeal + float(eta_k[k]) + cost_term + hunger_term
        row = {"meal": name}
        for a, attr in enumerate(population_mod.ATTR_ORDER):
            row[f"{attr} x g_{attr}"] = f"{z[a]:.2f} x {gamma_vec[a]:.2f} = {appeal_terms[a]:.3f}"
        row["appeal (sum)"] = round(appeal, 3)
        row["eta_k"] = round(float(eta_k[k]), 3)
        row["cost = -g_cost x price x e_k"] = (f"-{gamma_cost_val:.2f} x {ex_price:.2f} x "
                                                f"{meals.E_KWH[k]:.2f} = {cost_term:.3f}")
        row["hunger_term = alpha_k x hunger"] = f"{meals.ALPHA_K[k]:.2f} x {hunger2:.3f} = {hunger_term:.3f}"
        row["u (total)"] = round(u, 3)
        rows.append(row)
    df_u = pd.DataFrame(rows)
    st.dataframe(df_u, width="stretch", hide_index=True)

    u_values = df_u["u (total)"].to_numpy(dtype=float)
    exp_u = np.exp(u_values - u_values.max())
    probs = exp_u / exp_u.sum()
    st.write("softmax: prob_k = exp(u_k - max(u)) / sum_j exp(u_j - max(u))")
    df_probs = pd.DataFrame({"meal": meals.MEAL_NAMES, "u": u_values,
                              "exp(u - max)": exp_u, "softmax prob": probs})
    st.dataframe(df_probs, width="stretch", hide_index=True)

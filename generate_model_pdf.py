"""Generate a simple PDF reference explaining how the model works, with every
current parameter value. Uses only matplotlib (already a dependency) via its
multi-page PdfPages backend -- no new dependency for a one-off doc generator.

Run:  python generate_model_pdf.py   ->  writes out/model_reference.pdf
"""
from __future__ import annotations

import datetime
import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sim import config, meals
from sim.population import ATTR_ORDER, PERSONA_NAMES, persona_gamma_vector, persona_lam_vector

PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches
WRAP = 100


def _new_page():
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
    ax.axis("off")
    return fig, ax


def _wrapped(text: str, width: int = WRAP) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def title_page(pdf: PdfPages) -> None:
    fig, ax = _new_page()
    ax.text(0.5, 0.75, "Clean-Cooking Mini-Grid Tariff Simulator", ha="center", fontsize=22,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.70, "Model Reference", ha="center", fontsize=16, transform=ax.transAxes)
    ax.text(0.5, 0.63,
            "How the agent-based day-ahead tariff simulator works, and every parameter value\n"
            "currently configured in sim/config.py.",
            ha="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.5, 0.55, f"Generated {datetime.date.today().isoformat()}", ha="center", fontsize=9,
            color="gray", transform=ax.transAxes)

    n_tbd = len(config.tbd_params())
    n_total = len(config.all_params())
    summary = (
        f"Population: {config.N_AGENTS} agents across personas {', '.join(PERSONA_NAMES)}\n"
        f"Meal menu: {meals.K} meals (from master_table_Z.md)\n"
        f"Day resolution: {config.STATE.T} blocks of {config.STATE.block_minutes:g} minutes\n"
        f"Parameters: {n_total} total, {n_tbd} flagged as placeholder guesses (see the audit table)"
    )
    ax.text(0.5, 0.40, summary, ha="center", va="top", fontsize=10, family="monospace",
            transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def overview_page(pdf: PdfPages) -> None:
    fig, ax = _new_page()
    ax.text(0, 1.0, "How the model works", fontsize=16, fontweight="bold",
            transform=ax.transAxes, va="top")

    body = """
Each agent (household / school / kiosk) is a discrete-time Markov chain stepped
through one day in 5-minute blocks. Per-block state:

  t     -- block index, 0..T-1
  h     -- [h_B, h_L, h_D], each 0 (not yet eaten) or the 1-based index of the
            meal eaten at that stage (breakfast/lunch/dinner)
  tau   -- hours since the agent last ate (capped)

At every block, each agent goes through three steps:

  Stage 1 (fire) -- a hazard draw decides whether the agent starts cooking
      this block, IF the currently-active meal stage is still open (inside
      its clock-time window) AND that stage's slot in h is still empty.

  Stage 2 (which) -- if it fires, the agent picks ONE meal from the menu via
      a softmax over each meal's utility. The menu mixes electric meals
      (grid energy, subject to the swept tariff) and fire/wood meals (zero
      grid energy -- immune to the tariff, but not free: they carry a
      charcoal-cost disutility instead).

  Stage 3 (update) -- the chosen meal is written into h, tau resets to 0,
      and a cooking event (start block, meal, agent) is recorded. Recorded
      events feed the aggregate demand curve (via each meal's kW profile)
      and the wood-share / exceedance scoring.

A day-ahead TARIFF is a price(t) array announced at t=0. Three candidate
shapes are swept (flat / evening-peak / solar-following), all normalised to
the same time-average price, and scored by:

  score(tariff) = wood_share + PI * P_exceed

where wood_share is the fraction of all meals cooked on fire across the
whole Monte Carlo sweep, and P_exceed is the fraction of simulated days on
which aggregate demand ever exceeded the grid cap.
"""
    ax.text(0, 0.94, body.strip("\n"), fontsize=10, family="monospace", va="top",
            transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def equations_page(pdf: PdfPages) -> None:
    fig, ax = _new_page()
    ax.text(0, 1.0, "The equations, with current values substituted", fontsize=16,
            fontweight="bold", transform=ax.transAxes, va="top")

    bh = config.TIMING.bump_heights
    bc = config.TIMING.bump_centers_hr
    sw = config.TIMING.stage_windows_hr
    text = f"""
Hunger (drives both stages):
  n      = count of nonzero entries of h
  hunger = max(0, nbar(t) - n) + kappa * tau
           kappa = {config.HUNGER.kappa:g}
           nbar(t) = 0 before {config.HUNGER.nbar_step_times_hr['1']:g}h, 1 before {config.HUNGER.nbar_step_times_hr['2']:g}h,
                     2 before {config.HUNGER.nbar_step_times_hr['3']:g}h, else 3

Stage 1 -- firing hazard:
  w(t)   = overnight_base + sum over stages of (height_s - overnight_base)
           * exp(-0.5 * ((t - center_s) / width)^2)
           overnight_base = {config.TIMING.overnight_base_logit:g}, width = {config.TIMING.bump_width_hr:g}h
           breakfast: center={bc['breakfast']:g}h height={bh['breakfast']:g}   window={sw['breakfast']}
           lunch:     center={bc['lunch']:g}h height={bh['lunch']:g}   window={sw['lunch']}
           dinner:    center={bc['dinner']:g}h height={bh['dinner']:g}   window={sw['dinner']}
  logit  = w(t) + eta_t(t) + lam[persona, stage] + alpha0 * hunger
           alpha0 = {config.HUNGER.alpha0:g}
  q      = sigmoid(logit) * DELTA          DELTA = {config.TIMING.DELTA:g}
           (q is the probability this agent starts cooking in THIS block;
            only evaluated if the active stage's slot in h is still empty)

Stage 2 -- which meal (softmax choice), for each meal k with attributes z_k:
  z_k    = [{', '.join(ATTR_ORDER)}]
           (ing_cost/prep_min/kcal/fuelcost are normalised to ~0-1 by dividing
            by ING_COST_MAX_KES={config.ING_COST_MAX_KES:g}, PREP_MIN_MAX={config.PREP_MIN_MAX:g},
            KCAL_MAX={config.KCAL_MAX:g}, CHARCOAL_KES_MAX={config.CHARCOAL_KES_MAX:g} respectively;
            fuelcost is 0 for electric meals, charcoal_kes_norm for fire-only meals)
  alpha_k = ALPHA_SCALE * kcal_norm_k        ALPHA_SCALE = {config.ALPHA_SCALE:g}
  u_k    = gamma . z_k + eta_k - gamma_cost * price(t) * e_k + alpha_k * hunger
  P(k)   = exp(u_k) / sum_j exp(u_j)          (e_k = 0 for fire-only meals,
                                                so price(t) never affects them)

Scoring:
  score(tariff) = wood_share + PI * P_exceed        PI = {config.SCORING.PI:g}
  Monte Carlo:  R = {config.SCORING.R} independent days per tariff, fresh dice, same agents.
"""
    ax.text(0, 0.94, text.strip("\n"), fontsize=8.7, family="monospace", va="top",
            transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def persona_page(pdf: PdfPages) -> None:
    fig, ax = _new_page()
    ax.text(0, 1.0, "Personas", fontsize=16, fontweight="bold", transform=ax.transAxes, va="top")

    lines = ["Household base gamma (every other persona's offsets are added on top of this):"]
    for a in ATTR_ORDER:
        lines.append(f"  {a:10s} = {config.PERSONAS.base_gamma[a]:+.2f}")
    lines.append(f"  gamma_cost = {config.PERSONAS.base_gamma_cost:g}  (price sensitivity, shared by all personas)")
    lines.append(f"  sigma_ind  = {config.PERSONAS.sigma_ind:g}  (individual noise std dev, sampled once per agent)")
    lines.append("")

    for persona in PERSONA_NAMES:
        if persona == "household":
            continue
        lines.append(f"{persona} (offsets on the household base; blank = inherits household unchanged):")
        offsets = config.PERSONAS.persona_gamma_offsets.get(persona, {})
        if offsets:
            for attr, delta in offsets.items():
                effective = config.PERSONAS.base_gamma[attr] + delta
                lines.append(f"  {attr:10s} offset {delta:+.2f}  ->  effective {effective:+.2f}")
        else:
            lines.append("  (no gamma offsets)")
        lam_over = config.PERSONAS.persona_lam.get(persona, {})
        lines.append(f"  lam (per-stage hazard REPLACEMENT, not additive): {lam_over or '(none -- household timing)'}")
        lines.append("")

    lines.append("Population mix (agent counts, scaled to N_AGENTS):")
    for name, count in config.PERSONAS.mix.items():
        lines.append(f"  {name:10s} {count}")
    lines.append(f"  N_AGENTS   = {config.N_AGENTS}")

    ax.text(0, 0.94, "\n".join(lines), fontsize=9.5, family="monospace", va="top",
            transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def meal_table_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.88])
    ax.axis("off")
    ax.text(0, 1.02, f"Meal menu ({meals.K} meals, from master_table_Z.md)", fontsize=14,
            fontweight="bold", transform=ax.transAxes, va="bottom")

    cols = ["#", "meal", "type", "min", "kWh", "kcal", "ing.KES", "taste", "trad.", "kid", "batch", "fire_only"]
    rows = []
    for i, m in enumerate(config.MEALS):
        rows.append([str(m.idx), m.name, m.meal_type, f"{m.dbar_min:g}", f"{m.e_kwh:g}", f"{m.kcal:g}",
                     f"{m.ing_cost_kes:g}", f"{m.taste:g}", f"{m.tradition:g}", f"{m.kid:g}", f"{m.batch:g}",
                     "Y" if m.fire_only else ""])

    table = ax.table(cellText=rows, colLabels=cols, loc="upper left", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.auto_set_column_width(col=list(range(len(cols))))
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
        if c == 1:
            cell.set_text_props(ha="left")
            cell._loc = "left"
    pdf.savefig(fig)
    plt.close(fig)


def glossary_pages(pdf: PdfPages) -> None:
    line_height = 0.028
    for group in config.GROUP_ORDER:
        rows = [p for p in config.all_params() if p.group == group]
        if not rows:
            continue
        fig, ax = _new_page()
        y = 1.0
        ax.text(0, y, f"Parameter glossary -- {group}", fontsize=15, fontweight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.05

        for prm in rows:
            flag = "  [TBD]" if prm.tbd else ""
            entry = f"{prm.name}{flag}\n  value: {prm.value!r}   units: {prm.units}\n"
            entry += "  meaning: " + _wrapped(prm.meaning, WRAP - 11) .replace("\n", "\n           ") + "\n"
            entry += "  effect:  " + _wrapped(prm.effect, WRAP - 11).replace("\n", "\n           ")
            n_lines = entry.count("\n") + 1
            needed = n_lines * line_height + 0.02

            if y - needed < 0.03:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = _new_page()
                y = 1.0
                ax.text(0, y, f"Parameter glossary -- {group} (continued)", fontsize=15,
                        fontweight="bold", transform=ax.transAxes, va="top")
                y -= 0.05

            ax.text(0, y, entry, fontsize=8.3, family="monospace", va="top", transform=ax.transAxes)
            y -= needed

        pdf.savefig(fig)
        plt.close(fig)


def main(out_path: str = "out/model_reference.pdf") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with PdfPages(out_path) as pdf:
        title_page(pdf)
        overview_page(pdf)
        equations_page(pdf)
        persona_page(pdf)
        meal_table_page(pdf)
        glossary_pages(pdf)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

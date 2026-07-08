"""Generate a simple PDF reference explaining how the model works, with every
current parameter value. Uses only matplotlib (already a dependency) via its
multi-page PdfPages backend -- no new dependency for a one-off doc generator.

Run:  python generate_model_pdf.py   ->  writes model_reference.pdf (repo root)
"""
from __future__ import annotations

import datetime
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


def _render_flow(pdf: PdfPages, items: list[tuple[str, object]], page_title: str) -> None:
    """Lay out a mixed heading/paragraph/equation/table stream, paginating as needed."""
    state = {"fig": None, "ax": None, "y": 1.0}

    def start_page(continued: bool = False):
        fig, ax = _new_page()
        title = f"{page_title} (cont.)" if continued else page_title
        ax.text(0, 1.0, title, fontsize=16, fontweight="bold", va="top", transform=ax.transAxes)
        state["fig"], state["ax"], state["y"] = fig, ax, 1.0 - 0.055

    def new_page():
        pdf.savefig(state["fig"])
        plt.close(state["fig"])
        start_page(continued=True)

    start_page()
    for item in items:
        kind, payload = item[0], item[1]
        eq_height = item[2] if kind == "eq" and len(item) > 2 else 0.065
        ax = state["ax"]
        if kind == "heading":
            if state["y"] - 0.05 < 0.06:
                new_page()
                ax = state["ax"]
            state["ax"].text(0, state["y"], payload, fontsize=12.5, fontweight="bold", va="top",
                              transform=state["ax"].transAxes)
            state["y"] -= 0.045
        elif kind == "para":
            wrapped = _wrapped(payload, 96)
            n_lines = wrapped.count("\n") + 1
            need = n_lines * 0.0235 + 0.015
            if state["y"] - need < 0.06:
                new_page()
            state["ax"].text(0, state["y"], wrapped, fontsize=10.3, va="top",
                              transform=state["ax"].transAxes)
            state["y"] -= need
        elif kind == "eq":
            if state["y"] - eq_height < 0.06:
                new_page()
            state["ax"].text(0.5, state["y"], payload, fontsize=14, va="top", ha="center",
                              transform=state["ax"].transAxes)
            state["y"] -= eq_height
        elif kind == "space":
            state["y"] -= payload
    pdf.savefig(state["fig"])
    plt.close(state["fig"])


def notation_page(pdf: PdfPages) -> None:
    fig, ax = _new_page()
    ax.text(0, 1.0, "Notation", fontsize=16, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0, 0.955,
            "Every symbol used in the equations on the following pages, in order of first use.",
            fontsize=10, va="top", transform=ax.transAxes)

    rows = [
        (r"$t$", r"$\{0,\dots,T-1\}$", "Discrete block index within one simulated day"),
        (r"$s(t)$", r"$\{B,L,D\}$", "Active meal stage (breakfast/lunch/dinner) at time $t$ (clock-time only)"),
        (r"$\mathbf{h}_t$", r"$\{0,\dots,K\}^3$", r"Meal-slot state $(h_t^B,h_t^L,h_t^D)$; 0 = not yet eaten, else 1-based meal index"),
        (r"$\tau_t$", r"$[0,\tau_{\max}]$", "Hours since the agent last ate"),
        (r"$n_t$", r"$\{0,1,2,3\}$", r"Meals eaten so far today, $n_t=|\{s:h_t^s\neq 0\}|$"),
        (r"$\bar n(t)$", r"$\{0,1,2,3\}$", "Step function: meals a schedule-following agent should have eaten by $t$"),
        (r"$H_t$", r"$\mathbb{R}_{\geq 0}$", "Hunger"),
        (r"$\kappa$", "scalar", r"Weight of $\tau_t$ in hunger"),
        (r"$w(t)$", r"$\mathbb{R}$", "Baseline hazard logit: time-of-day profile (3 Gaussian bumps)"),
        (r"$\eta_t,\ \eta_k$", r"$\mathbb{R}$", "Scenario offsets: extra timing bias / extra meal-appeal"),
        (r"$\lambda_{p,s}$", r"$\mathbb{R}$", "Per-(persona,stage) hazard bias -- shapes WHEN a persona cooks"),
        (r"$\alpha_0$", "scalar", "Weight of hunger in the hazard logit"),
        (r"$\Delta$", r"$(0,1]$", "Hazard-to-probability scale (also the max per-block probability)"),
        (r"$\ell_t$", r"$\mathbb{R}$", "Hazard logit"),
        (r"$q_t$", r"$[0,\Delta]$", "Probability the agent starts cooking in block $t$"),
        (r"$K$", "integer", "Number of meals on the menu"),
        (r"$\mathbf{z}_k$", r"$\mathbb{R}^8$", "Meal $k$'s attribute vector (Table 2)"),
        (r"$\boldsymbol{\gamma}$", r"$\mathbb{R}^8$", "Agent's own weights on the 8 meal attributes -- shapes WHICH meal it picks"),
        (r"$\gamma_{\mathrm{cost}}$", "scalar", "Agent's price sensitivity"),
        (r"$p(t)$", r"$\mathbb{R}_{\geq 0}$", "Grid tariff price at time $t$ (currency/kWh)"),
        (r"$e_k$", r"$\mathbb{R}_{\geq 0}$", "Meal $k$'s grid energy draw (kWh); 0 for fire-only meals"),
        (r"$\alpha_k$", r"$\mathbb{R}$", "Meal $k$'s hunger-boost coefficient"),
        (r"$u_k$", r"$\mathbb{R}$", "Meal $k$'s utility"),
        (r"$P(k)$", r"$[0,1]$", "Probability of choosing meal $k$, conditional on firing"),
        (r"$\pi$", "tariff", "One candidate day-ahead tariff: a full price path $p(t)$, announced at $t=0$"),
    ]
    table = ax.table(cellText=[[s, d, m] for s, d, m in rows],
                      colLabels=["Symbol", "Domain", "Meaning"],
                      loc="upper left", cellLoc="left", colLoc="left",
                      colWidths=[0.16, 0.16, 0.68], bbox=[0, 0, 1, 0.90])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_text_props(ha="left")
        if r == 0:
            cell.set_text_props(fontweight="bold", ha="left")
        cell.PAD = 0.01
    pdf.savefig(fig)
    plt.close(fig)


def model_pages(pdf: PdfPages) -> None:
    bh, bc, sw = config.TIMING.bump_heights, config.TIMING.bump_centers_hr, config.TIMING.stage_windows_hr
    items: list[tuple[str, object]] = [
        ("heading", "1. State space"),
        ("para",
         "Each agent is simulated as an independent discrete-time stochastic process over one day, "
         "divided into $T$ blocks of $\\Delta t_{\\mathrm{blk}}$ minutes. The agent's state at block "
         "$t$ is the triple"),
        ("eq", r"$s_t = (t,\ \mathbf{h}_t,\ \tau_t)$"),
        ("para",
         "where $\\mathbf{h}_t=(h_t^{B},h_t^{L},h_t^{D})\\in\\{0,\\dots,K\\}^3$ records which menu item "
         "(if any) has been eaten at each of the three meal stages, and $\\tau_t$ is the number of hours "
         "since the agent last ate. Let $n_t=|\\{s:h_t^s\\neq 0\\}|$ be the number of meals eaten so far "
         "today. Hunger, which drives both stages below, is"),
        ("eq", r"$H_t = \max\left(0,\ \bar n(t) - n_t\right) + \kappa\,\tau_t$"),
        ("space", 0.01),

        ("heading", "2. Stage 1 -- cooking hazard"),
        ("para",
         "At each block, if the active stage $s(t)$ still has an empty slot ($h_t^{s(t)}=0$), the agent "
         "begins cooking this block with probability"),
        ("eq", r"$q_t = \Delta\cdot\sigma(\ell_t), \qquad \sigma(x)=\dfrac{1}{1+e^{-x}}$", 0.078),
        ("para", "where the hazard logit sums a time-of-day baseline, a scenario offset, a persona/stage "
                 "bias, and a hunger term:"),
        ("eq", r"$\ell_t = w(t) + \eta_t + \lambda_{p,s(t)} + \alpha_0 H_t$"),
        ("para",
         "$\\lambda_{p,s}$ is a fixed bias, specific to both the agent's persona $p$ (household, school "
         "or kiosk) and the currently-active stage $s$, added directly into the logit -- it does not "
         "depend on the meal chosen, only on who is cooking and when. It is how a persona's whole daily "
         "rhythm is shaped without writing separate code per persona: e.g. giving school a strongly "
         "negative $\\lambda$ at breakfast and dinner and a strongly positive one at lunch makes it fire "
         "almost exclusively around lunchtime, while household keeps $\\lambda_{p,s}=0$ at every stage."),
        ("para",
         "and $w(t)$ relaxes to an overnight floor $b$ away from three Gaussian bumps, one per meal "
         "stage $s\\in\\{B,L,D\\}$ with centre $c_s$, peak height $\\beta_s$ and shared width $\\sigma_w$:"),
        ("eq", r"$w(t) = b + \sum_{s} (\beta_s - b)\, \exp\left(-\frac{1}{2}\left(\frac{t-c_s}{\sigma_w}\right)^{2}\right)$", 0.085),
        ("para",
         f"Current calibration: $b={config.TIMING.overnight_base_logit:g}$, "
         f"$\\sigma_w={config.TIMING.bump_width_hr:g}$h; "
         f"$(c_B,\\beta_B)=({bc['breakfast']:g}\\mathrm{{h}}, {bh['breakfast']:g})$, "
         f"$(c_L,\\beta_L)=({bc['lunch']:g}\\mathrm{{h}}, {bh['lunch']:g})$, "
         f"$(c_D,\\beta_D)=({bc['dinner']:g}\\mathrm{{h}}, {bh['dinner']:g})$; "
         f"$\\alpha_0={config.HUNGER.alpha0:g}$, $\\kappa={config.HUNGER.kappa:g}$, "
         f"$\\Delta={config.TIMING.DELTA:g}$. Stage $s$ may only fire inside its own clock-time window "
         f"(breakfast {sw['breakfast']}, lunch {sw['lunch']}, dinner {sw['dinner']}, all in 24h clock hours)."),
        ("space", 0.01),

        ("heading", "3. Stage 2 -- meal choice"),
        ("para",
         "Conditional on firing, the agent chooses meal $k\\in\\{1,\\dots,K\\}$ via a multinomial-logit "
         "(softmax) discrete-choice model over utilities"),
        ("eq", r"$u_k = \boldsymbol{\gamma}^{\top}\mathbf{z}_k + \eta_k - \gamma_{\mathrm{cost}}\,p(t)\,e_k + \alpha_k H_t$"),
        ("eq", r"$P(k) = \dfrac{\exp(u_k)}{\sum_{j=1}^{K}\exp(u_j)}$", 0.1),
        ("para",
         "$\\mathbf{z}_k$ is meal $k$'s fixed 8-dimensional attribute vector -- taste, tradition, "
         "kid-acceptance, batch potential, ingredient cost, prep labour, kcal, fuel cost (Table 2) -- the "
         "same for every agent. $\\boldsymbol{\\gamma}$ is the opposite: it belongs to the agent, not the "
         "meal, and says how much that agent personally cares about each of those 8 attributes (e.g. a "
         "large positive weight on tradition, a negative weight on ingredient cost). Every persona has "
         "its own mean $\\boldsymbol{\\gamma}$ (Table 3), and every individual agent draws its own "
         "$\\boldsymbol{\\gamma}$ once at initialisation as a small random perturbation around its "
         "persona's mean -- so agents of the same persona behave similarly but not identically. "
         "$e_k=0$ for fire-only meals, so $p(t)$ never affects their utility: this is the single "
         "mechanism by which the grid tariff can push demand toward or away from wood."),
        ("para",
         f"Current calibration: $\\alpha_k$ = ALPHA_SCALE $\\times\\,\\mathrm{{kcal}}_k$ / KCAL_MAX, with "
         f"ALPHA_SCALE = {config.ALPHA_SCALE:g}. ing_cost, prep_min, kcal and fuel-cost are each "
         f"normalised to roughly $[0,1]$ by dividing by a reference maximum "
         f"(ING_COST_MAX_KES = {config.ING_COST_MAX_KES:g}, PREP_MIN_MAX = {config.PREP_MIN_MAX:g}, "
         f"KCAL_MAX = {config.KCAL_MAX:g}, CHARCOAL_KES_MAX = {config.CHARCOAL_KES_MAX:g}) so every "
         f"gamma-weighted feature sits on a comparable scale."),
        ("space", 0.01),

        ("heading", "4. Demand and tariffs"),
        ("para",
         "Every cooking event contributes a kW profile (a boxcar of height $e_k/D$ over its sampled "
         "duration $D$, by default) to the aggregate demand curve. A tariff $\\pi$ is simply a price "
         "path $p(t)$ over the $T$ blocks of the day, announced in full at $t=0$ (agents never see "
         "future prices before they're announced, but the whole path is fixed and known from block 0 "
         "onward). Three candidate price paths are defined:"),
        ("eq", r"$\mathrm{flat}:\quad p(t) = \bar p$", 0.05),
        ("eq", r"$\mathrm{evening\_peak}:\quad p(t) = p_{lo} + (p_{hi}-p_{lo})\cdot\mathbb{1}[t \in W_{peak}]$", 0.05),
        ("eq", r"$\mathrm{solar\_following}:\quad p(t) = p_{hi} - (p_{hi}-p_{lo})\cdot \dfrac{PV(t)}{\max_t PV(t)}$", 0.078),
        ("para",
         "where $\\bar p$ is a common target average price, $p_{lo}/p_{hi}$ are low/high price levels, "
         "$W_{peak}$ is a fixed evening window, and $PV(t)$ is a stylised solar-generation curve used "
         "only to shape this one tariff (it is not part of any agent's decision). Every candidate is "
         "then rescaled by a constant factor so its own time-average equals $\\bar p$ exactly -- this "
         "keeps the comparison about the shape of the price over the day, not its overall level."),
    ]
    _render_flow(pdf, items, "The model")


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
        lines.append("  lam (per-stage hazard REPLACEMENT, not additive):")
        if lam_over:
            lam_str = ", ".join(f"{stage}={val:+.1f}" for stage, val in lam_over.items())
            lines.append(f"    {lam_str}")
        else:
            lines.append("    (none -- household timing)")
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


def main(out_path: str = "model_reference.pdf") -> None:
    with PdfPages(out_path) as pdf:
        title_page(pdf)
        notation_page(pdf)
        model_pages(pdf)
        persona_page(pdf)
        meal_table_page(pdf)
        glossary_pages(pdf)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

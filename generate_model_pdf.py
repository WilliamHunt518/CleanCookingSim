"""Generate a simple PDF reference explaining how the model works, with every
current parameter value. Uses only matplotlib (already a dependency) via its
multi-page PdfPages backend -- no new dependency for a one-off doc generator.

Run:  python generate_model_pdf.py   ->  writes model_reference.pdf (repo root)
"""
from __future__ import annotations

import datetime
import re

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
    """Word-wrap text, treating each $...$ math span as one indivisible token so a
    line break can never land inside it (which would break mathtext rendering)."""
    if not text:
        return ""
    tokens = re.findall(r"\$[^$]*\$\S*|\S+", text)
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for tok in tokens:
        sep = 1 if current else 0
        if current and current_len + sep + len(tok) > width:
            lines.append(" ".join(current))
            current, current_len = [tok], len(tok)
        else:
            current.append(tok)
            current_len += sep + len(tok)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


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
    rows = [
        (r"$t$", r"$\{0,\dots,T-1\}$", "Discrete block index within one simulated day"),
        (r"$s(t)$", r"$\{B,L,D\}$", "Active meal stage (breakfast/lunch/dinner) at time $t$ (clock-time only)"),
        (r"$\mathbf{h}_t$", r"$\{0,\dots,K\}^3$", r"Meal-slot state $(h_t^B,h_t^L,h_t^D)$; 0 = not yet eaten, else 1-based meal index"),
        (r"$\tau_t$", r"$[0,\tau_{\max}]$", "Hours since the agent last ate"),
        (r"$n_t$", r"$\{0,1,2,3\}$", r"Meals eaten so far today, $n_t=|\{s:h_t^s\neq 0\}|$"),
        (r"$\bar n(t)$", r"$\{0,1,2,3\}$", "Step function: meals a schedule-following agent should have eaten by $t$"),
        (r"$H_t$", r"$\mathbb{R}_{\geq 0}$", "Hunger"),
        (r"$\kappa$", "scalar", r"Weight of $\tau_t$ in hunger"),
        (r"$\sigma(x)$", r"$(0,1)$", "Logistic sigmoid function -- squashes any real number into a probability"),
        (r"$\Delta$", r"$(0,1]$", "Hazard-to-probability scale (also the max per-block probability)"),
        (r"$\ell_t$", r"$\mathbb{R}$", "Hazard logit (an unbounded score, not itself a probability)"),
        (r"$w(t)$", r"$\mathbb{R}$", "Baseline hazard logit: time-of-day profile (3 Gaussian bumps)"),
        (r"$\eta_t,\ \eta_k$", r"$\mathbb{R}$", "Scenario offsets: extra timing bias / extra meal-appeal"),
        (r"$\lambda_{p,s}$", r"$\mathbb{R}$", "Per-(persona,stage) hazard bias -- shapes WHEN a persona cooks"),
        (r"$\alpha_0$", "scalar", "Weight of hunger in the hazard logit"),
        (r"$b$", "scalar", "Deeply negative overnight floor $w(t)$ relaxes to, away from any meal"),
        (r"$c_s$", "hour", "Clock time where stage $s$'s hazard bump peaks"),
        (r"$\beta_s$", "scalar", r"Peak value of $w(t)$ exactly at $c_s$ -- how strong the pull is at that peak"),
        (r"$\sigma_w$", "hours", r"Shared width of the 3 bumps (unrelated to the function $\sigma(\cdot)$ above)"),
        (r"$q_t$", r"$[0,\Delta]$", "Probability the agent starts cooking in block $t$"),
        (r"$K$", "integer", "Number of meals on the menu"),
        (r"$\mathbf{z}_k$", r"$\mathbb{R}^8$", "Meal $k$'s fixed attribute vector, same for every agent (Table 2)"),
        (r"$\boldsymbol{\gamma}$", r"$\mathbb{R}^8$", "Agent's own weights on the 8 meal attributes -- shapes WHICH meal it picks"),
        (r"$\gamma_{\mathrm{cost}}$", "scalar", "Agent's price sensitivity"),
        (r"$p(t)$", r"$\mathbb{R}_{\geq 0}$", "Grid tariff price at time $t$ (currency/kWh)"),
        (r"$e_k$", r"$\mathbb{R}_{\geq 0}$", "Meal $k$'s grid energy draw (kWh); 0 for fire-only meals"),
        (r"$\alpha_k$", r"$\mathbb{R}$", "Meal $k$'s hunger-boost coefficient"),
        (r"$u_k$", r"$\mathbb{R}$", "Meal $k$'s utility (meaningful only relative to other meals' $u_j$)"),
        (r"$P(k)$", r"$[0,1]$", "Probability of choosing meal $k$, conditional on firing"),
        (r"$\pi$", "tariff", "One candidate day-ahead tariff: a full price path $p(t)$, announced at $t=0$"),
        (r"$\bar p$", "currency/kWh", "Common target average price every tariff candidate is rescaled to"),
        (r"$p_{lo},\ p_{hi}$", "currency/kWh", "Low / high price levels used by the non-flat tariffs"),
        (r"$W_{peak}$", "time window", "Fixed evening window used by the evening_peak tariff (e.g. 17:00-21:00)"),
        (r"$PV(t)$", r"$\mathbb{R}_{\geq 0}$", "Stylised solar-output curve; shapes solar_following only, not agent choices"),
    ]
    _paginated_table(pdf, rows, page_title="Notation",
                      subtitle="Every symbol used in the equations on the following pages, in order of first use.")


def _paginated_table(pdf: PdfPages, rows: list[tuple[str, str, str]], page_title: str,
                      subtitle: str = "", rows_per_page: int = 20) -> None:
    chunks = [rows[i:i + rows_per_page] for i in range(0, len(rows), rows_per_page)]
    for page_i, chunk in enumerate(chunks):
        fig, ax = _new_page()
        title = page_title if page_i == 0 else f"{page_title} (cont.)"
        ax.text(0, 1.0, title, fontsize=16, fontweight="bold", va="top", transform=ax.transAxes)
        top = 0.90
        if page_i == 0 and subtitle:
            ax.text(0, 0.955, subtitle, fontsize=10, va="top", transform=ax.transAxes)
            top = 0.87
        table = ax.table(cellText=[[s, d, m] for s, d, m in chunk],
                          colLabels=["Symbol", "Domain", "Meaning"],
                          loc="upper left", cellLoc="left", colLoc="left",
                          colWidths=[0.16, 0.16, 0.68], bbox=[0, 0, 1, top])
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
         "divided into $T$ blocks of $\\Delta t_{\\mathrm{blk}}$ minutes ($T=288$ blocks of 5 minutes "
         "by default). The agent's state at block $t$ (an integer counting up from 0 at midnight) is "
         "the triple"),
        ("eq", r"$s_t = (t,\ \mathbf{h}_t,\ \tau_t)$"),
        ("para",
         "where $\\mathbf{h}_t=(h_t^{B},h_t^{L},h_t^{D})\\in\\{0,\\dots,K\\}^3$ is three numbers, one per "
         "meal stage (Breakfast/Lunch/Dinner): each is 0 if that meal hasn't been eaten yet today, or "
         "the 1-based index of whichever of the $K$ menu items was eaten there. $\\tau_t$ (tau) is simply "
         "the number of hours since the agent last ate anything at all, counting up from 0 every time it "
         "eats. Let $n_t=|\\{s:h_t^s\\neq 0\\}|$ (a count from 0 to 3) be how many of the three meals the "
         "agent has already eaten today. Hunger, which drives both stages below, combines both ideas:"),
        ("eq", r"$H_t = \max\left(0,\ \bar n(t) - n_t\right) + \kappa\,\tau_t$"),
        ("para",
         "$\\bar n(t)$ (n-bar) is a step function giving the number of meals an agent 'on schedule' should "
         "have eaten by clock time $t$ (0 before breakfast, 1 after breakfast, 2 after lunch, 3 after "
         "dinner) -- so the first term is 0 while the agent is keeping pace, and only turns positive once "
         "it has fallen behind. $\\kappa$ (kappa) is a single fixed constant that adds a small, steadily "
         "growing amount of hunger for every hour since the last meal, on top of that -- so hunger never "
         "actually reaches zero for long, even for an agent that is technically 'on schedule'."),
        ("space", 0.01),

        ("heading", "2. Stage 1 -- cooking hazard"),
        ("para",
         "At each block, if the active stage $s(t)$ (whichever of breakfast/lunch/dinner is open at the "
         "current clock time) still has an empty slot ($h_t^{s(t)}=0$, i.e. the agent hasn't already "
         "eaten that meal), the agent begins cooking this one block with probability"),
        ("eq", r"$q_t = \Delta\cdot\sigma(\ell_t), \qquad \sigma(x)=\dfrac{1}{1+e^{-x}}$", 0.078),
        ("para",
         "$\\sigma(x)$, the logistic sigmoid function, takes any real number $x$ -- negative, positive, "
         "huge or tiny -- and squashes it into the open interval $(0,1)$: very negative $x$ gives a "
         "result near 0, very positive $x$ gives a result near 1, and $x=0$ gives exactly 0.5. It exists "
         "so that the unbounded score $\\ell_t$ below can be turned into an actual probability. "
         "$\\Delta$ (a fixed constant between 0 and 1, e.g. 0.15) then scales that probability down "
         "further and doubles as a hard ceiling: however hungry or however deep into a meal-time peak "
         "the agent is, $q_t$ can never exceed $\\Delta$."),
        ("para",
         "$\\ell_t$ (a lower-case script L, read \"the hazard logit\") is the unbounded, arbitrary-scale "
         "score that $\\sigma(\\cdot)$ above converts into a probability -- it is not a probability "
         "itself, and can be any real number, very negative or very positive. It is built by adding up "
         "four separate influences, so each can be reasoned about on its own:"),
        ("eq", r"$\ell_t = w(t) + \eta_t + \lambda_{p,s(t)} + \alpha_0 H_t$"),
        ("para",
         "$w(t)$ (defined below) is a baseline pull toward cooking that depends only on the clock time "
         "$t$, the same for every agent regardless of persona. $\\eta_t$ (eta) is an optional, "
         "scenario-specific nudge to that baseline -- it is exactly 0 unless a non-default scenario is "
         "active. $\\lambda_{p,s}$ (lambda) is a fixed number specific to both the agent's persona $p$ "
         "(household, school or kiosk) and the currently-active stage $s$ -- it does not depend on which "
         "meal is eventually chosen, only on who is cooking and when. This single number is how a whole "
         "persona's daily rhythm is shaped without writing separate code per persona: e.g. giving school "
         "a strongly negative $\\lambda$ at breakfast and dinner, and a strongly positive one at lunch, "
         "makes it fire almost exclusively around lunchtime, while household keeps every "
         "$\\lambda_{p,s}=0$ (no bias at any stage). Finally $\\alpha_0$ (alpha-zero, a single fixed "
         "constant shared by everyone) controls how strongly the hunger score $H_t$ from Section 1 pushes "
         "up the logit -- and hence the firing probability -- as the agent gets hungrier."),
        ("para",
         "$w(t)$ itself relaxes to a very negative overnight floor $b$ away from three Gaussian-shaped "
         "\"bumps\", one per meal stage $s\\in\\{B,L,D\\}$:"),
        ("eq", r"$w(t) = b + \sum_{s} (\beta_s - b)\, \exp\left(-\frac{1}{2}\left(\frac{t-c_s}{\sigma_w}\right)^{2}\right)$", 0.085),
        ("para",
         "$b$ is a single, deeply negative constant (e.g. $-6$): far from every meal-time bump, "
         "$w(t)\\approx b$, so $\\ell_t$ is deeply negative, $\\sigma(\\ell_t)$ is essentially 0, and "
         "agents essentially never start cooking outside meal times. Each stage $s$ then contributes one "
         "bump, described by three numbers: $c_s$ is the clock time (in hours) at which that stage's pull "
         "is strongest -- e.g. $c_s=12.5$ means the lunch pull peaks at 12:30. $\\beta_s$ (beta) is the "
         "actual value $w(t)$ reaches exactly at that peak time $c_s$ -- so $\\beta_s$ directly sets how "
         "strong the pull to cook is at the busiest moment of that meal window (a bigger $\\beta_s$ means "
         "a bigger logit, and hence a higher firing probability, right at that time of day). And "
         "$\\sigma_w$ is a single width, shared by all three bumps, controlling how quickly that pull "
         "fades as $t$ moves away from $c_s$ in either direction -- a small $\\sigma_w$ gives a sharp, "
         "narrow peak, a large $\\sigma_w$ spreads the pull over a wider span of the day. (This $\\sigma_w$ "
         "is an unrelated constant to the sigmoid FUNCTION $\\sigma(\\cdot)$ above -- same Greek letter, "
         "different role: one is a fixed width, the other is a function applied to $\\ell_t$.) Away from "
         "all three peaks, the exponential term is essentially 0 and $w(t)$ returns to the floor $b$."),
        ("para",
         f"Current calibration: $b={config.TIMING.overnight_base_logit:g}$, "
         f"$\\sigma_w={config.TIMING.bump_width_hr:g}$h; "
         f"$(c_B,\\beta_B)=({bc['breakfast']:g}\\mathrm{{h}}, {bh['breakfast']:g})$, "
         f"$(c_L,\\beta_L)=({bc['lunch']:g}\\mathrm{{h}}, {bh['lunch']:g})$, "
         f"$(c_D,\\beta_D)=({bc['dinner']:g}\\mathrm{{h}}, {bh['dinner']:g})$; "
         f"$\\alpha_0={config.HUNGER.alpha0:g}$, $\\kappa={config.HUNGER.kappa:g}$, "
         f"$\\Delta={config.TIMING.DELTA:g}$. There is no hard clock-time window per stage: at every "
         f"block, stage $s^*=\\arg\\max_s w_s(t)$ (whichever stage's own bump is currently highest) is "
         f"the one this agent's decision is about, and it can only fire if that stage hasn't been eaten "
         f"yet today. Nominal reference windows for stage $s$ -- roughly where $s$ tends to win that "
         f"contest -- are breakfast {sw['breakfast']}, lunch {sw['lunch']}, dinner {sw['dinner']} (24h "
         f"clock hours)."),
        ("space", 0.01),

        ("heading", "3. Stage 2 -- meal choice"),
        ("para",
         "Conditional on firing, the agent chooses one meal $k\\in\\{1,\\dots,K\\}$ via a multinomial-logit "
         "(softmax) discrete-choice model: every meal $k$ gets a utility score $u_k$, and meals with a "
         "higher utility are more likely (but never certain) to be chosen."),
        ("eq", r"$u_k = \boldsymbol{\gamma}^{\top}\mathbf{z}_k + \eta_k - \gamma_{\mathrm{cost}}\,p(t)\,e_k + \alpha_k H_t$"),
        ("para",
         "$u_k$ has no inherent units or absolute meaning by itself -- only its size RELATIVE to the "
         "other meals' $u_j$ matters, via the softmax formula below. $\\mathbf{z}_k$ is meal $k$'s fixed "
         "attribute vector: 8 numbers -- taste, tradition, kid-acceptance, batch potential, ingredient "
         "cost, prep labour, kcal, fuel cost (Table 2) -- that describe the MEAL, and are the same for "
         "every agent. $\\boldsymbol{\\gamma}$ (gamma) is the opposite: it belongs to the AGENT, not the "
         "meal, and is also 8 numbers, one per attribute, saying how much this particular agent cares "
         "about each one. $\\boldsymbol{\\gamma}^{\\top}\\mathbf{z}_k$ (\"gamma dot z sub k\") is a dot "
         "product -- writing out both as 8 numbers each, it is just shorthand for the sum of each pair "
         "multiplied together:"),
        ("eq", r"$\boldsymbol{\gamma}^{\top}\mathbf{z}_k = \gamma_1 z_{k,1} + \gamma_2 z_{k,2} + \dots + \gamma_8 z_{k,8}$", 0.05),
        ("para",
         "i.e. each of the meal's 8 attributes gets multiplied by how much this agent personally weights "
         "that attribute, and the 8 results are added together. For example, a large positive $\\gamma$ "
         "on tradition combined with a meal's high tradition score $z_k$ adds a lot to $u_k$; a negative "
         "$\\gamma$ on ingredient cost means a pricier meal (higher $z_k$ on that attribute) SUBTRACTS "
         "from $u_k$ instead. Every persona has its own mean $\\boldsymbol{\\gamma}$ (Table 3), and every "
         "individual agent draws its own $\\boldsymbol{\\gamma}$ once at initialisation as a small random "
         "perturbation around its persona's mean -- so agents of the same persona behave similarly but "
         "not identically."),
        ("para",
         "$\\eta_k$ (eta) is a flat, scenario-specific bonus or penalty added to meal $k$'s utility -- "
         "zero in the default scenario, but e.g. a \"festival\" scenario could give a celebratory dish a "
         "positive $\\eta_k$ to represent extra festive appeal, independent of the agent's own "
         "$\\boldsymbol{\\gamma}$. $\\gamma_{\\mathrm{cost}}$ is a single non-negative number, specific to "
         "the agent, describing how price-sensitive it is; multiplying it by the current grid price "
         "$p(t)$ and by meal $k$'s grid energy draw $e_k$ (in kWh), then SUBTRACTING that product from "
         "$u_k$, makes pricier grid electricity less attractive -- more so for agents with a larger "
         "$\\gamma_{\\mathrm{cost}}$. $e_k=0$ for fire-only meals, so $p(t)$ never affects their utility "
         "at all: this is the single mechanism by which the grid tariff can push demand toward or away "
         "from wood. Finally $\\alpha_k$ (alpha sub $k$) adds extra utility to meal $k$ in proportion to "
         "the agent's current hunger $H_t$ (from Section 1) -- calibrated so more filling, higher-kcal "
         "meals get a bigger hunger boost than light ones."),
        ("eq", r"$P(k) = \dfrac{\exp(u_k)}{\sum_{j=1}^{K}\exp(u_j)}$", 0.1),
        ("para",
         "$P(k)$ turns the $K$ utility scores into an actual probability distribution over meals. "
         "Exponentiating each $u_k$ turns DIFFERENCES in utility into RATIOS of likelihood -- a meal 2 "
         "utility-points better than another becomes about $e^2\\approx 7.4$ times more likely to be "
         "chosen, whatever the two absolute utility values happen to be. Dividing by the sum over every "
         "meal $j$ then rescales the result so the $K$ probabilities add up to exactly 1."),
        ("para",
         f"Current calibration: $\\alpha_k$ = ALPHA_SCALE $\\times\\,\\mathrm{{kcal}}_k$ / KCAL_MAX, with "
         f"ALPHA_SCALE = {config.ALPHA_SCALE:g} (so a meal's hunger boost simply scales with how many "
         f"kcal it has). ing_cost, prep_min, kcal and fuel-cost are each normalised to roughly $[0,1]$ by "
         f"dividing by a reference maximum (ING_COST_MAX_KES = {config.ING_COST_MAX_KES:g}, "
         f"PREP_MIN_MAX = {config.PREP_MIN_MAX:g}, KCAL_MAX = {config.KCAL_MAX:g}, "
         f"CHARCOAL_KES_MAX = {config.CHARCOAL_KES_MAX:g}) so every gamma-weighted feature sits on a "
         f"comparable scale, and no single attribute dominates $u_k$ just because of its raw units."),
        ("space", 0.01),

        ("heading", "4. Demand and tariffs"),
        ("para",
         "Every cooking event contributes a kW profile (a boxcar of height $e_k/D$ over its sampled "
         "duration $D$, by default) to the aggregate demand curve. A tariff $\\pi$ (pi) is simply a "
         "price path $p(t)$: one number per block over the whole day, announced in full at $t=0$ (agents "
         "never see future prices before they're announced, but the whole path is fixed and known from "
         "block 0 onward, so there is no day-ahead uncertainty about price). Three candidate price paths "
         "are defined:"),
        ("eq", r"$\mathrm{flat}:\quad p(t) = \bar p$", 0.05),
        ("eq", r"$\mathrm{evening\_peak}:\quad p(t) = p_{lo} + (p_{hi}-p_{lo})\cdot\mathbb{1}[t \in W_{peak}]$", 0.05),
        ("eq", r"$\mathrm{solar\_following}:\quad p(t) = p_{hi} - (p_{hi}-p_{lo})\cdot \dfrac{PV(t)}{\max_t PV(t)}$", 0.078),
        ("para",
         "$\\bar p$ (p-bar) is a single target price, the same for every candidate, that the whole tariff "
         "will average out to (see the rescaling below). $p_{lo}$ and $p_{hi}$ are a low and a high price "
         "level used by the two non-flat tariffs. $\\mathbb{1}[t \\in W_{peak}]$ is an indicator: it "
         "equals 1 for every block $t$ inside the fixed evening window $W_{peak}$ (e.g. 17:00-21:00) and "
         "0 everywhere else, so evening_peak is cheap ($p_{lo}$) all day except that one expensive "
         "window. $PV(t)$ is a stylised solar-generation curve (a bell-shaped curve peaking at midday) "
         "used only to shape the solar_following price -- it has no other role and never feeds into any "
         "agent's decision directly; dividing by its own maximum $\\max_t PV(t)$ rescales it to $[0,1]$, "
         "so solar_following's price is cheapest when solar output is highest and most expensive when "
         "it's lowest. Every candidate is then rescaled by one constant factor so its own time-average "
         "equals $\\bar p$ exactly -- this keeps the comparison about the SHAPE of the price over the "
         "day, not its overall level."),
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

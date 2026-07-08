# Clean-Cooking Mini-Grid Tariff Simulator

Agent-based simulation of a mini-grid cooking population, stepped in 5-minute
blocks through one day. Each agent may start cooking (a hazard/"fire" draw);
if it fires, it picks a meal from a fixed menu (electric or wood) via softmax
over utilities. Wood draws zero grid energy, so it is immune to the
electricity tariff -- defection to wood under bad tariffs emerges from the
arithmetic, it is not scripted.

The experiment: sweep day-ahead tariffs, simulate the population, score each
tariff by wood share plus a penalty for the probability that aggregate demand
ever exceeds the grid cap.

## Quickstart

```
pip install numpy matplotlib pandas pytest streamlit
python -m sim explain          # full parameter glossary, grouped by topic
python -m sim audit            # just the [TBD] placeholder parameters
python -m sim run              # simulate + score + plot (writes ./out/)
python -m pytest tests/ -v     # unit tests
streamlit run app.py           # interactive tuning dashboard, see below
python generate_model_pdf.py   # writes out/model_reference.pdf -- equations + every parameter value
```

## Interactive dashboard (`app.py`)

`streamlit run app.py` opens a local browser dashboard, laid out as four tabs:

- **Simulation** -- the scoreboard, and a house map: every agent placed on a
  grid (circles = household, squares = school), coloured by instantaneous kW
  draw. A time-of-day slider scrubs through the day; **Run day -- step
  through at playback speed** animates it end to end at an adjustable
  blocks/sec rate, with a live digital clock next to the map. Switching the
  map's tariff re-simulates a single day directly (cheap, no Monte Carlo
  needed), so it updates immediately.
- **Plots** -- the four summary PNGs (load curves, wood share, meal timing,
  utility waterfall), regenerated on each run.
- **Parameters** -- where you actually craft a persona: household/school
  `gamma`, `gamma_cost`, `sigma_ind`, school `lam` overrides, `DELTA`, tariff
  levels, `PI`, population size/mix. All of this sits inside a form, so
  **nothing changes until you press Save** (or **Save & run simulation** to
  save and immediately re-sweep) -- dragging a slider here never silently
  recomputes anything.
- **Explainability** -- the full parameter glossary (grouped, with units/
  meaning/effect/tbd, same content as `python -m sim explain`), plus a
  worked example: pick a persona/hour/price/hunger state and see every term
  of the Stage 1 hazard and Stage 2 softmax choice substituted with real
  numbers -- literally what's multiplied by what, down to
  `taste x gamma_taste = ...` for each meal.

The sidebar only holds *run configuration* (scenario, which tariffs to
sweep, Monte Carlo runs `R`, seed, ablation switches) -- these are cheap
execution settings, not model design, so they apply immediately. Everything
that defines the model itself lives behind the Parameters form. If the live
config (however it was last changed) no longer matches the config used for
the last **Run simulation**, a warning banner appears above the tabs telling
you the scoreboard/plots/house map are stale -- so it's never ambiguous
whether what's on screen reflects the current sliders.

`python -m sim run` prints a scoreboard, writes `out/scoreboard.csv`, and
writes four plots to `out/`: `load_curves.png`, `wood_share.png`,
`meal_timing.png`, `utility_waterfall.png`.

Useful flags on `run`:

- `--tariffs flat evening_peak` -- restrict the sweep (default: all candidates)
- `--scenario festival_day` -- swap in the example alternative scenario
- `--seed N` -- reproducibility (same seed = identical output)
- `--R N` -- Monte Carlo days per tariff (default from config)
- `--n-agents N` -- population size
- `--trace AGENT_ID` -- log every block for one agent (first tariff, run 0)
  to `out/trace_agent_<id>.csv`, and pretty-print its eaten-meal rows
- `--no-cost`, `--no-hunger`, `--no-personas` -- ablation switches, see below

## Every parameter is TBD until someone checks it

Every number in `sim/config.py` carries a plain-English meaning, units, and a
`tbd` flag. `python -m sim audit` lists everything still flagged as a
placeholder guess rather than a sourced number (currently the large majority
of parameters -- this is a brand-new model). Before trusting the scoreboard
for a real decision, work through that list.

One parameter was already empirically re-tuned away from its initial guess:
`DELTA` (the hazard-to-probability scale) started at the spec's suggested
0.05, which produced only ~37% of agents eating all 3 meals in a day. It was
raised to 0.15, which reaches ~90%, matching the calibration target the spec
called for. It is still flagged `tbd=True` because 0.15 is itself a guess,
not a sourced number -- just a better-calibrated one.

## Ablation tuning workflow

This is the order to sanity-check the model when something looks wrong, or
when re-tuning after changing a parameter:

1. **Everything off**: `python -m sim run --no-cost --no-hunger --no-personas`.
   With hunger and personas neutralised, every agent behaves like the same
   noise-free household reacting only to time-of-day. Check `load_curves.png`:
   you should see three clean peaks at breakfast/lunch/dinner and nothing
   overnight. If the peaks are in the wrong place, fix `bump_centers_hr` /
   `stage_windows_hr`, not the cost or hunger logic.
2. **Enable cost**: `python -m sim run --no-hunger --no-personas`. Now
   `gamma_cost` is live. Compare `wood_share.png` across tariffs -- the
   `evening_peak` tariff should show a visibly suppressed dinner bump in
   `load_curves.png` (expensive electricity during 17:00-21:00 pushes some
   dinners to wood) and a higher wood share than `flat`. If cost has no
   effect, check `gamma_cost` and the `-gamma_cost * price(t) * e_k` term.
3. **Enable personas**: plain `python -m sim run`. Check `meal_timing.png`:
   the `school` row should be essentially unimodal at lunch -- breakfast and
   dinner `lam` are both set to -6 (same magnitude as the overnight base
   logit, i.e. "off"), with lunch boosted to +2.0. This is a `lam` (Stage 1
   hazard bias) tuning, not `gamma` (Stage 2 meal-choice weights) -- `lam`
   controls *whether* a stage fires at all, `gamma` only controls *which*
   meal gets picked once it does.

Each step isolates one mechanism, so if a plot looks wrong you know which
layer introduced the problem.

## Population

Default `N_AGENTS=100` split via `PERSONAS.mix` as **95 household / 5 school**
(both TBD placeholders -- tune in the sidebar or `config.py`).

## Layout

```
sim/config.py      all parameters: value, units, meaning, tbd flag
sim/meals.py       meal table (K=6), power profiles phi_k, duration sampler
sim/tariffs.py     flat / evening_peak / solar_following, PV stub, normalisation
sim/agent.py       pure functions: fire (hazard), which (softmax choice), update
sim/population.py  personas (household/school), individual sampling
sim/run.py         day loop, Monte Carlo sweep, demand assembly
sim/score.py       wood share, exceedance probability, scoreboard
sim/plots.py       the four output figures
sim/cli.py         explain / audit / run subcommands
prism_export.py    stretch goal: exports one persona as a PRISM .pm/.props model
tests/             pytest -- softmax direction, hunger dynamics, stage windows,
                   energy conservation, tariff normalisation, reproducibility
```

## What this deliberately does not model

No brownout dynamics, no batch cooking / meal prep, no PV term in agent
utility (the PV stub only shapes the `solar_following` tariff's price curve,
it never feeds agent decisions), no reinforcement learning. These were
scoped out of this build.

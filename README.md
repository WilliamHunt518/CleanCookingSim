# Clean-Cooking Mini-Grid Tariff Simulator

Agent-based simulation of a mini-grid cooking population, stepped in 5-minute
blocks through one day. Each agent may start cooking (a hazard/"fire" draw);
if it fires, it picks a meal from a fixed menu (electric or wood) via softmax
over utilities. Wood draws zero grid energy, so it is immune to the
electricity tariff -- defection to wood under bad tariffs emerges from the
arithmetic, it is not scripted.

The experiment: sweep day-ahead tariffs, simulate the population, and score
each tariff on two separate goals -- clean_cooking_share (the fraction of all
meals cooked electric rather than on fire; higher is better) and how well it
flattens the village's demand curve (peak_kw / load_factor; a tariff can win
on one and lose on the other).

## Quickstart

```
pip install numpy matplotlib pandas pytest streamlit
python -m sim explain          # full parameter glossary, grouped by topic
python -m sim audit            # just the [TBD] placeholder parameters
python -m sim run              # simulate + score + plot (writes ./out/)
python -m pytest tests/ -v     # unit tests
streamlit run app.py           # interactive tuning dashboard, see below
python generate_model_pdf.py   # writes model_reference.pdf -- equations + every parameter value
```

## Interactive dashboard (`app.py`)

`streamlit run app.py` opens a local browser dashboard as a single scrolling
page (no sidebar, no tabs) with a sticky nav -- Map, Live model, Field data,
Constraints, Menu, Scale, Parameters, Explainability -- following the same
layout as the project's showcase page:

- **Live model** -- the one dark section, holding everything that used to
  live in a sidebar plus the old Simulation/Plots tabs: *run configuration*
  (scenario, which tariffs to sweep, Monte Carlo runs `R`, seed, ablation
  switches, the Run button -- cheap execution settings that apply
  immediately, not model design), live run progress (a progress bar plus
  running clean-cooking-% / meals-simulated counters as the sweep executes),
  the scoreboard, a compact HTML5-canvas house-map day playback (its own
  play/pause/scrub, animated client-side, not through Streamlit reruns) for
  whichever tariff you pick, and the summary plots (load curve, clean
  cooking share, peakiness, meal timing, cooking-events/meal-type-over-time,
  utility waterfall) -- all rebuilt live from the current results, not
  cached PNGs.
- **Parameters** -- where you actually craft a persona: household/school/
  kiosk `gamma` offsets, `gamma_cost`, `kappa_price_time`, `sigma_ind`,
  school/kiosk `lam` overrides, `DELTA`, tariff levels, population size/mix.
  All of this sits inside a form, so **nothing changes until you press
  Save** (or **Save & run simulation** to save and immediately re-sweep) --
  dragging a slider here never silently recomputes anything.
- **Explainability** -- the complete Stage 1/Stage 2 maths (compact, four
  equations), the full parameter glossary (grouped, with units/meaning/
  effect/tbd, same content as `python -m sim explain`), a worked example
  (pick a persona/hour/price/hunger state and see every term substituted
  with real numbers, down to `taste x gamma_taste = ...` for each meal), a
  live PRISM-export demo (build the formal-verification DTMC for a chosen
  persona in-browser and see its actual reachable-state count), and a
  walkthrough of how one block's decision becomes the scoreboard.

If the live config (however it was last changed) no longer matches the
config used for the last **Run simulation**, a warning banner appears in the
Live model section telling you the scoreboard/plots/house map are stale --
so it's never ambiguous whether what's on screen reflects the current
sliders.

`python -m sim run` prints a scoreboard, writes `out/scoreboard.csv`, and
writes plots to `out/`: `load_curves.png`, `clean_cooking_share.png`,
`peakiness.png`, `meal_timing.png`, `events_over_time.png`,
`meal_type_over_time.png`, `utility_waterfall.png`.

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

Four parameters were already empirically re-tuned away from their initial
guess. `DELTA` (the hazard-to-probability scale) started at the spec's
suggested 0.05, which produced only ~37% of agents eating all 3 meals in a
day. It was raised to 0.15, which reaches ~90%, matching the calibration
target the spec called for. `base_gamma_cost` (price sensitivity, Stage 2 --
which meal) went through two passes: 1.2 gave only a ~17% relative gap in
clean_cooking_share (then still called wood_share) between tariffs, too
subtle to see on the scoreboard; a first re-tune to 2.5 helped but was still
muted once agents could dodge expensive hours by rescheduling rather than
switching fuel. It's now 4.5. `kappa_price_time` (new parameter, Stage 1 --
price's pull on *when* to cook, not just what) started at 0 -- price only
ever changed meal choice at a fixed time, so cooking timing was identical
across every tariff, which understates how a real evening-peak price would
displace the dinner rush rather than just push it onto wood. It's now 4.0,
clipped so a below-average price can only ever leave the hazard unchanged,
never boost it (see the comment in `sim/agent.py::fire` -- an earlier
unclipped version let cheap off-peak hours override a school's `lam=-6`
"lunch only" schedule constraint, since that constraint is institutional,
not economic). `p_hi`/`p_lo` (the evening_peak/solar_following price levels)
were widened from 0.45/0.10 to 0.60/0.05 alongside these. Together: the
share of evening_peak dinners still starting inside the 17-21h peak window
drops from ~85% to ~0%, and clean_cooking_share now spans ~58% (flat) to
~78% (evening_peak) across the three realistic tariffs -- note evening_peak
scores *best* on clean cooking, not worst, since its cheap off-peak hours
dominate its day and a price-responsive population reschedules into them
rather than eating electric at peak; check `peak_kw` instead for its actual
cost, since that redistributed load usually gives it the *tallest* peak of
the three (see "Scoring" below -- clean cooking and demand-curve flatness
are separate goals a tariff can win on one and lose on the other). There's
also a fifth, deliberately-not-realistic tariff candidate, `extreme_test`
(flat, `extreme_test_multiplier`=5x p_bar, opt-in via the "Tariffs to sweep"
picker): a sanity check that the price response actually saturates at an
extreme input rather than silently no-op'ing -- at the current
base_gamma_cost/kappa_price_time it drives cook-start events to exactly zero
across 100 independent simulated days (not just a low clean_cooking_share --
Stage 1 doesn't know which fuel an agent would pick until *after* it fires,
so an extreme enough price suppresses firing altogether rather than
diverting it to wood). All parameters are still flagged `tbd=True` because
the new values are themselves guesses, not sourced numbers -- just
better-calibrated ones.

## Realism noise and institutional scale

Two further additions, orthogonal to the price mechanics above:

- **`sigma_bump_center_jitter`** / **`sigma_logit_noise`** -- without any
  per-agent variation, every agent shares the exact same w(t) hazard curve,
  so a large population synchronises into an almost perfectly clean,
  razor-sharp aggregate peak at each meal time with dead silence between --
  too idealised to be believable. `sigma_bump_center_jitter` gives each
  agent its own small, fixed personal offset to each stage's bump centre
  (sampled once, like gamma -- some people habitually eat a bit
  earlier/later every day); `sigma_logit_noise` adds fresh per-block
  idiosyncratic noise on top (today's whim). Together they widen the std dev
  of lunch cook-start times from ~0.9h to ~1.1-1.3h and cut the tallest
  half-hour bin on the cooking-events-over-time chart by ~15-20%, while
  keeping meals/day within ~5% of the old, noise-free baseline.
  `repeat_meal_prob` adds a third, much smaller kind of irregularity: a tiny
  independent per-block chance that an already-eaten stage fires again
  anyway (a second helping / snack) -- the odd bit of messiness a hard
  one-meal-per-stage rule can otherwise never produce.
- **`meals_per_cook`** -- a school or kiosk *agent* is still one firing
  decision, but represents an institutional kitchen, not one household: a
  school "cooking lunch" is really a canteen serving many students at once.
  This multiplies that persona's energy (`e_kwh`) and power draw per event
  (household stays at 1x). It does *not* affect `clean_cooking_share`, which
  counts events, not energy. A literal per-student headcount (50-80x)
  pushed `peak_kw` to 350-450 (vs. ~42 unscaled) -- one school lunch
  swamping the entire village's demand curve into a single school-shaped
  spike rather than being a visible *contributor* to it. Settled on a more
  modest 5x (school) / 3x (kiosk), roughly doubling `peak_kw` over the
  unscaled baseline.

## Scoring

Two separate goals, two separate metrics -- a tariff can win on one and lose
on the other:

- **`clean_cooking_share`** -- the fraction of all meals (pooled across every
  Monte Carlo run) cooked electric rather than on fire. Higher is better. A
  tariff with zero cook events (e.g. `extreme_test`) scores 0%, not 100% --
  suppressing cooking altogether isn't clean cooking.
- **`peak_kw` / `load_factor`** -- these tariffs are also meant to flatten
  the village's demand curve, not just relocate fuel choice. `peak_kw` is
  the mean, across runs, of each day's peak aggregate demand; `load_factor`
  is the mean of each day's (average demand / peak demand), the standard
  grid-engineering flatness measure (1.0 = perfectly flat, closer to 0 = one
  sharp spike). Lower peak_kw / higher load_factor is better.

The scoreboard sorts by `clean_cooking_share` descending.

## Ablation tuning workflow

This is the order to sanity-check the model when something looks wrong, or
when re-tuning after changing a parameter:

1. **Everything off**: `python -m sim run --no-cost --no-hunger --no-personas`.
   With hunger and personas neutralised, every agent behaves like the same
   noise-free household reacting only to time-of-day. Check `load_curves.png`:
   you should see three clean peaks at breakfast/lunch/dinner and nothing
   overnight. If the peaks are in the wrong place, fix `bump_centers_hr` /
   `stage_windows_hr`, not the cost or hunger logic.
2. **Enable cost**: `python -m sim run --no-hunger --no-personas`. `--no-cost`
   zeroes each agent's `gamma_cost`, which is shared by *both* price channels
   -- Stage 2's `-gamma_cost * price(t) * e_k` (meal choice) and Stage 1's
   `-kappa_price_time * gamma_cost * max(price(t)-p_bar, 0)` (does it fire at
   all) -- so turning cost back on enables both at once. Check
   `load_curves.png`: `evening_peak` should show a visibly suppressed (at the
   current calibration, nearly flattened) dinner bump during its 17:00-21:00
   peak window. Check `clean_cooking_share.png`: don't assume `evening_peak`
   scores worse than `flat` -- at the current calibration it scores *better*,
   because agents displaced from the expensive window mostly reschedule into
   evening_peak's cheap off-peak hours and cook electric there rather than
   switching to wood. Check `peakiness.png` instead for evening_peak's actual
   cost: redistributed load usually gives it the *tallest* peak_kw of the
   three tariffs. If cost has no effect on any of these, check `gamma_cost`
   and both price terms above.
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

Default `N_AGENTS=100` split via `PERSONAS.mix` as **80 household / 10
school / 10 kiosk** (all TBD placeholders -- tune in the Parameters section
of `app.py`, or directly in `config.py`).

## Layout

```
sim/config.py      all parameters: value, units, meaning, tbd flag
sim/meals.py       meal table (K=18), power profiles phi_k, duration sampler
sim/tariffs.py     flat / evening_peak / solar_following / extreme_test, PV stub, normalisation
sim/agent.py       pure functions: fire (hazard), which (softmax choice), update
sim/population.py  personas (household/school/kiosk), individual sampling
sim/run.py         day loop, Monte Carlo sweep, demand assembly
sim/score.py       clean_cooking_share, peak_kw, load_factor, scoreboard
sim/plots.py       the seven output figures
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

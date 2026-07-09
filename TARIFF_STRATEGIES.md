# Forecast-driven tariff strategies — design spec

Formulations for five forecast-driven tariff candidates (A-E) plus a genetic-
algorithm search for near-optimal pricing. **This is a design document — none
of it is implemented yet.** It exists so that, when the end-to-end simulation
is ready, each strategy can be added to `sim/tariffs.py` mechanically, with
every formula, parameter, unit conversion, and edge case already decided.

Companion docs: `COMPONENT_API.md` (the `GridEnergyComponent` API these
strategies consume) and `README.md` (the SOC model itself).

---

## 0. Integration contract

### 0.1 What `sim/tariffs.py` expects

A tariff candidate is a zero-argument function returning a `np.ndarray` of
length `T = sim.config.STATE.T` (288 blocks of 5 minutes = one day) in sim's
internal currency units, registered in the `CANDIDATES` dict. Existing
candidates: `flat`, `evening_peak`, `solar_following`.

### 0.2 Where the new code goes

- **`grid_energy/pricing.py` (new):** pure price-curve builders. Each takes a
  `ForecastResult` and/or `SOCResult` (plus parameters) and returns a
  15-minute-resolution price array in **KES/kWh**. No `sim` imports —
  preserves the `sim -> grid_energy` dependency direction.
- **`sim/tariffs.py` (thin adapters):** one function per strategy that calls
  the `grid_energy` builder, then applies the two bridges below, and
  registers the result in `CANDIDATES`.

### 0.3 Resolution bridge (15-min week -> 5-min day)

`grid_energy` outputs 672 x 15-min blocks (one week). `sim` needs 288 x 5-min
blocks (one day). Adapter procedure: select day `d` (0-6) -> take its 96
blocks -> upsample x3 with `grid_energy.resample.resample_kw(prices, 15, 5)`
(repeat each value; already implemented and tested). Day-ahead semantics:
each simulated day uses only that day's slice of the forecast.

### 0.4 Unit bridge (KES/kWh -> sim currency units) — **load-bearing, not cosmetic**

Agents are calibrated against sim's internal price scale in **two separate
channels** (both from the merged collaborator work):

| channel | parameter | calibrated against |
|---|---|---|
| meal-choice utility | `base_gamma_cost = 4.5` | prices in [`p_lo`=0.05, `p_hi`=0.60], mean `p_bar`=0.25 |
| cook-timing hazard | `kappa_price_time = 4.0` | `max(price(t) - p_bar, 0)` — **anchored to `p_bar`** |

Feeding raw KES (0-80) into sim would scale both terms ~160x and destroy the
calibration. Conversion: **`sim_price = kes_price / KES_PER_SIM_UNIT`** with
`KES_PER_SIM_UNIT = 160.0` (from the anchor "KES 40 flat rate ≡ `p_bar`
0.25"). The KES band [0, 80] then maps to sim [0, 0.50] — inside the
calibrated [0.05, 0.60] envelope. Constant lives in one place (proposal:
`grid_energy/config.py`), applied only in the `sim/tariffs.py` adapters.

Note on `kappa_price_time`: because the hazard channel keys off
`price(t) - p_bar`, a strategy whose *average* sits far from KES 40 shifts
overall cooking frequency, not just meal choice/timing. This is a reason to
keep strategy averages roughly near KES 40 even though exact normalisation is
not required (user decision: no strict KES-40-mean constraint).

### 0.5 Normalisation policy

`_normalise()` (rescale to common mean `p_bar`) is **optional per strategy**.
Default: **unnormalised**, clamped to [`P_MIN`, `P_MAX`]. Add normalisation
only later if apples-to-apples revenue comparison against `flat` is wanted.
Every strategy's final step is a hard clamp regardless.

### 0.6 Shared parameters (proposed defaults)

| name | default | unit | meaning |
|---|---|---|---|
| `P_MIN` | 0 | KES/kWh | hard price floor |
| `P_MAX` | 80 | KES/kWh | hard price ceiling |
| `P_FLAT` | 40 | KES/kWh | current real flat rate (Oloika prepay), reference level |
| `P_DISC` | 30 | KES/kWh | discounted tier (paper's actual Green Light Hours rate) |
| `KES_PER_SIM_UNIT` | 160 | KES per sim-unit | unit bridge, section 0.4 |

### 0.7 The usage chicken-and-egg

Strategies C, D, E need a `SOCResult`, which needs a usage forecast — but
usage depends on the price being designed. For the POC: build the day-ahead
usage assumption from a reference simulation under `flat` (exactly what
`demo_forecast_week.py` already produces). The GA (section 7) closes this
loop properly by re-simulating usage under every candidate price.

### 0.8 Reference week (for grounding/edge cases)

All "observed behaviour" notes below refer to the live run of 2026-07-09..15
(60 kWp, 27.8 kWh battery, 18 agents): PV 314.6 vs usage 254.8 kWh/week
(~23% margin); day 1 = 2.68 kWh dawn deficit (SOC hits 0%); day 2 = net
shortfall (33.6 < 36.4, no surplus); days 3-7 = surpluses 2-17 kWh, socs
peaks 107-163%; battery cycles 40-65% overnight; dinner peak entirely
post-sunset.

---

## Strategy A — `green_light` (two-tier day-ahead surplus window) **[POC first]**

Real-world precedent: the Oloika paper's actual intervention (KES 30 during
"Green Light Hours" vs KES 40 otherwise).

| | |
|---|---|
| **inputs** | `SOCResult` for day `d` (forecast PV + reference usage) |
| **parameters** | `P_DISC`=30, `P_FLAT`=40, `pad_blocks`=0 (optional window dilation, 15-min units) |

**Formula.** Discount window `W_d = { t in day d : surplus_kwh[t] > 0 }`,
optionally dilated by `pad_blocks` on each side. Then:

```
price(t) = P_DISC   if t in W_d
           P_FLAT   otherwise
```

**Edge cases.** `W_d` empty (day 2, any cloudy day) -> flat `P_FLAT` all day:
the tariff self-adapts, no discount is offered on days with nothing to give
away. Multiple disjoint surplus intervals in one day are all discounted (no
contiguity requirement).

**Observed behaviour on reference week.** Discount window appears on days
3-7 only, a few midday hours each; days 1-2 price flat.

---

## Strategy B — `pv_following_real` (continuous, generation-indexed)

Same shape as the existing `solar_following` candidate, but driven by the
real quartz forecast instead of the synthetic stub, and with a tunable
response exponent.

| | |
|---|---|
| **inputs** | `ForecastResult` only (no usage needed — cheapest data requirement) |
| **parameters** | `p_lo`=`P_MIN`, `p_hi`=`P_MAX`, `gamma`=1.0, `P_ref` mode = day-max (alt: week 95th percentile) |

**Formula.**

```
price(t) = p_hi - (p_hi - p_lo) * (pv_kw(t) / P_ref) ^ gamma      clamped [p_lo, p_hi]
```

`gamma < 1` front-loads the discount (modest sun already cheap); `gamma > 1`
reserves it for true peak sun. `P_ref` = that day's max forecast PV
guarantees each day actually touches `p_lo`; the percentile alternative
guards against single-block forecast spikes (observed in earlier runs).

**Edge cases.** `P_ref = 0` (fully dark forecast day) -> flat `p_hi`; guard
required. Nighttime always prices at `p_hi`.

**Known blind spot (why C exists).** Prices generation, not storage: on day
2 it discounts the sunny morning even though that day ends in near-deficit.

---

## Strategy C — `soc_banded` (three-band battery-state pricing)

| | |
|---|---|
| **inputs** | `SOCResult` for day `d` |
| **parameters** | `theta_hi`=90 (% SOC), `theta_lo`=20 (% SOC), `p_cheap`=20, `p_mid`=`P_FLAT`, `p_dear`=70 |

**Formula.** With `s(t) = actual_soc_pct(t)` (the clipped, physically-real
tracker — not the unbounded `socs_pct`):

```
price(t) = p_cheap   if s(t) >= theta_hi
           p_dear    if s(t) <= theta_lo
           p_mid     otherwise
```

**Edge cases.** Battery never reaching either threshold -> flat `p_mid` day.
Rapid oscillation across a threshold could flip prices block-to-block; if
observed, add hysteresis (enter cheap at 90, exit at 85) — deferred unless
needed.

**Observed behaviour.** Cheap band ~midday when SOC pins at 100%; dear band
at day 1's dawn (SOC -> 0%) and late evenings dipping toward 40%.

---

## Strategy D — `residual_load` (net-load marginal-cost proxy)

The closest to a true marginal-cost signal: separates lunch (high usage but
PV-covered — nearly free to serve) from dinner (same usage, zero PV — every
kWh costs battery throughput).

| | |
|---|---|
| **inputs** | `SOCResult` for day `d` (needs both `pv_kw` and `usage_kw`; `net_kw` is already a field) |
| **parameters** | `p_lo`=`P_MIN`, `p_hi`=`P_MAX` |

**Formula.** Residual `r(t) = -net_kw(t) = usage_kw(t) - pv_kw(t)`,
normalised `r_hat(t) = r(t) / max_day |r(t)|` in [-1, 1], mapped affinely:

```
price(t) = (p_hi + p_lo)/2 + r_hat(t) * (p_hi - p_lo)/2       clamped [p_lo, p_hi]
```

Max surplus (`r_hat = -1`) -> `p_lo`; max unmet residual (`r_hat = +1`) ->
`p_hi`; balanced -> midpoint (40 with the 0/80 defaults — lands on `P_FLAT`
automatically).

**Edge cases.** `max|r| = 0` (identically balanced day, theoretical) -> flat
midpoint; guard required. Overnight (PV=0, low base usage) prices moderately
high, not maximal — only the dinner residual peak hits `p_hi`. That is the
desired distinction vs B, which prices all darkness identically.

---

## Strategy E — `deficit_guard` (flat + targeted scarcity surcharge)

Minimal deviation from the status quo; the conservative baseline A-D should
beat.

| | |
|---|---|
| **inputs** | `SOCResult` for day `d` |
| **parameters** | `P_FLAT`=40, `p_surge`=`P_MAX`=80, `lead_blocks`=8 (= 2h at 15-min) |

**Formula.** Deficit set `D_d = { t : deficit_kwh[t] > 0 }`. Surcharge
window = `D_d` **extended backward by `lead_blocks`**:

```
price(t) = p_surge   if any deficit block occurs in [t, t + lead_blocks]
           P_FLAT    otherwise
```

**Why the backward extension is essential (causality).** A deficit at block
`t` is caused by cumulative drain *before* `t`. Surcharging only the deficit
block itself acts after the battery is already empty; pricing the 2h lead-in
(e.g. pre-dawn cooking before day 1's dawn deficit) is what actually deters
the drain. `lead_blocks` is the strategy's real design variable.

**Edge cases.** No forecast deficit (days 2-7 of reference week) -> flat 40
all day, indistinguishable from `flat`. That is intended.

---

## 7. Genetic algorithm — `ga_optimal` (near-optimal pricing, closed-loop)

### 7.1 Why a GA fits this problem

The objective is **stochastic** (Monte Carlo agent days), **non-differentiable**
(discrete meal choices, clip/threshold nonlinearities in SOC), and
**feedback-coupled** (price changes usage, which changes the surplus/deficit
the price was justified by — critical in the reference week's tight 23%
margin regime). GAs need only fitness evaluations, tolerate noise, and
parallelise trivially across a population. Strategies A-E each fix a *shape*
and tune nothing; the GA searches shapes.

### 7.2 Chromosome encodings (two options, recommend (a) first)

**(a) Parametric (~7 genes) — recommended first.** Optimise the parameters
of the strategy families instead of raw prices, e.g.
`theta = (p_lo, p_hi, gamma, theta_hi, theta_lo, pad_blocks, lead_blocks)`.
Tiny search space -> converges in few generations; every individual is
smooth and explainable by construction. Limitation: can only reach shapes
the families express.

**(b) Direct (24 or 96 genes).** One gene per hour (24) or per 15-min block
(96), each clamped [`P_MIN`, `P_MAX`]. Add a smoothness penalty
`lambda * sum_t (p[t+1] - p[t])^2` to the fitness, else the GA returns
jagged, unusable curves. Use only after (a) plateaus.

### 7.3 Fitness function (minimise)

For candidate `theta`: build `price(t)` -> run `R` Monte Carlo sim days ->
feed each day's aggregate `demand_kw` through
`GridEnergyComponent.compute_soc_for_usage(..., forecast=prefetched)` ->
average:

```
F(theta) = w_def * total_deficit_kwh          (reliability: unmet demand)
         + w_sur * total_surplus_kwh          (waste: curtailed PV)
         + w_wood * wood_share                (adoption: e-cooking displaced to fire)
         [+ w_rev * |mean_paid_price - P_FLAT|]   (optional affordability anchor)
```

`mean_paid_price` = usage-weighted average price actually paid — the
economically meaningful anchor (not the time-average). Weights are the
experiment design surface; starting point `w_def >> w_sur ~ w_wood` since
unmet demand is the hard failure.

**The one live PV forecast is fetched once** and passed via the existing
`forecast=` argument to every evaluation — zero API calls inside the loop.

### 7.4 Noise handling — common random numbers (CRN)

Evaluate every candidate in a generation with the **same** set of `R` seeds
(sim already uses `SeedSequence` spawning, so this is natural). Fitness
differences then reflect price differences, not luck — the single most
effective variance-reduction available here. Schedule: `R = 10-20` during
search; re-rank the final top-5 at `R = 200` (matching `SCORING.R`) before
declaring a winner.

### 7.5 Operators and hyperparameters (starting values)

| element | choice |
|---|---|
| population | 20-30; **seeded with A-E's curves + flat-40**, remainder random — guarantees GA >= best heuristic by construction |
| selection | tournament, k=3 |
| elitism | copy best 1-2 unchanged |
| crossover | blend/BLX-alpha for real genes, prob 0.8 |
| mutation | Gaussian, sigma = 10% of each gene's range, per-gene prob ~1/n_genes, sigma annealed over generations |
| clamp | re-clamp every gene to its bounds after every operator |

### 7.6 Convergence and budget

Stop when best fitness improves by less than the CRN noise floor (estimated
from replicate evaluations of the elite) for 5 consecutive generations, or
at a hard cap of ~40 generations. Budget: 25 pop x 25 gen x R=15 ≈ **9,400
sim-days** ≈ the same order as one `python -m sim run` sweep repeated ~15x —
tractable overnight even single-threaded; population members parallelise if
needed.

### 7.7 Output artifact

The winner is a frozen price array (or parameter set), stored to a small
file and registered in `CANDIDATES` as `ga_optimal` like any other tariff —
it is *produced* offline by the search, not recomputed at sim time.

---

## 8. Proposed registration and rollout order

| order | `CANDIDATES` name | source | needs usage forecast? |
|---|---|---|---|
| 1 | `green_light` | strategy A | yes (reference sim under flat) |
| 2 | `pv_following_real` | strategy B | no |
| 3 | `soc_banded` | strategy C | yes |
| 4 | `residual_load` | strategy D | yes |
| 5 | `deficit_guard` | strategy E | yes |
| 6 | `ga_optimal` | section 7, offline search | yes (closed-loop) |

Rollout: implement A + B first (A = POC headline, B = trivial and
usage-free), score them against the existing `flat`/`evening_peak`/
`solar_following` with the standard sweep, then add C-E, then run the GA
seeded with all of them.

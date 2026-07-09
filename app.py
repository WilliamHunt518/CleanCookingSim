"""Single-page Streamlit dashboard: tune parameters, run the sim, watch a
house map animate through the day, and see the exact arithmetic behind any
one decision.

Launch with:  streamlit run app.py

Layout follows the colleague's oloika-showcase-v2.html showcase page: one
long scroll with a sticky top nav (no sidebar, no tabs) -- Map, Live model,
Field data, Constraints, Menu, Scale, Parameters, Explainability, in that
order. "Live model" is the one dark section (mirrors the showcase's #live)
and holds everything the sidebar used to: run configuration (scenario,
which tariffs to sweep, R, seed, ablation switches, the Run button -- cheap
execution settings that apply immediately), live run progress, the
scoreboard, the house map day playback, and the summary plots.

Parameters (what to simulate: gamma/lam/DELTA/tariff levels/population mix)
lives inside an st.form further down the page, so nothing changes until you
explicitly press Save -- no on-the-fly reactivity there. A staleness banner
in the Live model section compares the live config against a snapshot taken
at the last Run, so it's always visible when the scoreboard/plots are out of
date relative to the sliders.
"""
from __future__ import annotations

import json
import math
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import prism_export
from sim import agent, config, meals, plots, run as run_mod, score, tariffs as tariffs_mod
from sim import population as population_mod
import theme

st.set_page_config(page_title="Oloika E-Cooking — Tariff Simulator", layout="wide")
theme.inject()


def _snapshot(scenario_name, tariff_names, R, seed, no_cost, no_hunger, no_personas) -> dict:
    """Everything that affects a run's outcome, for staleness comparison."""
    return {
        "n_agents": config.N_AGENTS, "mix": dict(config.PERSONAS.mix),
        "meals_per_cook": dict(config.PERSONAS.meals_per_cook),
        "gamma": dict(config.PERSONAS.base_gamma), "gamma_cost": config.PERSONAS.base_gamma_cost,
        "sigma_ind": config.PERSONAS.sigma_ind,
        "persona_gamma_offsets": {p: dict(off) for p, off in config.PERSONAS.persona_gamma_offsets.items()},
        "persona_lam": {p: dict(lam) for p, lam in config.PERSONAS.persona_lam.items()},
        "DELTA": config.TIMING.DELTA, "kappa_price_time": config.TIMING.kappa_price_time,
        "sigma_bump_center_jitter": config.TIMING.sigma_bump_center_jitter,
        "sigma_logit_noise": config.TIMING.sigma_logit_noise,
        "repeat_meal_prob": config.TIMING.repeat_meal_prob,
        "tariff": (config.TARIFF.p_bar, config.TARIFF.p_lo, config.TARIFF.p_hi),
        "scenario": scenario_name, "tariffs": tuple(tariff_names), "R": R, "seed": seed,
        "no_cost": no_cost, "no_hunger": no_hunger, "no_personas": no_personas,
    }


def _display_name(raw: str) -> str:
    return " + ".join(w.capitalize() for w in raw.split("_"))


_OLOIKA_MAP_HTML = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  html,body{margin:0;font-family:Arial,sans-serif}
  #oloikaMap{height:440px;width:100%}
  .map-bar{display:flex;justify-content:space-between;align-items:center;gap:1rem;
    padding:.5rem .8rem;background:#181310;color:#EFE3CE;font:12px 'Space Mono',monospace;flex-wrap:wrap}
  .map-bar a{color:#F2B01E}
  .map-bar button{font:11px 'Space Mono',monospace;background:#F7EFE2;border:2px solid #F7EFE2;
    padding:.2rem .55rem;cursor:pointer}
  .map-bar button.on{background:#F2B01E;border-color:#F2B01E}
</style>
<div id="oloikaMap"></div>
<div class="map-bar">
  <span>OLOIKA &middot; -2.05&deg;, 36.13&deg; (approx.) &middot; south of Lake Magadi on the Shompole road</span>
  <span><button id="lyrSat" class="on">Satellite</button><button id="lyrTopo">Topo</button></span>
</div>
<script>
(function(){
  const map = L.map('oloikaMap', {scrollWheelZoom:false}).setView([-2.049, 36.132], 13);
  const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    {maxZoom:18, attribution:'Imagery &copy; Esri, Maxar, Earthstar Geographics'});
  const topo = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    {maxZoom:18, attribution:'&copy; Esri World Topographic Map'});
  sat.addTo(map);
  const mk = (ll, color, html, popup) => L.marker(ll, {icon:L.divIcon({className:'',
    html:`<div style="background:${color};border:3px solid #181310;color:#fff;font:700 10px Arial;padding:3px 7px;white-space:nowrap;box-shadow:3px 3px 0 #181310">${html}</div>`,
    iconAnchor:[10,10]})}).addTo(map).bindPopup(popup);
  mk([-2.049,36.132], '#C22A1E', 'OLOIKA MINI-GRID', '<b>Oloika power plant</b><br>25 kWp PV &middot; 54 kWh Li-ion &middot; 3-phase<br>72 metered + 12 sub-metered customers');
  mk([-2.028,36.152], '#E2711D', 'e-cooking premises', '<b>7 premises, 14 e-cookers</b><br>mostly restaurants -- EPCs & induction hobs');
  mk([-1.980,36.230], '#1F5FA8', 'LAKE MAGADI', '<b>Lake Magadi</b><br>soda lake, ~15 km NE on the tarmac road');
  mk([-2.105,36.075], '#2E7D4F', 'SHOMPOLE', '<b>Shompole centre</b><br>sister e4D mini-grid, ~10 km SW');
  L.circle([-2.049,36.132], {radius:1600, color:'#F2B01E', weight:3, dashArray:'8 6', fillColor:'#F2B01E', fillOpacity:.08})
    .addTo(map).bindPopup('Approximate mini-grid service area');
  const bS=document.getElementById('lyrSat'), bT=document.getElementById('lyrTopo');
  bS.onclick=()=>{map.addLayer(sat);map.removeLayer(topo);bS.classList.add('on');bT.classList.remove('on');};
  bT.onclick=()=>{map.addLayer(topo);map.removeLayer(sat);bT.classList.add('on');bS.classList.remove('on');};
})();
</script>
"""

# A compact HTML5-canvas playback of one simulated day -- house-shaped/square/triangle markers
# for household/school/kiosk, coloured by cook-fuel type, plus a demand-curve canvas -- all
# animated client-side (requestAnimationFrame) from data computed once in Python. Replaces a
# previous matplotlib-per-block version that needed several separate sliders/buttons and a
# Python time.sleep loop to "animate": this is one self-contained, compact component, closer to
# a small game-engine view than a chart.
_PLAYBACK_HTML_TEMPLATE = r"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Anton&family=Archivo:wght@400;600;700&family=Space+Mono:wght@400;700&display=swap');
  * { box-sizing: border-box; }
  body { margin: 0; font-family: 'Archivo', Arial, sans-serif; background: #181310; color: #EFE3CE; }
  .wrap { padding: .7rem .8rem .9rem; }
  .bar { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; margin-bottom: .6rem; }
  .bar button {
    font-family: 'Anton', sans-serif; letter-spacing: .04em; text-transform: uppercase; font-size: .78rem;
    border: 2px solid #EFE3CE; background: transparent; color: #EFE3CE; padding: .4rem .8rem; cursor: pointer;
  }
  .bar button.play { background: #C22A1E; border-color: #C22A1E; color: #fff; }
  .bar button.play.playing { background: #2E7D4F; border-color: #2E7D4F; }
  .bar label { font-family: 'Space Mono', monospace; font-size: .65rem; color: #a3937d; display: flex; align-items: center; gap: .4rem; }
  .clock { font-family: 'Anton', sans-serif; font-size: 1.5rem; color: #F2B01E; min-width: 4.4rem; text-align: center; }
  .tariff-label { font-family: 'Space Mono', monospace; font-size: .68rem; color: #F2B01E; margin-left: auto; }
  .panels { display: flex; gap: .9rem; flex-wrap: wrap; align-items: flex-start; }
  .panel { border: 2px solid #a3937d; background: #221A14; padding: .55rem; }
  .panel h4 {
    font-family: 'Anton', sans-serif; font-weight: 400; font-size: .72rem; letter-spacing: .05em;
    text-transform: uppercase; color: #F2B01E; margin: 0 0 .4rem;
  }
  canvas { display: block; }
  .legend { display: flex; gap: .8rem; flex-wrap: wrap; font-family: 'Space Mono', monospace; font-size: .6rem; color: #CBBFA9; margin-top: .45rem; }
  .legend i { display: inline-block; width: 9px; height: 9px; margin-right: .3rem; vertical-align: -1px; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; margin-top: .55rem; }
  .stat { border: 2px solid #a3937d; padding: .35rem .5rem; }
  .stat b { display: block; font-family: 'Anton', sans-serif; font-size: 1.05rem; color: #F2B01E; }
  .stat small { font-size: .58rem; text-transform: uppercase; letter-spacing: .06em; color: #a3937d; }
</style>
<div class="wrap">
  <div class="bar">
    <button class="play" id="playBtn">&#9654; Play</button>
    <button id="resetBtn">&#8634; Reset</button>
    <span class="clock" id="clock">00:00</span>
    <label>speed <input type="range" id="speed" min="1" max="60" value="12" style="accent-color:#F2B01E"></label>
    <label style="flex:1;min-width:160px">scrub <input type="range" id="scrub" min="0" max="__TMAX__" value="0" style="width:100%;accent-color:#C22A1E"></label>
    <span class="tariff-label">TARIFF: __TARIFF__</span>
  </div>
  <div class="panels">
    <div class="panel">
      <h4>The village &middot; who's cooking now</h4>
      <canvas id="agentCanvas" width="330" height="330"></canvas>
      <div class="legend">
        <span><i style="background:#4a3f35;border-radius:50%"></i>household</span>
        <span><i style="background:#4a3f35"></i>school</span>
        <span><i style="background:#4a3f35;transform:rotate(45deg)"></i>kiosk</span>
        <span><i style="background:#F2B01E;border-radius:50%"></i>electric</span>
        <span><i style="background:#C22A1E;border-radius:50%"></i>fire</span>
      </div>
    </div>
    <div class="panel" style="flex:1;min-width:320px">
      <h4>Village demand</h4>
      <canvas id="demandCanvas" width="500" height="240"></canvas>
      <div class="stats">
        <div class="stat"><b id="stTotal">0.00 kW</b><small>total draw</small></div>
        <div class="stat"><b id="stCooking">0</b><small>cooking now</small></div>
        <div class="stat"><b id="stClean">0%</b><small>clean cooking (today)</small></div>
      </div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
const T = DATA.T, BLOCK_HOURS = DATA.block_minutes / 60.0;
const PERSONA_NAMES = DATA.persona_names;

const agentCv = document.getElementById('agentCanvas'), actx = agentCv.getContext('2d');
const demandCv = document.getElementById('demandCanvas'), dctx = demandCv.getContext('2d');
const AW = agentCv.width, AH = agentCv.height, AM = 18;
const DM = {l:38, r:10, t:8, b:22}, DW = demandCv.width - DM.l - DM.r, DH = demandCv.height - DM.t - DM.b;
const demandMax = Math.max(...DATA.demand_kw, 1e-6);
const priceMax = Math.max(...DATA.price, 1e-6);

function ax_(nx){ return AM + nx * (AW - 2*AM); }
function ay_(ny){ return AM + ny * (AH - 2*AM); }
function dx_(t){ return DM.l + t/T * DW; }
function dy_(kw){ return DM.t + DH - Math.min(kw, demandMax)/demandMax * DH; }

function drawAgentShape(ctx, cx, cy, persona, r, fill) {
  ctx.fillStyle = fill; ctx.strokeStyle = '#181310'; ctx.lineWidth = 1.2;
  ctx.beginPath();
  if (persona === 0) { ctx.arc(cx, cy, r, 0, Math.PI*2); }
  else if (persona === 1) { ctx.rect(cx-r, cy-r, r*2, r*2); }
  else { ctx.moveTo(cx, cy-r*1.15); ctx.lineTo(cx+r*1.05, cy+r*0.85); ctx.lineTo(cx-r*1.05, cy+r*0.85); ctx.closePath(); }
  ctx.fill(); ctx.stroke();
}

function render(frame) {
  const active = DATA.activity[frame];
  const activeMap = new Map(active.map(a => [a[0], a[1]]));

  actx.fillStyle = '#221A14'; actx.fillRect(0, 0, AW, AH);
  for (let i = 0; i < DATA.positions.length; i++) {
    const [nx, ny] = DATA.positions[i];
    const persona = DATA.persona[i];
    const isActive = activeMap.has(i);
    const fill = isActive ? (activeMap.get(i) ? '#C22A1E' : '#F2B01E') : '#4a3f35';
    drawAgentShape(actx, ax_(nx), ay_(ny), persona, isActive ? 7.5 : 5, fill);
  }

  dctx.fillStyle = '#1B1410'; dctx.fillRect(0, 0, demandCv.width, demandCv.height);
  dctx.strokeStyle = '#4a3f35'; dctx.fillStyle = '#a3937d'; dctx.font = "9px 'Space Mono'"; dctx.lineWidth = 1;
  for (let kw = 0; kw <= demandMax; kw += Math.max(1, Math.round(demandMax/4))) {
    dctx.beginPath(); dctx.moveTo(DM.l, dy_(kw)); dctx.lineTo(DM.l+DW, dy_(kw)); dctx.stroke();
    dctx.fillText(kw.toFixed(0)+" kW", 2, dy_(kw)+3);
  }
  for (let h = 0; h <= 24; h += 6) {
    const xx = dx_(h*60/DATA.block_minutes);
    dctx.beginPath(); dctx.moveTo(xx, DM.t); dctx.lineTo(xx, DM.t+DH); dctx.stroke();
    dctx.fillText(h+"h", xx-6, demandCv.height-6);
  }
  dctx.strokeStyle = '#7a6a56'; dctx.setLineDash([4,4]); dctx.beginPath();
  for (let i = 0; i < T; i++) {
    const yy = DM.t + DH - (DATA.price[i]/priceMax) * DH * 0.4;
    i ? dctx.lineTo(dx_(i), yy) : dctx.moveTo(dx_(i), yy);
  }
  dctx.stroke(); dctx.setLineDash([]);
  dctx.strokeStyle = '#F2B01E'; dctx.lineWidth = 2; dctx.beginPath();
  for (let i = 0; i <= frame; i++) { const xx = dx_(i), yy = dy_(DATA.demand_kw[i]); i ? dctx.lineTo(xx,yy) : dctx.moveTo(xx,yy); }
  dctx.stroke();
  if (frame > 0) {
    dctx.lineTo(dx_(frame), dy_(0)); dctx.lineTo(dx_(0), dy_(0)); dctx.closePath();
    dctx.fillStyle = 'rgba(242,176,30,.12)'; dctx.fill();
  }
  dctx.strokeStyle = '#EFE3CE'; dctx.lineWidth = 1; dctx.beginPath();
  dctx.moveTo(dx_(frame), DM.t); dctx.lineTo(dx_(frame), DM.t+DH); dctx.stroke();

  const hour = frame * BLOCK_HOURS;
  const hh = String(Math.floor(hour)).padStart(2,'0'), mm = String(Math.round((hour%1)*60)).padStart(2,'0');
  document.getElementById('clock').textContent = hh+':'+mm;
  document.getElementById('stTotal').textContent = DATA.demand_kw[frame].toFixed(2)+' kW';
  document.getElementById('stCooking').textContent = active.length;
  let woodSoFar = 0, totalSoFar = 0;
  for (let i = 0; i <= frame; i++) { for (const a of DATA.activity_starts[i] || []) { totalSoFar++; if (a) woodSoFar++; } }
  document.getElementById('stClean').textContent = totalSoFar ? Math.round(100*(1-woodSoFar/totalSoFar))+'%' : '0%';
  document.getElementById('scrub').value = frame;
}

let frame = 0, playing = false, lastTs = 0, accum = 0;
const playBtn = document.getElementById('playBtn');
function loop(ts) {
  if (!playing) return;
  if (!lastTs) lastTs = ts;
  const dt = (ts - lastTs) / 1000; lastTs = ts;
  const speed = parseFloat(document.getElementById('speed').value);
  accum += dt * speed;
  while (accum >= 1 && frame < T - 1) { frame++; accum -= 1; }
  render(frame);
  if (frame >= T - 1) { stop(); } else { requestAnimationFrame(loop); }
}
function play() { if (playing) return; playing = true; lastTs = 0; playBtn.textContent = '❚❚ Pause'; playBtn.classList.add('playing'); requestAnimationFrame(loop); }
function stop() { playing = false; playBtn.textContent = '▶ Play'; playBtn.classList.remove('playing'); }
playBtn.onclick = () => { if (playing) stop(); else { if (frame >= T-1) frame = 0; play(); } };
document.getElementById('resetBtn').onclick = () => { stop(); frame = 0; render(0); };
document.getElementById('scrub').oninput = (e) => { stop(); frame = parseInt(e.target.value); render(frame); };
render(0);
</script>
"""


def _build_playback_html(day, population, positions: np.ndarray, tariff_name: str, price: np.ndarray) -> str:
    T = config.STATE.T
    block_hours = config.STATE.block_minutes / 60.0

    xs, ys = positions[:, 0], positions[:, 1]
    x_norm = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-6)
    y_norm = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-6)

    activity: list[list[list[int]]] = [[] for _ in range(T)]
    activity_starts: list[list[int]] = [[] for _ in range(T)]  # 1 entry per meal *start*, for wood-share-so-far
    for e in day.events:
        is_fire = int(meals.WOOD_MASK[e.meal_idx0])
        end = min(T, e.start_block + e.duration_blocks)
        for b in range(e.start_block, end):
            activity[b].append([int(e.agent_idx), is_fire])
        if e.start_block < T:
            activity_starts[e.start_block].append(is_fire)

    payload = {
        "T": T, "block_minutes": config.STATE.block_minutes,
        "positions": np.column_stack([x_norm, y_norm]).round(4).tolist(),
        "persona": population.persona_idx.tolist(),
        "persona_names": population_mod.PERSONA_NAMES,
        "activity": activity,
        "activity_starts": activity_starts,
        "demand_kw": [round(float(x), 4) for x in day.demand_kw],
        "price": [round(float(x), 4) for x in price],
    }
    html = _PLAYBACK_HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    html = html.replace("__TMAX__", str(T - 1)).replace("__TARIFF__", tariff_name)
    return html


# ---------------------------------------------------------------------------
# Nav + hero + map
# ---------------------------------------------------------------------------
theme.nav_bar()
theme.anchor("village")
st.markdown(
    '<div class="oloika-hero">'
    '<p class="eyebrow">MECS &middot; COSMO Phase 2 &middot; Shompole East ward, Kajiado West</p>'
    '<h1>Cooking with the <span class="hl">sun</span> in Oloika</h1>'
    '<div class="hero-stats">'
    '<div><b>25 kWp</b><small>PV array</small></div>'
    '<div><b>54 kWh</b><small>Li-ion storage</small></div>'
    '<div><b>14</b><small>e-cookers</small></div>'
    '<div><b>57%</b><small>daylight cooking</small></div>'
    '</div></div>', unsafe_allow_html=True)
st.markdown(theme.kanga("Where it happens", theme.RED) + "  \n### The village", unsafe_allow_html=True)
components.html(_OLOIKA_MAP_HTML, height=490)
st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Live model: run configuration, live progress, scoreboard, house map, plots
# ---------------------------------------------------------------------------
theme.anchor("live")
with st.container(key="live_panel"):
    st.markdown(theme.kanga("Watch the model think", theme.RED) + "  \n### Live model", unsafe_allow_html=True)
    st.caption("The real equations from sim/config.py. These are execution settings, not model design "
               "-- they apply immediately. Persona/tariff/timing values live in the Parameters section "
               "further down and need an explicit Save.")

    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1, 1])
    with c1:
        scenario_name = st.selectbox("Scenario", list(config.SCENARIOS.keys()))
    with c2:
        _default_tariffs = [t for t in tariffs_mod.CANDIDATES if t != "extreme_test"]
        tariff_names = st.multiselect(
            "Tariffs to sweep", list(tariffs_mod.CANDIDATES.keys()), default=_default_tariffs,
            help="extreme_test (5x p_bar, flat all day) is a sanity-check stress tariff, not a "
                 "realistic candidate -- opt in to see the model's price response saturate.")
    with c3:
        R = st.slider("Monte Carlo runs (R)", 5, 200, 20, 5,
                      help="Lower = faster iteration while tuning; raise for a stable final read.")
    with c4:
        seed = st.number_input("seed", value=config.DEFAULT_SEED, step=1)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
    with c1:
        no_cost = st.checkbox("--no-cost (zero price sensitivity)")
    with c2:
        no_hunger = st.checkbox("--no-hunger")
    with c3:
        no_personas = st.checkbox("--no-personas (everyone base household)")
    with c4:
        run_clicked = st.button("Run simulation", type="primary", width="stretch")

    if not tariff_names:
        st.error("Select at least one tariff to sweep.")
        st.stop()

    trigger_run = st.session_state.pop("trigger_run", False)
    if run_clicked or trigger_run or "results" not in st.session_state:
        progress_bar = st.progress(0.0)
        stats_ph = st.empty()
        total_steps = max(len(tariff_names), 1) * int(R)
        # Streamlit re-renders on every update -- throttle to ~12 pushes per tariff so a big R
        # doesn't spend more time drawing metrics than simulating.
        throttle = max(1, int(R) // 12)

        def _on_progress(info: dict) -> None:
            step = info["tariff_idx"] * info["R"] + info["run_idx"] + 1
            is_last = info["run_idx"] == info["R"] - 1
            if step % throttle != 0 and not is_last:
                return
            progress_bar.progress(min(step / total_steps, 1.0))
            with stats_ph.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tariff", f"{info['tariff_name']} ({info['tariff_idx'] + 1}/{info['n_tariffs']})")
                m2.metric("Monte Carlo run", f"{info['run_idx'] + 1}/{info['R']}")
                m3.metric("Meals simulated so far", f"{info['events_so_far']:,}")
                m4.metric("Running clean cooking %", f"{info['clean_share_so_far'] * 100:.1f}%")

        with st.spinner("Simulating..."):
            results, population = run_mod.run_sweep(
                tariff_names, scenario_name=scenario_name, seed=int(seed), R=int(R), n_agents=config.N_AGENTS,
                no_hunger=no_hunger, no_cost=no_cost, no_personas=no_personas,
                progress_callback=_on_progress,
            )
            board = score.scoreboard(results, population)
            st.session_state.update(results=results, population=population, board=board)
            st.session_state["last_run_snapshot"] = _snapshot(scenario_name, tariff_names, R, seed,
                                                                no_cost, no_hunger, no_personas)
            st.session_state.pop("day", None)      # population changed -- invalidate cached day
            st.session_state.pop("day_tariff", None)
        progress_bar.empty()
        stats_ph.empty()

    if "results" not in st.session_state:
        st.stop()

    current_snapshot = _snapshot(scenario_name, tariff_names, R, seed, no_cost, no_hunger, no_personas)
    if current_snapshot != st.session_state.get("last_run_snapshot"):
        st.warning("Parameters or run settings have changed since the last run -- the scoreboard, plots "
                   "and house map below are stale. Press **Run simulation** above to recompute.")

    population = st.session_state["population"]
    board = st.session_state["board"]
    results = st.session_state["results"]

    st.markdown("#### Scoreboard")
    st.caption("Sorted by clean_cooking_share descending -- higher means more meals cooked "
               "electric, not over fire. peak_kw and load_factor (mean/peak demand, 1.0 = "
               "perfectly flat) score how well a tariff flattens the village's demand curve, "
               "which is a separate goal from clean cooking -- a tariff can win on one and lose "
               "on the other.")
    st.dataframe(board, width="stretch", hide_index=True)

    st.markdown("#### House map -- day playback")
    tariff_for_map = st.selectbox("Tariff to visualize", list(tariffs_mod.CANDIDATES.keys()))
    st.caption("Re-simulates a single representative day for the chosen tariff -- cheap, so this "
               "updates immediately without needing Run simulation. Plays back client-side (its own "
               "play/pause/scrub), not through Streamlit reruns, so it stays smooth regardless of R.")

    if st.session_state.get("day_tariff") != tariff_for_map or "day" not in st.session_state:
        rng_day = np.random.default_rng(int(seed) + 999_983)
        price_for_map = tariffs_mod.build_tariff(tariff_for_map)
        scenario_obj = config.SCENARIOS[scenario_name]
        day = run_mod.simulate_day(population, price_for_map, scenario_obj, rng_day, no_hunger=no_hunger,
                                    track_agent_power=False)
        st.session_state["day"] = day
        st.session_state["day_price"] = price_for_map
        st.session_state["day_tariff"] = tariff_for_map
    day = st.session_state["day"]
    price_for_map = st.session_state["day_price"]

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

    playback_html = _build_playback_html(day, population, positions, tariff_for_map, price_for_map)
    components.html(playback_html, height=470, scrolling=False)

    st.markdown(theme.beads(), unsafe_allow_html=True)
    st.markdown("#### Summary plots")
    st.caption("Every chart here is rebuilt fresh from the current results each time this section "
               "renders -- not a cached PNG from disk.")

    st.markdown("###### Cooking events &amp; meal type over time", unsafe_allow_html=True)
    st.caption("Do agents shift *when* they cook under a pricier tariff, or *what* they cook at the "
               "same hour? These two charts split by tariff, unlike the pooled histograms below.")
    ev_col, type_col = st.columns(2)
    with ev_col:
        fig = plots.build_events_over_time_figure(results)
        st.pyplot(fig)
        plt.close(fig)
    with type_col:
        fig = plots.build_meal_type_over_time_figure(results)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("###### Load &amp; clean cooking", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        fig = plots.build_load_curves_figure(results)
        st.pyplot(fig)
        plt.close(fig)
    with cols[1]:
        fig = plots.build_clean_cooking_figure(results)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("###### How peaky is each tariff's demand curve?", unsafe_allow_html=True)
    st.caption("These tariffs are meant to flatten the village's demand curve, not just relocate "
               "fuel choice -- this is the metric that actually tests that goal, separate from "
               "clean cooking share above.")
    fig = plots.build_peakiness_figure(results)
    st.pyplot(fig)
    plt.close(fig)

    fig = plots.build_meal_timing_figure(results, population)
    st.pyplot(fig)
    plt.close(fig)

    fig = plots.build_utility_waterfall_figure()
    st.pyplot(fig)
    plt.close(fig)

st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Field data: real pilot measurements (not simulator output)
# ---------------------------------------------------------------------------
theme.anchor("evidence")
st.markdown(theme.kanga("What Oloika measured", theme.BLUE) + "  \n### Field data &middot; Aug '24 - Jul '25",
            unsafe_allow_html=True)
st.caption("Real pilot measurements from the deployed mini-grid -- not simulator output. "
           "Source: Univ. of Southampton & Kenya Power (MECS/FCDO, Oct 2025).")
st.markdown(
    '<div class="stat-row">'
    + theme.stat_tile("32&rarr;57", "kWh/day village load", "r")
    + theme.stat_tile("1.7", "kWh/day e-cooking (12.5 predicted)", "b")
    + theme.stat_tile("4.2 kW", "max combined cooker load", "y")
    + theme.stat_tile("KES 30/40", "green vs standard tariff", "g")
    + '</div>', unsafe_allow_html=True)
ev1, ev2 = st.columns(2)
with ev1:
    st.markdown(
        '<div class="oloika-card"><p style="font-family:var(--disp);text-transform:uppercase;'
        'font-size:.9rem;margin:0 0 .4rem">Monthly consumption (kWh)</p>'
        '<svg viewBox="0 0 760 290" role="img" aria-label="Monthly consumption growing from 1070 to 1764 kilowatt hours">'
        '<line x1="10" y1="240" x2="750" y2="240" stroke="#181310" stroke-width="3"/>'
        '<rect x="190" y="20" width="120" height="220" fill="#F2B01E" opacity=".22"/>'
        '<rect x="370" y="20" width="120" height="220" fill="#F2B01E" opacity=".22"/>'
        '<g fill="#1F5FA8">'
        '<rect x="12" y="109" width="40" height="131"/><rect x="72" y="112" width="40" height="128"/>'
        '<rect x="132" y="85" width="40" height="155"/><rect x="192" y="126" width="40" height="114" fill="#7fa3cd"/>'
        '<rect x="252" y="146" width="40" height="94" fill="#7fa3cd"/><rect x="312" y="92" width="40" height="148"/>'
        '<rect x="372" y="99" width="40" height="141" fill="#7fa3cd"/><rect x="432" y="118" width="40" height="122" fill="#7fa3cd"/>'
        '<rect x="492" y="87" width="40" height="153"/><rect x="552" y="58" width="40" height="182"/>'
        '<rect x="612" y="42" width="40" height="198"/><rect x="672" y="24" width="40" height="216"/>'
        '</g>'
        '<line x1="14" y1="118" x2="710" y2="32" stroke="#C22A1E" stroke-width="3" stroke-dasharray="8 6"/>'
        '<g font-size="9" fill="#181310" text-anchor="middle">'
        '<text x="32" y="258">A\'24</text><text x="92" y="258">S</text><text x="152" y="258">O</text>'
        '<text x="212" y="258">N</text><text x="272" y="258">D</text><text x="332" y="258">J\'25</text>'
        '<text x="392" y="258">F</text><text x="452" y="258">M</text><text x="512" y="258">A</text>'
        '<text x="572" y="258">M</text><text x="632" y="258">J</text><text x="692" y="258">J</text>'
        '</g></svg></div>', unsafe_allow_html=True)
with ev2:
    st.markdown(
        '<div class="oloika-card">'
        '<p style="font-family:var(--disp);text-transform:uppercase;font-size:.9rem;margin:0 0 .4rem">'
        "57% of e-cooking in Green Light Hours</p>"
        '<svg viewBox="0 0 380 120" role="img" aria-label="57 percent daytime cooking versus 43 percent other hours">'
        '<rect x="16" y="30" width="348" height="56" fill="#EFE3CE" stroke="#181310" stroke-width="3"/>'
        '<rect x="16" y="30" width="198" height="56" fill="#2E7D4F"/>'
        '<text x="30" y="64" font-size="18" fill="#fff" font-weight="bold">57% daylight</text>'
        '<text x="226" y="64" font-size="13">43% other</text></svg>'
        '<p style="font-family:var(--disp);text-transform:uppercase;font-size:.9rem;margin:1rem 0 .4rem">Financing recovery</p>'
        '<svg viewBox="0 0 380 96" role="img" aria-label="23 percent of the 95000 shilling investment repaid so far, full 112100 anticipated">'
        '<rect x="16" y="14" width="348" height="26" fill="#EFE3CE" stroke="#181310" stroke-width="2.5"/>'
        '<rect x="16" y="14" width="80" height="26" fill="#2E7D4F"/>'
        '<text x="104" y="32" font-size="11" font-weight="bold">KES 18,400 repaid &middot; 23% of 95,000</text>'
        '<rect x="16" y="52" width="348" height="26" fill="#F2B01E" stroke="#181310" stroke-width="2.5"/>'
        '<text x="24" y="70" font-size="11" font-weight="bold">KES 112,100 full repayment on track (Magadi Sacco)</text>'
        '</svg></div>', unsafe_allow_html=True)

st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constraints: local facts -> model knobs, pulled live from sim/config.py
# ---------------------------------------------------------------------------
theme.anchor("constraints")
st.markdown(theme.kanga("Grounded in Oloika", theme.GREEN) + "  \n### Local constraints &rarr; model knobs",
            unsafe_allow_html=True)
n_total = len(config.MEALS)
n_fire = sum(1 for m in config.MEALS if m.fire_only)
gc = config.PERSONAS.base_gamma_cost
peak_lo, peak_hi = config.TARIFF.w_peak_hr
school_off = config.PERSONAS.persona_gamma_offsets.get("school", {})
kiosk_off = config.PERSONAS.persona_gamma_offsets.get("kiosk", {})
school_lam = config.PERSONAS.persona_lam.get("school", {})
fest = config.FESTIVAL_DAY_SCENARIO
choma_fire_bonus = fest.eta_k.get("nyama_choma_open_fire", 0.0)
dinner_shift = fest.eta_t_hr_offsets.get("dinner", {}).get("centre_shift_hr", 0.0)
st.caption(f"Live from sim/config.py under the current **{scenario_name}** scenario -- "
           f"{sum(1 for r in config.tbd_params())} of {len(config.all_params())} parameters are "
           "honest placeholders awaiting survey data.")
st.markdown(
    '<div class="con-grid">'
    + theme.con_card("i-choma-fire", "Fuel stacking",
                      f"<code>{n_fire}</code> of {n_total} meals are fire-only, drawing "
                      "<code>0 kWh</code> -- immune to any tariff. Ugali &amp; choma flavour keeps "
                      "wood alive by design, not by rule.")
    + theme.con_card("i-price", "Price sensitivity",
                      f"<code>&gamma;_cost = {gc:g}</code> on <code>-p(t)&middot;e_k</code> -- the "
                      "meal-choice channel. Plus <code>&kappa;_price,time = "
                      f"{config.TIMING.kappa_price_time:g}</code> on <code>max(price(t)-p_bar, 0)</code> "
                      "in the firing hazard, so price can only ever delay/suppress cooking, never invent "
                      "an extra occasion -- price shifts *when* agents cook, not just what.")
    + theme.con_card("i-peak", "Evening price signal",
                      f"evening_peak window <code>{peak_lo:g}-{peak_hi:g}h @ {config.TARIFF.p_hi:g}</code> "
                      f"vs off-peak <code>{config.TARIFF.p_lo:g}</code> -- cuts dinners still starting "
                      "inside that window from ~85% to ~0%; scoreboard clean_cooking_share now spans "
                      "~58% (flat) to ~78% (evening_peak) -- though evening_peak's redistributed load "
                      "also gives it the *tallest* peak_kw of the three, a genuine clean-vs-flat tradeoff.")
    + theme.con_card("i-stew", "Real market prices",
                      f"Costs normalised to local maxima: ingredients <code>{config.ING_COST_MAX_KES:g} "
                      f"KES</code>, prep <code>{config.PREP_MIN_MAX:g} min</code>, charcoal "
                      f"<code>{config.CHARCOAL_KES_MAX:g} KES</code>.")
    + theme.con_card("i-vendor", "Restaurants &amp; school",
                      f"Kiosk: taste <code>+{kiosk_off.get('taste', 0):g}</code>, prep "
                      f"<code>{kiosk_off.get('prep_min', 0):g}</code>, hours unknown &rarr; inherits "
                      f"household timing. School: lunch-only <code>&lambda;={school_lam.get('lunch', 0):+g}</code>, "
                      f"ing_cost <code>{school_off.get('ing_cost', 0):g}</code>. Both are one *agent* but "
                      f"an institutional kitchen -- <code>meals_per_cook</code> scales school "
                      f"<code>&times;{config.PERSONAS.meals_per_cook['school']:g}</code> and kiosk "
                      f"<code>&times;{config.PERSONAS.meals_per_cook['kiosk']:g}</code> so one firing "
                      "decision draws energy like the canteen/vendor it represents, not one household.")
    + theme.con_card("i-choma", "Festivals",
                      f"Scenario offsets: choma <code>+{choma_fire_bonus:g}</code> appeal, dinner bump "
                      f"<code>+{dinner_shift:g} h</code> later -- a template for market days &amp; droughts.")
    + '</div>', unsafe_allow_html=True)

st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Menu: the real 18-meal table from sim/config.py, as a filterable gallery
# ---------------------------------------------------------------------------
theme.anchor("menu")
st.markdown(theme.kanga("Master Table Z", theme.ORANGE) + "  \n### The 18-meal menu", unsafe_allow_html=True)
st.caption("Local dishes at Magadi-area prices, straight from `sim/config.py`. Blue = mini-grid. "
           "Red = fire-only, the wood the tariff competes with.")
n_elec = sum(1 for m in config.MEALS if not m.fire_only)
n_fire = sum(1 for m in config.MEALS if m.fire_only)
menu_choice = st.radio("Filter", [f"All {len(config.MEALS)}", f"Electric ({n_elec})", f"Fire-only ({n_fire})"],
                        horizontal=True, label_visibility="collapsed")
if menu_choice.startswith("Electric"):
    menu_rows = [m for m in config.MEALS if not m.fire_only]
elif menu_choice.startswith("Fire-only"):
    menu_rows = [m for m in config.MEALS if m.fire_only]
else:
    menu_rows = config.MEALS
menu_cards = [
    theme.meal_card_html(_display_name(m.name), theme.ICON_BY_MEAL_IDX[m.idx - 1], m.fire_only,
                          m.dbar_min, m.kcal, m.ing_cost_kes, m.e_kwh, m.charcoal_kes)
    for m in menu_rows
]
st.markdown(f'<div class="menu-grid">{"".join(menu_cards)}</div>', unsafe_allow_html=True)

st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Scale: scalability narrative
# ---------------------------------------------------------------------------
theme.anchor("scale")
st.markdown(theme.kanga("Beyond one village", theme.RED) + "  \n### Scalability", unsafe_allow_html=True)
st.markdown(
    '<div class="scale-steps">'
    + theme.scale_step(1, "Capacity first",
                        "Model before you plug in: Oloika's upgrade (25 kWp / 54 kWh) left "
                        "~50% daytime headroom.")
    + theme.scale_step(2, "Finance locally",
                        "Sacco-backed sales: subsidies 40% &rarr; 25% &rarr; 0% within a year; "
                        "MSSL now invests its own capital.")
    + theme.scale_step(3, "Signal, don't lecture",
                        "Green LED + 25% discount &rarr; 57% daylight cooking, rising monthly.")
    + theme.scale_step(4, "Rehearse, then price",
                        "Re-run the simulator per site: new menu, personas, PV curve -- same two "
                        "scores: clean cooking share (higher is better) and load factor (flatter "
                        "demand curve, higher is better).")
    + '</div>', unsafe_allow_html=True)

st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Parameters: deliberate, form-gated editor ("craft a persona")
# ---------------------------------------------------------------------------
theme.anchor("parameters")
st.markdown(theme.kanga("Craft a persona", theme.GREEN) + "  \n### Model parameters", unsafe_allow_html=True)
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

    c1, c2 = st.columns(2)
    with c1:
        school_meals_per_cook_in = st.slider(
            "School meals per cook event", 1, 100, config.PERSONAS.meals_per_cook["school"],
            help="A school agent's one firing decision is a canteen serving many students at "
                 "once, not one household -- this multiplies both its energy (e_kwh) and power "
                 "draw. Doesn't affect clean_cooking_share (which counts events, not energy).")
    with c2:
        kiosk_meals_per_cook_in = st.slider(
            "Kiosk meals per cook event", 1, 100, config.PERSONAS.meals_per_cook["kiosk"],
            help="Same idea for a kiosk/vendor serving several customers per cooking session.")

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
        gamma_cost_in = st.slider("gamma_cost (price sensitivity -- THE tariff knob)", 0.0, 10.0,
                                   config.PERSONAS.base_gamma_cost, 0.1)
    with c2:
        sigma_ind_in = st.slider("sigma_ind (individual variation)", 0.0, 0.5,
                                  config.PERSONAS.sigma_ind, 0.01)

    st.markdown("#### School persona -- gamma offsets (additive on the household base) + timing")
    school_off_now = config.PERSONAS.persona_gamma_offsets.get("school", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        school_taste_off = st.slider("taste offset", -2.0, 2.0, school_off_now.get("taste", 0.0), 0.05,
                                      key="school_taste_off")
    with c2:
        school_kid_off = st.slider("kid offset", -2.0, 2.0, school_off_now.get("kid", 0.0), 0.05,
                                    key="school_kid_off")
    with c3:
        school_batch_off = st.slider("batch offset", -2.0, 2.0, school_off_now.get("batch", 0.0), 0.05,
                                      key="school_batch_off")
    with c4:
        school_ing_cost_off = st.slider("ing_cost offset", -3.0, 3.0, school_off_now.get("ing_cost", 0.0), 0.05,
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
    kiosk_off_now = config.PERSONAS.persona_gamma_offsets.get("kiosk", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kiosk_taste_off = st.slider("taste offset", -2.0, 2.0, kiosk_off_now.get("taste", 0.0), 0.05,
                                     key="kiosk_taste_off")
    with c2:
        kiosk_batch_off = st.slider("batch offset", -2.0, 2.0, kiosk_off_now.get("batch", 0.0), 0.05,
                                     key="kiosk_batch_off")
    with c3:
        kiosk_prep_min_off = st.slider("prep_min offset", -3.0, 3.0, kiosk_off_now.get("prep_min", 0.0), 0.05,
                                        key="kiosk_prep_min_off")
    with c4:
        kiosk_ing_cost_off = st.slider("ing_cost offset", -3.0, 3.0, kiosk_off_now.get("ing_cost", 0.0), 0.05,
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
    c1, c2 = st.columns(2)
    with c1:
        delta_in = st.slider("DELTA (hazard scale -- how often anyone cooks at all)", 0.01, 0.5,
                              config.TIMING.DELTA, 0.01)
    with c2:
        kappa_price_time_in = st.slider(
            "kappa_price_time (price's pull on WHEN to cook, not just what)", 0.0, 15.0,
            config.TIMING.kappa_price_time, 0.5,
            help="Multiplies each agent's own gamma_cost into the Stage 1 firing-hazard logit. "
                 "0 = tariffs only ever change meal choice at a fixed time (the old behaviour); "
                 "higher = a pricey hour visibly thins and delays cooking, not just its fuel.")

    st.markdown("###### Realism noise -- breaking up an otherwise too-clean, too-synchronised day")
    c1, c2, c3 = st.columns(3)
    with c1:
        sigma_bump_jitter_in = st.slider(
            "sigma_bump_center_jitter (personal mealtime offset)", 0.0, 4.0,
            config.TIMING.sigma_bump_center_jitter, 0.1,
            help="Each agent gets its own small, fixed offset to each stage's bump centre -- "
                 "some people habitually eat a bit earlier/later every day. 0 = every agent "
                 "shares the exact same w(t), which synchronises a large population into an "
                 "almost perfectly aligned, razor-sharp aggregate peak. This is what actually "
                 "widens the peak shape.")
    with c2:
        sigma_logit_noise_in = st.slider(
            "sigma_logit_noise (day-to-day whim)", 0.0, 4.0, config.TIMING.sigma_logit_noise, 0.1,
            help="Per-block idiosyncratic noise on top of the jitter above -- today's mood, a "
                 "visitor turning up. Redrawn fresh every block, so it perturbs which exact "
                 "block within an already-open window an agent fires on.")
    with c3:
        repeat_meal_prob_in = st.slider(
            "repeat_meal_prob (second helping / eating again)", 0.0, 0.01,
            config.TIMING.repeat_meal_prob, 0.0001, format="%.4f",
            help="Tiny per-block chance an agent fires again in a stage it already ate today -- "
                 "the odd bit of real-world messiness a hard one-meal-per-stage rule can't "
                 "otherwise produce. Not hunger-driven, doesn't affect hunger accounting.")

    st.markdown("#### Tariff")
    c1, c2, c3 = st.columns(3)
    with c1:
        p_bar_in = st.slider("p_bar", 0.05, 1.0, config.TARIFF.p_bar, 0.01)
    with c2:
        p_lo_in = st.slider("p_lo", 0.0, 1.0, config.TARIFF.p_lo, 0.01)
    with c3:
        p_hi_in = st.slider("p_hi", 0.0, 2.0, config.TARIFF.p_hi, 0.01)

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
    config.PERSONAS.meals_per_cook = {"household": 1, "school": school_meals_per_cook_in,
                                       "kiosk": kiosk_meals_per_cook_in}
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
    config.TIMING.kappa_price_time = kappa_price_time_in
    config.TIMING.sigma_bump_center_jitter = sigma_bump_jitter_in
    config.TIMING.sigma_logit_noise = sigma_logit_noise_in
    config.TIMING.repeat_meal_prob = repeat_meal_prob_in
    config.TARIFF.p_bar, config.TARIFF.p_lo, config.TARIFF.p_hi = p_bar_in, p_lo_in, p_hi_in
    st.success("Parameters saved.")
    if save_run_clicked:
        st.session_state["trigger_run"] = True
    st.rerun()

st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Explainability: the maths (compact) -> glossary -> worked example -> PRISM -> how
# the scoreboard gets built. In that order, per how this section reads best.
# ---------------------------------------------------------------------------
theme.anchor("explain")
st.markdown(theme.kanga("Show your working", theme.ORANGE) + "  \n### How the model actually works",
            unsafe_allow_html=True)

st.markdown("#### The complete maths")
st.caption("Every block, every agent runs this -- two stages, four equations. The worked example "
           "further down substitutes real numbers into exactly this.")

st.markdown("**State.** $h=(h_B,h_L,h_D)\\in\\{0,\\dots,K\\}^3$ (0 = that stage not yet eaten "
            "today, else the 1-based index of the meal eaten); $\\tau$ = hours since last meal.")

st.markdown("**Stage 1 -- does it fire this block?**")
st.latex(r"\text{hunger}(t) = \max\!\big(0,\ \bar n(t) - n\big) + \kappa \cdot \tau")
st.latex(r"w_i(t) = b + \sum_{s \in \{B,L,D\}} (h_s^{\text{ht}} - b)\, "
         r"\exp\!\Big(-\tfrac12\big(\tfrac{t - c_s - j_{i,s}}{\sigma_s}\big)^2\Big)")
st.latex(r"\ell_{i,t} = w_i(t) + \eta_t(t) + \lambda_{\text{persona},\,\text{stage}} "
         r"+ \alpha_0 \cdot \text{hunger}(t) "
         r"- \kappa_{\text{price,time}} \cdot \gamma_{\text{cost}} \cdot \max\!\big(\text{price}(t) - \bar p,\ 0\big) "
         r"+ \epsilon_{i,t}")
st.latex(r"q_{i,t} = \sigma(\ell_{i,t}) \cdot \Delta, \qquad \sigma(x) = \frac{1}{1+e^{-x}}, "
         r"\qquad \text{fired} \sim \text{Bernoulli}(q_{i,t}) \ \text{ if eligible, else } 0")
st.caption(
    "Two sources of realism noise sit in here, both off by default in the sense that "
    "$j_{i,s}=0,\\ \\epsilon_{i,t}=0$ would reproduce a perfectly clean, fully-synchronised "
    "population: $j_{i,s}$ is agent $i$'s own small, fixed-for-the-simulation personal offset to "
    "stage $s$'s bump centre (sampled once, like $\\gamma$ -- some people habitually eat a bit "
    "earlier/later every day); $\\epsilon_{i,t} \\sim \\mathcal{N}(0, \\sigma_{\\text{noise}})$ is "
    "fresh idiosyncratic noise redrawn every block (today's whim). *Eligible* is normally "
    "'this stage not yet eaten today', except for a small independent chance every block "
    "(`repeat_meal_prob`) that re-opens an already-eaten stage anyway -- a second helping or "
    "snack, logged as a real event but not double-counted in hunger."
)

st.markdown("**Stage 2 -- which meal, conditional on firing?**")
st.latex(r"u_k = \gamma \cdot z_k + \eta_k - \gamma_{\text{cost}} \cdot \text{price}(t) \cdot e_k "
         r"+ \alpha_k \cdot \text{hunger}(t)")
st.latex(r"P(\text{choice}=k) = \frac{\exp(u_k)}{\sum_{j=1}^{K} \exp(u_j)}")

st.markdown("**Update.** If fired: $h_{\\text{stage}} \\leftarrow \\text{choice}+1$, "
            "$\\tau \\leftarrow 0$. Otherwise: $\\tau \\leftarrow \\min(\\tau + \\Delta t,\\ 24)$.")

st.caption(
    "Symbols, tersely -- full values/meanings for every one of these are in the Parameter "
    "glossary below: $\\bar n(t)$ expected meals eaten by now &middot; $n$ meals actually eaten "
    "&middot; $\\kappa$ hunger-from-waiting weight &middot; $b$ overnight baseline logit &middot; "
    "$h_s^{\\text{ht}}, c_s, \\sigma_s$ each stage's hazard-bump height/centre/width &middot; "
    "$\\eta_t, \\eta_k$ scenario offsets (0 in the reference scenario) &middot; $\\lambda$ "
    "persona+stage timing bias &middot; $\\alpha_0, \\alpha_k$ hunger's pull on firing / on "
    "meal $k$ &middot; $\\kappa_{\\text{price,time}}, \\gamma_{\\text{cost}}$ price sensitivity "
    "(timing / meal-choice channels) &middot; $\\bar p$ tariff-average price &middot; $\\Delta$ "
    "hazard-to-probability scale and per-block cap &middot; $\\gamma, z_k$ this agent's taste "
    "weights and meal $k$'s fixed attributes &middot; $e_k$ meal $k$'s grid energy (0 for "
    "fire-only meals -- their only tariff exposure is through $\\text{price}(t)-\\bar p$ in "
    "Stage 1, since they have nothing to switch away from in Stage 2) &middot; $j_{i,s}$ agent "
    "$i$'s personal bump-centre offset for stage $s$ &middot; $\\epsilon_{i,t}$ fresh per-block "
    "idiosyncratic noise."
)

st.markdown(theme.beads(), unsafe_allow_html=True)
st.markdown("#### Parameter glossary")
st.caption("Every constant in the equations above, with its actual value, units, and a "
           "plain-English meaning/effect -- grouped the same way `python -m sim explain` prints it.")
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

st.markdown(theme.beads(), unsafe_allow_html=True)
st.markdown("### One agent, one decision -- the exact arithmetic")
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
st.caption("Assumes the currently-relevant stage's slot hasn't been eaten yet (still eligible to fire) -- "
           "this is an illustration of the arithmetic, not a full state replay.")

gamma_vec = population_mod.persona_gamma_vector(ex_persona)
gamma_cost_val = population_mod.persona_gamma_cost(ex_persona)
lam_vec = population_mod.persona_lam_vector(ex_persona)
scenario_obj = config.SCENARIOS[scenario_name]

# Which stage is "now" -- whichever stage's own bump is highest at this instant (no individual
# jitter, no hard stage_windows_hr clock gate), the same argmax test sim.agent.fire applies
# per-agent. See agent.stage_bump's docstring.
bump_vec = agent.stage_bump(ex_hour)
stage_idx = int(np.argmax(bump_vec))
stage_name = agent.STAGE_ORDER[stage_idx]

st.markdown("#### Stage 1 -- firing hazard (does the agent start cooking at all)")
st.write(f"Most-relevant stage right now: **{stage_name}** "
         f"(highest of the three bumps -- breakfast={bump_vec[0]:.2f}, lunch={bump_vec[1]:.2f}, "
         f"dinner={bump_vec[2]:.2f} -- no hard clock window, see Explainability notes above).")
nb = config.nbar(ex_hour)
hunger = max(0, nb - ex_n) + config.HUNGER.kappa * ex_tau
w = float(bump_vec[stage_idx])
eta_t = agent.eta_t_of_t(ex_hour, scenario_obj)
lam_val = float(lam_vec[stage_idx])
price_term_1 = -config.TIMING.kappa_price_time * gamma_cost_val * max(ex_price - config.TARIFF.p_bar, 0.0)
logit = w + eta_t + lam_val + config.HUNGER.alpha0 * hunger + price_term_1
sig = 1.0 / (1.0 + np.exp(-logit))
q = sig * config.TIMING.DELTA

st.latex(r"hunger = \max(0,\ \bar n(t) - n) + \kappa \cdot \tau")
st.write(f"= max(0, {nb} - {ex_n}) + {config.HUNGER.kappa:g} x {ex_tau:g}"
         f" = {max(0, nb - ex_n):.3f} + {config.HUNGER.kappa * ex_tau:.3f} = **{hunger:.3f}**")

st.latex(r"logit = w(t) + \eta_t + \lambda_{persona,stage} + \alpha_0 \cdot hunger"
         r" - \kappa_{price,time} \cdot \gamma_{cost} \cdot (price(t) - \bar p)")
st.caption(f"price(t) is penalised *relative to* p_bar = {config.TARIFF.p_bar:g}, the "
           "time-average price every candidate tariff is normalised to -- so a flat tariff "
           "(price(t) = p_bar always) contributes exactly 0 here; only a tariff's "
           "cheap/expensive *shape* around its own average pushes cooking earlier or later.")
st.write(f"= {w:.3f} + {eta_t:.3f} + {lam_val:.3f} + {config.HUNGER.alpha0:g} x {hunger:.3f}"
         f" + (-{config.TIMING.kappa_price_time:g} x {gamma_cost_val:.2f} x "
         f"({ex_price:.2f} - {config.TARIFF.p_bar:g}))"
         f" = {w:.3f} + {eta_t:.3f} + {lam_val:.3f} + {config.HUNGER.alpha0 * hunger:.3f}"
         f" + {price_term_1:.3f} = **{logit:.3f}**")

st.latex(r"q = \mathrm{sigmoid}(logit) \times DELTA")
st.write(f"= sigmoid({logit:.3f}) x {config.TIMING.DELTA:g} = {sig:.4f} x {config.TIMING.DELTA:g}"
         f" = **{q:.4f}** (probability this agent starts cooking in this one 5-minute block)")

st.markdown("#### Stage 2 -- which meal (softmax choice)")
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

st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PRISM export: formally checking the model, not just sampling it
# ---------------------------------------------------------------------------
st.markdown("### Checking it formally: the PRISM export")
st.markdown(
    "Everything above and in the Live model section is a *Monte Carlo* estimate -- simulate many "
    "random days, average the outcome. `prism_export.py` takes the same Stage 1/Stage 2 equations "
    "and instead builds an explicit **DTMC** (discrete-time Markov chain) for [PRISM]"
    "(https://www.prismmodelchecker.org/), a probabilistic model checker: given the chain and a "
    "formal query -- e.g. *\"what's the probability this agent finishes the day without eating all "
    "3 meals?\"* -- PRISM computes the **exact** answer via linear algebra over the whole state "
    "space, not a sampled approximation. It's a cross-check that the simulator's estimates converge "
    "to the right numbers, and a demonstration that the model is portable to formal-verification "
    "tooling, not just a bespoke simulator.\n\n"
    "PRISM's modelling language has no `exp`/`sigmoid`/`softmax`, so every transition probability is "
    "computed exactly in Python first (reusing `sim.agent`'s own hazard and choice formulas) and "
    "written out as an explicit numeric `.pm` file -- PRISM just plays the numbers back, it never "
    "re-derives them.\n\n"
    "To keep the state space small enough to enumerate exhaustively, this export is a **coarser "
    "companion model**, not the exact 288-block simulator: 30-minute blocks (48/day, vs. 288 "
    "5-minute blocks), $\\tau$ capped at 12h instead of 24h, **one** representative persona (its "
    "mean $\\gamma$/$\\gamma_{\\text{cost}}$/$\\lambda$, no per-agent noise) instead of the full "
    "heterogeneous population, and a single flat reference tariff/scenario (price $=\\bar p$, "
    "$\\eta=0$) instead of a sweep. Even with all of that, the state space is still large: softmax "
    "gives every one of the 18 meals *some* nonzero probability at each firing event, so almost "
    "every combination of (breakfast meal, lunch meal, dinner meal, block, hours-since-eating) ends "
    "up technically reachable over the course of a day -- that's why the button below reports "
    "roughly 900K+ states, not a tidy handful."
)

prism_persona = st.selectbox("Persona to export", population_mod.PERSONA_NAMES, key="prism_persona")


@st.cache_resource(show_spinner=False)
def _cached_prism_chain(persona: str):
    return prism_export.build_chain(persona)


if st.button(f"Build the PRISM chain for '{prism_persona}'", key="prism_build_btn"):
    with st.spinner("Enumerating the reachable state space (~5s the first time; cached after)..."):
        t0 = time.time()
        transitions, all_states = _cached_prism_chain(prism_persona)
        build_seconds = time.time() - t0
    n_transitions = sum(len(branches) for branches in transitions.values())

    m1, m2, m3 = st.columns(3)
    m1.metric("Reachable states", f"{len(all_states):,}")
    m2.metric("Transitions", f"{n_transitions:,}")
    m3.metric("Build time", f"{build_seconds:.2f}s" if build_seconds > 0.01 else "cached")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pm_path = f"{tmp_dir}/prism_{prism_persona}.pm"
        props_path = f"{tmp_dir}/prism_{prism_persona}.props"
        prism_export.write_pm(transitions, pm_path, prism_persona)
        prism_export.write_props(props_path)
        with open(pm_path) as f:
            pm_text = f.read()
        with open(props_path) as f:
            props_text = f.read()

    preview_lines = pm_text.splitlines()
    st.caption(f"First 20 of {len(preview_lines):,} lines of the generated `.pm` file "
               f"(module header + a few transition rules -- one `[] guard -> branches;` line per "
               "reachable state):")
    st.code("\n".join(preview_lines[:20]), language="text")
    st.caption("The two example PCTL properties (`.props`) this ships with:")
    st.code(props_text, language="text")

    dl1, dl2 = st.columns(2)
    dl1.download_button("Download .pm", pm_text, file_name=f"prism_{prism_persona}.pm")
    dl2.download_button("Download .props", props_text, file_name=f"prism_{prism_persona}.props")
    st.caption("Run with PRISM: `prism prism_PERSONA.pm prism_PERSONA.props`, or open the .pm in "
               "PRISM's GUI to explore the chain interactively.")
else:
    st.caption("Not built yet -- press the button (builds once per persona per session, then cached).")

st.markdown(theme.beads(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# How the results get built: one block -> one day -> R days -> the scoreboard
# ---------------------------------------------------------------------------
st.markdown("### From one decision to the scoreboard: how the results are built")
st.markdown(
    "**One day.** `sim.run.simulate_day` steps every agent through Stage 1 -> Stage 2 -> update, "
    "block by block, for all 288 blocks (24h at 5-minute resolution). Whenever an agent fires, its "
    "meal's power draw (a flat *boxcar*: `e_kwh / duration`, held for the meal's whole cook time) "
    "is added into that block's aggregate demand curve -- the same curve the Live model section "
    "plots.\n\n"
    "**Why more than one day.** A single day is noisy -- which particular households happen to get "
    "hungry and fire on any given random day varies. So `sim.run.run_sweep` (the sidebar's Monte "
    "Carlo runs `R`) simulates **R independent days per tariff**, and -- important for a fair "
    "comparison -- samples the population **once** and reuses those exact same agents across every "
    "tariff and every run: same people, fresh dice each day. A tariff never gets compared against a "
    "different fictional population.\n\n"
    "**Scoring -- two separate goals, two separate metrics.** `clean_cooking_share` pools every "
    "cook event from every one of the R runs for a tariff and divides electric events by the total "
    "-- a population-and-day-pooled fraction, not a per-agent average (0 events, e.g. under "
    "extreme_test, scores 0% clean, not 100% -- suppressing cooking altogether isn't clean cooking). "
    "`peak_kw` / `load_factor` score the *other* thing these tariffs are meant to do -- flatten the "
    "village's demand curve, not just relocate fuel choice -- as the mean, across runs, of each "
    "day's peak demand and its (average demand / peak demand) ratio (1.0 = perfectly flat). A tariff "
    "can win on one and lose on the other: evening_peak currently has the *best* clean_cooking_share "
    "but also the *tallest* peak_kw of the three real tariffs, because agents who dodge its expensive "
    "window pile back onto the grid at the cheap hours instead. `mean_daily_kwh_household` / "
    "`median_daily_kwh_household` pool each household agent's per-run daily kWh total across all R "
    "runs, then summarise. The scoreboard sorts tariffs by clean_cooking_share descending.\n\n"
    "**What you watch while it runs.** The progress bar and live metrics in the Live model section "
    "are this exact loop instrumented with a callback that fires after every simulated day (see "
    "`run_sweep`'s `progress_callback`) -- the running clean-cooking-% and meals-simulated counters "
    "are genuine partial sums of the same events that end up in the final scoreboard, not a separate "
    "estimate.\n\n"
    "**One more thing.** The House map day-playback panel is *not* one of the R runs behind the "
    "scoreboard -- it re-simulates one fresh representative day for whichever tariff you pick in "
    "that panel, which is why it updates instantly without needing Run simulation. Treat it as an "
    "illustration of a typical day, not literally one of the runs the scoreboard averaged."
)

theme.footer()

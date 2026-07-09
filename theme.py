"""Oloika Maasai-shuka visual theme for the Streamlit dashboard.

Ported from oloika-showcase-v2.html (the colleague's static showcase page) so
the real interactive tool (app.py) carries the same look: Anton/Archivo/Space
Mono type, the red/blue/green/yellow/black shuka palette, hard-shadow cards,
and the same 18-meal SVG icon set. This module only provides CSS + HTML
fragments -- all data still comes from sim/*, nothing here is simulated.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# palette (same values as the showcase's :root custom properties)
# ---------------------------------------------------------------------------
RED = "#C22A1E"
RED_DEEP = "#8F1B12"
BLUE = "#1F5FA8"
GREEN = "#2E7D4F"
YELLOW = "#F2B01E"
ORANGE = "#E2711D"
BLACK = "#181310"
SAND = "#F7EFE2"
SAND_2 = "#EFE3CE"
INK = "#241C16"
LINE = "#D8C8AC"

# icon id per config.MEALS position (1-based idx -> list index idx-1), the
# same mapping the showcase menu section uses for these 18 dishes.
ICON_BY_MEAL_IDX = [
    "i-uji", "i-mandazi", "i-chai", "i-mukimo", "i-ugali", "i-stew", "i-kuku",
    "i-chapati", "i-fish", "i-mukimo", "i-matoke", "i-soup", "i-choma",
    "i-choma-fire", "i-fish-fire", "i-soup-fire", "i-greens", "i-kachu",
]

_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Anton&family=Archivo:wght@400;500;600;700'
    '&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">'
)

_CSS = f"""
<style>
:root {{
  --red:{RED}; --red-deep:{RED_DEEP}; --blue:{BLUE}; --green:{GREEN};
  --yellow:{YELLOW}; --orange:{ORANGE}; --black:{BLACK}; --sand:{SAND};
  --sand-2:{SAND_2}; --ink:{INK}; --line:{LINE};
  --disp:'Anton',system-ui,sans-serif; --body:'Archivo',Arial,sans-serif;
  --mono:'Space Mono',ui-monospace,monospace;
}}

html, body, [class*="css"] {{ font-family: var(--body); }}
html, [data-testid="stMain"] {{ scroll-behavior: smooth; }}
[data-testid="stAppViewContainer"] {{ background: var(--sand); color: var(--ink); }}
/* Streamlit's default header is a transparent fixed overlay that sits on top of the page and
   intercepts clicks on our sticky nav underneath it -- this is a single-page site, not a dev
   dashboard, so drop the header chrome entirely rather than fight its z-index. */
[data-testid="stHeader"] {{ display: none; }}
.block-container {{ max-width: 1360px; padding-top: 0; padding-bottom: 3rem; }}
[data-testid="stSidebar"] {{ display: none; }}

/* ---- top nav (single-page site, replaces the sidebar) ---- */
.oloika-nav {{
  position: sticky; top: 0; z-index: 999; background: var(--black); color: #fff;
  display: flex; align-items: center; gap: 1.1rem; padding: .7rem 1rem; flex-wrap: wrap;
  margin: 0 0 1.4rem; border-bottom: 4px solid var(--yellow);
}}
.oloika-nav .brand {{ font-family: var(--disp); letter-spacing: .06em; color: var(--yellow); white-space: nowrap; }}
.oloika-nav .brand em {{ color: #fff; font-style: normal; }}
.oloika-nav a {{
  color: #EFE3CE; font-size: .78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .05em; text-decoration: none;
}}
.oloika-nav a:hover {{ color: var(--yellow); }}
.section-anchor {{ scroll-margin-top: 4.5rem; }}

/* ---- dark "live model" panel (the one section that mirrors the showcase's #live) ---- */
.st-key-live_panel {{
  background: var(--black); padding: 1.6rem 1.6rem 1.8rem; margin: 0 0 1.6rem;
  border: 4px solid var(--black); box-shadow: 8px 8px 0 var(--line);
}}
.st-key-live_panel h1, .st-key-live_panel h2, .st-key-live_panel h3, .st-key-live_panel h4 {{
  color: var(--yellow) !important;
}}
/* Scoped to stWidgetLabel/.stMarkdown so button/tag labels (which reuse the same
   stMarkdownContainer internals) are never caught by this -- a bare "p" or "span"
   selector here would make button text and multiselect pill text unreadable. */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"], [data-testid="stSidebar"] .stMarkdown p,
.st-key-live_panel [data-testid="stWidgetLabel"], .st-key-live_panel .stMarkdown p {{
  color: var(--sand-2) !important;
}}
/* BaseWeb inputs/selects keep their own light background -- don't force light text into them */
[data-testid="stSidebar"] [data-baseweb="select"] *, [data-testid="stSidebar"] [data-baseweb="input"] *,
[data-testid="stSidebar"] input,
.st-key-live_panel [data-baseweb="select"] *, .st-key-live_panel [data-baseweb="input"] *,
.st-key-live_panel input {{ color: var(--ink) !important; }}
.st-key-live_panel [data-testid="stMetric"] {{ background: #221A14; border-color: var(--sand-2); }}
.st-key-live_panel [data-testid="stMetricValue"] {{ color: var(--yellow); }}
.st-key-live_panel [data-testid="stMetricLabel"] {{ color: #a3937d !important; }}

h1, h2, h3 {{
  font-family: var(--disp) !important; text-transform: uppercase;
  letter-spacing: .02em; color: var(--ink); line-height: 1.05;
}}
h1 {{ font-size: clamp(1.9rem, 3.4vw, 2.7rem) !important; }}
h2 {{ font-size: clamp(1.3rem, 2.2vw, 1.7rem) !important; }}
h3 {{ font-size: 1.05rem !important; }}
code, .stCodeBlock, .stMarkdown code {{ font-family: var(--mono) !important; }}

/* ---- sidebar as the "nav" ---- */
[data-testid="stSidebar"] {{
  background: var(--black); color: var(--sand-2); border-right: 4px solid var(--black);
}}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] span {{
  color: var(--sand-2) !important;
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
  color: var(--yellow) !important; font-family: var(--disp) !important;
}}
[data-testid="stSidebar"] hr {{ border-color: #3a2f26; }}
/* BaseWeb inputs/selects keep their own light background -- don't force light text into them */
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="input"] *,
[data-testid="stSidebar"] input {{ color: var(--ink) !important; }}

/* ---- kanga beads divider ---- */
.beads {{
  height: 10px; margin: 1.4rem 0;
  background: repeating-linear-gradient(90deg,
    var(--red) 0 6%, var(--yellow) 6% 12%, var(--blue) 12% 18%,
    #fff 18% 24%, var(--green) 24% 30%, var(--black) 30% 36%);
}}

/* ---- kanga badge ---- */
.kanga {{
  display: inline-block; font-family: var(--mono); font-size: .7rem; letter-spacing: .2em;
  text-transform: uppercase; padding: .3rem .6rem; color: #fff; margin-bottom: .5rem;
}}

/* ---- hard-shadow cards ---- */
.oloika-card {{
  background: #fff; border: 3px solid var(--black); box-shadow: 7px 7px 0 var(--black);
  padding: 1.1rem 1.2rem; margin-bottom: 1rem;
}}

/* ---- hero ---- */
.oloika-hero {{
  color: #fff; padding: 2.2rem 1.6rem 1.8rem; margin-bottom: 1.2rem; border: 4px solid var(--black);
  box-shadow: 10px 10px 0 var(--black);
  background:
    repeating-linear-gradient(0deg, transparent 0 34px, rgba(24,19,16,.55) 34px 40px),
    repeating-linear-gradient(90deg, transparent 0 34px, rgba(24,19,16,.55) 34px 40px),
    repeating-linear-gradient(0deg, transparent 0 14px, rgba(31,95,168,.35) 14px 17px, transparent 17px 74px),
    repeating-linear-gradient(90deg, transparent 0 14px, rgba(31,95,168,.35) 14px 17px, transparent 17px 74px),
    var(--red);
}}
.oloika-hero .eyebrow {{
  font-family: var(--mono); font-size: .72rem; letter-spacing: .18em; text-transform: uppercase;
  color: #FBE2B0; margin-bottom: .6rem;
}}
.oloika-hero h1 {{ color: #fff !important; margin: 0 0 .9rem; }}
.oloika-hero h1 .hl {{ color: var(--yellow); }}
.hero-stats {{
  display: grid; grid-template-columns: repeat(4,1fr); gap: 1px;
  background: rgba(255,255,255,.25); max-width: 760px;
}}
.hero-stats div {{ background: var(--red-deep); padding: .7rem .8rem; }}
.hero-stats b {{ display: block; font-family: var(--disp); font-size: 1.4rem; color: var(--yellow); }}
.hero-stats small {{ font-size: .62rem; text-transform: uppercase; letter-spacing: .08em; color: #F6DFC4; }}

/* ---- stat tiles (field evidence) ---- */
.stat-row {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin: 1rem 0; }}
.stat {{ border: 3px solid var(--black); padding: .85rem; background: #fff; position: relative; }}
.stat::before {{ content: ""; position: absolute; top: 0; left: 0; right: 0; height: 7px; }}
.stat.r::before {{ background: var(--red); }} .stat.b::before {{ background: var(--blue); }}
.stat.g::before {{ background: var(--green); }} .stat.y::before {{ background: var(--yellow); }}
.stat b {{ font-family: var(--disp); font-size: 1.5rem; display: block; }}
.stat small {{ font-size: .66rem; text-transform: uppercase; letter-spacing: .06em; color: #5c5044; }}

/* ---- constraint cards ---- */
.con-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; }}
.con {{
  border: 3px solid var(--black); background: #fff; padding: 1rem; display: flex; gap: .8rem;
  align-items: flex-start;
}}
.con svg {{ flex: 0 0 40px; }}
.con h4 {{
  font-family: var(--disp); font-weight: 400; text-transform: uppercase; font-size: .85rem;
  letter-spacing: .03em; margin: 0 0 .2rem;
}}
.con p {{ font-size: .78rem; color: #4c4136; margin: 0; }}
.con code {{ font-family: var(--mono); font-size: .68rem; background: var(--sand-2); padding: 0 .3rem; }}

/* ---- menu cards ---- */
.menu-grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(200px,1fr)); gap: 1rem; }}
.meal {{ background: #fff; border: 3px solid var(--black); position: relative; }}
.meal .art {{
  height: 110px; display: flex; align-items: center; justify-content: center;
  border-bottom: 3px solid var(--black);
}}
.meal.elec .art {{ background: linear-gradient(160deg,#e9f1fa,#cfe0f2); }}
.meal.fire .art {{ background: linear-gradient(160deg,#fbe4dc,#f3c8ba); }}
.meal .art svg {{ width: 84px; height: 84px; }}
.meal .tag {{
  position: absolute; font-family: var(--mono); font-size: .58rem; letter-spacing: .07em;
  padding: .15rem .4rem; color: #fff; margin: .4rem;
}}
.meal.elec .tag {{ background: var(--blue); }} .meal.fire .tag {{ background: var(--red); }}
.meal h4 {{
  font-family: var(--disp); font-weight: 400; font-size: .84rem; line-height: 1.15;
  padding: .55rem .6rem .3rem; text-transform: uppercase; margin: 0;
}}
.meal .chips {{ display: flex; flex-wrap: wrap; gap: .3rem; padding: 0 .6rem .6rem; }}
.meal .chips span {{
  font-family: var(--mono); font-size: .58rem; border: 2px solid var(--black); padding: .06rem .35rem;
  background: var(--sand);
}}
.meal .chips span.kw {{ background: var(--yellow); }}
.meal .chips span.ch {{ background: #f3c8ba; }}

/* ---- scale steps ---- */
.scale-steps {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; }}
.step {{ border: 3px solid var(--black); background: #fff; padding: .9rem; }}
.step .n {{ font-family: var(--disp); font-size: 1.8rem; color: var(--red); line-height: 1; }}
.step h4 {{
  font-family: var(--disp); font-weight: 400; text-transform: uppercase; font-size: .85rem;
  margin: .3rem 0 .2rem;
}}
.step p {{ font-size: .76rem; color: #4c4136; margin: 0; }}

/* ---- buttons ---- */
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {{
  font-family: var(--disp) !important; letter-spacing: .04em; text-transform: uppercase;
  border: 3px solid var(--black) !important; border-radius: 0 !important;
  box-shadow: 4px 4px 0 var(--black); background: #fff; color: var(--ink);
  transition: transform .08s ease;
}}
[data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {{
  transform: translate(-2px,-2px); box-shadow: 6px 6px 0 var(--black);
}}
[data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {{
  background: var(--red) !important; color: #fff !important; border-color: var(--black) !important;
}}
[data-testid="baseButton-primary"] {{ background: var(--red) !important; color: #fff !important; }}

/* ---- tabs as nav pills ---- */
[data-testid="stTabs"] [role="tablist"] {{
  gap: .4rem; border-bottom: 3px solid var(--black); flex-wrap: wrap;
}}
[data-testid="stTabs"] button[role="tab"] {{
  font-family: var(--disp); text-transform: uppercase; letter-spacing: .03em; font-size: .82rem;
  background: var(--black); color: var(--sand-2); border: none; border-radius: 0; padding: .5rem 1rem;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
  background: var(--yellow); color: var(--black);
}}
[data-testid="stTabs"] button[role="tab"] p {{ font-family: var(--disp); }}

/* ---- metrics ---- */
[data-testid="stMetric"] {{
  background: #fff; border: 3px solid var(--black); box-shadow: 4px 4px 0 var(--black); padding: .6rem .8rem;
}}
[data-testid="stMetricValue"] {{ font-family: var(--disp) !important; color: var(--red-deep); }}
[data-testid="stMetricLabel"] {{
  font-family: var(--mono) !important; text-transform: uppercase; font-size: .68rem !important;
  letter-spacing: .05em; color: #5c5044;
}}

/* ---- dataframe / expander / form / alerts ---- */
[data-testid="stDataFrame"] {{ border: 3px solid var(--black); }}
[data-testid="stExpander"] {{ border: 3px solid var(--black) !important; background: #fff; }}
[data-testid="stForm"] {{
  border: 3px solid var(--black) !important; background: #fffdf8; box-shadow: 7px 7px 0 var(--black);
}}
[data-testid="stAlert"] {{ border: 3px solid var(--black) !important; border-radius: 0 !important; }}
[data-baseweb="slider"] [role="slider"] {{ background-color: var(--red) !important; }}
[data-baseweb="slider"] > div > div {{ background: var(--yellow) !important; }}

hr {{ border-color: var(--line) !important; }}
</style>
"""


def _no_blank_lines(html: str) -> str:
    """Streamlit's markdown renderer splits raw HTML into blocks on blank lines, and only
    blocks starting with a recognised tag get parsed as HTML -- the rest render as literal
    text. Collapsing blank lines keeps a <style>/<svg> block as one contiguous HTML block."""
    return "\n".join(line for line in html.splitlines() if line.strip())


def inject() -> None:
    """Load fonts + inject the reskin CSS + the shared meal-icon SVG sprite. Call once per run."""
    st.markdown(_no_blank_lines(_FONT_LINK + _CSS), unsafe_allow_html=True)
    st.markdown(_no_blank_lines(_ICON_SPRITE), unsafe_allow_html=True)


_NAV_SECTIONS = [
    ("village", "Map"), ("live", "Live model"), ("evidence", "Field data"),
    ("constraints", "Constraints"), ("menu", "Menu"), ("scale", "Scale"),
    ("parameters", "Parameters"), ("explain", "Explainability"),
]


def nav_bar() -> None:
    links = "".join(f'<a href="#{sid}">{label}</a>' for sid, label in _NAV_SECTIONS)
    st.markdown(_no_blank_lines(
        f'<nav class="oloika-nav" aria-label="Main">'
        f'<span class="brand">OLOIKA <em>E-COOKING</em></span>{links}</nav>'
    ), unsafe_allow_html=True)


def anchor(section_id: str) -> None:
    """An invisible landing point for #section_id nav links, offset for the sticky nav height."""
    st.markdown(f'<div id="{section_id}" class="section-anchor"></div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(_no_blank_lines(
        '<footer style="background:var(--black);color:#B9AD98;padding:1.6rem 1.2rem;'
        'margin-top:2rem;font-size:.75rem;display:flex;justify-content:space-between;'
        'gap:2rem;flex-wrap:wrap">'
        '<div><b style="color:var(--yellow);font-family:var(--disp);font-weight:400">'
        'OLOIKA E-COOKING &middot; COSMO PHASE 2</b>'
        '<p style="margin-top:.3rem">Field data: Univ. of Southampton &amp; Kenya Power (MECS/FCDO, '
        'Oct 2025). Simulator: sim/config.py.</p></div>'
        '<div><b style="color:var(--yellow);font-family:var(--disp);font-weight:400">SHUKA PALETTE</b>'
        '<p style="margin-top:.3rem">Red -- bravery &amp; strength &middot; Blue -- sky &amp; water '
        '&middot; Green -- land &middot; Yellow/Orange -- hospitality &middot; Black -- the people.</p>'
        '</div></footer>'
    ), unsafe_allow_html=True)


def kanga(text: str, color: str) -> str:
    return f'<span class="kanga" style="background:{color}">{text}</span>'


def beads() -> str:
    return '<div class="beads" aria-hidden="true"></div>'


def stat_tile(value: str, label: str, cls: str) -> str:
    return f'<div class="stat {cls}"><b>{value}</b><small>{label}</small></div>'


def con_card(icon_id: str, title: str, body_html: str) -> str:
    return (f'<div class="con"><svg viewBox="0 0 64 64"><use href="#{icon_id}"/></svg>'
            f'<div><h4>{title}</h4><p>{body_html}</p></div></div>')


def scale_step(n: int, title: str, body: str) -> str:
    return f'<div class="step"><span class="n">{n}</span><h4>{title}</h4><p>{body}</p></div>'


def meal_card_html(name: str, icon_id: str, is_fire: bool, min_: float, kcal: float, kes: float,
                    kwh: float, charcoal_kes: float) -> str:
    cls = "fire" if is_fire else "elec"
    tag = "FIRE-ONLY" if is_fire else "MINI-GRID"
    extra = (f'<span class="ch">charcoal KES {charcoal_kes:g}</span>' if is_fire
             else f'<span class="kw">{kwh:g} kWh</span>')
    return (f'<div class="meal {cls}"><span class="tag">{tag}</span>'
            f'<div class="art"><svg viewBox="0 0 64 64" role="img" aria-label="{name}">'
            f'<use href="#{icon_id}"/></svg></div>'
            f'<h4>{name}</h4>'
            f'<div class="chips"><span>{min_:g} min</span><span>{kcal:g} kcal</span>'
            f'<span>KES {kes:g}</span>{extra}</div></div>')


# ---------------------------------------------------------------------------
# shared SVG symbol sprite -- verbatim from oloika-showcase-v2.html, referenced
# by <use href="#i-..."/> in the meal cards / constraint icons above.
# ---------------------------------------------------------------------------
_ICON_SPRITE = """
<svg width="0" height="0" style="position:absolute" aria-hidden="true">
<defs>
<symbol id="i-uji" viewBox="0 0 64 64"><path d="M8 30h48c0 14-10 22-24 22S8 44 8 30z" fill="#E2711D"/><ellipse cx="32" cy="30" rx="24" ry="6" fill="#F2B01E"/><ellipse cx="32" cy="29" rx="18" ry="4" fill="#FBE2B0"/><path d="M24 14c-3 4 3 5 0 9M34 12c-3 4 3 5 0 9M44 14c-3 4 3 5 0 9" stroke="#8F1B12" stroke-width="2.4" fill="none" stroke-linecap="round"/></symbol>
<symbol id="i-mandazi" viewBox="0 0 64 64"><ellipse cx="32" cy="46" rx="26" ry="7" fill="#C22A1E" opacity=".25"/><path d="M14 40c0-10 8-18 18-18s18 8 18 18H14z" fill="#E2711D"/><path d="M18 40c0-8 6-14 14-14s14 6 14 14" fill="#F2B01E"/><circle cx="26" cy="33" r="1.6" fill="#8F1B12"/><circle cx="34" cy="30" r="1.6" fill="#8F1B12"/><circle cx="40" cy="35" r="1.6" fill="#8F1B12"/></symbol>
<symbol id="i-chai" viewBox="0 0 64 64"><path d="M14 24h32v14a12 12 0 0 1-12 12h-8a12 12 0 0 1-12-12V24z" fill="#1F5FA8"/><path d="M14 24h32v6H14z" fill="#7fa3cd"/><path d="M46 28h6a6 6 0 0 1 0 12h-6" fill="none" stroke="#1F5FA8" stroke-width="4"/><path d="M24 8c-3 4 3 5 0 9M34 8c-3 4 3 5 0 9" stroke="#8F1B12" stroke-width="2.4" fill="none" stroke-linecap="round"/></symbol>
<symbol id="i-ugali" viewBox="0 0 64 64"><ellipse cx="32" cy="42" rx="28" ry="12" fill="#EFE3CE" stroke="#181310" stroke-width="2"/><path d="M18 38c0-8 6-14 14-14s14 6 14 14c0 3-6 5-14 5s-14-2-14-5z" fill="#fff" stroke="#181310" stroke-width="2"/><path d="M12 44c4-3 10-3 14 0" stroke="#2E7D4F" stroke-width="4" fill="none" stroke-linecap="round"/><path d="M40 46c3-2 8-2 11 0" stroke="#2E7D4F" stroke-width="4" fill="none" stroke-linecap="round"/></symbol>
<symbol id="i-stew" viewBox="0 0 64 64"><path d="M10 28h44v8a18 14 0 0 1-18 14h-8a18 14 0 0 1-18-14v-8z" fill="#181310"/><ellipse cx="32" cy="28" rx="22" ry="6" fill="#8F1B12"/><ellipse cx="32" cy="27" rx="17" ry="4" fill="#C22A1E"/><circle cx="26" cy="27" r="2.4" fill="#F2B01E"/><circle cx="37" cy="26" r="2.4" fill="#2E7D4F"/><path d="M6 30l6-4M58 30l-6-4" stroke="#181310" stroke-width="4" stroke-linecap="round"/></symbol>
<symbol id="i-kuku" viewBox="0 0 64 64"><ellipse cx="32" cy="44" rx="26" ry="9" fill="#EFE3CE" stroke="#181310" stroke-width="2"/><path d="M22 40c-2-10 4-18 12-18 7 0 12 6 12 13 0 4-2 7-6 8l-14-1c-2 0-3-1-4-2z" fill="#E2711D"/><path d="M44 30l8-6-3 8 6 1-8 4" fill="#E2711D"/><path d="M28 44c4 2 8 2 12 0" stroke="#2E7D4F" stroke-width="3.4" fill="none" stroke-linecap="round"/></symbol>
<symbol id="i-chapati" viewBox="0 0 64 64"><ellipse cx="32" cy="46" rx="25" ry="7" fill="#E2711D"/><ellipse cx="32" cy="40" rx="25" ry="7" fill="#F2B01E"/><ellipse cx="32" cy="34" rx="25" ry="7" fill="#E2711D"/><ellipse cx="32" cy="28" rx="25" ry="7" fill="#F6D06A"/><circle cx="24" cy="27" r="1.7" fill="#8F1B12"/><circle cx="36" cy="29" r="1.7" fill="#8F1B12"/><circle cx="43" cy="26" r="1.4" fill="#8F1B12"/></symbol>
<symbol id="i-fish" viewBox="0 0 64 64"><ellipse cx="32" cy="44" rx="27" ry="9" fill="#EFE3CE" stroke="#181310" stroke-width="2"/><path d="M12 36c6-8 16-12 26-8l10-6-2 10 2 10-10-6c-10 4-20 0-26-8z" fill="#1F5FA8"/><circle cx="20" cy="34" r="2" fill="#fff"/><path d="M26 32c2 4 2 5 0 8M33 31c2 4 2 5 0 9" stroke="#7fa3cd" stroke-width="2" fill="none"/></symbol>
<symbol id="i-mukimo" viewBox="0 0 64 64"><ellipse cx="32" cy="44" rx="27" ry="10" fill="#EFE3CE" stroke="#181310" stroke-width="2"/><path d="M17 40c0-9 7-16 15-16s15 7 15 16c0 3-7 5-15 5s-15-2-15-5z" fill="#7fbf98"/><circle cx="26" cy="32" r="2.4" fill="#2E7D4F"/><circle cx="36" cy="29" r="2.4" fill="#F2B01E"/><circle cx="40" cy="36" r="2.4" fill="#2E7D4F"/></symbol>
<symbol id="i-matoke" viewBox="0 0 64 64"><path d="M14 40c8 6 28 6 36 0l-4 10c-8 4-20 4-28 0z" fill="#8F1B12"/><path d="M16 24c4-8 10-12 16-12-2 6-2 12 0 18-6 2-12-1-16-6z" fill="#2E7D4F"/><path d="M32 12c6 0 12 4 16 12-4 5-10 8-16 6 2-6 2-12 0-18z" fill="#7fbf98"/></symbol>
<symbol id="i-soup" viewBox="0 0 64 64"><path d="M12 26h40v6a16 16 0 0 1-16 16h-8a16 16 0 0 1-16-16v-6z" fill="#181310"/><ellipse cx="32" cy="26" rx="20" ry="5" fill="#E2711D"/><path d="M26 22c-1-5 3-6 2-10M36 22c-1-5 3-6 2-10" stroke="#8F1B12" stroke-width="2.4" fill="none" stroke-linecap="round"/><path d="M22 26l4 4M40 25l-3 5" stroke="#F6D06A" stroke-width="3" stroke-linecap="round"/></symbol>
<symbol id="i-choma" viewBox="0 0 64 64"><path d="M16 34c0-8 6-12 14-12 10 0 18 6 18 14 0 4-4 6-9 6H24c-5 0-8-3-8-8z" fill="#8F1B12"/><path d="M22 30c2-3 6-4 9-3" stroke="#C97B5A" stroke-width="3" fill="none" stroke-linecap="round"/><line x1="10" y1="48" x2="54" y2="48" stroke="#181310" stroke-width="3"/><line x1="16" y1="48" x2="16" y2="42" stroke="#181310" stroke-width="3"/><line x1="48" y1="48" x2="48" y2="42" stroke="#181310" stroke-width="3"/></symbol>
<symbol id="i-choma-fire" viewBox="0 0 64 64"><path d="M18 30c0-7 5-11 12-11 9 0 16 5 16 12 0 4-3 5-8 5H25c-4 0-7-2-7-6z" fill="#8F1B12"/><line x1="12" y1="40" x2="52" y2="40" stroke="#181310" stroke-width="3"/><path d="M22 58c-5-5-3-10 1-13-1 5 3 5 3 9 3-2 2-6 1-9 6 3 9 9 4 13z" fill="#E2711D"/><path d="M38 58c-5-5-3-10 1-13-1 5 3 5 3 9 3-2 2-6 1-9 6 3 9 9 4 13z" fill="#C22A1E"/></symbol>
<symbol id="i-fish-fire" viewBox="0 0 64 64"><path d="M10 28c6-8 16-12 26-8l10-6-2 10 2 10-10-6c-10 4-20 0-26-8z" fill="#1F5FA8"/><circle cx="18" cy="26" r="2" fill="#fff"/><line x1="12" y1="42" x2="52" y2="42" stroke="#181310" stroke-width="3"/><path d="M28 60c-5-5-3-10 1-13-1 5 3 5 3 9 3-2 2-6 1-9 6 3 9 9 4 13z" fill="#E2711D"/></symbol>
<symbol id="i-soup-fire" viewBox="0 0 64 64"><path d="M14 22h36v5a14 14 0 0 1-14 14h-8a14 14 0 0 1-14-14v-5z" fill="#181310"/><ellipse cx="32" cy="22" rx="18" ry="4.5" fill="#E2711D"/><path d="M24 60c-5-5-3-10 1-13-1 5 3 5 3 9 3-2 2-6 1-9 6 3 9 9 4 13z" fill="#C22A1E"/><path d="M38 60c-4-4-3-9 1-11-1 4 2 4 2 7 2-2 2-5 1-7 5 2 7 7 3 11z" fill="#E2711D"/></symbol>
<symbol id="i-greens" viewBox="0 0 64 64"><ellipse cx="32" cy="44" rx="27" ry="10" fill="#EFE3CE" stroke="#181310" stroke-width="2"/><path d="M18 40c2-8 8-13 14-12-3 5-3 10-1 14-5 2-10 1-13-2z" fill="#2E7D4F"/><path d="M33 28c6-2 12 1 15 8-4 4-10 5-14 2 1-4 1-7-1-10z" fill="#7fbf98"/><path d="M26 30c1-3 4-4 6-3" stroke="#1d5636" stroke-width="2" fill="none"/></symbol>
<symbol id="i-kachu" viewBox="0 0 64 64"><ellipse cx="32" cy="44" rx="27" ry="10" fill="#EFE3CE" stroke="#181310" stroke-width="2"/><circle cx="24" cy="38" r="6" fill="#C22A1E"/><circle cx="34" cy="34" r="6" fill="#E2711D"/><circle cx="42" cy="40" r="5" fill="#7fbf98"/><circle cx="24" cy="38" r="2.4" fill="#F6D06A"/></symbol>
<symbol id="i-price" viewBox="0 0 64 64"><path d="M32 6l6 12 14 2-10 10 2 14-12-6-12 6 2-14L12 20l14-2z" fill="#F2B01E" stroke="#181310" stroke-width="2"/></symbol>
<symbol id="i-peak" viewBox="0 0 64 64"><rect x="10" y="24" width="44" height="26" fill="#1F5FA8" stroke="#181310" stroke-width="2"/><path d="M22 24v-8h20v8" fill="none" stroke="#181310" stroke-width="2"/><circle cx="32" cy="37" r="7" fill="#F2B01E"/></symbol>
<symbol id="i-vendor" viewBox="0 0 64 64"><rect x="8" y="30" width="48" height="22" fill="#E2711D" stroke="#181310" stroke-width="2"/><path d="M8 30l24-16 24 16" fill="none" stroke="#181310" stroke-width="3"/><rect x="27" y="38" width="10" height="14" fill="#181310"/></symbol>
</defs>
</svg>
"""

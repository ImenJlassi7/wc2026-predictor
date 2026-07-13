"""
WC 2026 Predictor — Simple & Clean
"""
import json, math, random
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict

st.set_page_config(page_title="WC 2026 Predictor", page_icon="🏆", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.winner-card {
    background: linear-gradient(135deg, #0d3b1e, #1b5e20);
    border: 2px solid #2e7d32;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.winner-label { font-size: 0.85rem; color: #81c784; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.winner-name  { font-size: 2.4rem; font-weight: 900; color: #fff; margin: 8px 0; }
.winner-conf  { font-size: 1rem; color: #a5d6a7; }
.score-card {
    background: linear-gradient(135deg, #0a1929, #0d47a1);
    border: 2px solid #1565c0;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.score-label  { font-size: 0.85rem; color: #90caf9; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.score-num    { font-size: 3.5rem; font-weight: 900; color: #fff; letter-spacing: 6px; margin: 8px 0; }
.pen-card {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.pen-label { font-size: 0.75rem; color: #888; text-transform: uppercase; margin-bottom: 6px; }
.pen-winner { font-size: 1.1rem; font-weight: 700; color: #ffd54f; }
.hist-header { font-size: 1.1rem; font-weight: 700; color: #fff; margin: 1.5rem 0 0.75rem; }
.match-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; border-radius: 8px;
    margin-bottom: 6px; background: #1a1a2e;
    font-size: 0.88rem;
}
.res-W { background:#1b5e20; color:#a5d6a7; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.75rem; }
.res-D { background:#1a237e; color:#90caf9; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.75rem; }
.res-L { background:#b71c1c; color:#ef9a9a; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
LEAGUE_AVG = 1.35
FORM_WEIGHT= 0.30
MAX_GOALS  = 7
random.seed(42)

FLAG_IMGS = {
    "France":      "https://flagcdn.com/w40/fr.png",
    "Morocco":     "https://flagcdn.com/w40/ma.png",
    "Spain":       "https://flagcdn.com/w40/es.png",
    "Belgium":     "https://flagcdn.com/w40/be.png",
    "Norway":      "https://flagcdn.com/w40/no.png",
    "England":     "https://flagcdn.com/w40/gb-eng.png",
    "Argentina":   "https://flagcdn.com/w40/ar.png",
    "Switzerland": "https://flagcdn.com/w40/ch.png",
    "Croatia":     "https://flagcdn.com/w40/hr.png",
    "Ghana":       "https://flagcdn.com/w40/gh.png",
    "USA":         "https://flagcdn.com/w40/us.png",
    "Egypt":       "https://flagcdn.com/w40/eg.png",
    "Colombia":    "https://flagcdn.com/w40/co.png",
    "Spain":       "https://flagcdn.com/w40/es.png",
    "England":     "https://flagcdn.com/w40/gb-eng.png",
}

MATCHES = {
    "🏆 SF1 — Angleterre vs Argentine": ("England",   "Argentina", "Demi-Finale 1"),
    "🏆 SF2 — France vs Espagne":       ("France",    "Spain",     "Demi-Finale 2"),
    "🏆 QF1 — France vs Maroc":           ("France",       "Morocco",     "Demain 21h00"),
    "🏆 QF2 — Espagne vs Belgique":        ("Spain",        "Belgium",     "Ven. 10/07 20h00"),
    "🏆 QF3 — Norvège vs Angleterre":      ("Norway",       "England",     "Sam. 11/07 22h00"),
    "🏆 QF4 — Argentine vs Suisse":        ("Argentina",    "Switzerland", "Dim. 12/07 02h00"),
  
}

# ── CORE MATH ─────────────────────────────────────────────────────────────────
def poisson_prob(lam, k):
    return math.exp(-lam) * (lam**k) / math.factorial(k)

# ─── PENALTIES ────────────────────────────────────────────────────────────────
# Taux historiques réels par équipe (même source que SF_bivariate_predictor.py)
PENALTY_RATES = {
    "England":   0.720, "Argentina": 0.800, "France":    0.780, "Spain":     0.760,
    "Croatia":   0.790, "Ghana":     0.720, "Brazil":    0.750, "Portugal":  0.800,
    "Germany":   0.830, "Netherlands": 0.740, "Morocco":  0.780, "Switzerland": 0.760,
    "USA":       0.730, "Colombia":  0.770,
}
DEFAULT_RATE = 0.750

def get_penalty_rate(team: str) -> float:
    if team in PENALTY_RATES:
        return PENALTY_RATES[team]
    for key in PENALTY_RATES:
        if key.lower() in team.lower() or team.lower() in key.lower():
            return PENALTY_RATES[key]
    return DEFAULT_RATE

def compute_xg(a, b):
    xg_a = (a["overall_avg_scored"]/LEAGUE_AVG) * (b["overall_avg_conceded"]/LEAGUE_AVG) * LEAGUE_AVG
    xg_b = (b["overall_avg_scored"]/LEAGUE_AVG) * (a["overall_avg_conceded"]/LEAGUE_AVG) * LEAGUE_AVG
    xg_a = xg_a*(1-FORM_WEIGHT) + a["avg_scored_last5"]*FORM_WEIGHT
    xg_b = xg_b*(1-FORM_WEIGHT) + b["avg_scored_last5"]*FORM_WEIGHT
    return max(xg_a, 0.01), max(xg_b, 0.01)

def score_matrix(xg_a, xg_b):
    return {(i,j): poisson_prob(xg_a,i)*poisson_prob(xg_b,j)
            for i in range(MAX_GOALS) for j in range(MAX_GOALS)}

def outcome_probs(mat):
    return (sum(p for (i,j),p in mat.items() if i>j),
            sum(p for (i,j),p in mat.items() if i==j),
            sum(p for (i,j),p in mat.items() if i<j))

def simulate_penalties(team_a, team_b, n=50_000):
    rate_a = get_penalty_rate(team_a)
    rate_b = get_penalty_rate(team_b)
    random.seed(99)
    a_wins = 0
    for _ in range(n):
        def shoot(rate): return sum(1 for _ in range(5) if random.random()<rate)
        ga,gb = shoot(rate_a),shoot(rate_b)
        while ga==gb:
            ga += random.random()<rate_a
            gb += random.random()<rate_b
        if ga>gb: a_wins+=1
    return round(a_wins/n*100,1), round((1-a_wins/n)*100,1)


# ─── BIVARIATE POISSON ────────────────────────────────────────────────────────
LAM3 = 0.10  # corrélation knockout estimée

def bivariate_poisson_prob(lam1, lam2, lam3, x, y):
    if lam3 < 1e-9:
        return poisson_prob(lam1, x) * poisson_prob(lam2, y)
    k_max = min(x, y)
    total = 0.0
    for k in range(k_max + 1):
        try:
            term = (math.comb(x,k)*math.comb(y,k)*math.factorial(k)*((lam3/(lam1*lam2))**k))
            total += term
        except: break
    try:
        log_c = (-(lam1+lam2+lam3)+x*math.log(lam1)-math.lgamma(x+1)+y*math.log(lam2)-math.lgamma(y+1))
        return max(math.exp(log_c)*total, 0.0)
    except: return 0.0

def bivariate_matrix(lam1, lam2, lam3):
    mat = {(i,j): bivariate_poisson_prob(lam1,lam2,lam3,i,j) for i in range(MAX_GOALS) for j in range(MAX_GOALS)}
    total = sum(mat.values())
    return {k: v/total for k,v in mat.items()} if total>0 else mat

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    df = pd.read_csv(DATA_DIR/"wc_all_matches.csv").dropna(subset=["home_goals","away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    with open(DATA_DIR/"team_form.json") as f:
        form = json.load(f)
    return df.sort_values("date").reset_index(drop=True), form

df_all, form = load_all()

def get_team_history(team, n=8):
    """Last N WC matches for a team."""
    mask = (df_all["home_team"].str.contains(team, case=False, na=False) |
            df_all["away_team"].str.contains(team, case=False, na=False))
    matches = df_all[mask].tail(n).copy()
    rows = []
    for _, r in matches.iterrows():
        is_home = team.lower() in str(r["home_team"]).lower()
        scored   = r["home_goals"] if is_home else r["away_goals"]
        conceded = r["away_goals"] if is_home else r["home_goals"]
        opponent = r["away_team"] if is_home else r["home_team"]
        result   = "W" if scored>conceded else ("D" if scored==conceded else "L")
        rows.append({
            "Date":     r["date"].strftime("%d/%m/%Y"),
            "Opponent": opponent.strip(),
            "Score":    f"{scored}–{conceded}",
            "Result":   result,
            "Season":   int(r["season"]),
        })
    return list(reversed(rows))

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏆 WC 2026 Predictor")
    st.markdown("---")
    st.markdown("### Sélectionner le match")
    match_label = st.selectbox("", list(MATCHES.keys()), label_visibility="collapsed")

team_a, team_b, match_date = MATCHES[match_label]

# ── COMPUTE ───────────────────────────────────────────────────────────────────
a_stats = form.get(team_a)
b_stats = form.get(team_b)

if not a_stats or not b_stats:
    st.error(f"Données manquantes pour {team_a} ou {team_b}")
    st.stop()

xg_a, xg_b = compute_xg(a_stats, b_stats)
_is_sf = "SF" in match_label
if _is_sf:
    mat = bivariate_matrix(max(xg_a-LAM3,0.05), max(xg_b-LAM3,0.05), LAM3)
else:
    mat = score_matrix(xg_a, xg_b)
p_win, p_draw, p_lose = outcome_probs(mat)
sorted_scores = sorted(mat.items(), key=lambda x: x[1], reverse=True)
best          = sorted_scores[0][0]
pen_a, pen_b  = simulate_penalties(team_a, team_b)

winner     = team_a if p_win >= p_lose else team_b
winner_pct = p_win if p_win >= p_lose else p_lose
is_close   = abs(p_win - p_lose) < 0.08

# ── HEADER ────────────────────────────────────────────────────────────────────
fa = FLAG_IMGS.get(team_a,""); fb = FLAG_IMGS.get(team_b,"")
st.markdown(f"""
<div style="display:flex;align-items:center;gap:18px;margin-bottom:0.25rem">
    <img src="{fa}" width="56" style="border-radius:5px;box-shadow:0 2px 10px #0005">
    <h1 style="margin:0;font-size:2rem;font-weight:900">{team_a} <span style="color:#444;font-weight:300">vs</span> {team_b}</h1>
    <img src="{fb}" width="56" style="border-radius:5px;box-shadow:0 2px 10px #0005">
</div>
<p style="color:#666;margin:0 0 1.5rem">⚽ FIFA World Cup 2026 · Terrain neutre · {match_date}</p>
""", unsafe_allow_html=True)

# ── MAIN CARDS ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    if is_close:
        verdict_html = f"""
        <div class="winner-card" style="background:linear-gradient(135deg,#1a1a0d,#33300a);border-color:#f9a825">
            <div class="winner-label" style="color:#ffe082">⚖️ Match très serré</div>
            <div class="winner-name" style="font-size:1.8rem">Trop proche<br>pour trancher</div>
            <div class="winner-conf" style="color:#ffe082">{team_a} {p_win*100:.0f}% · {team_b} {p_lose*100:.0f}%</div>
        </div>"""
    else:
        fw = FLAG_IMGS.get(winner,"")
        verdict_html = f"""
        <div class="winner-card">
            <div class="winner-label">🏆 Vainqueur prédit</div>
            <div style="display:flex;align-items:center;justify-content:center;gap:12px;margin:10px 0">
                <img src="{fw}" width="44" style="border-radius:4px">
                <div class="winner-name">{winner}</div>
            </div>
            <div class="winner-conf">Confiance : {winner_pct*100:.0f}%</div>
        </div>"""
    st.markdown(verdict_html, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">⚽ Score prédit</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:10px;margin:6px 0">
            <img src="{fa}" width="32" style="border-radius:3px">
            <div class="score-num">{best[0]}–{best[1]}</div>
            <img src="{fb}" width="32" style="border-radius:3px">
        </div>
        <div style="font-size:0.8rem;color:#64b5f6;margin-top:6px">
            {team_a} {best[0]} · {team_b} {best[1]}<br>
            <span style="color:#546e7a">Probabilité : {mat[best]*100:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    fw_pen = team_a if pen_a >= pen_b else team_b
    fp_pen = FLAG_IMGS.get(fw_pen,"")
    st.markdown(f"""
    <div class="pen-card">
        <div class="pen-label">🥅 Si ça va aux pénalties</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0">
            <img src="{fp_pen}" width="30" style="border-radius:3px">
            <div class="pen-winner">{fw_pen}</div>
        </div>
        <div style="font-size:0.82rem;color:#aaa;margin-top:6px">
            {team_a} {pen_a}% · {team_b} {pen_b}%
        </div>
    </div>
    <div style="margin-top:10px;background:#1a1a2e;border-radius:12px;padding:14px">
        <div style="font-size:0.75rem;color:#888;margin-bottom:8px;text-transform:uppercase">Probas 90min</div>
        <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#ccc;margin-bottom:4px">
            <span>{team_a}</span><span style="font-weight:700;color:#42a5f5">{p_win*100:.0f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#ccc;margin-bottom:4px">
            <span>Draw</span><span style="font-weight:700;color:#9e9e9e">{p_draw*100:.0f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#ccc">
            <span>{team_b}</span><span style="font-weight:700;color:#ef5350">{p_lose*100:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── SCORES LES PLUS PROBABLES ─────────────────────────────────────────────────
st.markdown("###  Scores les plus probables")

TOP_N = 6
top_scores = sorted_scores[:TOP_N]

def result_type(i, j):
    if i > j:  return f"{team_a} W"
    if i < j:  return f"{team_b} W"
    return "Draw"

TYPE_COLORS = {f"{team_a} W": "#42a5f5", "Draw": "#9e9e9e", f"{team_b} W": "#ef5350"}

labels = [f"{i}–{j}" for (i, j), _ in top_scores]
probs  = [round(p*100, 2) for _, p in top_scores]
types  = [result_type(i, j) for (i, j), _ in top_scores]
colors = [TYPE_COLORS[t] for t in types]

fig = go.Figure(go.Bar(
    x=probs[::-1],
    y=labels[::-1],
    orientation="h",
    marker_color=colors[::-1],
    text=[f"{p}%" for p in probs[::-1]],
    textposition="outside",
    hovertext=types[::-1],
    hovertemplate="%{y} · %{hovertext}<br>%{x}%<extra></extra>",
))
fig.update_layout(
    height=90 + TOP_N*40,
    margin=dict(l=10, r=40, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0e0", family="Inter"),
    xaxis=dict(title="Probabilité (%)", gridcolor="#333", zeroline=False),
    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

legend_html = " &nbsp;&nbsp; ".join(
    f'<span style="color:{c}">●</span> {t}' for t, c in TYPE_COLORS.items()
)
st.markdown(f"<div style='font-size:0.85rem;color:#aaa;margin-top:-10px'>{legend_html}</div>", unsafe_allow_html=True)

st.markdown("---")

# ── HISTORIQUE ────────────────────────────────────────────────────────────────
st.markdown("### 📋 Historique WC des équipes")

col_ha, col_hb = st.columns(2)

def render_history(col, team):
    history = get_team_history(team, n=7)
    flag_url = FLAG_IMGS.get(team, "")
    with col:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
            <img src="{flag_url}" width="28" style="border-radius:3px">
            <span style="font-weight:700;font-size:1.05rem">{team}</span>
        </div>
        """, unsafe_allow_html=True)
        if not history:
            st.caption("Pas d'historique disponible")
            return
        for m in history:
            res_class = f"res-{m['Result']}"
            st.markdown(f"""
            <div class="match-row">
                <span class="{res_class}">{m['Result']}</span>
                <span style="color:#aaa;min-width:70px">{m['Date']}</span>
                <span style="color:#666">WC {m['Season']}</span>
                <span style="color:#fff;flex:1">vs {m['Opponent']}</span>
                <span style="font-weight:700;color:#e0e0e0">{m['Score']}</span>
            </div>
            """, unsafe_allow_html=True)

render_history(col_ha, team_a)
render_history(col_hb, team_b)

st.markdown("---")
st.caption("🌍 Data: OpenFootball · WC 2014–2026 · Modèle: Poisson + Monte Carlo")
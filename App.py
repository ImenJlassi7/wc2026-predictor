"""
Step 5 — Streamlit App
Football Score Predictor — WC 2026
Run: streamlit run app.py
"""

import json
import math
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import defaultdict

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Score Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-label { font-size: 0.8rem; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #fff; }
    .metric-sub   { font-size: 0.75rem; color: #666; margin-top: 4px; }
    .score-box {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .score-num { font-size: 4rem; font-weight: 900; color: #fff; letter-spacing: 8px; }
    .score-label { font-size: 0.9rem; color: #90caf9; margin-top: 8px; }
    .pill-win  { background:#1b5e20; color:#a5d6a7; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .pill-draw { background:#1a237e; color:#90caf9; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .pill-lose { background:#b71c1c; color:#ef9a9a; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    div[data-testid="stMetric"] { background:#1e2130; border-radius:10px; padding:1rem; border:1px solid #2d3250; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path("data")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
LEAGUE_AVG  = 1.35
FORM_WEIGHT = 0.30
MAX_GOALS   = 7
K_FACTOR    = 32

ELO_SEED = {
    "France":1900,"Brazil":1880,"England":1870,"Spain":1860,"Argentina":1850,
    "Portugal":1845,"Belgium":1830,"Netherlands":1820,"Germany":1810,"Italy":1800,
    "Uruguay":1770,"Croatia":1750,"Denmark":1740,"Switzerland":1730,"Mexico":1710,
    "USA":1700,"Colombia":1690,"Senegal":1780,"Morocco":1770,"Japan":1760,
    "South Korea":1750,"Australia":1730,"Serbia":1720,"Poland":1710,"Canada":1700,
    "Ecuador":1690,"Iran":1680,"Tunisia":1670,"Ghana":1660,"Nigeria":1650,
    "Cameroon":1640,"Algeria":1630,"Costa Rica":1620,"Peru":1610,"Sweden":1595,
    "Norway":1590,"Austria":1585,"Czech Republic":1580,"Turkey":1575,
    "Scotland":1570,"Wales":1565,"Russia":1560,"Iceland":1555,
    "Panama":1510,"Saudi Arabia":1505,"Qatar":1500,"Honduras":1490,
    "South Africa":1480,"Ivory Coast":1475,"DR Congo":1470,
    "Bosnia-Herzegovina":1550,"Cape Verde":1495,"New Zealand":1460,
    "Paraguay":1520,"Haiti":1450,"Jordan":1455,"Egypt":1530,
    "Iraq":1465,"Uzbekistan":1470,
}
DEFAULT_ELO = 1500

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────

def poisson_prob(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def expected_elo(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def update_elo(ra, rb, score_a):
    exp = expected_elo(ra, rb)
    return ra + K_FACTOR*(score_a-exp), rb + K_FACTOR*((1-score_a)-(1-exp))

def compute_xg(ovr_sc_h, ovr_cc_h, form_h, ovr_sc_a, ovr_cc_a, form_a):
    atk_h = ovr_sc_h / LEAGUE_AVG
    atk_a = ovr_sc_a / LEAGUE_AVG
    def_h = ovr_cc_h / LEAGUE_AVG
    def_a = ovr_cc_a / LEAGUE_AVG
    xg_h  = atk_h * def_a * LEAGUE_AVG
    xg_a  = atk_a * def_h * LEAGUE_AVG
    xg_h  = xg_h*(1-FORM_WEIGHT) + form_h*FORM_WEIGHT
    xg_a  = xg_a*(1-FORM_WEIGHT) + form_a*FORM_WEIGHT
    return max(xg_h, 0.01), max(xg_a, 0.01)

def score_matrix(xg_h, xg_a):
    m = {}
    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            m[(i,j)] = poisson_prob(xg_h, i) * poisson_prob(xg_a, j)
    return m

def outcome_probs(m):
    pw = sum(p for (i,j),p in m.items() if i>j)
    pd_ = sum(p for (i,j),p in m.items() if i==j)
    pl = sum(p for (i,j),p in m.items() if i<j)
    return pw, pd_, pl

def monte_carlo(xg_h, xg_a, n=50_000):
    random.seed(42)
    def sample(lam):
        L=math.exp(-lam); k,p=0,1.0
        while p>L: p*=random.random(); k+=1
        return k-1
    hw=dr=aw=0
    scores={}
    for _ in range(n):
        g1,g2=sample(xg_h),sample(xg_a)
        sc=(min(g1,6),min(g2,6))
        scores[sc]=scores.get(sc,0)+1
        if g1>g2: hw+=1
        elif g1==g2: dr+=1
        else: aw+=1
    return hw/n, dr/n, aw/n, scores

# ─── LOAD & PROCESS DATA ──────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR/"wc_all_matches.csv").dropna(subset=["home_goals","away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data
def build_team_histories(df):
    elo = defaultdict(lambda: DEFAULT_ELO, {k:float(v) for k,v in ELO_SEED.items()})
    stats = defaultdict(lambda: {"scored":[],"conceded":[],"results":[],"dates":[],"elo_history":[]})
    h2h   = defaultdict(lambda: {"hw":0,"dr":0,"aw":0,"n":0})

    for _,row in df.iterrows():
        home,away = str(row["home_team"]).strip(), str(row["away_team"]).strip()
        hg,ag = int(row["home_goals"]), int(row["away_goals"])
        date  = row["date"]
        sc    = 1.0 if hg>ag else (0.5 if hg==ag else 0.0)
        res_h = "W" if hg>ag else ("D" if hg==ag else "L")
        res_a = "L" if hg>ag else ("D" if hg==ag else "W")

        stats[home]["elo_history"].append(elo[home])
        stats[away]["elo_history"].append(elo[away])

        elo[home],elo[away] = update_elo(elo[home],elo[away],sc)

        stats[home]["scored"].append(hg); stats[home]["conceded"].append(ag)
        stats[home]["results"].append(res_h); stats[home]["dates"].append(date)
        stats[away]["scored"].append(ag); stats[away]["conceded"].append(hg)
        stats[away]["results"].append(res_a); stats[away]["dates"].append(date)

        key = tuple(sorted([home,away]))
        h2h[key]["n"]+=1
        if hg>ag: h2h[key]["hw"]+=1
        elif hg==ag: h2h[key]["dr"]+=1
        else: h2h[key]["aw"]+=1

    return dict(stats), dict(elo), dict(h2h)

@st.cache_data
def get_all_teams(df):
    t = set(df["home_team"].dropna().str.strip()) | set(df["away_team"].dropna().str.strip())
    return sorted(t)

def team_summary(stats, elo, team, w=5):
    ts = stats.get(team, {"scored":[],"conceded":[],"results":[],"dates":[],"elo_history":[]})
    n  = len(ts["scored"])
    if n == 0:
        return None
    def roll(lst): return float(np.mean(lst[-w:])) if lst else LEAGUE_AVG
    def winrate(r): recent=r[-w:]; return sum(1 for x in recent if x=="W")/len(recent) if recent else 0.33
    def pts(r):
        recent=r[-w:]; mp={"W":3,"D":1,"L":0}
        return float(np.mean([mp[x] for x in recent])) if recent else 1.0
    return {
        "elo":         float(elo.get(team, DEFAULT_ELO)),
        "n":           n,
        "ovr_sc":      float(np.mean(ts["scored"])),
        "ovr_cc":      float(np.mean(ts["conceded"])),
        "form_sc":     roll(ts["scored"]),
        "form_cc":     roll(ts["conceded"]),
        "win_rate":    winrate(ts["results"]),
        "pts":         pts(ts["results"]),
        "gd":          float(np.mean([s-c for s,c in zip(ts["scored"][-w:],ts["conceded"][-w:])])) if n else 0,
        "results":     ts["results"],
        "dates":       ts["dates"],
        "scored_hist": ts["scored"],
        "conc_hist":   ts["conceded"],
        "elo_history": ts["elo_history"],
    }

# ─── LOAD ─────────────────────────────────────────────────────────────────────

df_all  = load_data()
stats, elo_final, h2h_all = build_team_histories(df_all)
all_teams = get_all_teams(df_all)

# Try load XGB report
xgb_report = None
if (DATA_DIR/"xgb_model_report.json").exists():
    with open(DATA_DIR/"xgb_model_report.json") as f:
        xgb_report = json.load(f)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ WC 2026 Predictor")
    st.markdown("---")

    st.markdown("### Match")
    st.markdown("**🇭🇷 Croatia  vs  🇬🇭 Ghana**")
    st.caption("🌍 Neutral venue · WC 2026 · Group L")

    st.markdown("---")
    st.markdown("### Tune xG")
    xg_h_override = st.slider("xG Croatia", 0.1, 4.0, 1.9, 0.05)
    xg_a_override = st.slider("xG Ghana",   0.1, 4.0, 0.85, 0.05)

    st.markdown("---")
    st.markdown("### Blend weights")
    blend_pct = st.slider("Poisson weight %", 0, 100, 40, 5)
    blend = blend_pct / 100

    st.markdown("---")
    run_mc = st.checkbox("Run Monte Carlo (50k sims)", value=True)
    st.markdown("---")
    st.caption("Data: OpenFootball · WC 2014–2026")

# ─── COMPUTE ──────────────────────────────────────────────────────────────────

home_team = "Croatia"
away_team = "Ghana"
home_s = team_summary(stats, elo_final, home_team)
away_s = team_summary(stats, elo_final, away_team)

if home_s is None or away_s is None:
    st.error("Not enough data for one of the selected teams.")
    st.stop()

xg_h, xg_a = xg_h_override, xg_a_override
mat = score_matrix(xg_h, xg_a)
p_hw_p, p_d_p, p_aw_p = outcome_probs(mat)
sorted_scores = sorted(mat.items(), key=lambda x: x[1], reverse=True)
best = sorted_scores[0][0]

# XGB probs from report (if available and same match)
p_hw_x = p_d_x = p_aw_x = None
if xgb_report:
    pred = xgb_report["prediction"]
    p_hw_x = pred["xgboost"]["croatia_win"] / 100
    p_d_x  = pred["xgboost"]["draw"] / 100
    p_aw_x = pred["xgboost"]["ghana_win"] / 100

# Ensemble
if p_hw_x is not None:
    p_hw_e = blend*p_hw_p + (1-blend)*p_hw_x
    p_d_e  = blend*p_d_p  + (1-blend)*p_d_x
    p_aw_e = blend*p_aw_p + (1-blend)*p_aw_x
else:
    p_hw_e, p_d_e, p_aw_e = p_hw_p, p_d_p, p_aw_p

# Monte Carlo
if run_mc:
    mc_hw, mc_d, mc_aw, mc_scores = monte_carlo(xg_h, xg_a)

# H2H
h2h_key = tuple(sorted([home_team, away_team]))
h2h_entry = h2h_all.get(h2h_key, {"hw":0,"dr":0,"aw":0,"n":0})

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown(f"# ⚽ {home_team} vs {away_team}")
st.markdown(f"**FIFA World Cup 2026** · Poisson + XGBoost Ensemble")
st.markdown("---")

# ─── TOP KPIs ─────────────────────────────────────────────────────────────────

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("xG " + home_team, f"{xg_h:.2f}", delta=f"Elo {home_s['elo']:.0f}")
c2.metric("xG " + away_team, f"{xg_a:.2f}", delta=f"Elo {away_s['elo']:.0f}")
c3.metric(f"{home_team} win", f"{p_hw_e*100:.1f}%")
c4.metric("Draw", f"{p_d_e*100:.1f}%")
c5.metric(f"{away_team} win", f"{p_aw_e*100:.1f}%")

st.markdown("---")

# ─── PREDICTED SCORE + SCORELINES ─────────────────────────────────────────────

col_score, col_scores = st.columns([1, 2])

with col_score:
    st.markdown(f"""
    <div class="score-box">
        <div style="font-size:0.85rem;color:#90caf9;margin-bottom:12px">PREDICTED SCORE</div>
        <div class="score-num">{best[0]} – {best[1]}</div>
        <div class="score-label">{home_team} &nbsp;·&nbsp; {away_team}</div>
        <div style="margin-top:12px;font-size:0.8rem;color:#64b5f6">
            Probability: {mat[best]*100:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"**Model Ensemble** ({100-blend_pct}% XGB + {blend_pct}% Poisson)")

with col_scores:
    st.markdown("#### Top scorelines")
    scores_data = []
    for (i,j),p in sorted_scores[:8]:
        res = "🟢 " + home_team if i>j else ("🟡 Draw" if i==j else "🔴 " + away_team)
        scores_data.append({"Score": f"{i}–{j}", "Prob %": round(p*100,2), "Result": res})
    df_sc = pd.DataFrame(scores_data)

    def score_color(result):
        if home_team in result: return "#1565c0"
        if "Draw" in result: return "#616161"
        return "#b71c1c"

    fig_sc = go.Figure(go.Bar(
        x=df_sc["Prob %"],
        y=df_sc["Score"],
        orientation="h",
        marker_color=[score_color(r) for r in df_sc["Result"]],
        text=[f"{p:.2f}%" for p in df_sc["Prob %"]],
        textposition="outside",
    ))
    fig_sc.update_layout(
        height=280, margin=dict(l=10,r=60,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc", size=12),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

st.markdown("---")

# ─── HEATMAP + OUTCOME DONUT ──────────────────────────────────────────────────

col_hm, col_donut = st.columns(2)

with col_hm:
    st.markdown("#### Score probability heatmap")
    N = 6
    z = [[mat.get((i,j),0)*100 for j in range(N)] for i in range(N)]
    fig_hm = go.Figure(go.Heatmap(
        z=z,
        x=[f"GHA {j}" for j in range(N)],
        y=[f"CRO {i}" for i in range(N)],
        colorscale="Blues",
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        showscale=False,
    ))
    fig_hm.update_layout(
        height=300, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc", size=11),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with col_donut:
    st.markdown("#### Outcome probabilities")
    labels = [f"{home_team} win", "Draw", f"{away_team} win"]
    vals_p = [p_hw_p*100, p_d_p*100, p_aw_p*100]
    vals_e = [p_hw_e*100, p_d_e*100, p_aw_e*100]

    fig_do = go.Figure()
    fig_do.add_trace(go.Bar(name="Poisson", x=labels, y=vals_p,
                             marker_color=["#1565c0","#616161","#b71c1c"],
                             opacity=0.5))
    if p_hw_x:
        vals_x = [p_hw_x*100, p_d_x*100, p_aw_x*100]
        fig_do.add_trace(go.Bar(name="XGBoost", x=labels, y=vals_x,
                                 marker_color=["#42a5f5","#9e9e9e","#ef5350"],
                                 opacity=0.7))
    fig_do.add_trace(go.Bar(name="Ensemble", x=labels, y=vals_e,
                             marker_color=["#1e88e5","#757575","#e53935"]))
    fig_do.update_layout(
        barmode="group", height=300,
        margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="%", gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig_do, use_container_width=True)

st.markdown("---")

# ─── TEAM STATS ───────────────────────────────────────────────────────────────

st.markdown("#### Team stats (WC history)")
col_h, col_a = st.columns(2)

def render_team_card(col, team, s):
    with col:
        st.markdown(f"**{team}**")
        r1,r2,r3 = st.columns(3)
        r1.metric("Elo", f"{s['elo']:.0f}")
        r2.metric("WC Matches", s["n"])
        r3.metric("Win rate (L5)", f"{s['win_rate']*100:.0f}%")
        r4,r5,r6 = st.columns(3)
        r4.metric("Avg scored", f"{s['ovr_sc']:.2f}")
        r5.metric("Avg conceded", f"{s['ovr_cc']:.2f}")
        r6.metric("Goal diff (L5)", f"{s['gd']:+.2f}")

        # Form string
        form_str = " ".join(
            f"{'🟢' if r=='W' else '🟡' if r=='D' else '🔴'}"
            for r in s["results"][-5:]
        )
        st.markdown(f"Last 5: {form_str}")

        # Elo history chart
        if s["elo_history"]:
            fig_elo = go.Figure(go.Scatter(
                y=s["elo_history"], mode="lines+markers",
                line=dict(color="#42a5f5", width=2),
                marker=dict(size=5),
            ))
            fig_elo.update_layout(
                height=120, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
                yaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
            )
            st.plotly_chart(fig_elo, use_container_width=True)

render_team_card(col_h, home_team, home_s)
render_team_card(col_a, away_team, away_s)

# ─── H2H ──────────────────────────────────────────────────────────────────────

if h2h_entry["n"] > 0:
    st.markdown("---")
    st.markdown("#### Head to Head")
    h1,h2_,h3 = st.columns(3)
    h1.metric(f"{home_team} wins", h2h_entry["hw"])
    h2_.metric("Draws", h2h_entry["dr"])
    h3.metric(f"{away_team} wins", h2h_entry["aw"])

# ─── MONTE CARLO ──────────────────────────────────────────────────────────────

if run_mc:
    st.markdown("---")
    st.markdown("#### Monte Carlo validation (50,000 simulations)")
    mc1,mc2,mc3 = st.columns(3)
    mc1.metric(f"{home_team} win", f"{mc_hw*100:.1f}%", delta=f"Poisson: {p_hw_p*100:.1f}%")
    mc2.metric("Draw",             f"{mc_d*100:.1f}%",  delta=f"Poisson: {p_d_p*100:.1f}%")
    mc3.metric(f"{away_team} win", f"{mc_aw*100:.1f}%", delta=f"Poisson: {p_aw_p*100:.1f}%")

    top_mc = sorted(mc_scores.items(), key=lambda x: x[1], reverse=True)[:6]
    mc_df  = pd.DataFrame([{"Score": f"{i}–{j}", "Simulations": c,
                             "Freq %": round(c/50000*100,2)} for (i,j),c in top_mc])
    st.dataframe(mc_df, use_container_width=True, hide_index=True)

# ─── MODEL REPORT ─────────────────────────────────────────────────────────────

if xgb_report:
    st.markdown("---")
    st.markdown("#### XGBoost model performance")
    cv = xgb_report["cv_metrics"]
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("CV Log-Loss", cv["log_loss_mean"], delta=f"±{cv['log_loss_std']}")
    m2.metric("CV Accuracy", f"{cv['accuracy_mean']}%", delta=f"±{cv['accuracy_std']}%")
    m3.metric("Training matches", xgb_report["n_training_matches"])
    m4.metric("Features", len(xgb_report["features"]))

    # Feature importance chart
    fi = xgb_report["feature_importance"]
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]).sort_values("Importance",ascending=True).tail(12)
    fig_fi = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
        marker_color="#42a5f5",
        text=[f"{v:.3f}" for v in fi_df["Importance"]], textposition="outside"
    ))
    fig_fi.update_layout(
        height=320, margin=dict(l=10,r=60,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc",size=11),
        xaxis=dict(showgrid=False,showticklabels=False),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("---")
st.caption("Built with ⚽ · OpenFootball data · Poisson + XGBoost + Monte Carlo · WC 2026")
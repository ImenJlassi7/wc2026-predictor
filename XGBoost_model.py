"""
Step 4 — XGBoost Upgrade
Features : Poisson xG · rolling form · Elo ratings · goal diff · H2H
Output   : data/xgb_prediction.json + data/xgb_model_report.json
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

DATA_DIR = Path("data")
LEAGUE_AVG  = 1.35
FORM_WEIGHT = 0.30
MAX_GOALS   = 6

# ─── ELO RATINGS ──────────────────────────────────────────────────────────────
# Seeded from approximate real-world FIFA/Elo values (2024 baseline)
ELO_SEED = {
    "France": 2000, "Brazil": 1980, "England": 1970, "Spain": 1960,
    "Argentina": 1950, "Portugal": 1945, "Belgium": 1930, "Netherlands": 1920,
    "Germany": 1910, "Italy": 1900, "Uruguay": 1870, "Croatia": 1850,
    "Denmark": 1840, "Switzerland": 1830, "Mexico": 1810, "USA": 1800,
    "Colombia": 1790, "Senegal": 1780, "Morocco": 1770, "Japan": 1760,
    "South Korea": 1750, "Australia": 1730, "Serbia": 1720, "Poland": 1710,
    "Canada": 1700, "Ecuador": 1690, "Iran": 1680, "Tunisia": 1670,
    "Ghana": 1660, "Nigeria": 1650, "Cameroon": 1640, "Algeria": 1630,
    "Costa Rica": 1620, "Peru": 1610, "Chile": 1600, "Sweden": 1595,
    "Norway": 1590, "Austria": 1585, "Czech Republic": 1580, "Turkey": 1575,
    "Scotland": 1570, "Wales": 1565, "Russia": 1560, "Iceland": 1555,
    "Panama": 1510, "Saudi Arabia": 1505, "Qatar": 1500, "Honduras": 1490,
    "South Africa": 1480, "Ivory Coast": 1475, "DR Congo": 1470,
    "Bosnia-Herzegovina": 1550, "Cape Verde": 1495, "New Zealand": 1460,
    "Paraguay": 1520, "Haiti": 1450, "Jordan": 1455, "Egypt": 1530,
    "Iraq": 1465, "Uzbekistan": 1470,
}
DEFAULT_ELO = 1500

K_FACTOR = 32

def expected_elo(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a):
    """score_a: 1=win, 0.5=draw, 0=loss"""
    exp = expected_elo(rating_a, rating_b)
    new_a = rating_a + K_FACTOR * (score_a - exp)
    new_b = rating_b + K_FACTOR * ((1 - score_a) - (1 - exp))
    return new_a, new_b

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def poisson_prob(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def compute_xg(atk_scored, def_conceded, form_scored, atk_b_scored, def_b_conceded, form_b_scored):
    atk_a = atk_scored / LEAGUE_AVG
    atk_b = atk_b_scored / LEAGUE_AVG
    def_a = def_conceded / LEAGUE_AVG
    def_b = def_b_conceded / LEAGUE_AVG
    xg_a = atk_a * def_b * LEAGUE_AVG
    xg_b = atk_b * def_a * LEAGUE_AVG
    xg_a = xg_a * (1 - FORM_WEIGHT) + form_scored * FORM_WEIGHT
    xg_b = xg_b * (1 - FORM_WEIGHT) + form_b_scored * FORM_WEIGHT
    return max(xg_a, 0.01), max(xg_b, 0.01)

def poisson_outcome_probs(xg_a, xg_b):
    matrix = {}
    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            matrix[(i, j)] = poisson_prob(xg_a, i) * poisson_prob(xg_b, j)
    p_win  = sum(p for (i, j), p in matrix.items() if i > j)
    p_draw = sum(p for (i, j), p in matrix.items() if i == j)
    p_lose = sum(p for (i, j), p in matrix.items() if i < j)
    return p_win, p_draw, p_lose

def build_features(df_all: pd.DataFrame):
    """Build feature matrix with Elo, rolling stats, Poisson xG, H2H."""
    df_all = df_all.dropna(subset=["home_goals", "away_goals"]).copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values("date").reset_index(drop=True)

    # Live Elo dict (updated match by match)
    elo = defaultdict(lambda: DEFAULT_ELO, {k: float(v) for k, v in ELO_SEED.items()})

    # Rolling stats per team
    team_stats = defaultdict(lambda: {
        "goals_scored": [], "goals_conceded": [], "results": [], "dates": []
    })

    # H2H cache
    h2h = defaultdict(lambda: {"home_w": 0, "draw": 0, "away_w": 0, "matches": 0})

    records = []

    for _, row in df_all.iterrows():
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()
        hg   = int(row["home_goals"])
        ag   = int(row["away_goals"])
        date = row["date"]

        # ── Collect stats BEFORE this match (no leakage) ──
        hs = team_stats[home]
        as_ = team_stats[away]

        def rolling(lst, w=5):
            return np.mean(lst[-w:]) if lst else LEAGUE_AVG

        def win_rate(results, w=5):
            recent = results[-w:]
            return sum(1 for r in recent if r == "W") / len(recent) if recent else 0.33

        def pts_last(results, w=5):
            recent = results[-w:]
            pts = {"W": 3, "D": 1, "L": 0}
            return np.mean([pts[r] for r in recent]) if recent else 1.0

        n_home = len(hs["goals_scored"])
        n_away = len(as_["goals_scored"])

        if n_home < 2 or n_away < 2:
            # Update stats and Elo, then skip
            result_h = "W" if hg > ag else ("D" if hg == ag else "L")
            result_a = "L" if hg > ag else ("D" if hg == ag else "W")
            hs["goals_scored"].append(hg); hs["goals_conceded"].append(ag)
            hs["results"].append(result_h); hs["dates"].append(date)
            as_["goals_scored"].append(ag); as_["goals_conceded"].append(hg)
            as_["results"].append(result_a); as_["dates"].append(date)
            sc = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            elo[home], elo[away] = update_elo(elo[home], elo[away], sc)
            continue

        # ── Elo features ──
        elo_h = elo[home]
        elo_a = elo[away]
        elo_diff = elo_h - elo_a
        elo_win_prob = expected_elo(elo_h, elo_a)

        # ── Rolling form features ──
        avg_sc_h5   = rolling(hs["goals_scored"])
        avg_cc_h5   = rolling(hs["goals_conceded"])
        avg_sc_a5   = rolling(as_["goals_scored"])
        avg_cc_a5   = rolling(as_["goals_conceded"])
        wrate_h     = win_rate(hs["results"])
        wrate_a     = win_rate(as_["results"])
        pts_h       = pts_last(hs["results"])
        pts_a       = pts_last(as_["results"])

        # Overall averages (all history)
        ovr_sc_h  = np.mean(hs["goals_scored"])
        ovr_cc_h  = np.mean(hs["goals_conceded"])
        ovr_sc_a  = np.mean(as_["goals_scored"])
        ovr_cc_a  = np.mean(as_["goals_conceded"])

        # ── Poisson xG features ──
        xg_h, xg_a = compute_xg(ovr_sc_h, ovr_cc_h, avg_sc_h5,
                                  ovr_sc_a, ovr_cc_a, avg_sc_a5)
        p_hw, p_d, p_aw = poisson_outcome_probs(xg_h, xg_a)

        # Goal diff rolling
        gd_h = np.mean([s - c for s, c in zip(hs["goals_scored"][-5:], hs["goals_conceded"][-5:])]) if hs["goals_scored"] else 0
        gd_a = np.mean([s - c for s, c in zip(as_["goals_scored"][-5:], as_["goals_conceded"][-5:])]) if as_["goals_scored"] else 0

        # ── H2H ──
        key = tuple(sorted([home, away]))
        h2h_entry = h2h[key]
        h2h_total = h2h_entry["matches"]
        h2h_home_wr = h2h_entry["home_w"] / h2h_total if h2h_total > 0 else 0.33

        # ── Experience (WC matches played) ──
        exp_h = len(hs["goals_scored"])
        exp_a = len(as_["goals_scored"])

        # ── Target ──
        outcome = 0 if hg > ag else (1 if hg == ag else 2)  # 0=home win, 1=draw, 2=away win

        records.append({
            # Elo
            "elo_diff":      elo_diff,
            "elo_win_prob":  elo_win_prob,
            # Poisson xG
            "xg_home":       xg_h,
            "xg_away":       xg_a,
            "xg_diff":       xg_h - xg_a,
            "p_home_win":    p_hw,
            "p_draw":        p_d,
            "p_away_win":    p_aw,
            # Rolling form
            "avg_scored_h5": avg_sc_h5,
            "avg_conced_h5": avg_cc_h5,
            "avg_scored_a5": avg_sc_a5,
            "avg_conced_a5": avg_cc_a5,
            "win_rate_h":    wrate_h,
            "win_rate_a":    wrate_a,
            "pts_h":         pts_h,
            "pts_a":         pts_a,
            "gd_h":          gd_h,
            "gd_a":          gd_a,
            "gd_diff":       gd_h - gd_a,
            # H2H
            "h2h_home_wr":   h2h_home_wr,
            "h2h_matches":   h2h_total,
            # Experience
            "exp_h":         exp_h,
            "exp_a":         exp_a,
            # Meta
            "home":  home, "away": away,
            "hg": hg, "ag": ag,
            "date": date,
            "outcome": outcome,
        })

        # ── Update state AFTER recording ──
        result_h = "W" if hg > ag else ("D" if hg == ag else "L")
        result_a = "L" if hg > ag else ("D" if hg == ag else "W")
        hs["goals_scored"].append(hg); hs["goals_conceded"].append(ag)
        hs["results"].append(result_h); hs["dates"].append(date)
        as_["goals_scored"].append(ag); as_["goals_conceded"].append(hg)
        as_["results"].append(result_a); as_["dates"].append(date)

        sc = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo[home], elo[away] = update_elo(elo[home], elo[away], sc)

        # H2H update
        h2h[key]["matches"] += 1
        if hg > ag: h2h[key]["home_w"] += 1
        elif hg == ag: h2h[key]["draw"] += 1
        else: h2h[key]["away_w"] += 1

    return pd.DataFrame(records), elo, team_stats, h2h


# ─── LOAD DATA + BUILD FEATURES ───────────────────────────────────────────────
df_all = pd.read_csv(DATA_DIR / "wc_all_matches.csv")
print("=" * 55)
print("  STEP 4 — XGBoost Upgrade")
print("=" * 55)

df_feat, elo_final, team_stats_final, h2h_final = build_features(df_all)
print(f"\n  Features built: {len(df_feat)} matches × {len(df_feat.columns)-5} features")

FEATURE_COLS = [
    "elo_diff", "elo_win_prob",
    "xg_home", "xg_away", "xg_diff", "p_home_win", "p_draw", "p_away_win",
    "avg_scored_h5", "avg_conced_h5", "avg_scored_a5", "avg_conced_a5",
    "win_rate_h", "win_rate_a", "pts_h", "pts_a",
    "gd_h", "gd_a", "gd_diff",
    "exp_h", "exp_a",
]
# h2h_home_wr / h2h_matches retirées : importance mesurée à 0.0 (data/xgb_model_report.json),
# elles n'ajoutaient que du bruit sur un dataset de 101 lignes.

X = df_feat[FEATURE_COLS].values
y = df_feat["outcome"].values

# ─── TRAIN XGBOOST ────────────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    objective       = "multi:softprob",
    num_class       = 3,
    n_estimators    = 60,
    max_depth       = 2,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    min_child_weight= 5,
    gamma           = 0.1,
    reg_alpha       = 0.3,
    reg_lambda      = 2.0,
    use_label_encoder=False,
    eval_metric     = "mlogloss",
    random_state    = 42,
    verbosity       = 0,
)

# Calibrate probabilities
cal_model = CalibratedClassifierCV(xgb_model, method="isotonic", cv=3)
cal_model.fit(X, y)

print(f"  Model trained on {len(X)} samples")

# ─── CROSS-VALIDATION METRICS ─────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Log-loss via CV
xgb_raw = xgb.XGBClassifier(
    objective="multi:softprob", num_class=3, n_estimators=60,
    max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.1, reg_alpha=0.3, reg_lambda=2.0,
    use_label_encoder=False, eval_metric="mlogloss", random_state=42, verbosity=0
)
cv_logloss = -cross_val_score(xgb_raw, X, y, cv=cv, scoring="neg_log_loss")
cv_acc     =  cross_val_score(xgb_raw, X, y, cv=cv, scoring="accuracy")

# Full predictions for metrics
proba_all = cal_model.predict_proba(X)
y_pred    = cal_model.predict(X)

# Brier per class (one-vs-rest)
brier_scores = []
for cls in range(3):
    y_bin = (y == cls).astype(int)
    brier_scores.append(brier_score_loss(y_bin, proba_all[:, cls]))
brier_avg = np.mean(brier_scores)

# Log-loss overall
ll_train = log_loss(y, proba_all)

# Accuracy
acc_train = np.mean(y_pred == y)

print(f"\n  ┌─────────────────────────────────────────────────────────┐")
print(f"  │ Metric           XGBoost (CV)   Poisson step3  Baseline │")
print(f"  ├─────────────────────────────────────────────────────────┤")
print(f"  │ Log-Loss (CV)    {cv_logloss.mean():.4f}         1.2862         1.0986  │")
print(f"  │ Accuracy (CV)    {cv_acc.mean()*100:.1f}%          38.4%          33.3%  │")
print(f"  │ Brier (train)    {brier_avg:.4f}         0.7260         0.4444  │")
print(f"  │ LL (train)       {ll_train:.4f}                                │")
print(f"  └─────────────────────────────────────────────────────────┘")

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
importances = xgb_raw.fit(X, y).feature_importances_
feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)

print(f"\n  Feature importance (top 10):")
for feat, imp in feat_imp[:10]:
    bar = "█" * int(imp * 300)
    print(f"    {feat:<20s} {imp:.4f}  {bar}")

# ─── PREDICT CROATIA vs GHANA ─────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  PREDICTION: Croatia vs Ghana — WC 2026")
print(f"{'='*55}")

with open(DATA_DIR / "team_form.json") as f:
    form = json.load(f)

def get_latest_stats(team_stats, elo, team):
    ts = team_stats[team]
    n  = len(ts["goals_scored"])
    if n == 0:
        return None

    def rolling(lst, w=5):
        return float(np.mean(lst[-w:])) if lst else LEAGUE_AVG

    def win_rate(results, w=5):
        recent = results[-w:]
        return sum(1 for r in recent if r == "W") / len(recent) if recent else 0.33

    def pts_last(results, w=5):
        recent = results[-w:]
        pts = {"W": 3, "D": 1, "L": 0}
        return float(np.mean([pts[r] for r in recent])) if recent else 1.0

    return {
        "elo":          float(elo[team]),
        "avg_scored":   float(np.mean(ts["goals_scored"])),
        "avg_conced":   float(np.mean(ts["goals_conceded"])),
        "avg_scored_h5":rolling(ts["goals_scored"]),
        "avg_conced_h5":rolling(ts["goals_conceded"]),
        "win_rate":     win_rate(ts["results"]),
        "pts":          pts_last(ts["results"]),
        "gd":           float(np.mean([s-c for s,c in zip(ts["goals_scored"][-5:], ts["goals_conceded"][-5:])])),
        "n_matches":    n,
    }

cro_s = get_latest_stats(team_stats_final, elo_final, "Croatia")
gha_s = get_latest_stats(team_stats_final, elo_final, "Ghana")

xg_h, xg_a = compute_xg(
    cro_s["avg_scored"], cro_s["avg_conced"], cro_s["avg_scored_h5"],
    gha_s["avg_scored"], gha_s["avg_conced"], gha_s["avg_scored_h5"]
)
p_hw, p_d, p_aw = poisson_outcome_probs(xg_h, xg_a)

h2h_key = tuple(sorted(["Croatia", "Ghana"]))
h2h_entry = h2h_final[h2h_key]
h2h_total = h2h_entry["matches"]
h2h_home_wr = h2h_entry["home_w"] / h2h_total if h2h_total > 0 else 0.33

feat_vec = np.array([[
    cro_s["elo"] - gha_s["elo"],              # elo_diff
    expected_elo(cro_s["elo"], gha_s["elo"]), # elo_win_prob
    xg_h, xg_a, xg_h - xg_a,                 # xg features
    p_hw, p_d, p_aw,                          # poisson probs
    cro_s["avg_scored_h5"], cro_s["avg_conced_h5"],
    gha_s["avg_scored_h5"], gha_s["avg_conced_h5"],
    cro_s["win_rate"], gha_s["win_rate"],
    cro_s["pts"], gha_s["pts"],
    cro_s["gd"], gha_s["gd"], cro_s["gd"] - gha_s["gd"],
    cro_s["n_matches"], gha_s["n_matches"],
]])

xgb_proba = cal_model.predict_proba(feat_vec)[0]  # [home_win, draw, away_win]
p_cro_win_xgb = float(xgb_proba[0])
p_draw_xgb    = float(xgb_proba[1])
p_gha_win_xgb = float(xgb_proba[2])

# Ensemble: blend Poisson + XGBoost
BLEND = 0.70  # 70% Poisson, 30% XGBoost — Poisson généralise mieux en CV (log-loss 1.286 vs 1.345)
p_cro_final = BLEND * p_hw    + (1 - BLEND) * p_cro_win_xgb
p_draw_final= BLEND * p_d     + (1 - BLEND) * p_draw_xgb
p_gha_final = BLEND * p_aw    + (1 - BLEND) * p_gha_win_xgb

print(f"\n  Elo Croatia  : {cro_s['elo']:.0f}")
print(f"  Elo Ghana    : {gha_s['elo']:.0f}")
print(f"  xG Croatia   : {xg_h:.3f}")
print(f"  xG Ghana     : {xg_a:.3f}")
print(f"  H2H matches  : {h2h_total}")

print(f"\n  {'':20s}  {'Poisson':>8s}  {'XGBoost':>8s}  {'Ensemble':>9s}")
print(f"  {'─'*52}")
print(f"  {'Croatia win':20s}  {p_hw*100:>7.1f}%  {p_cro_win_xgb*100:>7.1f}%  {p_cro_final*100:>8.1f}%")
print(f"  {'Draw':20s}  {p_d*100:>7.1f}%  {p_draw_xgb*100:>7.1f}%  {p_draw_final*100:>8.1f}%")
print(f"  {'Ghana win':20s}  {p_aw*100:>7.1f}%  {p_gha_win_xgb*100:>7.1f}%  {p_gha_final*100:>8.1f}%")

# Best scoreline via blended xG still
matrix = {}
for i in range(MAX_GOALS):
    for j in range(MAX_GOALS):
        matrix[(i, j)] = poisson_prob(xg_h, i) * poisson_prob(xg_a, j)
sorted_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
best = sorted_scores[0][0]

print(f"\n  ★ Score prédit   : {best[0]}-{best[1]} (CRO-GHA)")
print(f"\n  Top 6 scorelines:")
for (i, j), p in sorted_scores[:6]:
    res = "CRO W" if i > j else ("Draw" if i == j else "GHA W")
    bar = "█" * int(p * 300)
    print(f"    {i}-{j}  {p*100:>5.2f}%  {res}  {bar}")

# ─── EXPORT ───────────────────────────────────────────────────────────────────
report = {
    "model": "XGBoost + Poisson Ensemble",
    "n_training_matches": len(X),
    "features": FEATURE_COLS,
    "cv_metrics": {
        "log_loss_mean": round(float(cv_logloss.mean()), 4),
        "log_loss_std":  round(float(cv_logloss.std()), 4),
        "accuracy_mean": round(float(cv_acc.mean() * 100), 1),
        "accuracy_std":  round(float(cv_acc.std() * 100), 1),
    },
    "feature_importance": {f: round(float(imp), 4) for f, imp in feat_imp},
    "prediction": {
        "match":       "Croatia vs Ghana",
        "competition": "FIFA World Cup 2026",
        "elo_croatia": round(cro_s["elo"], 0),
        "elo_ghana":   round(gha_s["elo"], 0),
        "xg_croatia":  round(xg_h, 3),
        "xg_ghana":    round(xg_a, 3),
        "poisson": {
            "croatia_win": round(p_hw * 100, 1),
            "draw":        round(p_d * 100, 1),
            "ghana_win":   round(p_aw * 100, 1),
        },
        "xgboost": {
            "croatia_win": round(p_cro_win_xgb * 100, 1),
            "draw":        round(p_draw_xgb * 100, 1),
            "ghana_win":   round(p_gha_win_xgb * 100, 1),
        },
        "ensemble": {
            "croatia_win": round(p_cro_final * 100, 1),
            "draw":        round(p_draw_final * 100, 1),
            "ghana_win":   round(p_gha_final * 100, 1),
        },
        "predicted_score": f"{best[0]}-{best[1]}",
        "top_scorelines": [
            {"score": f"{i}-{j}", "prob_pct": round(p*100, 2),
             "result": "Croatia W" if i > j else ("Draw" if i == j else "Ghana W")}
            for (i, j), p in sorted_scores[:6]
        ],
    }
}

with open(DATA_DIR / "xgb_model_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n✓ xgb_model_report.json exporté")
print("=" * 55)
"""
Step 3 — Model Evaluation
Metrics : Log-Loss · Brier Score · Calibration · Backtest WC 2014-2026
"""

import json
import math
import random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
LEAGUE_AVG  = 1.35
FORM_WEIGHT = 0.30
MAX_GOALS   = 6
random.seed(42)

# ─── POISSON CORE ─────────────────────────────────────────────────────────────

def poisson_prob(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def compute_xg(stats_a: dict, stats_b: dict) -> tuple[float, float]:
    atk_a = stats_a["overall_avg_scored"]   / LEAGUE_AVG
    atk_b = stats_b["overall_avg_scored"]   / LEAGUE_AVG
    def_a = stats_a["overall_avg_conceded"] / LEAGUE_AVG
    def_b = stats_b["overall_avg_conceded"] / LEAGUE_AVG

    xg_a = atk_a * def_b * LEAGUE_AVG
    xg_b = atk_b * def_a * LEAGUE_AVG

    xg_a = xg_a * (1 - FORM_WEIGHT) + stats_a["avg_scored_last5"] * FORM_WEIGHT
    xg_b = xg_b * (1 - FORM_WEIGHT) + stats_b["avg_scored_last5"] * FORM_WEIGHT
    return max(xg_a, 0.01), max(xg_b, 0.01)

def predict_probs(xg_a: float, xg_b: float) -> dict:
    matrix = {}
    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            matrix[(i, j)] = poisson_prob(xg_a, i) * poisson_prob(xg_b, j)
    p_win  = sum(p for (i, j), p in matrix.items() if i > j)
    p_draw = sum(p for (i, j), p in matrix.items() if i == j)
    p_lose = sum(p for (i, j), p in matrix.items() if i < j)
    return {"win": p_win, "draw": p_draw, "lose": p_lose}

# ─── LOAD ALL WC MATCHES ──────────────────────────────────────────────────────

df_all = pd.read_csv(DATA_DIR / "wc_all_matches.csv", parse_dates=["date"])
df_all = df_all.dropna(subset=["home_goals", "away_goals"])
df_all["home_goals"] = df_all["home_goals"].astype(int)
df_all["away_goals"] = df_all["away_goals"].astype(int)
df_all = df_all.sort_values("date").reset_index(drop=True)

# ─── BACKTEST LOOP ────────────────────────────────────────────────────────────
# Pour chaque match, on reconstruit les stats des 2 équipes
# sur les matchs WC PRÉCÉDENTS uniquement (no leakage)

def get_team_stats_before(df: pd.DataFrame, team: str, before_date) -> dict | None:
    """Calcule les stats d'une équipe sur ses matchs WC avant une date."""
    mask_home = df["home_team"].str.lower().str.contains(team.lower(), na=False)
    mask_away = df["away_team"].str.lower().str.contains(team.lower(), na=False)
    matches = df[(mask_home | mask_away) & (df["date"] < before_date)].copy()

    if len(matches) < 2:  # trop peu de données → skip
        return None

    rows = []
    for _, r in matches.iterrows():
        is_home = team.lower() in str(r["home_team"]).lower()
        scored   = r["home_goals"] if is_home else r["away_goals"]
        conceded = r["away_goals"] if is_home else r["home_goals"]
        rows.append({"scored": scored, "conceded": conceded})

    s = pd.DataFrame(rows)
    last5 = s.tail(5)

    return {
        "overall_avg_scored":   round(s["scored"].mean(), 3),
        "overall_avg_conceded": round(s["conceded"].mean(), 3),
        "avg_scored_last5":     round(last5["scored"].mean(), 3),
        "avg_conceded_last5":   round(last5["conceded"].mean(), 3),
        "n_matches": len(s)
    }

# Collecter toutes les prédictions vs réalité
results = []

for idx, row in df_all.iterrows():
    home = str(row["home_team"]).strip()
    away = str(row["away_team"]).strip()
    gh   = int(row["home_goals"])
    ga   = int(row["away_goals"])
    date = row["date"]
    season = row["season"]

    stats_home = get_team_stats_before(df_all, home, date)
    stats_away = get_team_stats_before(df_all, away, date)

    if stats_home is None or stats_away is None:
        continue

    xg_h, xg_a = compute_xg(stats_home, stats_away)
    probs = predict_probs(xg_h, xg_a)

    actual_outcome = "win" if gh > ga else ("draw" if gh == ga else "lose")

    results.append({
        "date":       date,
        "season":     season,
        "home":       home,
        "away":       away,
        "home_goals": gh,
        "away_goals": ga,
        "xg_home":    round(xg_h, 3),
        "xg_away":    round(xg_a, 3),
        "p_home_win": round(probs["win"], 4),
        "p_draw":     round(probs["draw"], 4),
        "p_away_win": round(probs["lose"], 4),
        "outcome":    actual_outcome,
    })

df_res = pd.DataFrame(results)
df_res.to_csv(DATA_DIR / "backtest_results.csv", index=False)

print(f"  Backtest: {len(df_res)} matches évalués ({df_all['season'].unique()})")

# ─── MÉTRIQUES ────────────────────────────────────────────────────────────────

def log_loss_score(df: pd.DataFrame) -> float:
    """Log-loss multiclass (3 outcomes)."""
    eps = 1e-9
    losses = []
    for _, r in df.iterrows():
        if r["outcome"] == "win":
            p = r["p_home_win"]
        elif r["outcome"] == "draw":
            p = r["p_draw"]
        else:
            p = r["p_away_win"]
        losses.append(-math.log(max(p, eps)))
    return round(np.mean(losses), 4)

def brier_score(df: pd.DataFrame) -> float:
    """Brier score multiclass."""
    scores = []
    for _, r in df.iterrows():
        y = [1 if r["outcome"] == o else 0 for o in ["win", "draw", "lose"]]
        p = [r["p_home_win"], r["p_draw"], r["p_away_win"]]
        scores.append(sum((pi - yi)**2 for pi, yi in zip(p, y)))
    return round(np.mean(scores), 4)

def accuracy_argmax(df: pd.DataFrame) -> float:
    """Accuracy: on prédit l'outcome avec la proba max."""
    correct = 0
    for _, r in df.iterrows():
        pred = max([("win", r["p_home_win"]), ("draw", r["p_draw"]), ("lose", r["p_away_win"])],
                   key=lambda x: x[1])[0]
        if pred == r["outcome"]:
            correct += 1
    return round(correct / len(df) * 100, 1)

def calibration_analysis(df: pd.DataFrame, n_bins: int = 5) -> list[dict]:
    """Calibration: compare prob prédite vs fréquence réelle par bin."""
    # On traite chaque outcome séparément
    bins_data = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for outcome, col in [("home win", "p_home_win"), ("draw", "p_draw"), ("away win", "p_away_win")]:
        actual_flag = df["outcome"].apply(
            lambda x: 1 if (outcome == "home win" and x == "win") or
                           (outcome == "draw"     and x == "draw") or
                           (outcome == "away win" and x == "lose") else 0
        )
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i+1]
            mask = (df[col] >= lo) & (df[col] < hi)
            n = mask.sum()
            if n == 0: continue
            pred_mean = df.loc[mask, col].mean()
            actual_mean = actual_flag[mask].mean()
            bins_data.append({
                "outcome":    outcome,
                "bin":        f"{lo:.1f}-{hi:.1f}",
                "n":          int(n),
                "pred_prob":  round(pred_mean, 3),
                "actual_freq":round(actual_mean, 3),
                "diff":       round(actual_mean - pred_mean, 3)
            })
    return bins_data

def rps_score(df: pd.DataFrame) -> float:
    """Ranked Probability Score (ordered: win > draw > lose)."""
    scores = []
    for _, r in df.iterrows():
        p = [r["p_home_win"], r["p_draw"], r["p_away_win"]]
        y = [1 if r["outcome"] == o else 0 for o in ["win", "draw", "lose"]]
        cum_p = [sum(p[:k+1]) for k in range(3)]
        cum_y = [sum(y[:k+1]) for k in range(3)]
        rps = sum((cp - cy)**2 for cp, cy in zip(cum_p, cum_y)) / (3 - 1)
        scores.append(rps)
    return round(np.mean(scores), 4)

# ─── COMPUTE ALL METRICS ──────────────────────────────────────────────────────

ll   = log_loss_score(df_res)
bs   = brier_score(df_res)
acc  = accuracy_argmax(df_res)
rps  = rps_score(df_res)
cal  = calibration_analysis(df_res)

# Baseline naif : toujours prédire 1/3 pour chaque outcome
naive_ll = -math.log(1/3)
naive_bs  = 2 * (1/3) * (2/3)  # approx multiclass uniform

print(f"\n{'='*55}")
print(f"  STEP 3 — MODEL EVALUATION")
print(f"{'='*55}")
print(f"\n  Matchs WC backtest : {len(df_res)}")
print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │ Metric          Model    Baseline (1/3) │")
print(f"  ├─────────────────────────────────────────┤")
print(f"  │ Log-Loss        {ll:<8.4f} {naive_ll:<8.4f}       │")
print(f"  │ Brier Score     {bs:<8.4f} {naive_bs:.4f}       │")
print(f"  │ RPS             {rps:<8.4f} 0.2222         │")
print(f"  │ Accuracy (max)  {acc:<6.1f}%                   │")
print(f"  └─────────────────────────────────────────┘")

# Calibration
print(f"\n  Calibration (predicted prob vs actual frequency):")
print(f"  {'Outcome':<12} {'Bin':<12} {'N':>4} {'Pred':>6} {'Actual':>8} {'Diff':>7}")
print(f"  {'─'*55}")
for b in cal:
    flag = " ⚠" if abs(b["diff"]) > 0.10 else ""
    print(f"  {b['outcome']:<12} {b['bin']:<12} {b['n']:>4} {b['pred_prob']:>6.3f} {b['actual_freq']:>8.3f} {b['diff']:>+7.3f}{flag}")

# Performance par saison
print(f"\n  Performance par saison:")
for season in sorted(df_res["season"].unique()):
    sub = df_res[df_res["season"] == season]
    if len(sub) < 3: continue
    sub_ll  = log_loss_score(sub)
    sub_acc = accuracy_argmax(sub)
    sub_rps = rps_score(sub)
    print(f"    WC {season}: {len(sub):>3} matchs │ LL={sub_ll:.3f} │ RPS={sub_rps:.3f} │ Acc={sub_acc:.1f}%")

# ─── ANALYSE CROATIA VS GHANA SPÉCIFIQUE ─────────────────────────────────────

print(f"\n{'='*55}")
print(f"  PREDICTION CROATIA vs GHANA (today WC 2026)")
print(f"{'='*55}")

with open(DATA_DIR / "team_form.json") as f:
    form = json.load(f)

xg_cro, xg_gha = compute_xg(form["Croatia"], form["Ghana"])
probs = predict_probs(xg_cro, xg_gha)

# Scoreline matrix
matrix = {}
for i in range(MAX_GOALS):
    for j in range(MAX_GOALS):
        matrix[(i, j)] = poisson_prob(xg_cro, i) * poisson_prob(xg_gha, j)

sorted_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
best_score    = sorted_scores[0][0]

print(f"\n  xG Croatia : {xg_cro:.3f}")
print(f"  xG Ghana   : {xg_gha:.3f}")
print(f"\n  Croatia win : {probs['win']*100:.1f}%")
print(f"  Draw        : {probs['draw']*100:.1f}%")
print(f"  Ghana win   : {probs['lose']*100:.1f}%")
print(f"\n  Score prédit : {best_score[0]}-{best_score[1]} (CRO-GHA)")

print(f"\n  Top 6 scorelines:")
for (i, j), p in sorted_scores[:6]:
    result = "CRO W" if i > j else ("Draw" if i == j else "GHA W")
    bar = "█" * int(p * 300)
    print(f"    {i}-{j}  {p*100:>5.2f}%  {result}  {bar}")

# ─── EXPORT JSON ──────────────────────────────────────────────────────────────

evaluation_report = {
    "n_matches_backtest": len(df_res),
    "seasons_covered":    sorted(df_res["season"].unique().tolist()),
    "metrics": {
        "log_loss":         ll,
        "brier_score":      bs,
        "rps":              rps,
        "accuracy_pct":     acc,
        "baseline_log_loss":round(naive_ll, 4),
        "baseline_brier":   round(naive_bs, 4),
    },
    "calibration": cal,
    "prediction_croatia_ghana": {
        "xg_croatia": round(xg_cro, 3),
        "xg_ghana":   round(xg_gha, 3),
        "croatia_win_pct": round(probs["win"] * 100, 1),
        "draw_pct":         round(probs["draw"] * 100, 1),
        "ghana_win_pct":    round(probs["lose"] * 100, 1),
        "predicted_score":  f"{best_score[0]}-{best_score[1]}",
        "top_scorelines": [
            {"score": f"{i}-{j}", "prob_pct": round(p*100, 2),
             "result": "Croatia W" if i > j else ("Draw" if i == j else "Ghana W")}
            for (i, j), p in sorted_scores[:6]
        ]
    }
}

with open(DATA_DIR / "evaluation_report.json", "w") as f:
    json.dump(evaluation_report, f, indent=2)

print(f"\n✓ evaluation_report.json exporté")
print(f"{'='*55}")
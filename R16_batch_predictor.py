"""
Round of 16 — Batch Predictor
Matches : USA vs Belgium | Argentina vs Egypt | Switzerland vs Colombia
Output  : data/r16_predictions.json
"""

import json
import math
import random
from pathlib import Path

DATA_DIR    = Path("data")
LEAGUE_AVG  = 1.35
FORM_WEIGHT = 0.30
MAX_GOALS   = 7
random.seed(42)

MATCHES = [
    ("USA",         "Belgium"),
    ("Argentina",   "Egypt"),
    ("Switzerland", "Colombia"),
]

# ─── CORE ─────────────────────────────────────────────────────────────────────

def poisson_prob(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def compute_xg(a, b):
    atk_a = a["overall_avg_scored"]   / LEAGUE_AVG
    atk_b = b["overall_avg_scored"]   / LEAGUE_AVG
    def_a = a["overall_avg_conceded"] / LEAGUE_AVG
    def_b = b["overall_avg_conceded"] / LEAGUE_AVG
    xg_a  = atk_a * def_b * LEAGUE_AVG
    xg_b  = atk_b * def_a * LEAGUE_AVG
    xg_a  = xg_a * (1 - FORM_WEIGHT) + a["avg_scored_last5"] * FORM_WEIGHT
    xg_b  = xg_b * (1 - FORM_WEIGHT) + b["avg_scored_last5"] * FORM_WEIGHT
    return max(xg_a, 0.01), max(xg_b, 0.01)

def score_matrix(xg_a, xg_b):
    return {(i, j): poisson_prob(xg_a, i) * poisson_prob(xg_b, j)
            for i in range(MAX_GOALS) for j in range(MAX_GOALS)}

def outcome_probs(mat):
    return (
        sum(p for (i, j), p in mat.items() if i > j),
        sum(p for (i, j), p in mat.items() if i == j),
        sum(p for (i, j), p in mat.items() if i < j),
    )

def monte_carlo(xg_a, xg_b, n=50_000):
    def sample(lam):
        L = math.exp(-lam); k, p = 0, 1.0
        while p > L: p *= random.random(); k += 1
        return k - 1
    hw = dr = aw = 0
    for _ in range(n):
        g1, g2 = sample(xg_a), sample(xg_b)
        if g1 > g2: hw += 1
        elif g1 == g2: dr += 1
        else: aw += 1
    return hw/n, dr/n, aw/n

# ─── LOAD FORM ────────────────────────────────────────────────────────────────

with open(DATA_DIR / "team_form.json") as f:
    form = json.load(f)

# ─── BATCH PREDICT ────────────────────────────────────────────────────────────

print("=" * 60)
print("  ROUND OF 16 — WC 2026 PREDICTIONS")
print("=" * 60)

all_predictions = []

for team_a, team_b in MATCHES:
    a = form[team_a]
    b = form[team_b]

    xg_a, xg_b  = compute_xg(a, b)
    mat          = score_matrix(xg_a, xg_b)
    p_win, p_draw, p_lose = outcome_probs(mat)
    mc_w, mc_d, mc_l      = monte_carlo(xg_a, xg_b)

    sorted_scores = sorted(mat.items(), key=lambda x: x[1], reverse=True)
    best = sorted_scores[0][0]

    print(f"\n  {'─'*55}")
    print(f"  {team_a} vs {team_b}")
    print(f"  {'─'*55}")
    print(f"  xG {team_a:<15s}: {xg_a:.3f}")
    print(f"  xG {team_b:<15s}: {xg_b:.3f}")
    print(f"\n  {'':20s} {'Poisson':>8s}  {'Monte Carlo':>12s}")
    print(f"  {team_a+' win':<20s} {p_win*100:>7.1f}%  {mc_w*100:>11.1f}%")
    print(f"  {'Draw':<20s} {p_draw*100:>7.1f}%  {mc_d*100:>11.1f}%")
    print(f"  {team_b+' win':<20s} {p_lose*100:>7.1f}%  {mc_l*100:>11.1f}%")
    print(f"\n  ★ Score prédit : {best[0]}-{best[1]} ({team_a}-{team_b})")
    print(f"\n  Top 6 scorelines:")
    for (i, j), p in sorted_scores[:6]:
        res = f"{team_a} W" if i > j else ("Draw" if i == j else f"{team_b} W")
        bar = "█" * int(p * 300)
        print(f"    {i}-{j}  {p*100:>5.2f}%  {res}  {bar}")

    all_predictions.append({
        "match":    f"{team_a} vs {team_b}",
        "team_a":   team_a,
        "team_b":   team_b,
        "xg": {
            team_a: round(xg_a, 3),
            team_b: round(xg_b, 3),
        },
        "probabilities": {
            f"{team_a}_win": round(p_win  * 100, 1),
            "draw":          round(p_draw * 100, 1),
            f"{team_b}_win": round(p_lose * 100, 1),
        },
        "monte_carlo": {
            f"{team_a}_win": round(mc_w * 100, 1),
            "draw":          round(mc_d * 100, 1),
            f"{team_b}_win": round(mc_l * 100, 1),
        },
        "predicted_score": f"{best[0]}-{best[1]}",
        "top_scorelines": [
            {
                "score":    f"{i}-{j}",
                "prob_pct": round(p * 100, 2),
                "result":   f"{team_a} W" if i > j else ("Draw" if i == j else f"{team_b} W"),
            }
            for (i, j), p in sorted_scores[:6]
        ],
    })

with open(DATA_DIR / "r16_predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=2)

print(f"\n\n{'='*60}")
print(f"  ✓ r16_predictions.json exporté")
print(f"{'='*60}")
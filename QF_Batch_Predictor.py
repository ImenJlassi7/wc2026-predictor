"""
Quarts de Finale — Batch Predictor WC 2026
Matches : France vs Morocco | Spain vs Belgium | Norway vs England | Argentina vs Switzerland
Output  : data/qf_predictions.json
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
    ("France",    "Morocco",     "Demain 21h00"),
    ("Spain",     "Belgium",     "Ven. 10/07 20h00"),
    ("Norway",    "England",     "Sam. 11/07 22h00"),
    ("Argentina", "Switzerland", "Dim. 12/07 02h00"),
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
    pw  = sum(p for (i,j),p in mat.items() if i > j)
    pd_ = sum(p for (i,j),p in mat.items() if i == j)
    pl  = sum(p for (i,j),p in mat.items() if i < j)
    return pw, pd_, pl

def monte_carlo(xg_a, xg_b, n=100_000):
    random.seed(42)
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

def simulate_penalties(n=100_000):
    """P(score per kick) = 0.75. Best of 5, then sudden death."""
    random.seed(99)
    a_wins = 0
    for _ in range(n):
        def shoot():
            return sum(1 for _ in range(5) if random.random() < 0.75)
        ga, gb = shoot(), shoot()
        while ga == gb:
            ga += random.random() < 0.75
            gb += random.random() < 0.75
        if ga > gb: a_wins += 1
    return round(a_wins/n*100, 1), round((1-a_wins/n)*100, 1)

# ─── LOAD ─────────────────────────────────────────────────────────────────────

with open(DATA_DIR / "team_form.json") as f:
    form = json.load(f)

# ─── BATCH ────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  QUARTS DE FINALE — WC 2026 PREDICTIONS")
print("=" * 60)

all_predictions = []

for team_a, team_b, date in MATCHES:
    a = form[team_a]
    b = form[team_b]

    xg_a, xg_b           = compute_xg(a, b)
    mat                   = score_matrix(xg_a, xg_b)
    p_win, p_draw, p_lose = outcome_probs(mat)
    mc_w, mc_d, mc_l      = monte_carlo(xg_a, xg_b)
    pen_a, pen_b          = simulate_penalties()

    sorted_scores = sorted(mat.items(), key=lambda x: x[1], reverse=True)
    best = sorted_scores[0][0]

    # Determine winner
    if p_win > p_lose:
        winner = team_a; conf = p_win
    else:
        winner = team_b; conf = p_lose

    print(f"\n  {'─'*55}")
    print(f"  {date}")
    print(f"  {team_a} vs {team_b}")
    print(f"  {'─'*55}")
    print(f"  xG {team_a:<15s}: {xg_a:.3f}")
    print(f"  xG {team_b:<15s}: {xg_b:.3f}")
    print(f"\n  {'':20s} {'Poisson':>8s}  {'MonteCarlo':>11s}")
    print(f"  {team_a+' win':<20s} {p_win*100:>7.1f}%  {mc_w*100:>10.1f}%")
    print(f"  {'Draw':<20s} {p_draw*100:>7.1f}%  {mc_d*100:>10.1f}%")
    print(f"  {team_b+' win':<20s} {p_lose*100:>7.1f}%  {mc_l*100:>10.1f}%")
    print(f"\n  ★ Score prédit  : {best[0]}-{best[1]} ({team_a}-{team_b})")
    print(f"  ★ Vainqueur     : {winner} ({conf*100:.0f}%)")
    print(f"  ★ Si pénalties  : {team_a} {pen_a}% / {team_b} {pen_b}%")
    print(f"\n  Top 6 scorelines:")
    for (i, j), p in sorted_scores[:6]:
        res = f"{team_a} W" if i > j else ("Draw" if i == j else f"{team_b} W")
        bar = "█" * int(p * 300)
        print(f"    {i}-{j}  {p*100:>5.2f}%  {res}  {bar}")

    all_predictions.append({
        "match":   f"{team_a} vs {team_b}",
        "date":    date,
        "team_a":  team_a,
        "team_b":  team_b,
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
            "draw":          round(mc_d  * 100, 1),
            f"{team_b}_win": round(mc_l  * 100, 1),
        },
        "predicted_score": f"{best[0]}-{best[1]}",
        "predicted_winner": winner,
        "winner_confidence_pct": round(conf * 100, 1),
        "penalties": {
            team_a: pen_a,
            team_b: pen_b,
        },
        "top_scorelines": [
            {
                "score":    f"{i}-{j}",
                "prob_pct": round(p * 100, 2),
                "result":   f"{team_a} W" if i > j else ("Draw" if i == j else f"{team_b} W"),
            }
            for (i, j), p in sorted_scores[:6]
        ],
    })

with open(DATA_DIR / "qf_predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=2)

print(f"\n\n{'='*60}")
print(f"  ✓ qf_predictions.json exporté")
print(f"{'='*60}")
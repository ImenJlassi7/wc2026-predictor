"""
Football Score Predictor — Poisson Model
Input  : data/team_form.json  (généré par football_data_collector.py)
Output : data/prediction.json + affichage complet
"""

import json
import math
import itertools
from pathlib import Path

DATA_DIR = Path("data")

# ─── 1. CHARGER LES DONNÉES ───────────────────────────────────────────────────
with open(DATA_DIR / "team_form.json") as f:
    form = json.load(f)

print("=" * 55)
print("  Poisson Score Predictor — Croatia vs Ghana")
print("=" * 55)
print("\n[1] Données chargées depuis team_form.json")
for team, stats in form.items():
    print(f"\n  {team}:")
    for k, v in stats.items():
        print(f"    {k:30s} : {v}")

# ─── 2. CALCULER LES xG (Expected Goals) ─────────────────────────────────────
# Formule :
#   xG_A = avg_scored_A  * (1 - defense_factor_B)  * home_factor
#   xG_B = avg_scored_B  * (1 - defense_factor_A)
#
# defense_factor = avg_conceded / league_avg  (normalisé)
# league_avg WC historique ~ 1.35 buts/match par équipe

LEAGUE_AVG    = 1.35   # moyenne historique buts/match WC
HOME_FACTOR   = 1.0    # match neutre (terrain tiers, Philadelphie)

cro = form["Croatia"]
gha = form["Ghana"]

# Attack strength (normalisé sur la moyenne WC)
atk_cro = cro["overall_avg_scored"]   / LEAGUE_AVG
atk_gha = gha["overall_avg_scored"]   / LEAGUE_AVG

# Defense weakness (+ élevé = défense pire)
def_cro = cro["overall_avg_conceded"] / LEAGUE_AVG
def_gha = gha["overall_avg_conceded"] / LEAGUE_AVG

# xG = attaque de A × faiblesse défensive de B × moyenne ligue × facteur terrain
xg_cro = atk_cro * def_gha * LEAGUE_AVG * HOME_FACTOR
xg_gha = atk_gha * def_cro * LEAGUE_AVG

# Ajustement forme récente (pondération 30%)
FORM_WEIGHT = 0.30
xg_cro = xg_cro * (1 - FORM_WEIGHT) + cro["avg_scored_last5"] * FORM_WEIGHT
xg_gha = xg_gha * (1 - FORM_WEIGHT) + gha["avg_scored_last5"] * FORM_WEIGHT

print(f"\n[2] Expected Goals (xG) calculés")
print(f"    Croatia xG : {xg_cro:.3f}")
print(f"    Ghana   xG : {xg_gha:.3f}")

# ─── 3. DISTRIBUTION DE POISSON ───────────────────────────────────────────────
# P(k buts) = e^(-λ) × λ^k / k!

def poisson_prob(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

MAX_GOALS = 6  # on calcule jusqu'à 5 buts par équipe

# Matrice de probabilités P(Croatia=i, Ghana=j)
matrix = {}
for i in range(MAX_GOALS):
    for j in range(MAX_GOALS):
        matrix[(i, j)] = poisson_prob(xg_cro, i) * poisson_prob(xg_gha, j)

# ─── 4. PROBABILITÉS ISSUES ───────────────────────────────────────────────────
p_croatia_win = sum(p for (i, j), p in matrix.items() if i > j)
p_draw        = sum(p for (i, j), p in matrix.items() if i == j)
p_ghana_win   = sum(p for (i, j), p in matrix.items() if i < j)

print(f"\n[3] Probabilités de résultat")
print(f"    Croatia win : {p_croatia_win*100:.1f}%")
print(f"    Draw        : {p_draw*100:.1f}%")
print(f"    Ghana win   : {p_ghana_win*100:.1f}%")

# ─── 5. TOP SCORELINES ────────────────────────────────────────────────────────
sorted_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)

print(f"\n[4] Top 10 scorelines les plus probables")
print(f"    {'Score':10s}  {'Prob':>8s}  {'Résultat'}")
print(f"    {'-'*40}")
for (i, j), p in sorted_scores[:10]:
    result = "Croatia W" if i > j else ("Draw" if i == j else "Ghana W")
    bar    = "█" * int(p * 200)
    print(f"    {i}-{j}{'(CRO-GHA)':>10s}  {p*100:>6.2f}%  {result:12s} {bar}")

# Score le plus probable
best_score = sorted_scores[0][0]
print(f"\n    ★ Score prédit : {best_score[0]}-{best_score[1]} (CRO-GHA)")

# ─── 6. HEATMAP ASCII ─────────────────────────────────────────────────────────
print(f"\n[5] Heatmap probabilités (Croatia en ligne, Ghana en colonne)")
print(f"    CRO\\GHA ", end="")
for j in range(5):
    print(f"  GHA={j}  ", end="")
print()
print(f"    {'─'*55}")
for i in range(5):
    print(f"    CRO={i}  ", end="")
    for j in range(5):
        p = matrix[(i, j)] * 100
        # Highlight les cellules importantes
        marker = "►" if (i, j) == best_score else " "
        print(f" {marker}{p:5.2f}% ", end="")
    print()

# ─── 7. MONTE CARLO VALIDATION ────────────────────────────────────────────────
import random
random.seed(42)

def poisson_sample(lam: float) -> int:
    """Tire un nombre de buts selon distribution Poisson."""
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        p *= random.random()
        k += 1
    return k - 1

N_SIM = 100_000
mc_cro_wins, mc_draws, mc_gha_wins = 0, 0, 0
mc_scores = {}

for _ in range(N_SIM):
    g_cro = poisson_sample(xg_cro)
    g_gha = poisson_sample(xg_gha)
    score = (min(g_cro, 5), min(g_gha, 5))

    mc_scores[score] = mc_scores.get(score, 0) + 1

    if g_cro > g_gha:   mc_cro_wins += 1
    elif g_cro == g_gha: mc_draws   += 1
    else:                mc_gha_wins += 1

print(f"\n[6] Validation Monte Carlo ({N_SIM:,} simulations)")
print(f"    Croatia win : {mc_cro_wins/N_SIM*100:.1f}%  "
      f"(Poisson théorique: {p_croatia_win*100:.1f}%)")
print(f"    Draw        : {mc_draws/N_SIM*100:.1f}%  "
      f"(Poisson théorique: {p_draw*100:.1f}%)")
print(f"    Ghana win   : {mc_gha_wins/N_SIM*100:.1f}%  "
      f"(Poisson théorique: {p_ghana_win*100:.1f}%)")

top_mc = sorted(mc_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\n    Top 5 scores (Monte Carlo):")
for (i, j), cnt in top_mc:
    print(f"      {i}-{j}  →  {cnt/N_SIM*100:.2f}%")

# ─── 8. EXPORT JSON ───────────────────────────────────────────────────────────
prediction = {
    "match":        "Croatia vs Ghana",
    "competition":  "FIFA World Cup 2026 — Group L",
    "xg": {
        "Croatia": round(xg_cro, 3),
        "Ghana":   round(xg_gha, 3),
    },
    "probabilities": {
        "croatia_win": round(p_croatia_win * 100, 1),
        "draw":        round(p_draw * 100, 1),
        "ghana_win":   round(p_ghana_win * 100, 1),
    },
    "predicted_score": {
        "Croatia": best_score[0],
        "Ghana":   best_score[1],
    },
    "top_scorelines": [
        {
            "score":   f"{i}-{j}",
            "prob_pct": round(p * 100, 2),
            "result":  "Croatia W" if i > j else ("Draw" if i == j else "Ghana W"),
        }
        for (i, j), p in sorted_scores[:6]
    ],
    "monte_carlo_n": N_SIM,
    "monte_carlo_probabilities": {
        "croatia_win": round(mc_cro_wins / N_SIM * 100, 1),
        "draw":        round(mc_draws    / N_SIM * 100, 1),
        "ghana_win":   round(mc_gha_wins / N_SIM * 100, 1),
    },
}

out = DATA_DIR / "prediction.json"
with open(out, "w") as f:
    json.dump(prediction, f, indent=2)

print(f"\n[7] Résultats exportés → {out}")
print("\n" + "=" * 55)
print(f"  PREDICTION FINALE : Croatia {best_score[0]}-{best_score[1]} Ghana")
print(f"  Croatia win {p_croatia_win*100:.0f}% | "
      f"Draw {p_draw*100:.0f}% | "
      f"Ghana win {p_ghana_win*100:.0f}%")
print("=" * 55)
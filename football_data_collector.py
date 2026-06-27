"""
Football Match Data Collector
Source : OpenFootball (GitHub) — ZERO inscription, ZERO clé API
Données WC 2010, 2014, 2018, 2022 + 2026 (en cours)
"""

import re
import requests
import pandas as pd
import json
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

TEAMS_FILTER = ["Croatia", "Ghana"]

# Fichiers OpenFootball disponibles (raw GitHub, accès libre)
WC_URLS = {
    2010: "https://raw.githubusercontent.com/openfootball/worldcup/master/2010--south-africa/cup.txt",
    2014: "https://raw.githubusercontent.com/openfootball/worldcup/master/2014--brazil/cup.txt",
    2018: "https://raw.githubusercontent.com/openfootball/worldcup/master/2018--russia/cup.txt",
    2022: "https://raw.githubusercontent.com/openfootball/worldcup/master/2022--qatar/cup.txt",
    2026: "https://raw.githubusercontent.com/openfootball/worldcup/master/2026--usa/cup.txt",
}

# ─── PARSER ───────────────────────────────────────────────────────────────────
# Format des lignes de match :
#   19:00   Croatia  4-1 (2-1)  Canada   @ Stadium...
#   16:00   Ghana    0-2        Uruguay  @ Stadium...
MATCH_RE = re.compile(
    r"[\d:]+\s+"                        # heure
    r"(?:UTC[+-]\d+\s+)?"               # timezone optionnelle
    r"([A-Za-z\s\-'\.]+?)"             # équipe domicile
    r"\s+(\d+)-(\d+)"                  # score
    r"(?:\s*\(\d+-\d+\))?"             # mi-temps optionnel
    r"\s+([A-Za-z\s\-'\.]+?)"          # équipe extérieur
    r"\s*(?:@|$)"                       # @ stade ou fin
)

def parse_wc_file(url: str, season: int) -> list[dict]:
    """Télécharge et parse un fichier WC OpenFootball."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  ⚠ Impossible de télécharger {url}: {e}")
        return []

    rows = []
    current_date = None

    for line in r.text.splitlines():
        line = line.strip()

        # Détecter les lignes de date  ex: "▪ Matchday 1 | Sun Nov 20"
        date_match = re.search(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)", line
        )
        if date_match:
            month_str = date_match.group(1)
            day       = date_match.group(2)
            # Déterminer l'année (WC se joue Nov-Dec sauf 2026 qui est Juin)
            year = season if season < 2026 else season
            try:
                current_date = pd.to_datetime(
                    f"{year} {month_str} {day}", format="%Y %b %d"
                ).strftime("%Y-%m-%d")
            except Exception:
                pass
            continue

        # Détecter les lignes de match
        m = MATCH_RE.search(line)
        if m and current_date:
            home     = m.group(1).strip()
            score_h  = int(m.group(2))
            score_a  = int(m.group(3))
            away     = m.group(4).strip()

            rows.append({
                "date":       current_date,
                "season":     season,
                "competition":"World Cup",
                "home_team":  home,
                "away_team":  away,
                "home_goals": score_h,
                "away_goals": score_a,
            })

    return rows


def build_team_df(rows: list, team: str) -> pd.DataFrame:
    """Filtre les matchs d'une équipe et oriente le dataframe."""
    records = []
    for r in rows:
        home, away = r["home_team"], r["away_team"]
        is_home = team.lower() in home.lower()
        is_away = team.lower() in away.lower()
        if not (is_home or is_away):
            continue

        scored    = r["home_goals"] if is_home else r["away_goals"]
        conceded  = r["away_goals"] if is_home else r["home_goals"]
        opponent  = away if is_home else home
        result    = "W" if scored > conceded else ("D" if scored == conceded else "L")

        records.append({
            "date":           r["date"],
            "season":         r["season"],
            "competition":    r["competition"],
            "team":           team,
            "opponent":       opponent,
            "is_home":        int(is_home),
            "goals_scored":   scored,
            "goals_conceded": conceded,
            "result":         result,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def add_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    pts_map = {"W": 3, "D": 1, "L": 0}
    df = df.copy()
    df["points"] = df["result"].map(pts_map)

    for col, src in [("avg_scored_last5",  "goals_scored"),
                     ("avg_conceded_last5", "goals_conceded"),
                     ("avg_pts_last5",      "points")]:
        df[col] = (df[src].shift(1)
                          .rolling(window, min_periods=1)
                          .mean().round(2))

    df["win_rate_last5"] = (
        df["result"].shift(1)
        .apply(lambda x: 1 if x == "W" else 0)
        .rolling(window, min_periods=1).mean().round(2)
    )
    return df


# ─── PIPELINE ─────────────────────────────────────────────────────────────────
def collect_data():
    print("=" * 55)
    print("  Football Data Collector — OpenFootball (no API key)")
    print("=" * 55)

    all_rows = []

    for season, url in WC_URLS.items():
        print(f"\n  Téléchargement WC {season}...")
        rows = parse_wc_file(url, season)
        print(f"  → {len(rows)} matchs parsés")
        all_rows.extend(rows)

    print(f"\n  Total matchs bruts : {len(all_rows)}")

    # Sauvegarde brute complète
    pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / "wc_all_matches.csv", index=False)

    datasets = {}
    for team in TEAMS_FILTER:
        df = build_team_df(all_rows, team)
        if df.empty:
            print(f"\n  ⚠ Aucun match trouvé pour {team}")
            continue

        df = add_form_features(df)
        datasets[team] = df

        path = OUTPUT_DIR / f"{team.lower()}_matches.csv"
        df.to_csv(path, index=False)

        print(f"\n{'─'*50}")
        print(f"  {team} — {len(df)} matchs World Cup")
        print(f"{'─'*50}")
        print(df[["date", "season", "opponent", "goals_scored",
                   "goals_conceded", "result",
                   "avg_scored_last5", "avg_conceded_last5",
                   "avg_pts_last5"]].to_string())

    # Summary JSON → input modèle Poisson
    summary = {}
    for team, df in datasets.items():
        last = df.iloc[-1]
        all_matches = len(df)
        summary[team] = {
            "avg_scored_last5":        round(float(last["avg_scored_last5"]), 2),
            "avg_conceded_last5":      round(float(last["avg_conceded_last5"]), 2),
            "avg_pts_last5":           round(float(last["avg_pts_last5"]), 2),
            "win_rate_last5":          round(float(last["win_rate_last5"]), 2),
            "overall_avg_scored":      round(df["goals_scored"].mean(), 2),
            "overall_avg_conceded":    round(df["goals_conceded"].mean(), 2),
            "total_wc_matches":        all_matches,
        }

    with open(OUTPUT_DIR / "team_form.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*55)
    print("  team_form.json (input prêt pour modèle Poisson):")
    print("="*55)
    print(json.dumps(summary, indent=2))
    print("\n✓ Done — fichiers dans data/")
    return datasets


if __name__ == "__main__":
    collect_data()
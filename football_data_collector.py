"""
Football Match Data Collector
Source : OpenFootball (GitHub) — ZERO inscription, ZERO clé API
"""

import re
import requests
import pandas as pd
import json
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

TEAMS_FILTER = ["Croatia", "Ghana", "USA", "Belgium", "Argentina", "Egypt", "Switzerland", "Colombia", "France", "Morocco", "Spain", "Norway", "England"]

WC_URLS = {
    2010: "https://raw.githubusercontent.com/openfootball/worldcup/master/2010--south-africa/cup.txt",
    2014: "https://raw.githubusercontent.com/openfootball/worldcup/master/2014--brazil/cup.txt",
    2018: "https://raw.githubusercontent.com/openfootball/worldcup/master/2018--russia/cup.txt",
    2022: "https://raw.githubusercontent.com/openfootball/worldcup/master/2022--qatar/cup.txt",
    2026: "https://raw.githubusercontent.com/openfootball/worldcup/master/2026--usa/cup.txt",
}

MATCH_RE = re.compile(
    r"[\d:]+\s+"
    r"(?:UTC[+-]\d+\s+)?"
    r"([A-Za-z\s\-'\.]+?)"
    r"\s+(\d+)-(\d+)"
    r"(?:\s*\(\d+-\d+\))?"
    r"\s+([A-Za-z\s\-'\.]+?)"
    r"\s*(?:@|$)"
)

def parse_wc_file(url, season):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  ⚠ {url}: {e}")
        return []

    rows = []
    current_date = None
    for line in r.text.splitlines():
        line = line.strip()
        date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)", line)
        if date_match:
            try:
                current_date = pd.to_datetime(f"{season} {date_match.group(1)} {date_match.group(2)}", format="%Y %b %d").strftime("%Y-%m-%d")
            except: pass
            continue
        m = MATCH_RE.search(line)
        if m and current_date:
            rows.append({"date": current_date, "season": season, "competition": "World Cup",
                         "home_team": m.group(1).strip(), "away_team": m.group(4).strip(),
                         "home_goals": int(m.group(2)), "away_goals": int(m.group(3))})
    return rows

def build_team_df(rows, team):
    records = []
    for r in rows:
        is_home = team.lower() in r["home_team"].lower()
        is_away = team.lower() in r["away_team"].lower()
        if not (is_home or is_away): continue
        scored = r["home_goals"] if is_home else r["away_goals"]
        conceded = r["away_goals"] if is_home else r["home_goals"]
        result = "W" if scored > conceded else ("D" if scored == conceded else "L")
        records.append({"date": r["date"], "season": r["season"], "competition": r["competition"],
                        "team": team, "opponent": r["away_team"] if is_home else r["home_team"],
                        "is_home": int(is_home), "goals_scored": scored, "goals_conceded": conceded, "result": result})
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def add_form_features(df, window=5):
    pts_map = {"W": 3, "D": 1, "L": 0}
    df = df.copy()
    df["points"] = df["result"].map(pts_map)
    for col, src in [("avg_scored_last5","goals_scored"),("avg_conceded_last5","goals_conceded"),("avg_pts_last5","points")]:
        df[col] = df[src].shift(1).rolling(window, min_periods=1).mean().round(2)
    df["win_rate_last5"] = df["result"].shift(1).apply(lambda x: 1 if x=="W" else 0).rolling(window, min_periods=1).mean().round(2)
    return df

def collect_data():
    all_rows = []
    for season, url in WC_URLS.items():
        print(f"  Fetching WC {season}...")
        rows = parse_wc_file(url, season)
        print(f"  → {len(rows)} matches")
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / "wc_all_matches.csv", index=False)

    datasets = {}
    for team in TEAMS_FILTER:
        df = build_team_df(all_rows, team)
        if df.empty: continue
        df = add_form_features(df)
        datasets[team] = df
        df.to_csv(OUTPUT_DIR / f"{team.lower()}_matches.csv", index=False)

    summary = {}
    for team, df in datasets.items():
        last = df.iloc[-1]
        summary[team] = {
            "avg_scored_last5": round(float(last["avg_scored_last5"]), 2),
            "avg_conceded_last5": round(float(last["avg_conceded_last5"]), 2),
            "avg_pts_last5": round(float(last["avg_pts_last5"]), 2),
            "win_rate_last5": round(float(last["win_rate_last5"]), 2),
            "overall_avg_scored": round(df["goals_scored"].mean(), 2),
            "overall_avg_conceded": round(df["goals_conceded"].mean(), 2),
            "total_wc_matches": len(df),
        }

    with open(OUTPUT_DIR / "team_form.json", "w") as f:
        json.dump(summary, f, indent=2)

    return datasets, all_rows

if __name__ == "__main__":
    collect_data()
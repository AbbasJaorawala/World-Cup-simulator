# pipeline/transform.py — Clean & enrich raw API data

import logging
import sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_FORMAT, LOG_LEVEL, ELO_DEFAULT_RATING

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Cleans, normalises, and enriches raw data fetched from APIs.

    Responsibilities:
    - Standardise team names across different APIs
    - Compute derived stats (form, head-to-head, goal averages)
    - Validate and fill missing values
    - Convert API-specific formats into our internal schema
    """

    # ── Team name aliases: maps API variants → our canonical name ─────────────
    TEAM_NAME_MAP = {
        # football-data.org names
        "Germany": "Germany",
        "Brazil": "Brazil",
        "France": "France",
        "England": "England",
        "Spain": "Spain",
        "Argentina": "Argentina",
        "Portugal": "Portugal",
        "Netherlands": "Netherlands",
        "Korea Republic": "South Korea",
        "Republic of Korea": "South Korea",
        "IR Iran": "Iran",
        "USA": "USA",
        "United States": "USA",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Czech Republic": "Czech Republic",
        "Czechia": "Czech Republic",
        # Club ELO names
        "EngPL": "England",
        "GerBL": "Germany",
        # Open Football names (already clean mostly)
    }

    def __init__(self):
        logger.info("DataTransformer initialized")

    # ── Name Normalisation ────────────────────────────────────────────────────

    def normalise_team_name(self, name: str) -> str:
        """Map any API variant of a team name to our canonical version."""
        return self.TEAM_NAME_MAP.get(name, name)

    def normalise_matches(self, matches: list) -> list:
        """Normalise team names in a list of match dicts."""
        cleaned = []
        for m in matches:
            m = m.copy()
            m["team1"] = self.normalise_team_name(m.get("team1", ""))
            m["team2"] = self.normalise_team_name(m.get("team2", ""))
            cleaned.append(m)
        return cleaned

    # ── Match Data Cleaning ───────────────────────────────────────────────────

    def clean_matches(self, matches: list) -> list:
        """
        Clean raw match data:
        - Remove matches with missing scores
        - Coerce scores to int
        - Add result field (win/draw/loss from team1's perspective)
        - Normalise team names
        """
        cleaned = []
        skipped = 0
        for m in matches:
            team1 = self.normalise_team_name(m.get("team1", ""))
            team2 = self.normalise_team_name(m.get("team2", ""))
            score1 = m.get("score1")
            score2 = m.get("score2")

            # Skip incomplete records
            if not team1 or not team2:
                skipped += 1
                continue
            if score1 is None or score2 is None:
                skipped += 1
                continue

            try:
                s1, s2 = int(score1), int(score2)
            except (ValueError, TypeError):
                skipped += 1
                continue

            result = "win" if s1 > s2 else ("draw" if s1 == s2 else "loss")

            cleaned.append({
                "team1": team1,
                "team2": team2,
                "score1": s1,
                "score2": s2,
                "result": result,          # from team1 perspective
                "goal_diff": s1 - s2,
                "total_goals": s1 + s2,
                "year": m.get("year"),
                "round": m.get("round", ""),
                "date": m.get("date", ""),
            })

        logger.info(f"Cleaned matches: {len(cleaned)} kept, {skipped} skipped")
        return cleaned

    # ── ELO Ratings Cleaning ──────────────────────────────────────────────────

    def clean_elo_ratings(self, raw_ratings: dict) -> dict:
        """
        Normalise team names in ELO ratings dict and
        fill any missing teams with the default rating.
        """
        cleaned = {}
        for team, elo in raw_ratings.items():
            canon = self.normalise_team_name(team)
            try:
                cleaned[canon] = float(elo) if elo else ELO_DEFAULT_RATING
            except (ValueError, TypeError):
                cleaned[canon] = ELO_DEFAULT_RATING
        logger.info(f"Cleaned ELO ratings for {len(cleaned)} teams")
        return cleaned

    # ── Derived Stats ─────────────────────────────────────────────────────────

    def compute_team_form(self, matches: list, last_n: int = 5) -> dict:
        """
        Compute recent form for each team as a win rate over last N matches.

        Returns:
            {team_name: form_score}  where form_score ∈ [0.0, 1.0]
            (1.0 = won all last N games, 0.0 = lost all)
        """
        # Collect matches per team (most recent first)
        team_matches = defaultdict(list)
        for m in matches:
            team_matches[m["team1"]].append(m["result"])
            # Reverse result for team2
            rev = {"win": "loss", "loss": "win", "draw": "draw"}
            team_matches[m["team2"]].append(rev[m["result"]])

        form = {}
        for team, results in team_matches.items():
            recent = results[-last_n:]   # last N
            points = sum(3 if r == "win" else (1 if r == "draw" else 0) for r in recent)
            max_points = len(recent) * 3
            form[team] = round(points / max_points, 4) if max_points > 0 else 0.5

        logger.info(f"Computed form for {len(form)} teams (last {last_n} matches)")
        return form

    def compute_head_to_head(self, matches: list) -> dict:
        """
        Compute head-to-head win rates between all team pairs.

        Returns:
            {(team_a, team_b): win_rate_a}  — fraction of matches team_a won
        """
        h2h = defaultdict(lambda: {"wins": 0, "total": 0})

        for m in matches:
            t1, t2 = m["team1"], m["team2"]
            key_ab = (t1, t2)
            key_ba = (t2, t1)
            h2h[key_ab]["total"] += 1
            h2h[key_ba]["total"] += 1

            if m["result"] == "win":
                h2h[key_ab]["wins"] += 1
            elif m["result"] == "loss":
                h2h[key_ba]["wins"] += 1
            # draw: no wins for either

        result = {}
        for pair, stats in h2h.items():
            result[pair] = round(stats["wins"] / stats["total"], 4) if stats["total"] > 0 else 0.5

        logger.info(f"Computed H2H for {len(result)} team pairs")
        return result

    def compute_goal_averages(self, matches: list) -> dict:
        """
        Compute average goals scored and conceded per match for each team.

        Returns:
            {team: {"avg_scored": float, "avg_conceded": float, "matches": int}}
        """
        stats = defaultdict(lambda: {"gf": 0, "ga": 0, "matches": 0})

        for m in matches:
            t1, t2 = m["team1"], m["team2"]
            s1, s2 = m["score1"], m["score2"]

            stats[t1]["gf"] += s1
            stats[t1]["ga"] += s2
            stats[t1]["matches"] += 1

            stats[t2]["gf"] += s2
            stats[t2]["ga"] += s1
            stats[t2]["matches"] += 1

        averages = {}
        for team, s in stats.items():
            n = s["matches"]
            averages[team] = {
                "avg_scored": round(s["gf"] / n, 3) if n > 0 else 1.2,
                "avg_conceded": round(s["ga"] / n, 3) if n > 0 else 1.0,
                "matches": n,
            }

        logger.info(f"Computed goal averages for {len(averages)} teams")
        return averages

    # ── Combined Transform ────────────────────────────────────────────────────

    def transform_all(self, raw: dict) -> dict:
        """
        Full transform pipeline: clean + enrich everything in one call.

        Args:
            raw: output of DataFetcher.fetch_all()

        Returns:
            Enriched data dict ready for the loader
        """
        logger.info("=== Starting data transformation ===")

        cleaned_matches = self.clean_matches(raw.get("historical_matches", []))
        cleaned_elos = self.clean_elo_ratings(raw.get("elo_ratings", {}))

        transformed = {
            "elo_ratings": cleaned_elos,
            "matches": cleaned_matches,
            "team_form": self.compute_team_form(cleaned_matches),
            "head_to_head": self.compute_head_to_head(cleaned_matches),
            "goal_averages": self.compute_goal_averages(cleaned_matches),
            "wc_teams": raw.get("wc_teams", []),
        }

        logger.info(f"=== Transformation complete: "
                    f"{len(cleaned_matches)} matches, "
                    f"{len(cleaned_elos)} ELO ratings ===")
        return transformed


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transformer = DataTransformer()

    # Minimal test with fake data
    sample_matches = [
        {"team1": "Brazil",    "team2": "Germany",  "score1": 1, "score2": 7, "year": 2014},
        {"team1": "Germany",   "team2": "Argentina","score1": 1, "score2": 0, "year": 2014},
        {"team1": "Brazil",    "team2": "Argentina","score1": 2, "score2": 1, "year": 2018},
        {"team1": "France",    "team2": "Croatia",  "score1": 4, "score2": 2, "year": 2018},
        {"team1": "Argentina", "team2": "France",   "score1": 3, "score2": 3, "year": 2022},
    ]

    cleaned = transformer.clean_matches(sample_matches)
    print(f"✅ Cleaned {len(cleaned)} matches")

    form = transformer.compute_team_form(cleaned)
    print("\nTeam Form:")
    for team, f in sorted(form.items(), key=lambda x: -x[1]):
        print(f"  {team:12s}: {f:.2f}")

    h2h = transformer.compute_head_to_head(cleaned)
    print(f"\nH2H pairs computed: {len(h2h)}")
    print(f"  Brazil vs Germany: {h2h.get(('Brazil','Germany'), 'N/A')}")

    avgs = transformer.compute_goal_averages(cleaned)
    print("\nGoal Averages:")
    for team, s in avgs.items():
        print(f"  {team:12s}: scored={s['avg_scored']:.2f}, conceded={s['avg_conceded']:.2f}")
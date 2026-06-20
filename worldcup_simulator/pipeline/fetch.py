# pipeline/fetch.py — Pull data from all free football APIs

import logging
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WC_2026_TEAMS, LOG_FORMAT, LOG_LEVEL
from api.football_api import FootballAPIFactory

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches raw data from all configured APIs and returns
    it as plain Python dicts/lists ready for transformation.
    """

    def __init__(self):
        self.api = FootballAPIFactory()
        logger.info("DataFetcher initialized")

    # ── ELO Ratings ───────────────────────────────────────────────────────────

    def fetch_elo_ratings(self, teams: list = None) -> dict:
        """
        Fetch ELO ratings for all WC teams from clubelo.com.
        Falls back to default ratings if API is unavailable.

        Returns:
            {team_name: elo_float}
        """
        teams = teams or WC_2026_TEAMS
        logger.info(f"Fetching ELO ratings for {len(teams)} teams...")
        ratings = self.api.get_team_elos(teams)
        fetched = sum(1 for v in ratings.values() if v != 1500.0)
        logger.info(f"ELO fetch complete: {fetched}/{len(teams)} live, rest defaulted to 1500")
        return ratings

    # ── Historical Match Data ─────────────────────────────────────────────────

    def fetch_historical_matches(self, years: list = None) -> list:
        """
        Fetch historical WC match results from Open Football (GitHub JSON).
        No API key required.

        Returns:
            List of match dicts: {round, date, team1, team2, score1, score2}
        """
        years = years or [2022, 2018, 2014]
        all_matches = []
        for year in years:
            logger.info(f"Fetching {year} WC match data...")
            matches = self.api.get_historical_matches(year)
            for m in matches:
                m["year"] = year
            all_matches.extend(matches)
            logger.info(f"  → {len(matches)} matches from {year}")
        logger.info(f"Total historical matches fetched: {len(all_matches)}")
        return all_matches

    # ── Competition Data ──────────────────────────────────────────────────────

    def fetch_wc_teams(self) -> list:
        """
        Fetch official WC team list from football-data.org.
        Requires FOOTBALL_DATA_API_KEY in config.

        Returns:
            List of team dicts from the API, or [] if key not set.
        """
        logger.info("Fetching WC team list from football-data.org...")
        teams = self.api.get_wc_teams()
        if teams:
            logger.info(f"Fetched {len(teams)} official teams")
        else:
            logger.warning("No teams fetched — check FOOTBALL_DATA_API_KEY in config.py")
        return teams

    def fetch_wc_matches(self, season: int = 2022) -> list:
        """
        Fetch WC match schedule/results from football-data.org.

        Returns:
            List of match dicts
        """
        logger.info(f"Fetching WC matches for season {season}...")
        matches = self.api.football_data.get_matches("WC", season=season)
        logger.info(f"Fetched {len(matches)} matches")
        return matches

    def fetch_wc_standings(self) -> list:
        """
        Fetch current WC group standings from football-data.org.

        Returns:
            List of group standing dicts
        """
        logger.info("Fetching WC standings...")
        standings = self.api.football_data.get_standings("WC")
        logger.info(f"Fetched {len(standings)} group standings")
        return standings

    # ── Team Form ─────────────────────────────────────────────────────────────

    def fetch_team_recent_form(self, team_id: int, limit: int = 5) -> list:
        """
        Fetch last N matches for a team to compute recent form.
        Uses football-data.org.

        Returns:
            List of recent match dicts
        """
        logger.info(f"Fetching last {limit} matches for team {team_id}...")
        matches = self.api.football_data.get_team_matches(team_id, limit=limit)
        logger.info(f"Fetched {len(matches)} recent matches")
        return matches

    # ── Convenience: Fetch Everything ────────────────────────────────────────

    def fetch_all(self, teams: list = None) -> dict:
        """
        Fetch all data needed for simulation in one call.

        Returns:
            {
                "elo_ratings": {...},
                "historical_matches": [...],
                "wc_teams": [...],
            }
        """
        logger.info("=== Starting full data fetch ===")
        payload = {
            "elo_ratings": self.fetch_elo_ratings(teams),
            "historical_matches": self.fetch_historical_matches(),
            "wc_teams": self.fetch_wc_teams(),
        }
        logger.info("=== Full data fetch complete ===")
        return payload


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fetcher = DataFetcher()

    print("Testing Open Football (no key needed)...")
    matches = fetcher.fetch_historical_matches(years=[2022])
    print(f"  ✅ {len(matches)} matches from 2022 WC")
    if matches:
        print(f"  Sample: {matches[0]}")

    print("\nTesting Club ELO (no key needed)...")
    elos = fetcher.fetch_elo_ratings(["Brazil", "France", "Germany"])
    for team, elo in elos.items():
        print(f"  {team}: {elo}")
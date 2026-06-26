# pipeline/fetch.py — Pull data from all free football APIs

import logging
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_FORMAT, LOG_LEVEL
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

    def fetch_elo_ratings(self, teams: list = None, max_cache_age_hours: float = 24) -> dict:
        """
        Fetch starting ELO-proxy ratings for WC teams.

        Pulls current national-team Elo ratings from eloratings.net and caches
        them in SQLite. Missing teams fall back to 1500 individually.

        Returns:
            {team_name: elo_float}
        """
        from pipeline.loader import DataLoader

        loader = DataLoader()
        cached = loader.load_teams_if_fresh(max_age_hours=max_cache_age_hours)
        if cached:
            requested = set(teams or cached.keys())
            cached_subset = {team: cached[team] for team in requested if team in cached}
            all_requested_cached = len(cached_subset) == len(requested)
            all_defaults = cached_subset and all(float(value) == 1500.0 for value in cached_subset.values())
            if all_requested_cached and not all_defaults:
                logger.info(f"Using cached ratings for {len(cached_subset)} teams (skipping API call)")
                return cached_subset
            logger.info("Cached ratings are missing teams or only defaults - refreshing World Elo ratings")

        logger.info("No fresh cache - fetching World Elo ratings...")
        ratings = self.api.get_team_elos(teams or self.fetch_team_names())

        if not ratings:
            logger.warning(
                "No ratings available - "
                "falling back to any teams list provided with default 1500"
            )
            fallback_teams = teams or self.fetch_team_names()
            return {team: 1500.0 for team in fallback_teams}

        loader.save_teams(ratings)
        logger.info(f"Fetched and cached ratings for {len(ratings)} teams")
        return ratings

    def fetch_team_names(self) -> list:
        """
        Fetch the live list of qualified team names from football-data.org.
        Returns [] if the API has no data yet (e.g. before official squads
        are confirmed) — caller should handle empty list gracefully.
        """
        raw_teams = self.fetch_wc_teams()
        names = [t.get("name") for t in raw_teams if t.get("name")]
        if not names:
            logger.warning("No team names returned by API — draw may not be published yet")
        return names

    def fetch_wc_groups(self) -> dict:
        """
        Fetch the official 2026 group stage draw live from football-data.org.

        Returns:
            {group_name: [team1, team2, team3, team4]}
            Empty dict if standings/draw aren't published yet.
        """
        logger.info("Fetching official WC group draw...")
        groups = self.api.get_wc_groups()
        if groups:
            logger.info(f"Fetched {len(groups)} groups from API")
        else:
            logger.warning("No group draw available from API yet")
        return groups

    def fetch_team_squads(self) -> dict:
        """
        Fetch real player squads for all World Cup teams from football-data.org.

        Returns:
            {team_name: [{"name": player_name, "position": position}, ...], ...}
        """
        logger.info("Fetching team squads from football-data.org...")
        try:
            squads = self.api.get_team_squads()
            if squads:
                logger.info(f"Fetched squads for {len(squads)} teams")
                return squads
            else:
                logger.warning("No squads available from API")
                return {}
        except Exception as e:
            logger.warning(f"Failed to fetch squads: {e}")
            return {}

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

    def fetch_sportapi_scheduled_events(self, date: str = None) -> list:
        """
        Fetch football scheduled events from SportAPI via RapidAPI.

        Args:
            date: YYYY-MM-DD. Defaults to SPORTAPI_DEFAULT_DATE from config.

        Returns:
            List of normalised fixture dicts.
        """
        logger.info(f"Fetching SportAPI scheduled events for {date or 'default date'}...")
        fixtures = self.api.sport_api.get_fixtures(date=date)
        logger.info(f"Fetched {len(fixtures)} SportAPI fixtures")
        return fixtures

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

    def fetch_all(self) -> dict:
        """
        Fetch all data needed for simulation in one call.
        Team names and groups are pulled live from the API — nothing hardcoded.

        Returns:
            {
                "team_names": [...],
                "groups": {...},
                "elo_ratings": {...},
                "historical_matches": [...],
                "wc_teams": [...],
            }
        """
        logger.info("=== Starting full data fetch ===")
        team_names = self.fetch_team_names()
        payload = {
            "team_names": team_names,
            "groups": self.fetch_wc_groups(),
            "elo_ratings": self.fetch_elo_ratings(team_names or None),
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

# api/football_api.py — Unified client for all free football APIs

import requests
import logging
import time
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL,
    RAPIDAPI_KEY, RAPIDAPI_BASE_URL,
    CLUB_ELO_BASE_URL, OPEN_FOOTBALL_BASE_URL,
    LOG_FORMAT, LOG_LEVEL
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ─── Base Request Helper ───────────────────────────────────────────────────────

def _get(url: str, headers: dict = None, params: dict = None, retries: int = 3) -> Optional[dict]:
    """Generic GET with retry logic and rate-limit awareness."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None


# ─── football-data.org Client ──────────────────────────────────────────────────

class FootballDataClient:
    """
    Client for football-data.org (free tier).
    Provides: competitions, teams, standings, matches.
    """

    def __init__(self):
        self.base = FOOTBALL_DATA_BASE_URL
        self.headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}

    def get_competition(self, code: str = "WC") -> Optional[dict]:
        """Fetch competition metadata. Default: FIFA World Cup."""
        url = f"{self.base}/competitions/{code}"
        data = _get(url, headers=self.headers)
        if data:
            logger.info(f"Fetched competition: {data.get('name')}")
        return data

    def get_teams(self, competition_code: str = "WC") -> Optional[list]:
        """Fetch all teams in a competition."""
        url = f"{self.base}/competitions/{competition_code}/teams"
        data = _get(url, headers=self.headers)
        if data:
            teams = data.get("teams", [])
            logger.info(f"Fetched {len(teams)} teams for {competition_code}")
            return teams
        return []

    def get_standings(self, competition_code: str = "WC") -> Optional[list]:
        """Fetch group standings for a competition."""
        url = f"{self.base}/competitions/{competition_code}/standings"
        data = _get(url, headers=self.headers)
        if data:
            standings = data.get("standings", [])
            logger.info(f"Fetched standings: {len(standings)} groups")
            return standings
        return []

    def get_matches(self, competition_code: str = "WC", season: int = None) -> Optional[list]:
        """Fetch all matches for a competition, optionally filtered by season."""
        url = f"{self.base}/competitions/{competition_code}/matches"
        params = {}
        if season:
            params["season"] = season
        data = _get(url, headers=self.headers, params=params)
        if data:
            matches = data.get("matches", [])
            logger.info(f"Fetched {len(matches)} matches")
            return matches
        return []

    def get_team_matches(self, team_id: int, limit: int = 10) -> Optional[list]:
        """Fetch recent matches for a specific team."""
        url = f"{self.base}/teams/{team_id}/matches"
        params = {"limit": limit, "status": "FINISHED"}
        data = _get(url, headers=self.headers, params=params)
        if data:
            matches = data.get("matches", [])
            logger.info(f"Fetched {len(matches)} recent matches for team {team_id}")
            return matches
        return []


# ─── API-Football (RapidAPI) Client ────────────────────────────────────────────

class APIFootballClient:
    """
    Client for api-football via RapidAPI (free: 100 req/day).
    Provides: fixtures, player stats, predictions, injuries, odds.
    """

    def __init__(self):
        self.base = RAPIDAPI_BASE_URL
        self.headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }

    def get_fixtures(self, league_id: int = 1, season: int = 2026) -> Optional[list]:
        """Fetch fixtures for a league/season. League 1 = World Cup."""
        url = f"{self.base}/fixtures"
        params = {"league": league_id, "season": season}
        data = _get(url, headers=self.headers, params=params)
        if data:
            fixtures = data.get("response", [])
            logger.info(f"Fetched {len(fixtures)} fixtures")
            return fixtures
        return []

    def get_team_statistics(self, team_id: int, league_id: int = 1, season: int = 2022) -> Optional[dict]:
        """Fetch team statistics for a given season."""
        url = f"{self.base}/teams/statistics"
        params = {"team": team_id, "league": league_id, "season": season}
        data = _get(url, headers=self.headers, params=params)
        if data:
            stats = data.get("response", {})
            logger.info(f"Fetched stats for team {team_id}")
            return stats
        return {}

    def get_standings(self, league_id: int = 1, season: int = 2022) -> Optional[list]:
        """Fetch league standings."""
        url = f"{self.base}/standings"
        params = {"league": league_id, "season": season}
        data = _get(url, headers=self.headers, params=params)
        if data:
            standings = data.get("response", [])
            logger.info(f"Fetched standings for league {league_id}")
            return standings
        return []

    def get_predictions(self, fixture_id: int) -> Optional[dict]:
        """Fetch AI predictions for a specific fixture."""
        url = f"{self.base}/predictions"
        params = {"fixture": fixture_id}
        data = _get(url, headers=self.headers, params=params)
        if data:
            predictions = data.get("response", [{}])
            return predictions[0] if predictions else {}
        return {}


# ─── Club ELO Client ───────────────────────────────────────────────────────────

class ClubELOClient:
    """
    Client for clubelo.com — free ELO ratings for national teams.
    No API key needed.
    """

    def __init__(self):
        self.base = CLUB_ELO_BASE_URL

    def get_team_elo(self, team_name: str) -> Optional[float]:
        """
        Fetch current ELO rating for a team.
        team_name must match Club ELO naming (e.g. 'Brazil', 'France').
        """
        url = f"{self.base}/{team_name}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            # Club ELO returns CSV: Rank,Club,Country,Level,Elo,From,To
            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                logger.warning(f"No ELO data for {team_name}")
                return None
            latest = lines[-1].split(",")
            elo = float(latest[4])
            logger.info(f"ELO for {team_name}: {elo:.1f}")
            return elo
        except Exception as e:
            logger.error(f"Failed to fetch ELO for {team_name}: {e}")
            return None

    def get_all_elos(self, teams: list) -> dict:
        """Fetch ELO ratings for a list of teams. Returns {team_name: elo}."""
        elos = {}
        for team in teams:
            elo = self.get_team_elo(team)
            elos[team] = elo if elo else 1500.0  # Default ELO if not found
            time.sleep(0.5)  # Be polite to the free API
        logger.info(f"Fetched ELO ratings for {len(elos)} teams")
        return elos


# ─── Open Football (Static JSON) Client ────────────────────────────────────────

class OpenFootballClient:
    """
    Client for Open Football GitHub — historical WC data as static JSON.
    Completely free, no key needed.
    """

    def __init__(self):
        self.base = OPEN_FOOTBALL_BASE_URL

    def get_historical_wc(self, year: int = 2022) -> Optional[dict]:
        """Fetch historical World Cup data for a given year."""
        url = f"{self.base}/{year}/worldcup.json"
        data = _get(url)
        if data:
            logger.info(f"Fetched {year} WC data: {len(data.get('rounds', []))} rounds")
        return data

    def get_all_groups(self, year: int = 2022) -> Optional[list]:
        """Extract group stage data from historical WC."""
        data = self.get_historical_wc(year)
        if not data:
            return []
        rounds = data.get("rounds", [])
        group_rounds = [r for r in rounds if "Group" in r.get("name", "")]
        return group_rounds

    def get_all_matches(self, year: int = 2022) -> list:
        """Extract all match results from historical WC."""
        data = self.get_historical_wc(year)
        if not data:
            return []
        matches = []
        for round_ in data.get("rounds", []):
            for match in round_.get("matches", []):
                matches.append({
                    "round": round_.get("name"),
                    "date": match.get("date"),
                    "team1": match.get("team1", {}).get("name"),
                    "team2": match.get("team2", {}).get("name"),
                    "score1": match.get("score1"),
                    "score2": match.get("score2"),
                })
        logger.info(f"Extracted {len(matches)} matches from {year} WC")
        return matches


# ─── Unified API Factory ───────────────────────────────────────────────────────

class FootballAPIFactory:
    """Single entry point to access all API clients."""

    def __init__(self):
        self.football_data = FootballDataClient()
        self.api_football = APIFootballClient()
        self.club_elo = ClubELOClient()
        self.open_football = OpenFootballClient()

    def get_team_elos(self, teams: list) -> dict:
        return self.club_elo.get_all_elos(teams)

    def get_historical_matches(self, year: int = 2022) -> list:
        return self.open_football.get_all_matches(year)

    def get_wc_teams(self) -> list:
        return self.football_data.get_teams("WC")


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Open Football (no API key needed)...")
    client = OpenFootballClient()
    matches = client.get_all_matches(2022)
    if matches:
        print(f"✅ Fetched {len(matches)} matches from 2022 WC")
        print(f"   Sample: {matches[0]}")
    else:
        print("❌ No matches fetched — check network connection")

    print("\nTesting Club ELO (no API key needed)...")
    elo_client = ClubELOClient()
    elo = elo_client.get_team_elo("Brazil")
    if elo:
        print(f"✅ Brazil ELO: {elo}")
    else:
        print("❌ ELO fetch failed")
# api/football_api.py — Unified client for all free football APIs

import requests
import logging
import time
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL,
    RAPIDAPI_KEY, SPORTAPI_BASE_URL, SPORTAPI_RAPIDAPI_HOST,
    SPORTAPI_FOOTBALL_CATEGORY_ID, SPORTAPI_DEFAULT_DATE,
    CLUB_ELO_BASE_URL, OPEN_FOOTBALL_BASE_URL, WORLD_ELO_BASE_URL,
    LOG_FORMAT, LOG_LEVEL
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

ELO_DEFAULT_RATING = 1500
FORM_RATING_SPREAD = 400
GOAL_DIFF_WEIGHT = 15


def _derive_rating_from_stats(stats: dict) -> float:
    """
    Convert an API-Football team_statistics response into a starting
    ELO-like rating. Replaces the old ClubELOClient-based approach —
    Club ELO only covers club football, not national teams, so every
    lookup failed and the whole tournament ran on a flat 1500.

    Degrades gracefully PER TEAM: a missing/incomplete response for one
    team returns ELO_DEFAULT_RATING for that team only.
    """
    try:
        fixtures = stats.get("fixtures", {})
        played = fixtures.get("played", {}).get("total", 0)
        if not played:
            return ELO_DEFAULT_RATING

        wins = fixtures.get("wins", {}).get("total", 0)
        draws = fixtures.get("draws", {}).get("total", 0)

        win_rate = wins / played
        draw_rate = draws / played
        points_rate = win_rate + (draw_rate * 0.5)

        goals_for = stats.get("goals", {}).get("for", {}).get("total", {}).get("total", 0) or 0
        goals_against = stats.get("goals", {}).get("against", {}).get("total", {}).get("total", 0) or 0
        goal_diff_per_game = (goals_for - goals_against) / played

        rating = (
            ELO_DEFAULT_RATING
            + (points_rate - 0.5) * FORM_RATING_SPREAD
            + goal_diff_per_game * GOAL_DIFF_WEIGHT
        )
        return round(rating, 2)
    except (TypeError, ZeroDivisionError, AttributeError):
        return ELO_DEFAULT_RATING
# ─── Base Request Helper ───────────────────────────────────────────────────────

def _get(url: str, headers: dict = None, params: dict = None, retries: int = 2) -> Optional[dict]:
    """Generic GET with retry logic and rate-limit awareness."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=8)
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 30))
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            if response.status_code == 403:
                # Invalid/missing API key or forbidden — won't succeed on retry
                logger.error(f"403 Forbidden for {url} — check your API key in config.py")
                return None
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
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

    def get_team_squad(self, team_id: int) -> Optional[list]:
        """Fetch squad/players for a specific team."""
        url = f"{self.base}/teams/{team_id}"
        data = _get(url, headers=self.headers)
        if data:
            squad = data.get("squad", [])
            logger.info(f"Fetched {len(squad)} players for team {team_id}")
            return squad
        return []

    def get_teams_with_squads(self, competition_code: str = "WC") -> dict:
        """Fetch all teams and their squads for a competition.
        
        Returns:
            {team_name: [player1, player2, ...], ...}
        """
        teams = self.get_teams(competition_code)
        squads = {}
        for team in teams or []:
            team_name = team.get("name")
            team_id = team.get("id")
            if team_name and team_id:
                squad = self.get_team_squad(team_id)
                if squad:
                    # Extract player names and positions
                    players = []
                    for player in squad:
                        player_name = player.get("name")
                        position = player.get("position", "Unknown")
                        if player_name:
                            players.append({"name": player_name, "position": position})
                    if players:
                        squads[team_name] = players
        return squads


# ─── SportAPI (RapidAPI) Client ────────────────────────────────────────────────

class SportAPIClient:
    """
    Client for SportAPI via RapidAPI.

    Uses /api/v1/category/{category_id}/scheduled-events/{date}.
    Scheduled events provide fixtures, but not API-Football-style team
    statistics or predictions.
    """

    def __init__(self):
        self.base = SPORTAPI_BASE_URL.rstrip("/")
        self.default_category_id = SPORTAPI_FOOTBALL_CATEGORY_ID
        self.default_date = SPORTAPI_DEFAULT_DATE
        self.headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": SPORTAPI_RAPIDAPI_HOST,
            "Content-Type": "application/json",
        }

    def get_scheduled_events(self, date: str = None, category_id: int = None) -> list:
        """Fetch raw SportAPI scheduled events for a category/date."""
        date = date or self.default_date
        category_id = category_id or self.default_category_id
        url = f"{self.base}/category/{category_id}/scheduled-events/{date}"
        data = _get(url, headers=self.headers)

        if not data:
            return []
        if isinstance(data, list):
            events = data
        else:
            events = data.get("events") or data.get("response") or data.get("data") or []

        logger.info(f"Fetched {len(events)} SportAPI scheduled events for {date}")
        return events

    def get_events_for_date_range(self, start_date: str, end_date: str, category_id: int = None) -> list:
        """Fetch scheduled events for every date in an inclusive YYYY-MM-DD range."""
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if end < start:
            raise ValueError("end_date must be on or after start_date")

        events = []
        current = start
        while current <= end:
            events.extend(self.get_scheduled_events(current.isoformat(), category_id))
            current += timedelta(days=1)
            time.sleep(0.2)
        return events

    @staticmethod
    def _team_name(team: dict) -> str:
        if not isinstance(team, dict):
            return ""
        return team.get("name") or team.get("shortName") or team.get("slug") or ""

    @staticmethod
    def _team_id(team: dict):
        return team.get("id") if isinstance(team, dict) else None

    @staticmethod
    def _score_value(score: dict):
        if not isinstance(score, dict):
            return None
        return score.get("current") if score.get("current") is not None else score.get("display")

    @staticmethod
    def normalise_event(event: dict) -> dict:
        """Convert one SportAPI event into the simulator's simple match shape."""
        home_team = event.get("homeTeam", {}) or {}
        away_team = event.get("awayTeam", {}) or {}
        status = event.get("status") or {}
        timestamp = event.get("startTimestamp")
        date = None
        if timestamp:
            try:
                date = datetime.utcfromtimestamp(int(timestamp)).date().isoformat()
            except (TypeError, ValueError, OSError):
                date = None

        return {
            "id": event.get("id"),
            "date": date,
            "status": status.get("description") if isinstance(status, dict) else status,
            "team1": SportAPIClient._team_name(home_team),
            "team2": SportAPIClient._team_name(away_team),
            "team1_id": SportAPIClient._team_id(home_team),
            "team2_id": SportAPIClient._team_id(away_team),
            "score1": SportAPIClient._score_value(event.get("homeScore", {})),
            "score2": SportAPIClient._score_value(event.get("awayScore", {})),
            "tournament": (event.get("tournament") or {}).get("name"),
            "raw": event,
        }

    def get_fixtures(self, date: str = None, category_id: int = None, **_) -> list:
        """Fetch scheduled events and return normalised fixture dicts."""
        events = self.get_scheduled_events(date=date, category_id=category_id)
        fixtures = [self.normalise_event(event) for event in events]
        return [fixture for fixture in fixtures if fixture["team1"] and fixture["team2"]]

    @staticmethod
    def _is_world_cup_fixture(fixture: dict) -> bool:
        tournament = (fixture.get("tournament") or "").lower()
        return "world cup" in tournament or "fifa" in tournament

    def get_world_cup_fixtures(self, date: str = None, category_id: int = None) -> list:
        """
        Fetch SportAPI fixtures for a date and keep only World Cup fixtures.

        SportAPI category 1 is all football, so this filter prevents English
        league teams from being treated as World Cup participants.
        """
        return [
            fixture
            for fixture in self.get_fixtures(date=date, category_id=category_id)
            if self._is_world_cup_fixture(fixture)
        ]

    def get_team_statistics(self, *_, **__) -> dict:
        logger.warning("SportAPI scheduled-events endpoint does not provide team statistics")
        return {}

    def get_team_id_map(self, date: str = None, category_id: int = None, **_) -> dict:
        """
        Build {team_name: team_id} from SportAPI scheduled events.
        """
        team_ids = {}
        for fixture in self.get_fixtures(date=date, category_id=category_id):
            if fixture.get("team1_id"):
                team_ids[fixture["team1"]] = fixture["team1_id"]
            if fixture.get("team2_id"):
                team_ids[fixture["team2"]] = fixture["team2_id"]
        logger.info(f"Fetched {len(team_ids)} SportAPI team IDs")
        return team_ids

    def get_world_cup_team_id_map(self, date: str = None, category_id: int = None) -> dict:
        """Build {team_name: team_id} from SportAPI World Cup fixtures only."""
        team_ids = {}
        for fixture in self.get_world_cup_fixtures(date=date, category_id=category_id):
            if fixture.get("team1_id"):
                team_ids[fixture["team1"]] = fixture["team1_id"]
            if fixture.get("team2_id"):
                team_ids[fixture["team2"]] = fixture["team2_id"]
        logger.info(f"Fetched {len(team_ids)} SportAPI World Cup team IDs")
        return team_ids

    def get_all_team_ratings(self, teams: list = None, date: str = None, category_id: int = None, **_) -> dict:
        """
        Return default ELO ratings for supplied or inferred SportAPI teams.
        """
        if teams is None:
            teams = list(self.get_team_id_map(date=date, category_id=category_id).keys())
        ratings = {team: float(ELO_DEFAULT_RATING) for team in teams}
        logger.info(f"Created {len(ratings)} default ratings from SportAPI team list")
        return ratings

    def get_standings(self, *_, **__) -> list:
        logger.warning("SportAPI scheduled-events endpoint does not provide standings")
        return []

    def get_predictions(self, *_, **__) -> dict:
        logger.warning("SportAPI scheduled-events endpoint does not provide predictions")
        return {}


# Backwards-compatible name for existing imports/call sites.
APIFootballClient = SportAPIClient


# ─── World Football Elo Ratings Client ─────────────────────────────────────────

class WorldELOClient:
    """
    Client for eloratings.net national-team Elo ratings.

    Uses public TSV files:
    - World.tsv: current rankings and ratings
    - en.teams.tsv: team-code to English name/alias mapping
    """

    NAME_ALIASES = {
        "usa": "united states",
        "us": "united states",
        "united states of america": "united states",
        "turkiye": "turkey",
        "south korea": "korea republic",
        "korea republic": "korea republic",
        "ivory coast": "cote divoire",
        "cote d ivoire": "cote divoire",
        "cote divoire": "cote divoire",
        "curacao": "curacao",
        "curaçao": "curacao",
        "cape verde": "cabo verde",
        "cabo verde": "cabo verde",
        "czech republic": "czechia",
        "iran": "iran",
        "ir iran": "iran",
        "uae": "united arab emirates",
        "dr congo": "congo dr",
        "democratic republic of congo": "congo dr",
        "congo kinshasa": "congo dr",
        "north macedonia": "macedonia",
    }

    def __init__(self):
        self.base = WORLD_ELO_BASE_URL.rstrip("/")
        self._code_to_names = None
        self._ratings = None

    @staticmethod
    def _normalise_name(name: str) -> str:
        value = unicodedata.normalize("NFKD", name or "")
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        value = value.lower().replace("&", " and ")
        value = re.sub(r"[^a-z0-9]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        value = value.replace("cote d ivoire", "cote divoire")
        return WorldELOClient.NAME_ALIASES.get(value, value)

    def _get_text(self, path: str) -> str:
        url = f"{self.base}/{path.lstrip('/')}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text

    def _load_team_names(self) -> dict:
        if self._code_to_names is not None:
            return self._code_to_names

        code_to_names = {}
        text = self._get_text("en.teams.tsv")
        for line in text.splitlines():
            parts = [part.strip() for part in line.split("\t") if part.strip()]
            if len(parts) < 2:
                continue
            code_to_names[parts[0]] = parts[1:]

        self._code_to_names = code_to_names
        logger.info(f"Loaded {len(code_to_names)} World Elo team names")
        return code_to_names

    def get_current_ratings(self) -> dict:
        """Return {team_name_or_alias: elo_rating} for all current national teams."""
        if self._ratings is not None:
            return self._ratings

        code_to_names = self._load_team_names()
        ratings = {}
        text = self._get_text("World.tsv")

        for line in text.splitlines():
            parts = [part.strip() for part in line.split("\t")]
            if len(parts) < 4:
                continue
            code = parts[2]
            try:
                rating = float(parts[3])
            except ValueError:
                continue

            for team_name in code_to_names.get(code, [code]):
                ratings[team_name] = rating
                ratings[self._normalise_name(team_name)] = rating

        self._ratings = ratings
        logger.info(f"Loaded {len(ratings)} World Elo rating name mappings")
        return ratings

    def get_team_elo(self, team_name: str) -> Optional[float]:
        ratings = self.get_current_ratings()
        return ratings.get(team_name) or ratings.get(self._normalise_name(team_name))

    def get_all_elos(self, teams: list) -> dict:
        """Fetch current national-team Elo ratings for the requested teams."""
        ratings = self.get_current_ratings()
        result = {}
        missing = []

        for team in teams or []:
            rating = ratings.get(team) or ratings.get(self._normalise_name(team))
            if rating is None:
                missing.append(team)
                rating = float(ELO_DEFAULT_RATING)
            result[team] = rating

        if missing:
            logger.warning(
                "No World Elo rating found for %s; defaulted to %s",
                ", ".join(missing),
                ELO_DEFAULT_RATING,
            )
        logger.info(f"Fetched World Elo ratings for {len(result) - len(missing)}/{len(result)} teams")
        return result


# ─── Club ELO Client ───────────────────────────────────────────────────────────

class ClubELOClient:
    """
    Client for clubelo.com — free ELO ratings for national teams.
    No API key needed.
    """

    def __init__(self):
        self.base = CLUB_ELO_BASE_URL

    def get_team_elo(self, team_name: str, retries: int = 1) -> Optional[float]:
        """
        Fetch current ELO rating for a team.
        team_name must match Club ELO naming (e.g. 'Brazil', 'France').
        Uses a single attempt by default — if clubelo.com is unreachable,
        callers fall back to the default rating rather than retrying
        48 times across a full team list.
        """
        url = f"{self.base}/{team_name}"
        try:
            response = requests.get(url, timeout=5)
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
        """
        Fetch ELO ratings for a list of teams. Returns {team_name: elo}.
        Does a quick reachability check first — if clubelo.com is down or
        blocked, fails fast with defaults instead of retrying per-team.
        """
        elos = {}

        # Fast reachability probe using the first team only
        probe_team = teams[0] if teams else "Brazil"
        probe = self.get_team_elo(probe_team)
        if probe is None:
            logger.warning(
                "clubelo.com unreachable — defaulting all teams to ELO 1500. "
                "Check your network/firewall if this is unexpected."
            )
            return {team: 1500.0 for team in teams}
        elos[probe_team] = probe

        for team in teams:
            if team == probe_team:
                continue
            elo = self.get_team_elo(team)
            elos[team] = elo if elo else 1500.0
            time.sleep(0.3)  # Be polite to the free API

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
        self.sport_api = SportAPIClient()
        self.api_football = self.sport_api  # compatibility with older call sites
        self.world_elo = WorldELOClient()
        self.club_elo = ClubELOClient()
        self.open_football = OpenFootballClient()

    def get_team_elos(self, teams: list) -> dict:
        return self.world_elo.get_all_elos(teams)

    def get_historical_matches(self, year: int = 2022) -> list:
        return self.open_football.get_all_matches(year)

    def get_wc_teams(self) -> list:
        return self.football_data.get_teams("WC")

    def get_wc_groups(self, competition_code: str = "WC") -> dict:
        """
        Fetch the official group stage draw from football-data.org standings.
        Returns {group_name: [team1, team2, ...]} built from live API data.
        Falls back to {} if the API has no standings yet (e.g. draw not made).
        """
        standings = self.football_data.get_standings(competition_code)
        groups = {}
        for entry in standings or []:
            group_label = entry.get("group")  # e.g. "GROUP_A"
            if not group_label:
                continue
            group_name = group_label.replace("GROUP_", "")
            teams = [row["team"]["name"] for row in entry.get("table", [])]
            if teams:
                groups[group_name] = teams
        return groups

    def get_team_squads(self, competition_code: str = "WC") -> dict:
        """
        Fetch real player squads for all teams in a competition.
        
        Returns:
            {team_name: [{name: player_name, position: position}, ...], ...}
        """
        return self.football_data.get_teams_with_squads(competition_code)


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

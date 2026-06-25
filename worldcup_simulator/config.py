# config.py — Central configuration for the World Cup Simulator

import os

# ─── API Configuration ─────────────────────────────────────────────────────────

# football-data.org (free tier: 10 req/min)
# Get your key at: https://www.football-data.org/client/register
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

# SportAPI via RapidAPI
# Set RAPIDAPI_KEY in your environment; do not commit real keys.
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
SPORTAPI_RAPIDAPI_HOST = os.getenv("SPORTAPI_RAPIDAPI_HOST", "sportapi7.p.rapidapi.com")
SPORTAPI_BASE_URL = os.getenv("SPORTAPI_BASE_URL", f"https://{SPORTAPI_RAPIDAPI_HOST}/api/v1")
SPORTAPI_FOOTBALL_CATEGORY_ID = int(os.getenv("SPORTAPI_FOOTBALL_CATEGORY_ID", "1"))
SPORTAPI_DEFAULT_DATE = os.getenv("SPORTAPI_DEFAULT_DATE", "2026-05-03")

# Club ELO (no key needed, free)
CLUB_ELO_BASE_URL = "http://api.clubelo.com"

# Open Football (static GitHub JSON, no key needed)
OPEN_FOOTBALL_BASE_URL = "https://raw.githubusercontent.com/openfootball/world-cup.json/master"

# World Football Elo Ratings (national teams, no key needed)
WORLD_ELO_BASE_URL = "https://www.eloratings.net"

# ─── Tournament Configuration ──────────────────────────────────────────────────

TOURNAMENT_YEAR = 2026                # Upcoming FIFA World Cup
NUM_GROUPS = 12                       # 2026 WC has 12 groups
TEAMS_PER_GROUP = 4
TEAMS_ADVANCE_PER_GROUP = 2          # Top 2 from each group advance automatically
BEST_THIRD_PLACE_QUALIFIERS = 8      # 8 best 3rd-placed teams also advance
TOTAL_TEAMS = 48                      # 48 teams in 2026 WC
KNOCKOUT_BRACKET_SIZE = 32           # 24 direct + 8 best-thirds = 32-team knockout
SIMULATION_RUNS = 10_000             # Number of Monte Carlo iterations

# ─── ELO Configuration ─────────────────────────────────────────────────────────

ELO_K_FACTOR = 40                    # Sensitivity of ELO updates
ELO_HOME_ADVANTAGE = 100             # Points added for home advantage
ELO_DEFAULT_RATING = 1500            # Starting ELO for unknown teams
ELO_SCALE = 400                      # Standard ELO scale factor

# ─── Data Storage ──────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "worldcup.db")

# ─── Simulation Weights ────────────────────────────────────────────────────────
# These weights combine different signals for the final match prediction

WEIGHT_ELO = 0.40
WEIGHT_FIFA_RANK = 0.20
WEIGHT_RECENT_FORM = 0.25
WEIGHT_HEAD_TO_HEAD = 0.15

# ─── 2026 World Cup Teams ───────────────────────────────────────────────────────
# Team list and group draw are fetched live via api/football_api.py
# (FootballAPIFactory.get_wc_teams() / get_wc_groups()).
# No hardcoded roster — falls back to tournament/groups.py only if the API
# has no data yet (e.g. before the official draw is made).

# ─── Match Result Constants ─────────────────────────────────────────────────────

HOME_WIN = "home_win"
AWAY_WIN = "away_win"
DRAW = "draw"

# ─── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

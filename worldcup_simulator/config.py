# config.py — Central configuration for the World Cup Simulator

import os

# ─── API Configuration ─────────────────────────────────────────────────────────

# football-data.org (free tier: 10 req/min)
# Get your key at: https://www.football-data.org/client/register
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "YOUR_KEY_HERE")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

# API-Football via RapidAPI (free tier: 100 req/day)
# Get your key at: https://rapidapi.com/api-sports/api/api-football
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "YOUR_KEY_HERE")
RAPIDAPI_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

# Club ELO (no key needed, free)
CLUB_ELO_BASE_URL = "http://api.clubelo.com"

# Open Football (static GitHub JSON, no key needed)
OPEN_FOOTBALL_BASE_URL = "https://raw.githubusercontent.com/openfootball/world-cup.json/master"

# ─── Tournament Configuration ──────────────────────────────────────────────────

TOURNAMENT_YEAR = 2026                # Upcoming FIFA World Cup
NUM_GROUPS = 12                       # 2026 WC has 12 groups
TEAMS_PER_GROUP = 4
TEAMS_ADVANCE_PER_GROUP = 2          # Top 2 from each group advance
TOTAL_TEAMS = 48                      # 48 teams in 2026 WC
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

# ─── 2026 World Cup Teams (48 teams, grouped) ──────────────────────────────────
# Placeholder groups — update once draw is official

WC_2026_TEAMS = [
    "Brazil", "Argentina", "France", "England", "Spain", "Germany",
    "Portugal", "Netherlands", "Belgium", "Italy", "Croatia", "Uruguay",
    "Mexico", "USA", "Canada", "Colombia", "Chile", "Ecuador",
    "Peru", "Paraguay", "Bolivia", "Venezuela", "Costa Rica", "Panama",
    "Morocco", "Senegal", "Nigeria", "Cameroon", "Ghana", "Tunisia",
    "Egypt", "Ivory Coast", "Japan", "South Korea", "Australia", "Iran",
    "Saudi Arabia", "Qatar", "South Africa", "Algeria", "Serbia", "Poland",
    "Switzerland", "Denmark", "Sweden", "Ukraine", "Wales", "Slovakia",
]

# ─── Match Result Constants ─────────────────────────────────────────────────────

HOME_WIN = "home_win"
AWAY_WIN = "away_win"
DRAW = "draw"

# ─── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
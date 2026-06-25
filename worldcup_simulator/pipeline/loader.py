# pipeline/loader.py — Save & load all data to/from SQLite + JSON cache

import sqlite3
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, DATA_DIR, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Persists all simulation data to SQLite and JSON cache files.

    Tables:
    - teams          : team metadata + ELO ratings
    - matches        : historical match results
    - team_stats     : form, goal averages, H2H aggregates
    - simulations    : Monte Carlo run results
    - simulation_runs: individual run outcomes
    """

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.db_path = db_path
        self._init_db()
        logger.info(f"DataLoader connected to {db_path}")

    # ── Database Setup ────────────────────────────────────────────────────────

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row     # Allows dict-like access
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        return conn

    def load_teams_if_fresh(self, max_age_hours: float = 24) -> dict:
        """
        Load cached team ELO ratings from the DB, but only if EVERY row's
        `updated_at` is within max_age_hours. Otherwise returns {} so the
        caller knows to re-fetch from the API.

        Exists to avoid burning API-Football's 100-requests/day free
        quota re-fetching all ~48 teams' statistics on every run.

        Returns:
            {team_name: elo_rating} if cache is fresh and non-empty, else {}
        """
        from datetime import datetime, timedelta

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, elo_rating, updated_at FROM teams"
            ).fetchall()

        if not rows:
            logger.info("No cached team ratings found — fetch required")
            return {}

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        for row in rows:
            updated_at = row["updated_at"]
            if not updated_at:
                logger.info("Cached ratings missing timestamp — fetch required")
                return {}
            try:
                ts = datetime.fromisoformat(updated_at)
            except ValueError:
                logger.warning(f"Unparseable updated_at '{updated_at}' — fetch required")
                return {}
            if ts < cutoff:
                logger.info(
                    f"Cached rating for '{row['name']}' is older than "
                    f"{max_age_hours}h — fetch required"
                )
                return {}

        result = {row["name"]: row["elo_rating"] for row in rows}
        logger.info(f"Using {len(result)} cached team ratings (fresh, <{max_age_hours}h old)")
        return result

    def _init_db(self):
        """Create all tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS teams (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT UNIQUE NOT NULL,
                    elo_rating  REAL DEFAULT 1500,
                    fifa_rank   INTEGER,
                    group_name  TEXT,
                    updated_at  TEXT
                );

                CREATE TABLE IF NOT EXISTS matches (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    team1       TEXT NOT NULL,
                    team2       TEXT NOT NULL,
                    score1      INTEGER,
                    score2      INTEGER,
                    result      TEXT,
                    goal_diff   INTEGER,
                    total_goals INTEGER,
                    round       TEXT,
                    year        INTEGER,
                    date        TEXT
                );

                CREATE TABLE IF NOT EXISTS team_stats (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    team            TEXT UNIQUE NOT NULL,
                    avg_scored      REAL DEFAULT 1.2,
                    avg_conceded    REAL DEFAULT 1.0,
                    form_score      REAL DEFAULT 0.5,
                    matches_played  INTEGER DEFAULT 0,
                    updated_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS simulations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at          TEXT,
                    n_simulations   INTEGER,
                    method          TEXT,
                    results_json    TEXT
                );

                CREATE TABLE IF NOT EXISTS simulation_runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id   INTEGER REFERENCES simulations(id),
                    team            TEXT,
                    win_probability REAL,
                    semi_final_rate REAL,
                    quarter_final_rate REAL,
                    round_of_16_rate   REAL,
                    group_exit_rate    REAL,
                    elo_rating         REAL
                );
            """)
        logger.info("Database schema initialised")

    # ── Teams ─────────────────────────────────────────────────────────────────

    def save_teams(self, teams_data: dict, groups: dict = None):
        """
        Save team ELO ratings and group assignments.

        Args:
            teams_data: {team_name: elo_rating}
            groups: {group_name: [team, ...]} — optional group assignments
        """
        # Build reverse lookup: team → group
        team_group = {}
        if groups:
            for grp, members in groups.items():
                for t in members:
                    team_group[t] = grp

        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            for team, elo in teams_data.items():
                conn.execute("""
                    INSERT INTO teams (name, elo_rating, group_name, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        elo_rating  = excluded.elo_rating,
                        group_name  = excluded.group_name,
                        updated_at  = excluded.updated_at
                """, (team, float(elo), team_group.get(team), now))

        logger.info(f"Saved {len(teams_data)} team records")

    def load_teams(self) -> dict:
        """Load all team ELO ratings. Returns {team: elo}."""
        with self._connect() as conn:
            rows = conn.execute("SELECT name, elo_rating FROM teams").fetchall()
        result = {row["name"]: row["elo_rating"] for row in rows}
        logger.info(f"Loaded {len(result)} teams from DB")
        return result

    def load_teams_full(self) -> list:
        """Load full team records as list of dicts."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM teams ORDER BY elo_rating DESC").fetchall()
        return [dict(row) for row in rows]

    # ── Matches ───────────────────────────────────────────────────────────────

    def save_matches(self, matches: list):
        """Save cleaned match records to DB (skips duplicates)."""
        with self._connect() as conn:
            inserted = 0
            for m in matches:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO matches
                            (team1, team2, score1, score2, result,
                             goal_diff, total_goals, round, year, date)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (
                        m.get("team1"), m.get("team2"),
                        m.get("score1"), m.get("score2"),
                        m.get("result"), m.get("goal_diff"),
                        m.get("total_goals"), m.get("round"),
                        m.get("year"), m.get("date"),
                    ))
                    inserted += 1
                except sqlite3.Error as e:
                    logger.warning(f"Skipping match {m.get('team1')} vs {m.get('team2')}: {e}")

        logger.info(f"Saved {inserted}/{len(matches)} match records")

    def load_matches(self, year: int = None, team: str = None) -> list:
        """
        Load match records, optionally filtered by year or team.

        Returns:
            List of match dicts
        """
        query = "SELECT * FROM matches WHERE 1=1"
        params = []
        if year:
            query += " AND year = ?"
            params.append(year)
        if team:
            query += " AND (team1 = ? OR team2 = ?)"
            params.extend([team, team])
        query += " ORDER BY year DESC, date ASC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        result = [dict(row) for row in rows]
        logger.info(f"Loaded {len(result)} matches from DB")
        return result

    # ── Team Stats ────────────────────────────────────────────────────────────

    def save_team_stats(self, form: dict, goal_avgs: dict):
        """Save computed form and goal averages for all teams."""
        now = datetime.utcnow().isoformat()
        all_teams = set(form.keys()) | set(goal_avgs.keys())

        with self._connect() as conn:
            for team in all_teams:
                avgs = goal_avgs.get(team, {})
                conn.execute("""
                    INSERT INTO team_stats
                        (team, avg_scored, avg_conceded, form_score, matches_played, updated_at)
                    VALUES (?,?,?,?,?,?)
                    ON CONFLICT(team) DO UPDATE SET
                        avg_scored     = excluded.avg_scored,
                        avg_conceded   = excluded.avg_conceded,
                        form_score     = excluded.form_score,
                        matches_played = excluded.matches_played,
                        updated_at     = excluded.updated_at
                """, (
                    team,
                    avgs.get("avg_scored", 1.2),
                    avgs.get("avg_conceded", 1.0),
                    form.get(team, 0.5),
                    avgs.get("matches", 0),
                    now,
                ))

        logger.info(f"Saved stats for {len(all_teams)} teams")

    def load_team_stats(self) -> dict:
        """Load team stats. Returns {team: {avg_scored, avg_conceded, form_score}}."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM team_stats").fetchall()
        result = {row["team"]: dict(row) for row in rows}
        logger.info(f"Loaded stats for {len(result)} teams")
        return result

    # ── Simulation Results ────────────────────────────────────────────────────

    def save_simulation_results(self, results: dict, n_simulations: int, method: str = "monte_carlo"):
        """
        Save Monte Carlo simulation results to DB.

        Args:
            results: output of MonteCarloSimulator.run()
            n_simulations: how many runs
            method: simulation method name
        """
        now = datetime.utcnow().isoformat()

        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO simulations (run_at, n_simulations, method, results_json)
                VALUES (?, ?, ?, ?)
            """, (now, n_simulations, method, json.dumps(results)))
            sim_id = cursor.lastrowid

            for team, stats in results.items():
                conn.execute("""
                    INSERT INTO simulation_runs
                        (simulation_id, team, win_probability, semi_final_rate,
                         quarter_final_rate, round_of_16_rate, group_exit_rate, elo_rating)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    sim_id,
                    team,
                    stats.get("win_probability", 0),
                    stats.get("semi_final_rate", 0),
                    stats.get("quarter_final_rate", 0),
                    stats.get("round_of_16_rate", 0),
                    stats.get("group_exit_rate", 0),
                    stats.get("elo_rating", 1500),
                ))

        logger.info(f"Saved simulation #{sim_id}: {n_simulations:,} runs via {method}")
        return sim_id

    def load_latest_simulation(self) -> dict:
        """Load the most recent simulation results."""
        with self._connect() as conn:
            sim = conn.execute(
                "SELECT * FROM simulations ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not sim:
                logger.warning("No simulation results found in DB")
                return {}
            return json.loads(sim["results_json"])

    def load_simulation_history(self) -> list:
        """Load metadata for all past simulations."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, run_at, n_simulations, method FROM simulations ORDER BY id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    # ── JSON Cache ────────────────────────────────────────────────────────────

    def cache_write(self, key: str, data):
        """Write any data to a JSON cache file in the data/ directory."""
        path = os.path.join(DATA_DIR, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Cache written: {path}")

    def cache_read(self, key: str):
        """Read a JSON cache file. Returns None if not found."""
        path = os.path.join(DATA_DIR, f"{key}.json")
        if not os.path.exists(path):
            logger.warning(f"Cache miss: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Cache hit: {path}")
        return data

    # ── Full Pipeline Save ────────────────────────────────────────────────────

    def save_all(self, transformed: dict, groups: dict = None):
        """
        Save the full transformed dataset in one call.

        Args:
            transformed: output of DataTransformer.transform_all()
            groups: tournament group configuration
        """
        logger.info("=== Saving all data to DB ===")
        self.save_teams(transformed.get("elo_ratings", {}), groups)
        self.save_matches(transformed.get("matches", []))
        self.save_team_stats(
            transformed.get("team_form", {}),
            transformed.get("goal_averages", {}),
        )
        # Also write a JSON cache for quick access
        self.cache_write("elo_ratings", transformed.get("elo_ratings", {}))
        self.cache_write("team_stats", {
            t: {
                "form": transformed["team_form"].get(t, 0.5),
                **transformed["goal_averages"].get(t, {}),
            }
            for t in set(transformed["team_form"]) | set(transformed["goal_averages"])
        })
        logger.info("=== All data saved ===")


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = DataLoader()

    # Save some test data
    test_ratings = {"Brazil": 2100, "France": 2070, "Germany": 1995}
    loader.save_teams(test_ratings, groups={"A": ["Brazil", "Germany"], "B": ["France"]})

    test_matches = [
        {"team1": "Brazil", "team2": "Germany", "score1": 1, "score2": 7,
         "result": "loss", "goal_diff": -6, "total_goals": 8, "year": 2014, "round": "Semi Final", "date": "2014-07-08"},
        {"team1": "France", "team2": "Croatia", "score1": 4, "score2": 2,
         "result": "win", "goal_diff": 2, "total_goals": 6, "year": 2018, "round": "Final", "date": "2018-07-15"},
    ]
    loader.save_matches(test_matches)

    test_form = {"Brazil": 0.6, "France": 0.8, "Germany": 0.7}
    test_avgs = {
        "Brazil": {"avg_scored": 1.8, "avg_conceded": 0.9, "matches": 7},
        "France": {"avg_scored": 2.0, "avg_conceded": 0.8, "matches": 7},
        "Germany": {"avg_scored": 2.3, "avg_conceded": 1.2, "matches": 7},
    }
    loader.save_team_stats(test_form, test_avgs)

    # Load back
    teams = loader.load_teams()
    print(f"\n✅ Teams in DB: {teams}")

    matches = loader.load_matches()
    print(f"✅ Matches in DB: {len(matches)}")

    stats = loader.load_team_stats()
    print(f"✅ Team stats in DB: {list(stats.keys())}")
# tournament/simulator.py — Full Tournament Orchestrator

import logging
import sys, os
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WC_2026_TEAMS, NUM_GROUPS, TEAMS_PER_GROUP,
    WEIGHT_ELO, WEIGHT_FIFA_RANK, WEIGHT_RECENT_FORM, WEIGHT_HEAD_TO_HEAD,
    LOG_FORMAT, LOG_LEVEL
)
from simulator.elo import ELOEngine
from simulator.monte_carlo import MonteCarloSimulator
from simulator.ml_model import MLPredictor

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ─── Default 2026 World Cup Groups (placeholder until official draw) ────────────

world_cup_groups = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

# Default ELO ratings (sourced from Club ELO approximations)
DEFAULT_ELO_RATINGS = {
    "Argentina": 2140,
    "Spain": 2115,
    "France": 2105,
    "England": 2050,
    "Brazil": 2035,
    "Portugal": 2025,
    "Germany": 1995,
    "Netherlands": 1985,
    "Belgium": 1965,
    "Croatia": 1940,
    "Morocco": 1935,
    "Uruguay": 1925,
    "Colombia": 1915,
    "Japan": 1905,
    "Switzerland": 1885,
    "Austria": 1875,
    "Mexico": 1860,
    "Norway": 1855,
    "Senegal": 1845,
    "United States": 1835,
    "Iran": 1825,
    "Sweden": 1815,
    "South Korea": 1805,
    "Turkey": 1795,
    "Ecuador": 1785,
    "Egypt": 1775,
    "Paraguay": 1765,
    "Scotland": 1755,
    "Algeria": 1745,
    "Ivory Coast": 1735,
    "Canada": 1725,
    "Tunisia": 1715,
    "Bosnia and Herzegovina": 1705,
    "Ghana": 1695,
    "Iraq": 1685,
    "Jordan": 1675,
    "Uzbekistan": 1665,
    "Panama": 1655,
    "South Africa": 1645,
    "New Zealand": 1635,
    "Saudi Arabia": 1625,
    "Qatar": 1615,
    "Cape Verde": 1605,
    "DR Congo": 1595,
    "Czechia": 1585,
    "Curacao": 1575,
    "Haiti": 1565,
    "Australia": 1810
}


class TournamentSimulator:
    """
    Full World Cup Tournament Simulator.

    Orchestrates:
    - Group stage simulation
    - Knockout rounds
    - Multiple simulation modes (ELO, Monte Carlo, ML)
    - Detailed result tracking
    """

    def __init__(
        self,
        groups: dict = None,
        elo_ratings: dict = None,
        n_simulations: int = 10_000
    ):
        self.groups = groups or world_cup_groups
        self.elo_engine = ELOEngine(elo_ratings or DEFAULT_ELO_RATINGS)
        self.mc_simulator = MonteCarloSimulator(self.elo_engine, n_simulations)
        self.ml_predictor = MLPredictor()
        self.n_simulations = n_simulations

        # Try loading a saved ML model
        if not self.ml_predictor.load():
            logger.info("No saved ML model found — will train on demand")

        logger.info(f"Tournament Simulator ready: {len(self.groups)} groups, "
                    f"{sum(len(t) for t in self.groups.values())} teams")

    def simulate_group_stage(self, group: list) -> dict:
        """Simulate a group stage and return full standings."""
        standings = self.mc_simulator.simulate_group(group)
        # Sort by points → GD → GF → ELO
        return dict(sorted(
            standings.items(),
            key=lambda x: (
                x[1]["points"], x[1]["gd"],
                x[1]["gf"], self.elo_engine.get_rating(x[0])
            ),
            reverse=True
        ))

    def simulate_knockout_bracket(self, teams: list) -> dict:
        """
        Simulate knockout bracket from a list of qualified teams.

        Returns detailed bracket with all match results.
        """
        bracket = {"rounds": []}
        remaining = teams.copy()
        round_names = [
            "Round of 32", "Round of 16", "Quarter Finals",
            "Semi Finals", "Final"
        ]
        round_idx = 0

        while len(remaining) > 1:
            round_name = round_names[round_idx] if round_idx < len(round_names) else f"Round {round_idx+1}"
            round_matches = []
            next_round = []

            # Pair teams: 1st group A vs 2nd group B, etc.
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    team_a, team_b = remaining[i], remaining[i+1]
                    winner, ga, gb = self.mc_simulator.simulate_match(
                        team_a, team_b, knockout=True
                    )
                    match_result = {
                        "team_a": team_a,
                        "team_b": team_b,
                        "goals_a": ga,
                        "goals_b": gb,
                        "winner": winner,
                        "was_penalty": ga == gb
                    }
                    round_matches.append(match_result)
                    next_round.append(winner)
                else:
                    next_round.append(remaining[i])  # Bye

            bracket["rounds"].append({
                "name": round_name,
                "matches": round_matches
            })
            remaining = next_round
            round_idx += 1

        bracket["winner"] = remaining[0] if remaining else "Unknown"
        return bracket

    def run_full_simulation(self) -> dict:
        """
        Run one full World Cup simulation.
        Returns complete tournament results.
        """
        logger.info("Running full tournament simulation...")

        # Group stage
        all_group_results = {}
        qualified = []

        for group_name, teams in self.groups.items():
            standings = self.simulate_group_stage(teams)
            all_group_results[group_name] = standings

            # Top 2 advance
            qualifiers = list(standings.keys())[:2]
            qualified.extend(qualifiers)
            logger.debug(f"Group {group_name}: {qualifiers[0]} and {qualifiers[1]} advance")

        # Knockout stage
        bracket = self.simulate_knockout_bracket(qualified)
        winner = bracket["winner"]
        logger.info(f"Tournament winner: {winner}")

        return {
            "group_stage": all_group_results,
            "knockout": bracket,
            "winner": winner,
            "groups_config": self.groups,
        }

    def run_monte_carlo(self) -> dict:
        """
        Run full Monte Carlo simulation across n_simulations tournaments.
        Returns win probabilities and stage progression rates.
        """
        logger.info(f"Starting Monte Carlo: {self.n_simulations:,} simulations")
        return self.mc_simulator.run(self.groups)

    def train_ml_model(self, historical_matches: list = None):
        """Train ML model on historical match data."""
        if historical_matches:
            X, y = self.ml_predictor.prepare_training_data_from_matches(
                historical_matches,
                self.elo_engine.ratings
            )
            self.ml_predictor.train(X, y, use_synthetic=True)
        else:
            self.ml_predictor.train(use_synthetic=True)

    def predict_match_all_methods(self, team_a: str, team_b: str) -> dict:
        """
        Compare predictions from all 3 methods for a single match.
        """
        # ELO prediction
        elo_pred = self.elo_engine.predict_match(team_a, team_b)

        # ML prediction
        if not self.ml_predictor.is_trained:
            self.train_ml_model()

        ml_pred = self.ml_predictor.predict(
            team_a, team_b,
            elo_a=self.elo_engine.get_rating(team_a),
            elo_b=self.elo_engine.get_rating(team_b),
        )

        # Monte Carlo single match (run 1000 simulations of this matchup)
        mc_wins_a, mc_draws, mc_wins_b = 0, 0, 0
        for _ in range(1000):
            result, _, _ = self.mc_simulator.simulate_match(team_a, team_b)
            if result == team_a:
                mc_wins_a += 1
            elif result == "draw":
                mc_draws += 1
            else:
                mc_wins_b += 1

        return {
            "match": f"{team_a} vs {team_b}",
            "elo": {
                "win": elo_pred["win_prob"],
                "draw": elo_pred["draw_prob"],
                "loss": elo_pred["loss_prob"],
            },
            "ml": {
                "win": ml_pred["win_prob"],
                "draw": ml_pred["draw_prob"],
                "loss": ml_pred["loss_prob"],
            },
            "monte_carlo": {
                "win": round(mc_wins_a / 1000, 4),
                "draw": round(mc_draws / 1000, 4),
                "loss": round(mc_wins_b / 1000, 4),
            },
            "elo_ratings": {
                team_a: self.elo_engine.get_rating(team_a),
                team_b: self.elo_engine.get_rating(team_b),
            }
        }

    def get_team_elo_rankings(self) -> list:
        """Return all teams sorted by ELO."""
        return self.elo_engine.get_rankings()


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = TournamentSimulator(n_simulations=1000)

    print("=== Single Tournament Simulation ===")
    result = sim.run_full_simulation()
    print(f"Winner: {result['winner']}")

    print("\n=== Group A Standings ===")
    for team, stats in result["group_stage"]["A"].items():
        print(f"  {team:12s}: {stats['points']}pts | GD:{stats['gd']:+d} | W{stats['wins']}D{stats['draws']}L{stats['losses']}")

    print("\n=== Match Comparison: Brazil vs France ===")
    comparison = sim.predict_match_all_methods("Brazil", "France")
    for method, probs in comparison.items():
        if isinstance(probs, dict) and "win" in probs:
            print(f"  {method:12s}: Win={probs['win']:.1%}  Draw={probs['draw']:.1%}  Loss={probs['loss']:.1%}")
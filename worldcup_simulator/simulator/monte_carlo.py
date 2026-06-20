# simulator/monte_carlo.py — Monte Carlo Simulation Engine

import numpy as np
import logging
from typing import Tuple
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SIMULATION_RUNS, LOG_FORMAT, LOG_LEVEL
from simulator.elo import ELOEngine

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo engine: runs thousands of tournament simulations
    and returns probability distributions for each team's outcomes.

    Combines ELO ratings with randomness to simulate realistic
    football results including upsets, draws, and penalty shootouts.
    """

    def __init__(self, elo_engine: ELOEngine, n_simulations: int = SIMULATION_RUNS):
        self.elo = elo_engine
        self.n_simulations = n_simulations
        logger.info(f"Monte Carlo Simulator initialized: {n_simulations:,} runs")

    def _simulate_goals(self, expected_goals: float) -> int:
        """
        Simulate goals scored using a Poisson distribution.
        Football goals follow Poisson distribution closely.
        """
        return int(np.random.poisson(max(0.1, expected_goals)))

    def simulate_match(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
        knockout: bool = False
    ) -> Tuple[str, int, int]:
        """
        Simulate a single match between two teams.

        Args:
            team_a, team_b: Team names
            neutral: Neutral venue
            knockout: If True, resolve draws via penalties (no draws allowed)

        Returns:
            Tuple of (winner_or_draw, goals_a, goals_b)
            winner_or_draw is "draw" in group stage, team name in knockouts
        """
        pred = self.elo.predict_match(team_a, team_b, neutral)
        goals_a = self._simulate_goals(pred["expected_goals_a"])
        goals_b = self._simulate_goals(pred["expected_goals_b"])

        # Handle draw in knockout: go to penalties
        if knockout and goals_a == goals_b:
            pen_winner = self._simulate_penalties(team_a, team_b, pred["win_prob"])
            return pen_winner, goals_a, goals_b

        if goals_a > goals_b:
            return team_a, goals_a, goals_b
        elif goals_b > goals_a:
            return team_b, goals_a, goals_b
        else:
            return "draw", goals_a, goals_b

    def _simulate_penalties(self, team_a: str, team_b: str, win_prob_a: float) -> str:
        """
        Simulate a penalty shootout.
        Uses ELO-derived probability with added randomness (penalty luck factor).
        """
        # Penalties are more random — regress toward 50/50
        pen_prob_a = 0.5 * 0.4 + win_prob_a * 0.6
        return team_a if np.random.random() < pen_prob_a else team_b

    def simulate_group(self, teams: list, neutral: bool = True) -> dict:
        """
        Simulate a full group stage (round-robin) for a list of teams.

        Returns:
            dict of {team: {points, gf, ga, gd, wins, draws, losses}}
        """
        standings = {
            team: {"points": 0, "gf": 0, "ga": 0, "gd": 0,
                   "wins": 0, "draws": 0, "losses": 0}
            for team in teams
        }

        # Round-robin: every team plays every other team once
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a, team_b = teams[i], teams[j]
                result, ga, gb = self.simulate_match(team_a, team_b, neutral=neutral)

                standings[team_a]["gf"] += ga
                standings[team_a]["ga"] += gb
                standings[team_b]["gf"] += gb
                standings[team_b]["ga"] += ga

                if result == team_a:
                    standings[team_a]["points"] += 3
                    standings[team_a]["wins"] += 1
                    standings[team_b]["losses"] += 1
                elif result == team_b:
                    standings[team_b]["points"] += 3
                    standings[team_b]["wins"] += 1
                    standings[team_a]["losses"] += 1
                else:
                    standings[team_a]["points"] += 1
                    standings[team_b]["points"] += 1
                    standings[team_a]["draws"] += 1
                    standings[team_b]["draws"] += 1

        for team in standings:
            standings[team]["gd"] = standings[team]["gf"] - standings[team]["ga"]

        return standings

    def get_group_qualifiers(self, standings: dict, n_advance: int = 2) -> list:
        """
        Get teams that advance from group stage, sorted by:
        1. Points  2. Goal difference  3. Goals for  4. ELO (tiebreaker)
        """
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (
                x[1]["points"],
                x[1]["gd"],
                x[1]["gf"],
                self.elo.get_rating(x[0])
            ),
            reverse=True
        )
        return [team for team, _ in sorted_teams[:n_advance]]

    def simulate_tournament(self, groups: dict) -> str:
        """
        Simulate a full World Cup tournament from groups to final.

        Args:
            groups: {group_name: [team1, team2, team3, team4]}

        Returns:
            Name of the tournament winner
        """
        # Group stage
        qualified = []
        for group_name, teams in groups.items():
            standings = self.simulate_group(teams)
            qualifiers = self.get_group_qualifiers(standings, n_advance=2)
            qualified.extend(qualifiers)

        # Knockout rounds
        remaining = qualified.copy()
        while len(remaining) > 1:
            next_round = []
            np.random.shuffle(remaining)  # Randomise bracket pairing
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    team_a, team_b = remaining[i], remaining[i+1]
                    winner, _, _ = self.simulate_match(team_a, team_b, knockout=True)
                    next_round.append(winner)
                else:
                    next_round.append(remaining[i])  # Bye (for odd numbers)
            remaining = next_round

        return remaining[0] if remaining else "Unknown"

    def run(self, groups: dict) -> dict:
        """
        Run n_simulations full tournaments and aggregate results.

        Args:
            groups: {group_name: [team1, team2, team3, team4]}

        Returns:
            dict with win probabilities, semi-final rates, etc. per team
        """
        logger.info(f"Starting {self.n_simulations:,} Monte Carlo simulations...")

        results = defaultdict(int)
        all_teams = [team for teams in groups.values() for team in teams]
        stage_counts = {team: defaultdict(int) for team in all_teams}

        for sim in range(self.n_simulations):
            if sim % 1000 == 0 and sim > 0:
                logger.info(f"  Completed {sim:,}/{self.n_simulations:,} simulations...")

            # Track detailed stage progression for this simulation
            qualified = []
            for group_name, teams in groups.items():
                standings = self.simulate_group(teams)
                qualifiers = self.get_group_qualifiers(standings, n_advance=2)
                qualified.extend(qualifiers)
                for team in qualifiers:
                    stage_counts[team]["round_of_32"] += 1

            remaining = qualified.copy()
            stage_names = ["round_of_16", "quarter_final", "semi_final", "final", "winner"]
            stage_idx = 0

            while len(remaining) > 1:
                next_round = []
                np.random.shuffle(remaining)
                for i in range(0, len(remaining), 2):
                    if i + 1 < len(remaining):
                        team_a, team_b = remaining[i], remaining[i+1]
                        winner, _, _ = self.simulate_match(team_a, team_b, knockout=True)
                        next_round.append(winner)
                        if stage_idx < len(stage_names):
                            stage_counts[winner][stage_names[stage_idx]] += 1
                    else:
                        next_round.append(remaining[i])
                remaining = next_round
                stage_idx += 1

            if remaining:
                results[remaining[0]] += 1

        # Build output probabilities
        output = {}
        for team in all_teams:
            output[team] = {
                "win_probability": round(results[team] / self.n_simulations * 100, 2),
                "semi_final_rate": round(stage_counts[team]["semi_final"] / self.n_simulations * 100, 2),
                "quarter_final_rate": round(stage_counts[team]["quarter_final"] / self.n_simulations * 100, 2),
                "round_of_16_rate": round(stage_counts[team]["round_of_16"] / self.n_simulations * 100, 2),
                "group_exit_rate": round(
                    (1 - stage_counts[team]["round_of_32"] / self.n_simulations) * 100, 2
                ),
                "elo_rating": self.elo.get_rating(team),
                "total_wins": results[team],
            }

        logger.info(f"Simulation complete. Top team: {max(output, key=lambda t: output[t]['win_probability'])}")
        return dict(sorted(output.items(), key=lambda x: x[1]["win_probability"], reverse=True))


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulator.elo import ELOEngine

    ratings = {
        "Brazil": 2100, "Argentina": 2080, "France": 2060, "England": 2010,
        "Germany": 1990, "Spain": 2020, "Portugal": 2000, "Netherlands": 1970,
        "Belgium": 1960, "Italy": 1950, "Croatia": 1930, "Uruguay": 1940,
        "Mexico": 1900, "USA": 1890, "Senegal": 1870, "Morocco": 1860,
    }

    elo = ELOEngine(ratings)
    mc = MonteCarloSimulator(elo, n_simulations=1000)

    groups = {
        "A": ["Brazil", "Germany", "Mexico", "USA"],
        "B": ["Argentina", "France", "Uruguay", "Senegal"],
        "C": ["Spain", "England", "Croatia", "Morocco"],
        "D": ["Portugal", "Netherlands", "Belgium", "Italy"],
    }

    print(f"Running 1,000 simulations...")
    results = mc.run(groups)

    print("\n=== World Cup Win Probabilities ===")
    for team, stats in list(results.items())[:8]:
        print(f"  {team:15s}: {stats['win_probability']:5.1f}% | "
              f"SF: {stats['semi_final_rate']:4.1f}% | "
              f"QF: {stats['quarter_final_rate']:4.1f}%")
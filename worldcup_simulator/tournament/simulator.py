# tournament/simulator.py — Full Tournament Orchestrator

import logging
import sys, os
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NUM_GROUPS, TEAMS_PER_GROUP, BEST_THIRD_PLACE_QUALIFIERS,
    WEIGHT_ELO, WEIGHT_FIFA_RANK, WEIGHT_RECENT_FORM, WEIGHT_HEAD_TO_HEAD,
    LOG_FORMAT, LOG_LEVEL
)
from simulator.elo import ELOEngine
from simulator.monte_carlo import MonteCarloSimulator
from simulator.ml_model import MLPredictor
from tournament.groups import GroupStage
from tournament.knockout import KnockoutStage
from pipeline.fetch import DataFetcher

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ─── Fallback Groups (used ONLY if the live API has no draw published yet) ─────
# This is a placeholder so the simulator is still runnable before the
# official 2026 draw — it is never preferred over live API data.

_FALLBACK_GROUPS = {
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
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def _is_complete_2026_draw(groups: dict) -> bool:
    """Return True only for a complete 12-group, 48-team draw."""
    if not isinstance(groups, dict) or len(groups) != NUM_GROUPS:
        return False

    teams = []
    for group_teams in groups.values():
        if not isinstance(group_teams, list) or len(group_teams) != TEAMS_PER_GROUP:
            return False
        teams.extend(group_teams)

    return len(teams) == NUM_GROUPS * TEAMS_PER_GROUP and len(set(teams)) == len(teams)


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
        n_simulations: int = 10_000,
        use_live_data: bool = True,
    ):
        """
        Args:
            groups: Override group draw (skips API fetch if provided)
            elo_ratings: Override ELO ratings (skips API fetch if provided)
            n_simulations: Monte Carlo run count
            use_live_data: If True, fetch groups/ELO from live APIs when
                           not explicitly provided. If the API has no data
                           yet (e.g. draw not published), falls back to a
                           placeholder with a clear warning.
        """
        fetcher = None
        if groups is None and use_live_data:
            fetcher = DataFetcher()
            groups = fetcher.fetch_wc_groups()
            if not _is_complete_2026_draw(groups):
                logger.warning(
                    "No complete 2026 group draw available from API - using "
                    "built-in 2026 World Cup groups"
                )
                groups = _FALLBACK_GROUPS
        self.groups = groups or _FALLBACK_GROUPS

        if elo_ratings is None and use_live_data:
            fetcher = fetcher or DataFetcher()
            all_teams = [t for grp in self.groups.values() for t in grp]
            elo_ratings = fetcher.fetch_elo_ratings(all_teams)

        # Fetch real player squads if available
        squads = {}
        if use_live_data:
            fetcher = fetcher or DataFetcher()
            try:
                squads = fetcher.fetch_team_squads()
            except Exception as exc:
                logger.warning(f"Failed to fetch team squads from API: {exc}")

        self.elo_engine = ELOEngine(elo_ratings or {})
        self.mc_simulator = MonteCarloSimulator(self.elo_engine, n_simulations, squads=squads)
        self.group_stage = GroupStage(self.groups, self.elo_engine, self.mc_simulator)
        self.knockout_stage = KnockoutStage(self.elo_engine, self.mc_simulator)
        self.ml_predictor = MLPredictor()
        self.n_simulations = n_simulations

        # Try loading a saved ML model
        try:
            if not self.ml_predictor.load():
                logger.info("No saved ML model found — will train on demand")
        except Exception as exc:
            self.ml_predictor = MLPredictor()
            logger.warning(
                "Failed to load ML model cache. Falling back to ELO-based simulation. Error: %s",
                exc,
            )

        logger.info(f"Tournament Simulator ready: {len(self.groups)} groups, "
                    f"{sum(len(t) for t in self.groups.values())} teams")

    def run_full_simulation(self) -> dict:
        """
        Run one full World Cup simulation: 12-group stage (with the 8 best
        3rd-placed teams advancing) followed by a proper 32-team knockout
        bracket. Delegates to GroupStage and KnockoutStage so the bracket
        math (32→16→8→4→2→1) is always correct and no teams are dropped.

        Returns complete tournament results.
        """
        logger.info("Running full tournament simulation...")

        # Group stage — includes best-third-place logic for 32-team bracket
        group_results = self.group_stage.simulate_all_groups(
            n_advance=2,
            n_best_thirds=BEST_THIRD_PLACE_QUALIFIERS,
        )

        # Knockout stage — seeded bracket built from qualifiers + best thirds
        bracket_order = self.knockout_stage.build_bracket(
            group_results["qualified"],
            best_thirds=group_results["best_thirds"],
        )
        knockout_result = self.knockout_stage.simulate_bracket(bracket_order)
        winner = knockout_result["winner"]
        logger.info(f"Tournament winner: {winner}")

        return {
            "group_stage": group_results["standings"],
            "best_thirds": group_results["best_thirds"],
            "group_events": group_results.get("group_events", []),
            "knockout": knockout_result,
            "knockout_matches": knockout_result["all_matches"],
            "winner": winner,
            "runner_up": knockout_result.get("runner_up"),
            "third_place": knockout_result.get("third_place_team"),
            "groups_config": self.groups,
        }

    def run_monte_carlo(self) -> dict:
        """
        Run full Monte Carlo simulation across n_simulations tournaments.
        Returns win probabilities and stage progression rates.

        Note: uses the same 12-group + best-thirds + 32-team bracket logic
        as run_full_simulation, repeated n_simulations times.
        """
        logger.info(f"Starting Monte Carlo: {self.n_simulations:,} simulations")

        from collections import defaultdict
        results = defaultdict(int)
        all_teams = [team for teams in self.groups.values() for team in teams]
        stage_counts = {team: defaultdict(int) for team in all_teams}

        for sim in range(self.n_simulations):
            if sim % 1000 == 0 and sim > 0:
                logger.info(f"  Completed {sim:,}/{self.n_simulations:,} simulations...")

            group_results = self.group_stage.simulate_all_groups(
                n_advance=2, n_best_thirds=BEST_THIRD_PLACE_QUALIFIERS
            )
            for team in group_results["all_qualified"]:
                stage_counts[team]["round_of_32"] += 1

            bracket_order = self.knockout_stage.build_bracket(
                group_results["qualified"], best_thirds=group_results["best_thirds"]
            )
            knockout_result = self.knockout_stage.simulate_bracket(bracket_order)

            for rnd in knockout_result["rounds"]:
                stage_key = rnd["round"].lower().replace(" ", "_")
                for winner in rnd["winners"]:
                    stage_counts[winner][stage_key] += 1

            results[knockout_result["winner"]] += 1

        output = {}
        for team in all_teams:
            output[team] = {
                "win_probability": round(results[team] / self.n_simulations * 100, 2),
                "semi_final_rate": round(stage_counts[team].get("semi_finals", 0) / self.n_simulations * 100, 2),
                "quarter_final_rate": round(stage_counts[team].get("quarter_finals", 0) / self.n_simulations * 100, 2),
                "round_of_16_rate": round(stage_counts[team].get("round_of_16", 0) / self.n_simulations * 100, 2),
                "round_of_32_rate": round(stage_counts[team]["round_of_32"] / self.n_simulations * 100, 2),
                "elo_rating": self.elo_engine.get_rating(team),
                "total_wins": results[team],
            }

        logger.info(f"Monte Carlo complete. Top team: "
                    f"{max(output, key=lambda t: output[t]['win_probability'])}")
        return dict(sorted(output.items(), key=lambda x: x[1]["win_probability"], reverse=True))

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
    # use_live_data=False for a fast offline test with placeholder groups/ELO.
    # In production (main.py), leave use_live_data=True (the default) to
    # pull the real draw and ratings from the APIs.
    sim = TournamentSimulator(n_simulations=1000, use_live_data=False)

    print("=== Single Tournament Simulation ===")
    result = sim.run_full_simulation()
    print(f"Winner: {result['winner']}")
    print(f"Runner-up: {result['runner_up']}")
    print(f"Third place: {result['third_place']}")
    print(f"Best thirds that qualified: {result['best_thirds']}")

    print("\n=== Group A Standings ===")
    for team, stats in result["group_stage"]["A"].items():
        if team.startswith("_"):
            continue
        print(f"  {team:12s}: {stats['points']}pts | GD:{stats['gd']:+d} | W{stats['wins']}D{stats['draws']}L{stats['losses']}")

    print("\n=== Match Comparison: Brazil vs France ===")
    comparison = sim.predict_match_all_methods("Brazil", "France")
    for method, probs in comparison.items():
        if isinstance(probs, dict) and "win" in probs:
            print(f"  {method:12s}: Win={probs['win']:.1%}  Draw={probs['draw']:.1%}  Loss={probs['loss']:.1%}")

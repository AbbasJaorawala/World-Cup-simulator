# tournament/knockout.py — Knockout Stage Logic

import logging
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0] = project_root
from config import LOG_FORMAT, LOG_LEVEL
from simulator.elo import ELOEngine
from simulator.monte_carlo import MonteCarloSimulator
from tournament.groups import GroupStage
from tournament.simulator import world_cup_groups

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Round names for a 48-team WC (24 advance → 5 knockout rounds)
ROUND_NAMES = {
    24: "Round of 32",       # 24 → 12  (actually named R32 with byes in 2026)
    16: "Round of 16",
    8:  "Quarter Finals",
    4:  "Semi Finals",
    2:  "Final",
}


class KnockoutStage:
    """
    Manages the knockout stage of the World Cup.

    Handles:
    - Bracket construction from qualified teams
    - Per-match simulation with penalties
    - Full bracket traversal until one winner remains
    - Detailed result tracking for every round
    """

    def __init__(
        self,
        elo_engine: ELOEngine,
        mc_simulator: MonteCarloSimulator,
        seeding: str = "group_order",
    ):
        """
        Args:
            elo_engine: ELOEngine with ratings
            mc_simulator: MonteCarloSimulator for matches
            seeding: "group_order" = 1A vs 2B etc. | "random" = random bracket
        """
        self.elo = elo_engine
        self.mc = mc_simulator
        self.seeding = seeding
        logger.info(f"KnockoutStage initialised (seeding: {seeding})")

    # ── Bracket Construction ──────────────────────────────────────────────────

    def build_bracket(self, qualified: dict) -> list:
        """
        Build a seeded knockout bracket from group stage qualifiers.

        FIFA 2026 pairing: 1A vs 2C, 1B vs 2D, etc. (simplified here as
        alternating 1st/2nd place across groups).

        Args:
            qualified: {group_name: [1st_place, 2nd_place]}

        Returns:
            Ordered list of teams for round-of-32 matchups
        """
        group_names = sorted(qualified.keys())
        winners = [qualified[g][0] for g in group_names]   # All group winners
        runners_up = [qualified[g][1] for g in group_names] # All runners-up

        if self.seeding == "group_order":
            # Pair 1st of group A vs 2nd of group B, etc. (cross-group)
            bracket = []
            n = len(group_names)
            for i in range(n):
                bracket.append(winners[i])
                bracket.append(runners_up[(i + n // 2) % n])
        else:
            # Random seeding
            import random
            all_teams = winners + runners_up
            random.shuffle(all_teams)
            bracket = all_teams

        logger.info(f"Bracket built: {len(bracket)} teams, {len(bracket)//2} first-round matches")
        return bracket

    # ── Match Simulation ──────────────────────────────────────────────────────

    def simulate_match(self, team_a: str, team_b: str) -> dict:
        """
        Simulate a single knockout match (penalties resolve draws).

        Returns:
            {team_a, team_b, goals_a, goals_b, winner, went_to_penalties,
             elo_a, elo_b, win_prob_a}
        """
        winner, goals_a, goals_b = self.mc.simulate_match(
            team_a, team_b, neutral=True, knockout=True
        )
        win_prob = self.elo.predict_match(team_a, team_b)["win_prob"]

        return {
            "team_a": team_a,
            "team_b": team_b,
            "goals_a": goals_a,
            "goals_b": goals_b,
            "winner": winner,
            "went_to_penalties": goals_a == goals_b,
            "elo_a": self.elo.get_rating(team_a),
            "elo_b": self.elo.get_rating(team_b),
            "win_prob_a": win_prob,
            "was_upset": (winner == team_b and win_prob > 0.5)
                         or (winner == team_a and win_prob < 0.5),
        }

    # ── Round Simulation ──────────────────────────────────────────────────────

    def simulate_round(self, teams: list, round_name: str) -> dict:
        """
        Simulate one knockout round.

        Args:
            teams: Ordered list of teams (paired as [0v1, 2v3, ...])
            round_name: Display name for this round

        Returns:
            {
                "round": round_name,
                "matches": [match_result, ...],
                "winners": [team, ...]
            }
        """
        matches = []
        winners = []

        for i in range(0, len(teams), 2):
            if i + 1 < len(teams):
                match = self.simulate_match(teams[i], teams[i + 1])
                matches.append(match)
                winners.append(match["winner"])
            else:
                # Bye: odd number of teams (shouldn't happen in a real WC)
                winners.append(teams[i])
                logger.warning(f"{teams[i]} gets a bye in {round_name}")

        upsets = [m for m in matches if m["was_upset"]]
        logger.info(f"{round_name}: {len(matches)} matches, {len(upsets)} upset(s)")
        return {"round": round_name, "matches": matches, "winners": winners}

    # ── Full Knockout Bracket ─────────────────────────────────────────────────

    def simulate_bracket(self, qualified_teams: list) -> dict:
        """
        Simulate the entire knockout stage from a flat list of qualified teams.

        Args:
            qualified_teams: Ordered list (from build_bracket or flat qualified list)

        Returns:
            {
                "rounds": [round_result, ...],
                "third_place": {match_result},   # Optional 3rd place play-off
                "final": {match_result},
                "winner": team_name,
                "runner_up": team_name,
                "third_place_team": team_name,
            }
        """
        remaining = qualified_teams.copy()
        all_rounds = []
        semi_losers = []

        while len(remaining) > 1:
            round_name = ROUND_NAMES.get(len(remaining), f"Round of {len(remaining)}")
            round_result = self.simulate_round(remaining, round_name)
            all_rounds.append(round_result)

            # Track semi-final losers for 3rd place play-off
            if len(remaining) == 4:
                semi_losers = [
                    m["team_a"] if m["winner"] == m["team_b"] else m["team_b"]
                    for m in round_result["matches"]
                ]

            remaining = round_result["winners"]

        winner = remaining[0]
        runner_up = None
        final_match = None

        # Identify runner-up from the final round
        for rnd in all_rounds:
            if rnd["round"] == "Final" and rnd["matches"]:
                final_match = rnd["matches"][0]
                runner_up = (
                    final_match["team_a"]
                    if final_match["winner"] == final_match["team_b"]
                    else final_match["team_b"]
                )

        # Simulate 3rd place play-off if we have semi losers
        third_place_result = None
        third_place_team = None
        if len(semi_losers) == 2:
            third_place_result = self.simulate_match(semi_losers[0], semi_losers[1])
            third_place_team = third_place_result["winner"]
            logger.info(f"3rd place: {third_place_team}")

        logger.info(f"Champion: {winner} | Runner-up: {runner_up} | 3rd: {third_place_team}")

        return {
            "rounds": all_rounds,
            "final": final_match,
            "third_place": third_place_result,
            "winner": winner,
            "runner_up": runner_up,
            "third_place_team": third_place_team,
        }

    def simulate_from_groups(self, groups: dict, n_advance: int = 2) -> dict:
        """
        Simulate group stage qualifiers and then run the knockout stage.

        Args:
            groups: {group_name: [team1, team2, team3, team4]}
            n_advance: Number of teams advancing from each group

        Returns:
            {
                "group_stage": {
                    "standings": ..., "qualified": ..., "all_qualified": ...
                },
                "knockout": { ... }
            }
        """
        group_stage = GroupStage(groups, self.elo, self.mc)
        group_results = group_stage.simulate_all_groups(n_advance=n_advance)
        bracket_order = self.build_bracket(group_results["qualified"])
        knockout_results = self.simulate_bracket(bracket_order)

        return {
            "group_stage": group_results,
            "knockout": knockout_results,
            "bracket_order": bracket_order,
        }

    # ── Display Helper ────────────────────────────────────────────────────────

    def format_bracket(self, bracket_result: dict) -> str:
        """Format the full bracket as a readable string."""
        lines = ["\n  ══════════ KNOCKOUT STAGE ══════════\n"]
        for rnd in bracket_result["rounds"]:
            lines.append(f"  ── {rnd['round']} ──")
            for m in rnd["matches"]:
                pen = " (pens)" if m["went_to_penalties"] else ""
                upset = " 🚨 UPSET!" if m["was_upset"] else ""
                lines.append(
                    f"    {m['team_a']:16s} {m['goals_a']} – {m['goals_b']} "
                    f"{m['team_b']:16s}  →  {m['winner']}{pen}{upset}"
                )
            lines.append("")

        if bracket_result.get("third_place"):
            m = bracket_result["third_place"]
            pen = " (pens)" if m["went_to_penalties"] else ""
            lines.append(f"  🥉 3rd Place Play-off:")
            lines.append(f"    {m['team_a']:16s} {m['goals_a']} – {m['goals_b']} "
                         f"{m['team_b']:16s}  →  {m['winner']}{pen}")
            lines.append("")

        lines.append(f"  🥇 CHAMPION  : {bracket_result['winner']}")
        lines.append(f"  🥈 Runner-up : {bracket_result.get('runner_up', 'N/A')}")
        lines.append(f"  🥉 Third     : {bracket_result.get('third_place_team', 'N/A')}")
        return "\n".join(lines)


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulator.elo import ELOEngine
    from simulator.monte_carlo import MonteCarloSimulator

    ratings = {
        "Brazil": 2100, "Argentina": 2085, "France": 2070, "England": 2020,
        "Spain": 2030, "Germany": 1995, "Portugal": 2010, "Netherlands": 1975,
        "Belgium": 1965, "Italy": 1950, "Croatia": 1935, "Uruguay": 1945,
        "Mexico": 1905, "USA": 1895, "Morocco": 1870, "Senegal": 1855,
    }

    elo = ELOEngine(ratings)
    mc = MonteCarloSimulator(elo, n_simulations=1)
    ks = KnockoutStage(elo, mc)

    print("Simulating group stage qualifiers from default 2026 groups...")
    group_stage = GroupStage(world_cup_groups, elo, mc)
    group_results = group_stage.simulate_all_groups()
    qualified_groups = group_results["qualified"]

    print("Qualified teams by group:")
    for group_name, teams in qualified_groups.items():
        print(f"  Group {group_name}: {teams[0]} (1st), {teams[1]} (2nd)")

    bracket_order = ks.build_bracket(qualified_groups)
    result = ks.simulate_bracket(bracket_order)
    print(ks.format_bracket(result))
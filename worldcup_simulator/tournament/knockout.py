# tournament/knockout.py — Knockout Stage Logic

import logging
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_FORMAT, LOG_LEVEL
from simulator.elo import ELOEngine
from simulator.monte_carlo import MonteCarloSimulator

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Round names by number of teams entering that round (FIFA 2026: 32-team bracket)
ROUND_NAMES = {
    32: "Round of 32",
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

    def build_bracket(self, qualified: dict, best_thirds: list = None) -> list:
        """
        Build a seeded 32-team knockout bracket from group stage results.

        FIFA 2026 format: 12 groups × top-2 = 24 direct qualifiers,
        plus the 8 best 3rd-placed teams = 32 teams total.

        Group winners are seeded against either a runner-up or a
        best-third team (never against another group winner in R32),
        mirroring how FIFA seeds the actual bracket.

        Args:
            qualified: {group_name: [1st_place, 2nd_place]}
            best_thirds: list of the 8 best 3rd-placed teams (or None/[]
                         for an 8-group/16-team format with no thirds)

        Returns:
            Ordered list of teams for round-of-32 (or round-of-16) matchups,
            paired as [0v1, 2v3, 4v5, ...]. Every input team appears exactly
            once — none are ever dropped.
        """
        best_thirds = best_thirds or []
        group_names = sorted(qualified.keys())
        winners = [qualified[g][0] for g in group_names]
        runners_up = [qualified[g][1] for g in group_names]

        bracket_size = len(winners) + len(runners_up) + len(best_thirds)
        if bracket_size > 0 and bracket_size & (bracket_size - 1) != 0:
            logger.warning(
                f"Bracket size {bracket_size} is not a power of 2 — "
                f"check group_count / best_thirds configuration"
            )

        # Pool of "second tier" opponents: runners-up + best thirds, shuffled
        # so winners don't face a predictable pairing every simulation run.
        # By design this pool is LARGER than `winners` whenever best_thirds
        # is used (e.g. 12 winners vs 12 runners-up + 8 thirds = 20) — every
        # winner still gets exactly one opponent; extra second-tier teams
        # are paired against each other in the lines below.
        import random
        second_tier = runners_up + list(best_thirds)
        random.shuffle(second_tier)

        bracket = []
        # Step 1: pair every group winner with one second-tier opponent
        for i, winner in enumerate(winners):
            bracket.append(winner)
            bracket.append(second_tier[i])

        # Step 2: any remaining second-tier teams (the "extra" runners-up/
        # thirds beyond what winners can cover) are paired against each
        # other, two at a time — nobody is dropped.
        leftover = second_tier[len(winners):]
        bracket.extend(leftover)

        if len(bracket) != bracket_size:
            logger.warning(
                f"Bracket built with {len(bracket)} teams but expected "
                f"{bracket_size} — check input data for duplicates/gaps"
            )

        logger.info(f"Bracket built: {len(bracket)} teams, {len(bracket) // 2} first-round matches")
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


        all_matches = []
        for rnd in all_rounds:
            for m in rnd["matches"]:
                m_with_round = dict(m)
                m_with_round["round"] = rnd["round"]
                all_matches.append(m_with_round)
        if third_place_result:
            tp_with_round = dict(third_place_result)
            tp_with_round["round"] = "Third Place Play-off"
            all_matches.append(tp_with_round)

        return {
            "rounds": all_rounds,
            "final": final_match,
            "third_place": third_place_result,
            "winner": winner,
            "runner_up": runner_up,
            "third_place_team": third_place_team,
            "all_matches": all_matches,
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
    from tournament.groups import GroupStage

   
    elo = ELOEngine(ratings)
    mc = MonteCarloSimulator(elo, n_simulations=1)

    # Full pipeline: simulate groups (with best-thirds) → build bracket → knockout
    gs = GroupStage(groups, elo, mc)
    group_results = gs.simulate_all_groups(n_advance=2, n_best_thirds=8)

    print(f"24 direct qualifiers + {len(group_results['best_thirds'])} best thirds "
          f"= {len(group_results['all_qualified'])} teams in knockout\n")

    ks = KnockoutStage(elo, mc)
    bracket_order = ks.build_bracket(
        group_results["qualified"],
        best_thirds=group_results["best_thirds"]
    )
    result = ks.simulate_bracket(bracket_order)
    print(ks.format_bracket(result))
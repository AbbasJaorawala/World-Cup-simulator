# tournament/groups.py — Group Stage Logic

import logging
import sys, os
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_FORMAT, LOG_LEVEL
from simulator.elo import ELOEngine
from simulator.monte_carlo import MonteCarloSimulator

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class GroupStage:
    """
    Manages the full group stage of the World Cup.

    Handles:
    - Round-robin match scheduling within each group
    - Points / GD / GF tiebreaking rules (FIFA rules)
    - Head-to-head tiebreaker when points are level
    - Returning the ranked standings and qualified teams
    """

    def __init__(self, groups: dict, elo_engine: ELOEngine, mc_simulator: MonteCarloSimulator):
        """
        Args:
            groups: {group_name: [team1, team2, team3, team4]}
            elo_engine: ELOEngine instance with ratings loaded
            mc_simulator: MonteCarloSimulator for match simulation
        """
        self.groups = groups
        self.elo = elo_engine
        self.mc = mc_simulator
        logger.info(f"GroupStage initialised: {len(groups)} groups")

    # ── Match Schedule ────────────────────────────────────────────────────────

    def get_fixtures(self, group_name: str) -> list:
        """
        Return all round-robin fixtures for a group.

        Returns:
            List of (team_a, team_b) tuples
        """
        teams = self.groups.get(group_name, [])
        return list(combinations(teams, 2))

    def get_all_fixtures(self) -> dict:
        """Return all fixtures across every group."""
        return {grp: self.get_fixtures(grp) for grp in self.groups}

    # ── Standings Computation ─────────────────────────────────────────────────

    def _empty_record(self) -> dict:
        return {
            "played": 0, "wins": 0, "draws": 0, "losses": 0,
            "gf": 0, "ga": 0, "gd": 0, "points": 0,
            "match_results": [],  # list of (opponent, result, gf, ga)
        }

    def simulate_group(self, group_name: str) -> dict:
        """
        Simulate all matches in one group and return full standings.

        Returns:
            {team: {played, wins, draws, losses, gf, ga, gd, points}}
        """
        teams = self.groups.get(group_name, [])
        if not teams:
            logger.warning(f"Group {group_name} not found")
            return {}

        table = {team: self._empty_record() for team in teams}

        for team_a, team_b in combinations(teams, 2):
            result, ga, gb = self.mc.simulate_match(team_a, team_b, neutral=True)

            # Update played counts
            table[team_a]["played"] += 1
            table[team_b]["played"] += 1

            # Goals
            table[team_a]["gf"] += ga
            table[team_a]["ga"] += gb
            table[team_b]["gf"] += gb
            table[team_b]["ga"] += ga

            # Track match result for H2H tiebreaker
            table[team_a]["match_results"].append((team_b, result if result == team_a else ("draw" if result == "draw" else "loss"), ga, gb))
            table[team_b]["match_results"].append((team_a, result if result == team_b else ("draw" if result == "draw" else "loss"), gb, ga))

            # Points
            if result == team_a:
                table[team_a]["wins"] += 1
                table[team_a]["points"] += 3
                table[team_b]["losses"] += 1
            elif result == team_b:
                table[team_b]["wins"] += 1
                table[team_b]["points"] += 3
                table[team_a]["losses"] += 1
            else:
                table[team_a]["draws"] += 1
                table[team_b]["draws"] += 1
                table[team_a]["points"] += 1
                table[team_b]["points"] += 1

        # Compute goal difference
        for team in table:
            table[team]["gd"] = table[team]["gf"] - table[team]["ga"]

        logger.debug(f"Group {group_name} simulated")
        return table

    # ── Tiebreaking ───────────────────────────────────────────────────────────

    def _h2h_points(self, team: str, rivals: list, table: dict) -> int:
        """
        Compute head-to-head points between tied teams only.
        FIFA rule: if points are level, compare record between the tied teams.
        """
        points = 0
        for result_entry in table[team]["match_results"]:
            opponent, result, _, _ = result_entry
            if opponent in rivals:
                if result == "win":
                    points += 3
                elif result == "draw":
                    points += 1
        return points

    def _h2h_gd(self, team: str, rivals: list, table: dict) -> int:
        """Head-to-head goal difference among tied teams."""
        gd = 0
        for result_entry in table[team]["match_results"]:
            opponent, _, gf, ga = result_entry
            if opponent in rivals:
                gd += gf - ga
        return gd

    def rank_group(self, table: dict) -> list:
        """
        Rank teams in a group using FIFA tiebreaking rules:
        1. Points
        2. Goal difference (overall)
        3. Goals for (overall)
        4. Head-to-head points (among tied teams)
        5. Head-to-head goal difference
        6. ELO rating (final tiebreaker)

        Returns:
            Ordered list of team names (1st → last)
        """
        teams = list(table.keys())

        def sort_key(team):
            rivals = [t for t in teams if t != team]
            return (
                table[team]["points"],
                table[team]["gd"],
                table[team]["gf"],
                self._h2h_points(team, rivals, table),
                self._h2h_gd(team, rivals, table),
                self.elo.get_rating(team),
            )

        return sorted(teams, key=sort_key, reverse=True)

    def get_qualifiers(self, table: dict, n_advance: int = 2) -> list:
        """Return the top N teams from a group that advance."""
        ranked = self.rank_group(table)
        return ranked[:n_advance]

    # ── Simulate All Groups ───────────────────────────────────────────────────

    def simulate_all_groups(self, n_advance: int = 2, n_best_thirds: int = 8) -> dict:
        """
        Simulate every group in the tournament.

        2026 FIFA format: top 2 from each of the 12 groups advance
        automatically (24 teams), PLUS the 8 best 3rd-placed teams across
        all groups also advance — giving a 32-team knockout bracket.

        Args:
            n_advance: teams guaranteed to advance per group (default 2)
            n_best_thirds: number of best 3rd-place teams to also advance
                           (default 8, per 2026 format). Set to 0 to disable
                           (e.g. for a classic 8-group/16-team format).

        Returns:
            {
                "standings": {group_name: {team: stats, "_ranked": [...]}},
                "qualified": {group_name: [top teams that auto-advance]},
                "third_place_candidates": [{team, group, points, gd, gf}, ...],
                "best_thirds": [team, ...],          # teams that made the cut
                "all_qualified": [team1, team2, ...], # flat list, 24 or 32 teams
            }
        """
        standings = {}
        qualified_by_group = {}
        all_qualified = []
        third_place_candidates = []

        for group_name in self.groups:
            table = self.simulate_group(group_name)
            ranked = self.rank_group(table)
            advancers = ranked[:n_advance]

            standings[group_name] = {
                team: {k: v for k, v in stats.items() if k != "match_results"}
                for team, stats in table.items()
            }
            standings[group_name]["_ranked"] = ranked

            qualified_by_group[group_name] = advancers
            all_qualified.extend(advancers)

            # Track the 3rd-placed team as a candidate for "best thirds"
            if n_best_thirds > 0 and len(ranked) >= 3:
                third_team = ranked[2]
                stats = table[third_team]
                third_place_candidates.append({
                    "team": third_team,
                    "group": group_name,
                    "points": stats["points"],
                    "gd": stats["gd"],
                    "gf": stats["gf"],
                    "elo": self.elo.get_rating(third_team),
                })

            logger.info(f"Group {group_name}: {advancers[0]} (1st), {advancers[1]} (2nd) qualify")

        # Rank all 3rd-placed teams across groups and take the best N
        best_thirds = []
        if n_best_thirds > 0 and third_place_candidates:
            ranked_thirds = sorted(
                third_place_candidates,
                key=lambda t: (t["points"], t["gd"], t["gf"], t["elo"]),
                reverse=True,
            )
            best_thirds = [t["team"] for t in ranked_thirds[:n_best_thirds]]
            all_qualified.extend(best_thirds)
            logger.info(f"Best {len(best_thirds)} third-placed teams qualify: {best_thirds}")

        return {
            "standings": standings,
            "qualified": qualified_by_group,
            "third_place_candidates": third_place_candidates,
            "best_thirds": best_thirds,
            "all_qualified": all_qualified,
        }

    # ── Display Helpers ───────────────────────────────────────────────────────

    def format_standings_table(self, group_name: str, table: dict) -> str:
        """Format a group standings table as a readable string."""
        ranked = self.rank_group(table)
        header = f"\n  Group {group_name}\n"
        header += f"  {'Team':<15} {'P':>2} {'W':>2} {'D':>2} {'L':>2} {'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}\n"
        header += "  " + "-" * 48 + "\n"
        rows = ""
        for i, team in enumerate(ranked):
            s = table[team]
            qualifier = " ✅" if i < 2 else ""
            rows += (f"  {team:<15} {s['played']:>2} {s['wins']:>2} {s['draws']:>2} "
                     f"{s['losses']:>2} {s['gf']:>3} {s['ga']:>3} {s['gd']:>+4} {s['points']:>4}{qualifier}\n")
        return header + rows


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulator.elo import ELOEngine
    from simulator.monte_carlo import MonteCarloSimulator



    elo = ELOEngine(ratings)
    mc = MonteCarloSimulator(elo, n_simulations=1)
    gs = GroupStage(groups, elo, mc)

    print("Simulating Group A...")
    table_a = gs.simulate_group("A")
    print(gs.format_standings_table("A", table_a))

    print("Simulating All Groups...")
    results = gs.simulate_all_groups()
    print("\nQualified Teams:")
    for grp, teams in results["qualified"].items():
        print(f"  Group {grp}: {teams[0]} (1st), {teams[1]} (2nd)")
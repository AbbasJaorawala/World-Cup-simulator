# tournament/groups.py — Group Stage Logic

import logging
import sys, os
from itertools import combinations

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[0] = project_root
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

    def simulate_all_groups(self, n_advance: int = 2) -> dict:
        """
        Simulate every group in the tournament.

        Returns:
            {
                "standings": {group_name: {team: stats}},
                "qualified":  {group_name: [team1, team2]},  # teams advancing
                "all_qualified": [team1, team2, ...],         # flat list
            }
        """
        standings = {}
        qualified_by_group = {}
        all_qualified = []

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

            logger.info(f"Group {group_name}: {advancers[0]} (1st), {advancers[1]} (2nd) qualify")

        return {
            "standings": standings,
            "qualified": qualified_by_group,
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

    ratings = {
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


    groups = {
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
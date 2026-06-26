# simulator/elo.py — ELO Rating Engine for World Cup Simulation

import math
import logging
from typing import Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ELO_K_FACTOR, ELO_HOME_ADVANTAGE, ELO_DEFAULT_RATING, ELO_SCALE, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ELOEngine:
    """
    ELO Rating System for national football teams.

    Implements FIFA-style ELO with:
    - Expected score calculation
    - K-factor scaling by match importance
    - Goal difference weighting
    - Neutral venue / home advantage handling
    """

    # K-factor multipliers by match type (FIFA-style)
    MATCH_IMPORTANCE = {
        "friendly": 20,
        "qualifier": 25,
        "confederation": 35,
        "world_cup_group": 40,
        "world_cup_knockout": 50,
        "world_cup_final": 60,
    }

    def __init__(self, initial_ratings: dict = None):
        """
        Args:
            initial_ratings: {team_name: elo_rating}. Uses ELO_DEFAULT_RATING if not provided.
        """
        self.ratings = initial_ratings or {}
        logger.info(f"ELO Engine initialized with {len(self.ratings)} team ratings")

    def get_rating(self, team: str) -> float:
        """Get current ELO rating for a team (defaults to ELO_DEFAULT_RATING)."""
        return self.ratings.get(team, ELO_DEFAULT_RATING)

    def set_rating(self, team: str, rating: float):
        """Set ELO rating for a team."""
        self.ratings[team] = round(rating, 2)

    def expected_score(self, rating_a: float, rating_b: float, home_advantage: bool = False) -> float:
        """
        Calculate expected score (win probability) for team A vs team B.

        Args:
            rating_a: ELO of team A
            rating_b: ELO of team B
            home_advantage: If True, adds HOME_ADVANTAGE points to team A

        Returns:
            Float between 0 and 1 (probability team A wins or draws)
        """
        adj_rating_a = rating_a + (ELO_HOME_ADVANTAGE if home_advantage else 0)
        expected = 1.0 / (1.0 + 10 ** ((rating_b - adj_rating_a) / ELO_SCALE))
        return round(expected, 4)

    def win_probability(self, team_a: str, team_b: str, neutral: bool = True) -> Tuple[float, float, float]:
        """
        Calculate win/draw/loss probabilities for team_a vs team_b.

        Returns:
            Tuple of (win_prob, draw_prob, loss_prob)
        """
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        home_adv = not neutral

        exp_a = self.expected_score(ra, rb, home_advantage=home_adv)

        # Estimate draw probability based on ELO closeness
        elo_diff = abs(ra - rb)
        draw_base = 0.25
        draw_prob = max(0.10, draw_base - (elo_diff / 2000))

        # Remaining probability split between win and loss
        win_prob = exp_a * (1 - draw_prob)
        loss_prob = (1 - exp_a) * (1 - draw_prob)

        # Normalize to ensure they sum to 1
        total = win_prob + draw_prob + loss_prob
        return round(win_prob / total, 4), round(draw_prob / total, 4), round(loss_prob / total, 4)

    def goal_diff_multiplier(self, goal_diff: int) -> float:
        """
        Weight K-factor by goal difference (FIFA-style).
        Bigger wins = more ELO movement.
        """
        if goal_diff == 0:
            return 1.0
        elif goal_diff == 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:
            return 1.75 + (goal_diff - 3) * 0.25  # Caps growth for blowouts

    def actual_score(self, goals_a: int, goals_b: int) -> float:
        """
        Convert match result to actual score for ELO calculation.
        Win = 1.0, Draw = 0.5, Loss = 0.0
        """
        if goals_a > goals_b:
            return 1.0
        elif goals_a == goals_b:
            return 0.5
        else:
            return 0.0

    def update_ratings(
        self,
        team_a: str,
        team_b: str,
        goals_a: int,
        goals_b: int,
        match_type: str = "world_cup_group",
        neutral: bool = True
    ) -> Tuple[float, float]:
        """
        Update ELO ratings after a match.

        Args:
            team_a, team_b: Team names
            goals_a, goals_b: Goals scored
            match_type: Type of match (controls K-factor)
            neutral: Whether match is at neutral venue

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)

        k = self.MATCH_IMPORTANCE.get(match_type, ELO_K_FACTOR)
        home_adv = not neutral

        exp_a = self.expected_score(ra, rb, home_advantage=home_adv)
        exp_b = 1 - exp_a

        actual_a = self.actual_score(goals_a, goals_b)
        actual_b = 1 - actual_a

        gd_mult = self.goal_diff_multiplier(abs(goals_a - goals_b))

        new_ra = ra + k * gd_mult * (actual_a - exp_a)
        new_rb = rb + k * gd_mult * (actual_b - exp_b)

        self.set_rating(team_a, new_ra)
        self.set_rating(team_b, new_rb)

        logger.debug(f"{team_a} {goals_a}-{goals_b} {team_b} | "
                     f"ELO: {ra:.0f}→{new_ra:.0f}, {rb:.0f}→{new_rb:.0f}")

        return round(new_ra, 2), round(new_rb, 2)

    def train_on_historical(self, matches: list, match_type: str = "world_cup_group"):
        """
        Train ELO ratings using historical match data.

        Args:
            matches: List of dicts with keys: team1, team2, score1, score2
            match_type: Match importance category
        """
        updated = 0
        for match in matches:
            team1 = match.get("team1")
            team2 = match.get("team2")
            score1 = match.get("score1")
            score2 = match.get("score2")

            if not all([team1, team2, score1 is not None, score2 is not None]):
                continue

            try:
                self.update_ratings(team1, team2, int(score1), int(score2), match_type)
                updated += 1
            except Exception as e:
                logger.warning(f"Skipping match {team1} vs {team2}: {e}")

        logger.info(f"ELO trained on {updated}/{len(matches)} historical matches")

    def get_rankings(self) -> list:
        """Return all teams sorted by ELO rating (descending)."""
        ranked = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return [{"rank": i+1, "team": team, "elo": elo} for i, (team, elo) in enumerate(ranked)]

    def predict_match(self, team_a: str, team_b: str, neutral: bool = True) -> dict:
        """
        Full match prediction with probabilities and expected goals.

        Returns:
            dict with win/draw/loss probs, expected goals, and ELO ratings
        """
        win_p, draw_p, loss_p = self.win_probability(team_a, team_b, neutral)
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)

        # Expected goals: realistic World Cup estimation based on ELO difference
        # Real World Cup matches average 2.7 goals total (~1.35 per team)
        elo_diff = ra - rb
        base_goals = 1.1  # Conservative base for World Cup matches
        exp_goals_a = base_goals + (elo_diff / 1500)  # Reduced ELO weight
        exp_goals_b = base_goals - (elo_diff / 1500)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "elo_a": ra,
            "elo_b": rb,
            "win_prob": win_p,
            "draw_prob": draw_p,
            "loss_prob": loss_p,
            "expected_goals_a": round(max(0.3, exp_goals_a), 2),
            "expected_goals_b": round(max(0.3, exp_goals_b), 2),
        }


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize with some ratings
    ratings = {
        "Brazil": 2100, "Argentina": 2080, "France": 2060,
        "England": 2010, "Germany": 1990, "Spain": 2020,
        "Portugal": 2000, "Netherlands": 1970
    }

    engine = ELOEngine(ratings)

    print("=== Match Prediction: Brazil vs Germany ===")
    pred = engine.predict_match("Brazil", "Germany")
    for k, v in pred.items():
        print(f"  {k}: {v}")

    print("\n=== Simulating Brazil 2-1 Germany ===")
    new_ra, new_rb = engine.update_ratings("Brazil", "Germany", 2, 1)
    print(f"  Brazil ELO: {ratings['Brazil']} → {new_ra}")
    print(f"  Germany ELO: {ratings['Germany']} → {new_rb}")

    print("\n=== Top 5 Rankings ===")
    for r in engine.get_rankings()[:5]:
        print(f"  #{r['rank']} {r['team']}: {r['elo']}")
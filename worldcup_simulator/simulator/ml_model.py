# simulator/ml_model.py — ML Prediction Engine (scikit-learn)

import numpy as np
import logging
import os, sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Run: pip install scikit-learn joblib")


class MLPredictor:
    """
    ML-based match outcome predictor.

    Features used:
    - ELO difference between teams
    - FIFA ranking difference
    - Recent form (last 5 matches win rate)
    - Head-to-head win rate
    - Goals scored / conceded averages
    - Tournament stage (group / knockout)
    - Neutral venue flag

    Target: 0 = team_a loss, 1 = draw, 2 = team_a win
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml_model.pkl")
    SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "scaler.pkl")

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.feature_names = [
            "elo_diff",
            "fifa_rank_diff",
            "form_a",
            "form_b",
            "h2h_win_rate_a",
            "avg_goals_scored_a",
            "avg_goals_conceded_a",
            "avg_goals_scored_b",
            "avg_goals_conceded_b",
            "is_knockout",
            "is_neutral",
        ]
        logger.info("ML Predictor initialized")

    def _build_features(
        self,
        elo_a: float, elo_b: float,
        rank_a: int, rank_b: int,
        form_a: float, form_b: float,
        h2h_win_rate_a: float,
        avg_gf_a: float, avg_ga_a: float,
        avg_gf_b: float, avg_ga_b: float,
        is_knockout: bool = False,
        is_neutral: bool = True,
    ) -> np.ndarray:
        """Build the feature vector for a match."""
        return np.array([[
            elo_a - elo_b,
            rank_b - rank_a,         # Higher diff = team_a is better ranked
            form_a,
            form_b,
            h2h_win_rate_a,
            avg_gf_a,
            avg_ga_a,
            avg_gf_b,
            avg_ga_b,
            int(is_knockout),
            int(is_neutral),
        ]])

    def generate_synthetic_training_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data based on ELO theory.
        Used when real historical data isn't available.
        In production, replace with real match data from APIs.
        """
        np.random.seed(42)
        X, y = [], []

        for _ in range(n_samples):
            elo_diff = np.random.normal(0, 200)
            rank_diff = np.random.randint(-50, 50)
            form_a = np.random.uniform(0.1, 0.9)
            form_b = np.random.uniform(0.1, 0.9)
            h2h = np.random.uniform(0.2, 0.8)
            avg_gf_a = np.random.uniform(0.5, 3.0)
            avg_ga_a = np.random.uniform(0.5, 2.5)
            avg_gf_b = np.random.uniform(0.5, 3.0)
            avg_ga_b = np.random.uniform(0.5, 2.5)
            is_knockout = np.random.choice([0, 1])
            is_neutral = np.random.choice([0, 1])

            # Derive win probability from ELO (ground truth)
            win_prob_a = 1 / (1 + 10 ** (-elo_diff / 400))
            win_prob_a = win_prob_a * 0.7 + form_a * 0.3  # Blend with form

            # Add noise to simulate real-world upsets
            noise = np.random.normal(0, 0.1)
            win_prob_a = np.clip(win_prob_a + noise, 0.05, 0.95)

            draw_prob = max(0.1, 0.25 - abs(elo_diff) / 2000)
            adjusted_win = win_prob_a * (1 - draw_prob)
            adjusted_loss = (1 - win_prob_a) * (1 - draw_prob)

            outcome = np.random.choice(
                [2, 1, 0],  # win, draw, loss for team_a
                p=[adjusted_win, draw_prob, adjusted_loss]
            )

            X.append([elo_diff, rank_diff, form_a, form_b, h2h,
                      avg_gf_a, avg_ga_a, avg_gf_b, avg_ga_b,
                      is_knockout, is_neutral])
            y.append(outcome)

        return np.array(X), np.array(y)

    def prepare_training_data_from_matches(self, matches: list, elo_ratings: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from real historical match records.

        Args:
            matches: List of match dicts (team1, team2, score1, score2)
            elo_ratings: {team: elo} ratings

        Returns:
            X (features), y (labels: 0=team1 loss, 1=draw, 2=team1 win)
        """
        X, y = [], []
        default_elo = 1500

        for match in matches:
            t1, t2 = match.get("team1"), match.get("team2")
            s1, s2 = match.get("score1"), match.get("score2")
            if not all([t1, t2, s1 is not None, s2 is not None]):
                continue

            try:
                elo_a = elo_ratings.get(t1, default_elo)
                elo_b = elo_ratings.get(t2, default_elo)
                elo_diff = elo_a - elo_b

                # Use neutral defaults if detailed stats aren't available
                X.append([elo_diff, 0, 0.5, 0.5, 0.5, 1.3, 1.0, 1.3, 1.0, 0, 1])

                s1, s2 = int(s1), int(s2)
                y.append(2 if s1 > s2 else (1 if s1 == s2 else 0))
            except Exception as e:
                logger.warning(f"Skipping match {t1} vs {t2}: {e}")

        logger.info(f"Prepared {len(X)} training samples from real matches")
        return np.array(X), np.array(y)

    def train(self, X: np.ndarray = None, y: np.ndarray = None, use_synthetic: bool = True):
        """
        Train the ML model.

        Args:
            X, y: Training data (optional, uses synthetic if None)
            use_synthetic: Generate synthetic data if real data isn't enough
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available. Cannot train model.")
            return

        if X is None or len(X) < 100:
            if use_synthetic:
                logger.info("Using synthetic training data...")
                X_syn, y_syn = self.generate_synthetic_training_data(5000)
                X = np.vstack([X, X_syn]) if X is not None and len(X) > 0 else X_syn
                y = np.concatenate([y, y_syn]) if y is not None and len(y) > 0 else y_syn
            else:
                logger.error("Insufficient training data and synthetic disabled.")
                return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test_scaled))
        logger.info(f"Model trained! Accuracy: {acc:.1%}")

        self.is_trained = True
        self._save()

    def _save(self):
        """Save trained model and scaler to disk."""
        if not SKLEARN_AVAILABLE:
            return
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, self.MODEL_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)
        logger.info("Model saved to disk")

    def load(self) -> bool:
        """Load a previously trained model from disk."""
        if not SKLEARN_AVAILABLE:
            return False
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.SCALER_PATH):
            self.model = joblib.load(self.MODEL_PATH)
            self.scaler = joblib.load(self.SCALER_PATH)
            self.is_trained = True
            logger.info("Model loaded from disk")
            return True
        return False

    def predict(
        self,
        team_a: str,
        team_b: str,
        elo_a: float, elo_b: float,
        rank_a: int = 10, rank_b: int = 10,
        form_a: float = 0.5, form_b: float = 0.5,
        h2h_win_rate_a: float = 0.5,
        avg_gf_a: float = 1.3, avg_ga_a: float = 1.0,
        avg_gf_b: float = 1.3, avg_ga_b: float = 1.0,
        is_knockout: bool = False,
        is_neutral: bool = True,
    ) -> dict:
        """
        Predict match outcome.

        Returns:
            dict with win/draw/loss probabilities
        """
        if not self.is_trained:
            logger.warning("Model not trained. Falling back to ELO-based prediction.")
            exp = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
            return {
                "team_a": team_a, "team_b": team_b,
                "win_prob": round(exp * 0.75, 4),
                "draw_prob": 0.25,
                "loss_prob": round((1 - exp) * 0.75, 4),
                "predicted_outcome": "win" if exp > 0.5 else "loss",
                "model": "elo_fallback"
            }

        X = self._build_features(
            elo_a, elo_b, rank_a, rank_b,
            form_a, form_b, h2h_win_rate_a,
            avg_gf_a, avg_ga_a, avg_gf_b, avg_ga_b,
            is_knockout, is_neutral
        )
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]

        # Classes: 0=loss, 1=draw, 2=win for team_a
        classes = list(self.model.classes_)
        prob_map = {c: p for c, p in zip(classes, probs)}

        win_p = prob_map.get(2, 0.0)
        draw_p = prob_map.get(1, 0.0)
        loss_p = prob_map.get(0, 0.0)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "win_prob": round(win_p, 4),
            "draw_prob": round(draw_p, 4),
            "loss_prob": round(loss_p, 4),
            "predicted_outcome": "win" if win_p > loss_p else ("draw" if draw_p > max(win_p, loss_p) else "loss"),
            "model": "gradient_boosting"
        }

    def feature_importance(self) -> Optional[dict]:
        """Return feature importances from the trained model."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return None
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        ))


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = MLPredictor()

    if not predictor.load():
        print("Training new model with synthetic data...")
        predictor.train(use_synthetic=True)

    pred = predictor.predict(
        "Brazil", "Germany",
        elo_a=2100, elo_b=1990,
        form_a=0.8, form_b=0.6
    )

    print("\n=== ML Prediction: Brazil vs Germany ===")
    for k, v in pred.items():
        print(f"  {k}: {v}")

    fi = predictor.feature_importance()
    if fi:
        print("\n=== Feature Importance ===")
        for feat, imp in list(fi.items())[:5]:
            print(f"  {feat}: {imp:.4f}")
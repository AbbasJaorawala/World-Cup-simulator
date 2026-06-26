import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    print("This is a Streamlit app. Run it with:")
    print("  streamlit run worldcup_simulator/dashboard/app.py")
    sys.exit(0)

from pipeline.loader import DataLoader
from tournament.simulator import TournamentSimulator, _FALLBACK_GROUPS


st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon="WC",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.65rem; }
    div[data-testid="stDataFrame"] { border: 1px solid rgba(49, 51, 63, .12); }
    .small-note { color: #5f6368; font-size: .88rem; }

    @media (max-width: 768px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        .stButton>button, .stButton>div>button { width: 100% !important; }
        .stSidebar .css-1d391kg, .css-1d391kg { width: 100% !important; }
        [data-testid="stMetric"]:not(.stMetric) { min-width: 0 !important; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
        .css-1e5imcs { flex-direction: column !important; }
        .css-1n76uvr { width: 100% !important; }
        .css-yk16xz { width: 100% !important; }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div { width: 100% !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fallback_groups() -> dict:
    return {group: list(teams) for group, teams in _FALLBACK_GROUPS.items()}


def team_list(groups: dict) -> list:
    return [team for teams in groups.values() for team in teams]


@st.cache_data(show_spinner=False)
def load_cached_ratings() -> dict:
    try:
        return DataLoader().load_teams()
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def load_live_simulator(n_simulations: int) -> TournamentSimulator:
    return TournamentSimulator(n_simulations=n_simulations, use_live_data=True)


def build_base_simulator(n_simulations: int, data_mode: str) -> tuple[TournamentSimulator, str]:
    if data_mode == "Live APIs":
        try:
            return load_live_simulator(n_simulations), "Live APIs"
        except Exception as exc:
            st.sidebar.warning(f"Live data unavailable. Using cached ratings. ({exc})")

    groups = fallback_groups()
    cached = load_cached_ratings()
    ratings = {team: float(cached.get(team, 1500.0)) for team in team_list(groups)}
    sim = TournamentSimulator(
        groups=groups,
        elo_ratings=ratings,
        n_simulations=n_simulations,
        use_live_data=False,
    )
    return sim, "Cached ratings"


def apply_rating_overrides(base_sim: TournamentSimulator, overrides: dict, n_simulations: int) -> TournamentSimulator:
    ratings = {
        team: float(base_sim.elo_engine.get_rating(team))
        for team in team_list(base_sim.groups)
    }
    ratings.update(overrides)
    return TournamentSimulator(
        groups=base_sim.groups,
        elo_ratings=ratings,
        n_simulations=n_simulations,
        use_live_data=False,
    )


def standings_frame(result: dict) -> pd.DataFrame:
    rows = []
    best_thirds = set(result.get("best_thirds", []))
    for group, table in result.get("group_stage", {}).items():
        ranked = table.get("_ranked", [])
        for position, team in enumerate(ranked, start=1):
            stats = table[team]
            if position <= 2:
                status = "Auto"
            elif team in best_thirds:
                status = "Best third"
            else:
                status = ""

            rows.append(
                {
                    "Group": group,
                    "Pos": position,
                    "Team": team,
                    "P": stats["played"],
                    "W": stats["wins"],
                    "D": stats["draws"],
                    "L": stats["losses"],
                    "GF": stats["gf"],
                    "GA": stats["ga"],
                    "GD": stats["gd"],
                    "Pts": stats["points"],
                    "Qualifies": status,
                }
            )
    return pd.DataFrame(rows)


def knockout_frame(result: dict) -> pd.DataFrame:
    rows = []
    for match in result.get("knockout_matches", []):
        rows.append(
            {
                "Round": match["round"],
                "Team A": match["team_a"],
                "Score": f"{match['goals_a']}-{match['goals_b']}",
                "Team B": match["team_b"],
                "Winner": match["winner"],
                "Pens": "Yes" if match["went_to_penalties"] else "",
                "Upset": "Yes" if match["was_upset"] else "",
                "ELO A": round(match["elo_a"], 0),
                "ELO B": round(match["elo_b"], 0),
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_frame(results: dict) -> pd.DataFrame:
    rows = []
    for team, stats in results.items():
        rows.append(
            {
                "Team": team,
                "Win probability": stats["win_probability"],
                "Semi-final": stats["semi_final_rate"],
                "Quarter-final": stats["quarter_final_rate"],
                "Round of 16": stats["round_of_16_rate"],
                "Round of 32": stats.get("round_of_32_rate", 0),
                "ELO": round(stats["elo_rating"], 0),
                "Titles": stats["total_wins"],
            }
        )
    return pd.DataFrame(rows)


def ratings_frame(sim: TournamentSimulator) -> pd.DataFrame:
    rows = []
    for group, teams in sim.groups.items():
        for team in teams:
            rows.append(
                {
                    "Group": group,
                    "Team": team,
                    "ELO": round(sim.elo_engine.get_rating(team), 0),
                }
            )
    return pd.DataFrame(rows).sort_values(["ELO", "Team"], ascending=[False, True])


def group_config_frame(groups: dict) -> pd.DataFrame:
    rows = []
    for group, teams in groups.items():
        for slot, team in enumerate(teams, start=1):
            rows.append({"Group": group, "Slot": slot, "Team": team})
    return pd.DataFrame(rows)


def seed_random(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)


def run_single(sim: TournamentSimulator, seed: int | None) -> dict:
    seed_random(seed)
    return sim.run_full_simulation()


def run_monte_carlo(sim: TournamentSimulator, seed: int | None) -> dict:
    seed_random(seed)
    return sim.run_monte_carlo()


st.title("World Cup 2026 Simulator")

with st.sidebar:
    st.header("Simulation")
    n_simulations = st.slider(
        "Monte Carlo runs",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
    )
    data_mode = st.radio(
        "Data source",
        ["Cached ratings", "Live APIs"],
        help="Live mode can use Streamlit secrets or environment variables for API keys.",
    )
    seed_enabled = st.checkbox("Use random seed", value=False)
    seed_value = (
        st.number_input("Seed", min_value=0, max_value=999999, value=2026, step=1)
        if seed_enabled
        else None
    )

base_sim, active_source = build_base_simulator(n_simulations, data_mode)
teams = sorted(team_list(base_sim.groups))
base_ratings = {team: base_sim.elo_engine.get_rating(team) for team in teams}

with st.sidebar:
    st.caption(f"Active source: {active_source}")
    with st.expander("Adjust team ratings"):
        selected_teams = st.multiselect("Teams", teams, default=[])
        rating_overrides = {}
        for team in selected_teams:
            rating_overrides[team] = float(
                st.slider(
                    team,
                    min_value=1200,
                    max_value=2300,
                    value=int(round(base_ratings.get(team, 1500))),
                    step=5,
                )
            )

sim = apply_rating_overrides(base_sim, rating_overrides, n_simulations)
seed = int(seed_value) if seed_value is not None else None
config_signature = (
    active_source,
    n_simulations,
    seed,
    tuple(sorted((team, int(rating)) for team, rating in rating_overrides.items())),
)

if st.session_state.get("config_signature") != config_signature:
    st.session_state.single_result = run_single(sim, seed)
    st.session_state.mc_results = None
    st.session_state.match_prediction = None
    st.session_state.config_signature = config_signature
elif "single_result" not in st.session_state:
    st.session_state.single_result = run_single(sim, seed)
if "mc_results" not in st.session_state:
    st.session_state.mc_results = None
if "match_prediction" not in st.session_state:
    st.session_state.match_prediction = None

controls = st.columns([1, 1, 2])
with controls[0]:
    if st.button("Run tournament", type="primary", width="stretch"):
        st.session_state.single_result = run_single(sim, seed)
with controls[1]:
    if st.button("Run Monte Carlo", width="stretch"):
        with st.spinner(f"Running {n_simulations:,} tournaments..."):
            st.session_state.mc_results = run_monte_carlo(sim, seed)
with controls[2]:
    st.write("")

single_result = st.session_state.single_result

metric_cols = st.columns(4)
metric_cols[0].metric("Champion", single_result.get("winner", "N/A"))
metric_cols[1].metric("Runner-up", single_result.get("runner_up", "N/A"))
metric_cols[2].metric("Third place", single_result.get("third_place", "N/A"))
metric_cols[3].metric("Best thirds", len(single_result.get("best_thirds", [])))

tabs = st.tabs(
    [
        "Overview",
        "Groups",
        "Knockout",
        "Monte Carlo",
        "Match Predictor",
        "Ratings",
        "Deploy",
    ]
)

with tabs[0]:
    standings_df = standings_frame(single_result)
    knockout_df = knockout_frame(single_result)
    left, right = st.columns([1.15, 1])
    with left:
        qualifiers = standings_df[standings_df["Qualifies"] != ""]
        fig = px.bar(
            qualifiers,
            x="Team",
            y="Pts",
            color="Qualifies",
            facet_col="Group",
            facet_col_wrap=4,
            title="Qualified teams by group-stage points",
            height=520,
        )
        fig.update_layout(showlegend=True, margin=dict(l=20, r=20, t=60, b=20))
        fig.update_xaxes(matches=None, showticklabels=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Final path")
        final_rounds = knockout_df[knockout_df["Round"].isin(["Semi Finals", "Final", "Third Place Play-off"])]
        st.dataframe(final_rounds, width="stretch", hide_index=True)

with tabs[1]:
    standings_df = standings_frame(single_result)
    st.dataframe(standings_df, width="stretch", hide_index=True)
    st.download_button(
        "Download standings CSV",
        data=standings_df.to_csv(index=False).encode("utf-8"),
        file_name="worldcup_group_standings.csv",
        mime="text/csv",
    )

with tabs[2]:
    knockout_df = knockout_frame(single_result)
    round_order = [
        "Round of 32",
        "Round of 16",
        "Quarter Finals",
        "Semi Finals",
        "Final",
        "Third Place Play-off",
    ]
    selected_rounds = st.multiselect("Rounds", round_order, default=round_order)
    filtered = knockout_df[knockout_df["Round"].isin(selected_rounds)]
    st.dataframe(filtered, width="stretch", hide_index=True)
    upset_count = int((knockout_df["Upset"] == "Yes").sum()) if not knockout_df.empty else 0
    pens_count = int((knockout_df["Pens"] == "Yes").sum()) if not knockout_df.empty else 0
    c1, c2 = st.columns(2)
    c1.metric("Upsets", upset_count)
    c2.metric("Penalty shootouts", pens_count)

with tabs[3]:
    mc_results = st.session_state.mc_results
    if mc_results:
        mc_df = monte_carlo_frame(mc_results)
        top = mc_df.head(16)
        fig = px.bar(
            top,
            x="Win probability",
            y="Team",
            color="ELO",
            orientation="h",
            title="Title probability",
            height=560,
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(mc_df, width="stretch", hide_index=True)
        st.download_button(
            "Download Monte Carlo CSV",
            data=mc_df.to_csv(index=False).encode("utf-8"),
            file_name="worldcup_monte_carlo.csv",
            mime="text/csv",
        )
    else:
        st.info("Run Monte Carlo to populate probability tables.")

with tabs[4]:
    left, right = st.columns([1, 1])
    with left:
        team_a = st.selectbox("Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
        available_b = [team for team in teams if team != team_a]
        default_b = available_b.index("France") if "France" in available_b else 0
        team_b = st.selectbox("Team B", available_b, index=default_b)
        if st.button("Predict match", width="stretch"):
            with st.spinner("Comparing methods..."):
                st.session_state.match_prediction = sim.predict_match_all_methods(team_a, team_b)
    with right:
        prediction = st.session_state.match_prediction
        if prediction:
            pred_team_a, pred_team_b = list(prediction["elo_ratings"].keys())
            rows = []
            for method in ["elo", "ml", "monte_carlo"]:
                probs = prediction[method]
                rows.extend(
                    [
                        {"Method": method.replace("_", " ").title(), "Outcome": f"{pred_team_a} win", "Probability": probs["win"] * 100},
                        {"Method": method.replace("_", " ").title(), "Outcome": "Draw", "Probability": probs["draw"] * 100},
                        {"Method": method.replace("_", " ").title(), "Outcome": f"{pred_team_b} win", "Probability": probs["loss"] * 100},
                    ]
                )
            pred_df = pd.DataFrame(rows)
            fig = px.bar(
                pred_df,
                x="Method",
                y="Probability",
                color="Outcome",
                barmode="group",
                title=prediction["match"],
                range_y=[0, 100],
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pred_df, width="stretch", hide_index=True)

with tabs[5]:
    ratings_df = ratings_frame(sim)
    fig = px.scatter(
        ratings_df,
        x="Group",
        y="ELO",
        color="Group",
        hover_name="Team",
        title="Team ratings by group",
        height=460,
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(ratings_df, width="stretch", hide_index=True)

    st.subheader("Group draw")
    st.dataframe(group_config_frame(sim.groups), width="stretch", hide_index=True)

with tabs[6]:
    st.subheader("Streamlit Community Cloud")
    st.code(
        """Main file path:
worldcup_simulator/dashboard/app.py

Required secrets, only if Live APIs need keys:
FOOTBALL_DATA_API_KEY = "..."
RAPIDAPI_KEY = "..."
SPORTAPI_RAPIDAPI_HOST = "sportapi7.p.rapidapi.com"
""",
        language="toml",
    )
    st.markdown(
        """
        1. Push this project to GitHub.
        2. Create a new Streamlit Community Cloud app from the repository.
        3. Set the main file path to `worldcup_simulator/dashboard/app.py`.
        4. Add optional API keys in the app secrets panel.
        """
    )

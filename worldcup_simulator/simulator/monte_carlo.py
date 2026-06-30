# simulator/monte_carlo.py — Monte Carlo Simulation Engine

import numpy as np
import logging
import hashlib
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

    # Realistic player names by country/region for synthetic rosters
    PLAYER_NAMES = {
        "Brazil": {
            "forwards": ["Vinícius Júnior", "Rodrygo", "Raphinha", "Neymar"],
            "midfielders": ["Bruno Guimarães", "Paquetá", "Casemiro", "Lucas Paquetá"],
            "defenders": ["Marquinhos", "Éder Militão", "Danilo", "Renan Lodi"],
            "goalkeeper": ["Alisson"],
        },
        "Argentina": {
            "forwards": ["Lautaro Martínez", "Julián Álvarez", "Lionel Messi", "Ángel Correa"],
            "midfielders": ["Alexis Mac Allister", "Enzo Fernández", "Rodrigo De Paul", "Exequiel Palacios"],
            "defenders": ["Nicolás Otamendi", "Nicolás Tagliafico", "Cristian Romero", "Nicolás Otamendi"],
            "goalkeeper": ["Emiliano Martínez"],
        },
        "France": {
            "forwards": ["Kylian Mbappé", "Antoine Griezmann", "Ousmane Dembélé", "Randal Kolo Muani"],
            "midfielders": ["Aurélien Tchouaméni", "Eduardo Camavinga", "Adrien Rabiot", "N’Golo Kanté"],
            "defenders": ["Dayot Upamecano", "Jules Koundé", "Theo Hernandez", "Raphaël Varane"],
            "goalkeeper": ["Hugo Lloris"],
        },
        "Germany": {
            "forwards": ["Harry Kane", "Florian Wirtz", "Jamal Musiala", "Leroy Sané"],
            "midfielders": ["Joshua Kimmich", "İlkay Gündoğan", "Julian Brandt", "Florian Neuhaus"],
            "defenders": ["Antonio Rüdiger", "Nico Schlotterbeck", "David Raum", "Matthias Ginter"],
            "goalkeeper": ["Manuel Neuer"],
        },
        "Spain": {
            "forwards": ["Lamine Yamal", "Nico Williams", "Dani Olmo", "Ferran Torres"],
            "midfielders": ["Rodri", "Pedri", "Gavi", "Martín Zubimendi"],
            "defenders": ["Dani Carvajal", "Aymeric Laporte", "Pau Cubarsí", "Alejandro Grimaldo"],
            "goalkeeper": ["Unai Simón"],
        },
        "England": {
            "forwards": ["Harry Kane", "Bukayo Saka", "Phil Foden", "Cole Palmer"],
            "midfielders": ["Declan Rice", "Jude Bellingham", "Conor Gallagher", "Trent Alexander-Arnold"],
            "defenders": ["Kyle Walker", "John Stones", "Ben Chilwell", "Marc Guehi"],
            "goalkeeper": ["Jordan Pickford"],
        },
        "Italy": {
            "forwards": ["Gianluca Scamacca", "Moise Kean", "Mateo Retegui", "Federico Chiesa"],
            "midfielders": ["Nicolò Barella", "Jorginho", "Sandro Tonali", "Davide Frattesi"],
            "defenders": ["Alessandro Bastoni", "Francesco Acerbi", "Giovanni Di Lorenzo", "Federico Dimarco"],
            "goalkeeper": ["Gianluigi Donnarumma"],
        },
        "Netherlands": {
            "forwards": ["Cody Gakpo", "Memphis Depay", "Donyell Malen", "Wout Weghorst"],
            "midfielders": ["Frenkie de Jong", "Teun Koopmeiners", "Ryan Gravenberch", "Xavi Simons"],
            "defenders": ["Virgil van Dijk", "Matthijs de Ligt", "Nathan Aké", "Stefan de Vrij"],
            "goalkeeper": ["Bart Verbruggen"],
        },
        "Portugal": {
            "forwards": ["Cristiano Ronaldo", "Rafael Leão", "João Félix", "Bernardo Silva"],
            "midfielders": ["Bruno Fernandes", "Vitinha", "Ruben Neves", "João Palhinha"],
            "defenders": ["Rúben Dias", "Nuno Mendes", "João Cancelo", "Pepe"],
            "goalkeeper": ["Diogo Costa"],
        },
        "Belgium": {
            "forwards": ["Romelu Lukaku", "Leandro Trossard", "Loïs Openda", "Jeremy Doku"],
            "midfielders": ["Kevin De Bruyne", "Youri Tielemans", "Amadou Onana", "Orel Mangala"],
            "defenders": ["Wout Faes", "Arthur Theate", "Zeno Debast", "Timothy Castagne"],
            "goalkeeper": ["Thibaut Courtois"],
        },
        "Mexico": {
            "forwards": ["Santiago Giménez", "Henry Martín", "Alexis Vega", "Julián Quiñones"],
            "midfielders": ["Edson Álvarez", "Luis Chávez", "Roberto Alvarado", "Erick Sánchez"],
            "defenders": ["Jorge Sánchez", "César Montes", "Johan Vásquez", "Néstor Araujo"],
            "goalkeeper": ["Guillermo Ochoa"],
        },
        "United States": {
            "forwards": ["Christian Pulisic", "Tim Weah", "Ricardo Pepi", "Haji Wright"],
            "midfielders": ["Weston McKennie", "Yunus Musah", "Gio Reyna", "Tyler Adams"],
            "defenders": ["Antonee Robinson", "Chris Richards", "Tim Ream", "Sergiño Dest"],
            "goalkeeper": ["Matt Turner"],
        },
        "Sweden": {
            "forwards": ["Alexander Isak", "Viktor Gyökeres", "Anthony Elanga", "Dejan Kulusevski"],
            "midfielders": ["Jesper Karlsson", "Emil Forsberg", "Kristoffer Olsson", "Sebastian Nanasi"],
            "defenders": ["Victor Lindelöf", "Emil Krafth", "Gabriel Gudmundsson", "Linus Wahlqvist"],
            "goalkeeper": ["Robin Olsen"],
        },
        "Ecuador": {
            "forwards": ["Enner Valencia", "Kevin Rodríguez", "Michael Estrada", "Romario Ibarra"],
            "midfielders": ["Carlos Gruezo", "Alan Franco", "Pedro Vite", "Joao Ortiz"],
            "defenders": ["Piero Hincapié", "Félix Torres", "Robert Arboleda", "Diego Palacios"],
            "goalkeeper": ["Hernán Galíndez"],
        },
        "South Africa": {
            "forwards": ["Lyle Foster", "Percy Tau", "Themba Zwane", "Kgaogelo Sekgota"],
            "midfielders": ["Teboho Mokoena", "Mihlali Mayambela", "Sphephelo Sithole", "Aubrey Modiba"],
            "defenders": ["Sydney Mobbie", "Rushine de Reuck", "Nkosinathi Sibisi", "Siyabonga Ngezana"],
            "goalkeeper": ["Ronwen Williams"],
        },
        "Morocco": {
            "forwards": ["Youssef En-Nesyri", "Hakim Ziyech", "Sofiane Boufal", "Amine Adli"],
            "midfielders": ["Amine Harit", "Bilal El Khannouss", "Azzedine Ounahi", "Selim Amallah"],
            "defenders": ["Nayef Aguerd", "Romain Saïss", "Jawad El Yamiq", "Noussair Mazraoui"],
            "goalkeeper": ["Yassine Bounou"],
        },
        "Canada": {
            "forwards": ["Jonathan David", "Cyle Larin", "Tani Oluwaseyi", "Theo Corbeanu"],
            "midfielders": ["Stephen Eustáquio", "Ismaël Koné", "Mathieu Choinière", "Ali Ahmed"],
            "defenders": ["Alistair Johnston", "Richie Laryea", "Derek Cornelius", "Moïse Bombito"],
            "goalkeeper": ["Dayne St. Clair"],
        },
    }

    def __init__(self, elo_engine: ELOEngine, n_simulations: int = SIMULATION_RUNS, squads: dict = None):
        self.elo = elo_engine
        self.n_simulations = n_simulations
        self._player_rosters = {}
        self.api_squads = squads or {}
        # Tracks every synthetic full name already assigned to any team, so
        # two different countries sharing the same region's name pool never
        # end up with an identical full name (which would otherwise make a
        # forward on one team look identical to a goalkeeper on another).
        self._used_synthetic_names = set()
        logger.info(f"Monte Carlo Simulator initialized: {n_simulations:,} runs")

    def _build_team_roster(self, team: str) -> list[dict]:
        if team in self._player_rosters:
            return self._player_rosters[team]

        roster = []
        
        # First, try to use real players from API squads
        if team in self.api_squads:
            api_squad = self.api_squads[team]
            if isinstance(api_squad, list) and api_squad:
                # API squads are [{"name": player_name, "position": position}, ...]
                for player_dict in api_squad:
                    player_name = player_dict.get("name") if isinstance(player_dict, dict) else player_dict
                    position = player_dict.get("position", "Unknown") if isinstance(player_dict, dict) else "Unknown"
                    
                    # Normalize position to our roles
                    if position in ("GK", "Goalkeeper"):
                        role = "Goalkeeper"
                    elif position in ("DEF", "Defender", "CB", "LB", "RB", "RWB", "LWB"):
                        role = "Defender"
                    elif position in ("MID", "Midfielder", "CM", "CAM", "CDM", "LM", "RM"):
                        role = "Midfielder"
                    elif position in ("FWD", "Forward", "ST", "CF", "LW", "RW"):
                        role = "Forward"
                    else:
                        role = "Midfielder"  # Default
                    
                    if player_name:
                        roster.append({"name": player_name, "role": role})
                
                # If we got enough players, use this roster
                if len(roster) >= 10:
                    self._player_rosters[team] = roster
                    return roster
        
        # Fall back to hardcoded names or synthetic names
        names = self.PLAYER_NAMES.get(team)
        
        if names:
            # Use hardcoded player names from the mapping
            for player_name in names.get("forwards", []):
                roster.append({"name": player_name, "role": "Forward"})
            for player_name in names.get("midfielders", []):
                roster.append({"name": player_name, "role": "Midfielder"})
            for player_name in names.get("defenders", []):
                roster.append({"name": player_name, "role": "Defender"})
            for player_name in names.get("goalkeeper", []):
                roster.append({"name": player_name, "role": "Goalkeeper"})
        else:
            # Fallback: generate realistic-sounding, per-team-unique names for
            # countries not in PLAYER_NAMES and without API squad data.
            #
            # Earlier version 1: one fixed list of 12 names shared by every
            # unmapped country — different countries' goals all landed on
            # the same literal name strings.
            #
            # Earlier version 2: fixed that by shuffling per-team from one
            # global pool, but the pool mixed first/last names from many
            # different naming traditions (e.g. Korean "Jin" + West African
            # "Traoré" landing on the same Iranian player), which read as
            # obviously wrong even though the bug it fixed (shared names)
            # was gone.
            #
            # This version draws first/last names from a region-matched
            # pool per team, so synthetic names stay culturally coherent
            # while still being unique per country (deterministic shuffle
            # seeded by team name).
            region_name_pools = {
                "latin_america": (
                    ["Mateo", "Diego", "Santiago", "Nicolás", "Lucas", "Tomás",
                     "Joaquín", "Emiliano", "Agustín", "Bruno", "Gael", "Iker"],
                    ["Silva", "Navarro", "Santos", "Alvarez", "Reyes", "Vargas",
                     "Castro", "Romero", "Flores", "Medina", "Ortega", "Cabrera"],
                ),
                "western_europe": (
                    ["Liam", "Noah", "Jules", "Leon", "Felix", "Lucas",
                     "Theo", "Max", "Elias", "Hugo", "Mathis", "Tobias"],
                    ["Carter", "Bennett", "Laurent", "Voss", "Becker", "Martin",
                     "Dubois", "Fischer", "Moreau", "Schmidt", "Keller", "Roux"],
                ),
                "balkans_eastern_europe": (
                    ["Ivan", "Pavel", "Marko", "Luka", "Nikola", "Aleksandar",
                     "Filip", "Stefan", "Bojan", "Vuk", "Damir", "Tomislav"],
                    ["Nowak", "Popescu", "Horvat", "Novak", "Petrov", "Kovač",
                     "Dimitrov", "Stanković", "Jovanović", "Marić", "Babić", "Vidić"],
                ),
                "middle_east": (
                    ["Omar", "Sami", "Karim", "Yusuf", "Ali", "Hamza",
                     "Tariq", "Rashid", "Zayd", "Faisal", "Adel", "Bilal"],
                    ["Hassan", "Rahman", "Khalil", "Saleh", "Nasser", "Farouk",
                     "Aziz", "Hadi", "Mansour", "Qureshi", "Haddad", "Sultan"],
                ),
                "sub_saharan_africa": (
                    ["Kwame", "Amadou", "Kofi", "Chidi", "Sekou", "Emeka",
                     "Abdoulaye", "Yaw", "Ibrahima", "Ousmane", "Kojo", "Moussa"],
                    ["Diallo", "Mensah", "Traoré", "Okafor", "Keita", "Diop",
                     "Boateng", "Camara", "Toure", "Sangare", "Adeyemi", "Conde"],
                ),
                "east_asia": (
                    ["Jin", "Hiroshi", "Wei", "Min-jun", "Kenji", "Yusuke",
                     "Tae-yang", "Haruto", "Sho", "Jun", "Ryo", "Daiki"],
                    ["Park", "Sato", "Kim", "Watanabe", "Tanaka", "Lee",
                     "Suzuki", "Yamamoto", "Choi", "Nakamura", "Kobayashi", "Jung"],
                ),
                "south_asia": (
                    ["Arjun", "Ravi", "Rohan", "Aarav", "Vikram", "Karan",
                     "Aditya", "Rahul", "Sanjay", "Aryan", "Dev", "Ishaan"],
                    ["Singh", "Kumar", "Sharma", "Patel", "Gupta", "Khan",
                     "Verma", "Reddy", "Nair", "Chowdhury", "Malik", "Joshi"],
                ),
                "oceania": (
                    ["Liam", "Jack", "Noah", "Cooper", "Mason", "Ethan",
                     "Lachlan", "Hayden", "Riley", "Tyler", "Connor", "Blake"],
                    ["Mitchell", "Anderson", "Walker", "Wright", "Robinson", "Clarke",
                     "Bishop", "Fletcher", "Hayes", "Marsh", "Pratt", "Sinclair"],
                ),
            }

            # Map each unmapped country to the closest-matching region pool.
            country_region = {
                "algeria": "middle_east", "austria": "western_europe",
                "bosnia and herzegovina": "balkans_eastern_europe", "cape verde": "sub_saharan_africa",
                "colombia": "latin_america", "croatia": "balkans_eastern_europe",
                "curacao": "latin_america", "czechia": "balkans_eastern_europe",
                "dr congo": "sub_saharan_africa", "egypt": "middle_east",
                "ghana": "sub_saharan_africa", "haiti": "latin_america",
                "iran": "middle_east", "iraq": "middle_east",
                "ivory coast": "sub_saharan_africa", "japan": "east_asia",
                "jordan": "middle_east", "new zealand": "oceania",
                "norway": "western_europe", "panama": "latin_america",
                "paraguay": "latin_america", "qatar": "middle_east",
                "saudi arabia": "middle_east", "scotland": "western_europe",
                "senegal": "sub_saharan_africa", "south korea": "east_asia",
                "switzerland": "western_europe", "tunisia": "middle_east",
                "turkey": "middle_east", "uruguay": "latin_america",
                "uzbekistan": "south_asia", "australia": "oceania",
                "south africa": "sub_saharan_africa", "morocco": "middle_east",
                "nigeria": "sub_saharan_africa", "cameroon": "sub_saharan_africa",
                "venezuela": "latin_america", "bolivia": "latin_america",
                "costa rica": "latin_america", "chile": "latin_america",
                "ecuador": "latin_america", "peru": "latin_america",
                "poland": "balkans_eastern_europe", "serbia": "balkans_eastern_europe",
                "ukraine": "balkans_eastern_europe", "denmark": "western_europe",
                "wales": "western_europe", "slovakia": "balkans_eastern_europe",
            }

            region = country_region.get(team.lower(), "western_europe")
            first_names, last_names = region_name_pools[region]

            # Bug fix history:
            # v1 — one shared list of 12 names for every unmapped country
            #      (different countries' goals landed on identical names).
            # v2 — per-team shuffle of one global mixed pool, fixed the
            #      sharing but produced culturally incoherent names (e.g. a
            #      Korean first name + West African surname for Iran).
            # v3 — region-matched pools, but only zipped 12 first x 12 last
            #      names per team (12 combos), so countries sharing a region
            #      (e.g. 9 Middle East teams) frequently collided on the
            #      exact same full name — a forward on one team could be
            #      named identically to the goalkeeper on another, which
            #      looked like "the goalkeeper scored" even though the
            #      scorer-selection logic was correctly excluding keepers.
            #
            # v4 (current): sample from the region's full first x last
            # cross-product, skipping any full name already assigned to
            # ANY other team this session, so names are unique across the
            # whole tournament, not just within one team.
            all_pairs = [(f, l) for f in first_names for l in last_names]
            seed = int(hashlib.sha256(team.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            shuffled_order = rng.permutation(len(all_pairs))

            fallback_names = []
            for idx in shuffled_order:
                candidate = f"{all_pairs[idx][0]} {all_pairs[idx][1]}"
                if candidate not in self._used_synthetic_names:
                    fallback_names.append(candidate)
                    self._used_synthetic_names.add(candidate)
                if len(fallback_names) == 12:
                    break

            # If a region's pool is exhausted (e.g. many teams sharing a
            # small region with no API/hardcoded data), disambiguate with a
            # deterministic suffix rather than reusing another team's name.
            if len(fallback_names) < 12:
                for idx in shuffled_order:
                    if len(fallback_names) == 12:
                        break
                    base = f"{all_pairs[idx][0]} {all_pairs[idx][1]}"
                    candidate = f"{base} ({team[:3].upper()})"
                    if candidate not in self._used_synthetic_names:
                        fallback_names.append(candidate)
                        self._used_synthetic_names.add(candidate)

            for name in fallback_names[:4]:
                roster.append({"name": name, "role": "Forward"})
            for name in fallback_names[4:8]:
                roster.append({"name": name, "role": "Midfielder"})
            for name in fallback_names[8:11]:
                roster.append({"name": name, "role": "Defender"})
            roster.append({"name": fallback_names[11], "role": "Goalkeeper"})

        self._player_rosters[team] = roster
        return roster

    def _choose_scorer(self, roster: list[dict], goal_counts: dict = None) -> str:
        """
        Pick a scorer for one goal event.

        Base weights favour attacking players, with diminishing returns
        based on goals this player has already scored in this match —
        a hat-trick is possible but increasingly unlikely beyond that,
        rather than the same player being equally likely for every goal
        in a high-scoring match.
        """
        goal_counts = goal_counts or {}
        names = [p["name"] for p in roster]
        role_weight = {"Forward": 3, "Midfielder": 2, "Defender": 1, "Goalkeeper": 0.0}
        base_weights = np.array(
            [role_weight.get(p["role"], 1) for p in roster],
            dtype=float,
        )
        fatigue = np.array([0.55 ** goal_counts.get(n, 0) for n in names])
        weights = base_weights * fatigue
        weights = weights / weights.sum()
        return np.random.choice(names, p=weights)

    def _choose_assist(self, roster: list[dict], scorer: str) -> str:
        """
        Pick an assist provider for a goal.

        Bug fix: this previously checked `"Forward" in p` where `p` was the
        player's NAME string (e.g. "Jordan Pickford"), not their role — that
        substring check is never true, so every player including goalkeepers
        got equal weight. Now uses the actual role field, and excludes
        goalkeepers entirely (real-world goalkeeper assists are vanishingly
        rare and not worth modeling here).
        """
        options = [p for p in roster if p["name"] != scorer and p["role"] != "Goalkeeper"]
        if not options:
            return ""
        weights = [2 if p["role"] in ("Forward", "Midfielder") else 1 for p in options]
        names = [p["name"] for p in options]
        return np.random.choice(names, p=np.array(weights) / sum(weights))

    def _simulate_goal_events(self, team: str, goals: int, opponent: str, round_name: str) -> list[dict]:
        roster = self._build_team_roster(team)
        minutes = sorted(np.random.randint(1, 91, size=max(goals, 0)).tolist())
        events = []
        goal_counts_this_match = defaultdict(int)
        for minute in minutes:
            scorer = self._choose_scorer(roster, goal_counts_this_match)
            goal_counts_this_match[scorer] += 1
            assist = self._choose_assist(roster, scorer)
            events.append({
                "round": round_name,
                "team": team,
                "opponent": opponent,
                "minute": int(minute),
                "player": scorer,
                "assist": assist,
                "event": "goal",
            })
        return events

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
            pen_winner, _ = self._simulate_penalty_shootout(team_a, team_b, pred["win_prob"])
            return pen_winner, goals_a, goals_b

        if goals_a > goals_b:
            return team_a, goals_a, goals_b
        elif goals_b > goals_a:
            return team_b, goals_a, goals_b
        else:
            return "draw", goals_a, goals_b

    def simulate_match_with_events(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = True,
        knockout: bool = False,
        round_name: str = "Knockout"
    ) -> dict:
        """
        Simulate a single match and return detailed event data.

        Returns:
            {winner, goals_a, goals_b, events, penalty_shootout}
        """
        pred = self.elo.predict_match(team_a, team_b, neutral)
        goals_a = self._simulate_goals(pred["expected_goals_a"])
        goals_b = self._simulate_goals(pred["expected_goals_b"])

        events = []
        events.extend(self._simulate_goal_events(team_a, goals_a, team_b, round_name))
        events.extend(self._simulate_goal_events(team_b, goals_b, team_a, round_name))

        penalty_shootout = []
        winner = "draw"
        if knockout and goals_a == goals_b:
            winner, penalty_shootout = self._simulate_penalty_shootout(team_a, team_b, pred["win_prob"])
            events.extend(penalty_shootout)
        elif goals_a > goals_b:
            winner = team_a
        elif goals_b > goals_a:
            winner = team_b

        return {
            "winner": winner,
            "goals_a": goals_a,
            "goals_b": goals_b,
            "events": events,
            "penalty_shootout": penalty_shootout,
        }

    def _simulate_penalty_shootout(self, team_a: str, team_b: str, win_prob_a: float) -> tuple[str, list[dict]]:
        roster_a = self._build_team_roster(team_a)
        roster_b = self._build_team_roster(team_b)
        kick_order = []
        results = []
        score_a = score_b = 0

        prob_a = max(0.3, min(0.85, 0.55 + (win_prob_a - 0.5) * 0.2))
        prob_b = max(0.3, min(0.85, 0.55 - (win_prob_a - 0.5) * 0.2))

        takers_a = [p["name"] for p in roster_a if p["role"] in ("Forward", "Midfielder")][:5]
        takers_b = [p["name"] for p in roster_b if p["role"] in ("Forward", "Midfielder")][:5]
        keepers = {
            team_a: next(p["name"] for p in roster_a if p["role"] == "Goalkeeper"),
            team_b: next(p["name"] for p in roster_b if p["role"] == "Goalkeeper"),
        }

        max_rounds = 5
        for i in range(max_rounds):
            kick_order.append((team_a, takers_a[i] if i < len(takers_a) else f"{team_a} Penalty {i+1}"))
            kick_order.append((team_b, takers_b[i] if i < len(takers_b) else f"{team_b} Penalty {i+1}"))

        for idx, (team, player) in enumerate(kick_order, start=1):
            prob = prob_a if team == team_a else prob_b
            scored = np.random.random() < prob
            if scored:
                if team == team_a:
                    score_a += 1
                else:
                    score_b += 1
                result = "scored"
            else:
                result = np.random.choice(["saved", "missed"], p=[0.6, 0.4])

            results.append({
                "kick": idx,
                "team": team,
                "player": player,
                "keeper": keepers[team_b if team == team_a else team_a],
                "result": result,
                "score": f"{score_a}-{score_b}",
            })

            if idx >= 10:
                if score_a != score_b:
                    break
            else:
                remaining_a = max_rounds - (idx // 2 + (1 if idx % 2 == 0 and team == team_a else 0))
                remaining_b = max_rounds - ((idx + 1) // 2)
                if score_a > score_b + remaining_b or score_b > score_a + remaining_a:
                    break

        winner = team_a if score_a > score_b else team_b
        return winner, results

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
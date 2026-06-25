# main.py — Entry point for the World Cup Simulator
# Run this from the worldcup_simulator/ folder:
#   python main.py

import os
import sys

# Ensure the project root is always on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tournament.simulator import TournamentSimulator


def main():
    print("=" * 55)
    print("       🏆  FIFA WORLD CUP 2026 SIMULATOR  🏆")
    print("=" * 55)

    # ── Initialize simulator (1000 runs for quick test, use 10000 for full) ──
    # use_live_data=True will try the football-data.org API for the group
    # draw and World Football Elo ratings for team strength. If live group
    # data is unavailable/incomplete, it falls back to the built-in 2026 draw.
    sim = TournamentSimulator(n_simulations=1000, use_live_data=True)

    # ── Single tournament simulation ──────────────────────────────────────────
    print("\n📋  Running single tournament simulation...")
    result = sim.run_full_simulation()
    print(f"\n🥇  Winner: {result['winner']}")

    print("\n📊  Group Stage Results:")
    if not result["group_stage"]:
        print("   ⚠️  No group stage data available.")
    else:
        # Don't assume "A" exists — live API data may use different keys,
        # or the draw may not be published yet. Just show the first group
        # alphabetically, whatever it's actually called.
        group_name = sorted(result["group_stage"].keys())
        for group_name in group_name:
            print(f"   (showing Group {group_name})")
            for team, stats in result["group_stage"][group_name].items():
                if team.startswith("_"):
                    continue
                print(f"   {team:15s}: {stats['points']}pts | "
                  f"GD:{stats['gd']:+d} | "
                  f"W{stats['wins']} D{stats['draws']} L{stats['losses']}")

    # ── Knockout bracket summary ──────────────────────────────────────────────
    # Print a full, nicely formatted bracket (includes every knockout match)
    print(sim.knockout_stage.format_bracket(result["knockout"]))
    

    # ── Raw flat match list (every knockout match, in play order) ────────────
    # Same data as the formatted bracket above, but as plain dicts —
    # useful for feeding the dashboard, CSV export, or SQLite, instead of
    # re-parsing the printed string.
    print(f"\n📂  Raw knockout match data ({len(result['knockout_matches'])} matches):")
    for m in result["knockout_matches"]:
        pen = " (pens)" if m["went_to_penalties"] else ""
        upset = " 🚨" if m["was_upset"] else ""
        print(f"   [{m['round']:22s}] {m['team_a']:14s} {m['goals_a']}-{m['goals_b']} "
              f"{m['team_b']:14s} → {m['winner']}{pen}{upset}")

    

    if result.get("best_thirds"):
        print(f"\n🥉  Best 3rd-place qualifiers: {', '.join(result['best_thirds'])}")
    # Runner-up and third place are included in the formatted bracket output

    # ── Match prediction comparison ───────────────────────────────────────────
    print("\n🔮  Match Prediction Comparison: Brazil vs France")
    comp = sim.predict_match_all_methods("Brazil", "France")
    print(f"   {'Method':12s}  {'Win':>6}  {'Draw':>6}  {'Loss':>6}")
    print(f"   {'-'*35}")
    for method in ["elo", "ml", "monte_carlo"]:
        p = comp[method]
        print(f"   {method:12s}  {p['win']:>5.1%}  {p['draw']:>5.1%}  {p['loss']:>5.1%}")

    # ── ELO Rankings ─────────────────────────────────────────────────────────
    #participants = sorted({team for group in sim.groups.values() for team in group})
    #print(f"\n🌍  World Cup participating teams ({len(participants)}):")
    #for team in participants:
    #    print(f"   {team}")

    print("\n🌍  Top 20 ELO Rankings:")
    for r in sim.get_team_elo_rankings()[:20]:
        print(f"   #{r['rank']:2d}  {r['team']:15s}  ELO: {r['elo']:.0f}")

    print("\n✅  Done! Run the dashboard with:")
    print("    streamlit run dashboard/app.py")
    print("=" * 55)



if __name__ == "__main__":
    main()

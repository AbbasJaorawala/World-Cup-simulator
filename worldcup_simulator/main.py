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
    sim = TournamentSimulator(n_simulations=1000)

    # ── Single tournament simulation ──────────────────────────────────────────
    print("\n📋  Running single tournament simulation...")
    result = sim.run_full_simulation()
    print(f"\n🥇  Winner: {result['winner']}")

    print("\n📊  Group Stage Results (Group A):")
    for team, stats in result["group_stage"]["A"].items():
        print(f"   {team:15s}: {stats['points']}pts | "
              f"GD:{stats['gd']:+d} | "
              f"W{stats['wins']} D{stats['draws']} L{stats['losses']}")

    # ── Knockout bracket summary ──────────────────────────────────────────────
    print("\n⚔️   Knockout Rounds:")
    for rnd in result["knockout"]["rounds"]:
        print(f"\n  {rnd['name']}:")
        for match in rnd["matches"]:
            pen = " (pens)" if match["was_penalty"] else ""
            print(f"    {match['team_a']:15s} {match['goals_a']} - "
                  f"{match['goals_b']} {match['team_b']:15s}"
                  f"  →  {match['winner']}{pen}")

    # ── Match prediction comparison ───────────────────────────────────────────
    print("\n🔮  Match Prediction Comparison: Brazil vs France")
    comp = sim.predict_match_all_methods("Brazil", "France")
    print(f"   {'Method':12s}  {'Win':>6}  {'Draw':>6}  {'Loss':>6}")
    print(f"   {'-'*35}")
    for method in ["elo", "ml", "monte_carlo"]:
        p = comp[method]
        print(f"   {method:12s}  {p['win']:>5.1%}  {p['draw']:>5.1%}  {p['loss']:>5.1%}")

    # ── ELO Rankings ─────────────────────────────────────────────────────────
    print("\n🌍  Top 10 ELO Rankings:")
    for r in sim.get_team_elo_rankings()[:10]:
        print(f"   #{r['rank']:2d}  {r['team']:15s}  ELO: {r['elo']:.0f}")

    print("\n✅  Done! Run the dashboard with:")
    print("    streamlit run dashboard/app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
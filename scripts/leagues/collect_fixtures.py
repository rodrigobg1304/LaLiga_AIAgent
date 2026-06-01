"""
Collect upcoming fixture metadata for a round and store it in the Matches table.

Inserts all matches of the round (played or not) with their scheduled kick-off times.
Scores are NULL for unplayed matches. Safe to re-run — uses ON DUPLICATE KEY UPDATE.

Usage:
    cd scripts/leagues

    # Single league
    python collect_fixtures.py --league 8 --season 77559 --round 31

    # Multiple leagues at once (same round)
    python collect_fixtures.py --league 8 17 23 --season 77559 76986 76457 --round 31

    # Different rounds per league
    python collect_fixtures.py --league 8 17 23 --season 77559 76986 76457 --round 31 32 31

Environment variables required:
    DB_HOST, DB_PORT (default 3306), DB_USER, DB_PASSWORD, DB_DATABASE
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sofascore_client import collect_round_fixtures, league_dict
from db_utils import create_matches_table_if_not_exists, insert_match_metadata


def cmd_collect_fixtures(league_id: int, season_id: int, round_id) -> int:
    league_name = league_dict.get(str(league_id), f"League {league_id}")
    print(f"\n{'─' * 50}")
    print(f"League : {league_name} (id={league_id})")
    print(f"Season : {season_id}  Round : {round_id}")
    print(f"{'─' * 50}")

    fixtures = collect_round_fixtures(league_id, season_id, round_id)

    if not fixtures:
        print("  No matches found for this round.")
        return 0

    inserted = insert_match_metadata(fixtures)

    played   = sum(1 for f in fixtures if f["homeScore"] is not None)
    unplayed = len(fixtures) - played

    print(f"  {len(fixtures)} matches found  ({played} played, {unplayed} upcoming)")
    for f in fixtures:
        score = (f"{f['homeScore']}-{f['awayScore']}"
                 if f["homeScore"] is not None else "vs")
        print(f"  · {f['MatchDateLocal'] or f['MatchDate']}  "
              f"{f['homeTeam']} {score} {f['awayTeam']}")

    print(f"  {inserted} rows affected in Matches table.")
    return inserted


def main():
    parser = argparse.ArgumentParser(
        description="Collect fixture metadata (kick-off times, teams) for an upcoming round.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--league", type=int, nargs="+", required=True,
                        help="League ID(s): 8=LaLiga, 17=Premier League, 23=Serie A")
    parser.add_argument("--season", type=int, nargs="+", required=True,
                        help="Sofascore season ID(s), one per league")
    parser.add_argument("--round", type=str, nargs="+", dest="rounds", required=True,
                        help="Round(s): one per league or a single value applied to all")
    args = parser.parse_args()

    if len(args.season) != len(args.league):
        parser.error("Provide one --season per --league (same order)")

    if len(args.rounds) == 1:
        round_per_league = args.rounds * len(args.league)
    elif len(args.rounds) == len(args.league):
        round_per_league = args.rounds
    else:
        parser.error("--round accepts 1 value (applied to all leagues) or one per --league")

    create_matches_table_if_not_exists()

    total = 0
    for league_id, season_id, round_id in zip(args.league, args.season, round_per_league):
        total += cmd_collect_fixtures(league_id, season_id, round_id)

    print(f"\nTotal rows affected: {total}")


if __name__ == "__main__":
    main()

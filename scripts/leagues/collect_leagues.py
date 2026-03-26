"""
Data collection script for domestic leagues — fetches match statistics and metadata
from Sofascore and inserts them into MySQL (Leagues + Matches tables).

Usage:
    python collect_leagues.py --league 8 --season 77559 --round 27
    python collect_leagues.py --league 8 17 23 --season 77559 76986 76457 --round 27 29 28
    python collect_leagues.py --league 8 --season 77559 --round 9 --match 14083675
    python collect_leagues.py --league 8 --season 77559 --round-start 1 --round-end 38
    python collect_leagues.py --league 8 --list-seasons
    python collect_leagues.py --league 8 --season 77559 --list-rounds

Environment variables required:
    DB_HOST, DB_PORT (default 3306), DB_USER, DB_PASSWORD, DB_DATABASE
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sofascore_client import (
    get_seasons_dict_result,
    get_rounds_list,
    collect_data_statistics_from,
    collect_match_statistics,
    collect_round_metadata,
    league_dict,
)
from db_utils import (
    create_matches_table_if_not_exists,
    insert_statistics,
    insert_match_metadata,
)


# ─────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────

def cmd_list_rounds(league_id: int, season_id: int):
    rounds = get_rounds_list(league_id, season_id)
    league_name = league_dict.get(str(league_id), f"League {league_id}")
    print(f"\nAvailable rounds for {league_name} (id={league_id}), season {season_id}:")
    for r in rounds:
        print(f"  {r}")


def cmd_list_seasons(league_id: int):
    result = get_seasons_dict_result(league_id)
    league_name = league_dict.get(str(league_id), f"League {league_id}")
    print(f"\nAvailable seasons for {league_name} (id={league_id}):")
    for entry in result[str(league_id)]:
        for season_id, year in entry.items():
            print(f"  season_id={season_id}  year={year}")


def cmd_collect_round(league_id: int, season_id: int, round_id) -> int:
    get_seasons_dict_result(league_id)
    league_name = league_dict.get(str(league_id), f"League {league_id}")
    print(f"\n{'─' * 50}")
    print(f"League : {league_name} (id={league_id})")
    print(f"Season : {season_id}  Round : {round_id}")
    print(f"{'─' * 50}")

    df = collect_data_statistics_from(my_league=league_id, my_season=season_id, my_round=round_id)
    metadata = collect_round_metadata(my_league=league_id, my_season=season_id, my_round=round_id)

    if df.empty:
        print("  No completed matches found.")
        return 0

    stats_inserted    = insert_statistics(df)
    metadata_inserted = insert_match_metadata(metadata)
    print(f"  {len(df)} stat rows collected, {stats_inserted} new inserted.")
    print(f"  {len(metadata)} matches collected, {metadata_inserted} new inserted into Matches.")
    return stats_inserted


def cmd_collect_match(league_id: int, season_id: int, round_id, match_id: int) -> int:
    get_seasons_dict_result(league_id)
    league_name = league_dict.get(str(league_id), f"League {league_id}")
    print(f"\n{'─' * 50}")
    print(f"League : {league_name} (id={league_id})")
    print(f"Season : {season_id}  Round : {round_id}  Match : {match_id}")
    print(f"{'─' * 50}")

    df = collect_match_statistics(my_league=league_id, my_season=season_id, my_round=round_id, my_match_id=match_id)
    metadata = collect_round_metadata(my_league=league_id, my_season=season_id, my_round=round_id, my_match_id=match_id)

    if df.empty:
        print("  No data collected.")
        return 0

    stats_inserted    = insert_statistics(df)
    metadata_inserted = insert_match_metadata(metadata)
    print(f"  {len(df)} stat rows collected, {stats_inserted} new inserted.")
    print(f"  {len(metadata)} match metadata collected, {metadata_inserted} new inserted into Matches.")
    return stats_inserted


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect domestic league statistics from Sofascore into MySQL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--league", type=int, nargs="+", required=True,
                        help="League ID(s): 8=LaLiga, 17=Premier League, 23=Serie A")
    parser.add_argument("--season", type=int, nargs="+",
                        help="Sofascore season ID(s), one per league")
    parser.add_argument("--round", type=str, nargs="+", dest="rounds",
                        help="Round(s): one per league or a single value applied to all")
    parser.add_argument("--round-start", type=int, help="First round in range (single league only)")
    parser.add_argument("--round-end",   type=int, help="Last round in range, inclusive (single league only)")
    parser.add_argument("--match",       type=int, help="Collect a single match by ID")
    parser.add_argument("--list-seasons", action="store_true", help="List available seasons and exit")
    parser.add_argument("--list-rounds",  action="store_true", help="List available rounds and exit")
    args = parser.parse_args()

    if args.list_seasons:
        for league_id in args.league:
            cmd_list_seasons(league_id)
        return

    if not args.season:
        parser.error("--season is required unless --list-seasons is used")
    if len(args.season) != len(args.league):
        parser.error("Provide one --season per --league (same order)")

    if args.list_rounds:
        for league_id, season_id in zip(args.league, args.season):
            cmd_list_rounds(league_id, season_id)
        return

    create_matches_table_if_not_exists()

    if args.match:
        if len(args.league) != 1:
            parser.error("--match requires exactly one --league and one --season")
        if not args.rounds or len(args.rounds) != 1:
            parser.error("--match requires exactly one --round")
        cmd_collect_match(args.league[0], args.season[0], args.rounds[0], args.match)
        return

    if args.round_start is not None and args.round_end is not None:
        if len(args.league) != 1:
            parser.error("--round-start/--round-end only works with a single --league")
        total = 0
        for round_id in range(args.round_start, args.round_end + 1):
            total += cmd_collect_round(args.league[0], args.season[0], round_id)
        print(f"\nTotal new stat rows inserted: {total}")
        return

    if not args.rounds:
        parser.error("Provide --round, --round-start/--round-end, or --match")

    if len(args.rounds) == 1:
        round_per_league = args.rounds * len(args.league)
    elif len(args.rounds) == len(args.league):
        round_per_league = args.rounds
    else:
        parser.error("--round accepts 1 value (applied to all leagues) or one per --league")

    total = 0
    for league_id, season_id, round_id in zip(args.league, args.season, round_per_league):
        total += cmd_collect_round(league_id, season_id, round_id)
    print(f"\nTotal new stat rows inserted: {total}")


if __name__ == "__main__":
    main()

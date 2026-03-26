"""
Data collection script for tournaments — fetches match statistics and metadata
from Sofascore and inserts them into MySQL (Leagues + Matches tables).

Types:
  regular   — Continuous historical tournaments (Champions League, Europa League, Conference)
  qualifier — International qualifiers held every 4 years (Euro, World Cup qualifying phases)

Usage:
    python collect_tournaments.py --type regular --league 7 --list-seasons
    python collect_tournaments.py --type regular --league 7 --season 61644 --list-rounds
    python collect_tournaments.py --type regular --league 7 --season 61644 --round 1
    python collect_tournaments.py --type regular --league 7 --season 61644 --round "100/slug/round-of-16"
    python collect_tournaments.py --type qualifier --league 1 --season 56953 --all-rounds
    python collect_tournaments.py --type regular --league 7 --season 61644 --round 1 --match 12345678

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

TOURNAMENT_TYPES = {
    "regular":   "Torneo regular (Champions, Europa League, Conference)",
    "qualifier": "Clasificatoria internacional (Eurocopa, Mundial)",
}


# ─────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────

def cmd_list_rounds(league_id: int, season_id: int):
    rounds = get_rounds_list(league_id, season_id)
    tournament_name = league_dict.get(str(league_id), f"Tournament {league_id}")
    print(f"\nAvailable rounds for {tournament_name} (id={league_id}), season {season_id}:")
    for r in rounds:
        print(f"  {r}")


def cmd_list_seasons(league_id: int):
    result = get_seasons_dict_result(league_id)
    tournament_name = league_dict.get(str(league_id), f"Tournament {league_id}")
    print(f"\nAvailable seasons for {tournament_name} (id={league_id}):")
    for entry in result[str(league_id)]:
        for season_id, year in entry.items():
            print(f"  season_id={season_id}  year={year}")


def cmd_collect_round(tournament_type: str, league_id: int, season_id: int, round_id) -> int:
    get_seasons_dict_result(league_id)
    tournament_name = league_dict.get(str(league_id), f"Tournament {league_id}")
    print(f"\n{'─' * 50}")
    print(f"Type       : {TOURNAMENT_TYPES[tournament_type]}")
    print(f"Tournament : {tournament_name} (id={league_id})")
    print(f"Season     : {season_id}  Round : {round_id}")
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


def cmd_collect_all_rounds(tournament_type: str, league_id: int, season_id: int) -> int:
    get_seasons_dict_result(league_id)
    rounds = get_rounds_list(league_id, season_id)
    tournament_name = league_dict.get(str(league_id), f"Tournament {league_id}")
    print(f"\n{'─' * 50}")
    print(f"Type       : {TOURNAMENT_TYPES[tournament_type]}")
    print(f"Tournament : {tournament_name} (id={league_id})")
    print(f"Season     : {season_id}")
    print(f"Rounds     : {rounds}")
    print(f"{'─' * 50}")

    total = 0
    for round_id in rounds:
        print(f"\nRound {round_id}:")
        df = collect_data_statistics_from(my_league=league_id, my_season=season_id, my_round=round_id)
        metadata = collect_round_metadata(my_league=league_id, my_season=season_id, my_round=round_id)
        if df.empty:
            print("  No completed matches found.")
            continue
        stats_inserted    = insert_statistics(df)
        metadata_inserted = insert_match_metadata(metadata)
        total += stats_inserted
        print(f"  {len(df)} stat rows collected, {stats_inserted} new inserted.")
        print(f"  {len(metadata)} matches collected, {metadata_inserted} new inserted into Matches.")
    return total


def cmd_collect_match(tournament_type: str, league_id: int, season_id: int, round_id, match_id: int) -> int:
    get_seasons_dict_result(league_id)
    tournament_name = league_dict.get(str(league_id), f"Tournament {league_id}")
    print(f"\n{'─' * 50}")
    print(f"Type       : {TOURNAMENT_TYPES[tournament_type]}")
    print(f"Tournament : {tournament_name} (id={league_id})")
    print(f"Season     : {season_id}  Round : {round_id}  Match : {match_id}")
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
        description="Collect tournament statistics from Sofascore into MySQL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--type", required=True, choices=["regular", "qualifier"],
                        help="Tournament type: 'regular' or 'qualifier'")
    parser.add_argument("--league", type=int, required=True,
                        help="Tournament ID in Sofascore")
    parser.add_argument("--season", type=int, help="Sofascore season ID")
    parser.add_argument("--round", type=str, dest="round_id",
                        help="Round: integer or slug string (e.g. '100/slug/round-of-16')")
    parser.add_argument("--all-rounds",   action="store_true", help="Collect all rounds automatically")
    parser.add_argument("--match",        type=int, help="Collect a single match by ID")
    parser.add_argument("--list-seasons", action="store_true", help="List available seasons and exit")
    parser.add_argument("--list-rounds",  action="store_true", help="List available rounds and exit")
    args = parser.parse_args()

    if args.list_seasons:
        cmd_list_seasons(args.league)
        return

    if not args.season:
        parser.error("--season is required unless --list-seasons is used")

    if args.list_rounds:
        cmd_list_rounds(args.league, args.season)
        return

    create_matches_table_if_not_exists()

    if args.all_rounds:
        total = cmd_collect_all_rounds(args.type, args.league, args.season)
        print(f"\nTotal new stat rows inserted: {total}")
        return

    if not args.round_id:
        parser.error("Provide --round, --all-rounds, --list-rounds, or --list-seasons")

    if args.match:
        cmd_collect_match(args.type, args.league, args.season, args.round_id, args.match)
        return

    total = cmd_collect_round(args.type, args.league, args.season, args.round_id)
    print(f"\nTotal new stat rows inserted: {total}")


if __name__ == "__main__":
    main()

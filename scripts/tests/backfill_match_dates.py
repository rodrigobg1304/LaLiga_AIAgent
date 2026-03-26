"""
Backfill script — populates the Matches table for all matches already in the Leagues table.

Reads unique (LeagueId, SeasonId, Round) combinations from Leagues,
calls get_matches_by_round once per combination (no per-match API calls),
and inserts metadata into the Matches table.

Ejecución:
    cd scripts
    python tests/backfill_match_dates.py
    python tests/backfill_match_dates.py --dry-run   # muestra cuántas jornadas se procesarían
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sofascore_client import get_seasons_dict_result, collect_round_metadata
from db_utils import get_connection, create_matches_table_if_not_exists, insert_match_metadata


def get_pending_rounds(league_id: int = None) -> list[tuple]:
    """
    Returns list of (LeagueId, SeasonId, Round) combinations present in Leagues
    but with no entry yet in Matches. Optionally filtered by league_id.
    """
    conn = get_connection()
    cursor = conn.cursor()
    if league_id:
        cursor.execute("""
            SELECT DISTINCT l.LeagueId, l.SeasonId, l.Round
            FROM Leagues l
            LEFT JOIN Matches m ON l.MatchId = m.MatchId
            WHERE m.MatchId IS NULL AND l.LeagueId = %s
            ORDER BY l.SeasonId, l.Round
        """, (league_id,))
    else:
        cursor.execute("""
            SELECT DISTINCT l.LeagueId, l.SeasonId, l.Round
            FROM Leagues l
            LEFT JOIN Matches m ON l.MatchId = m.MatchId
            WHERE m.MatchId IS NULL
            ORDER BY l.LeagueId, l.SeasonId, l.Round
        """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Backfill Matches table from existing Leagues data.")
    parser.add_argument("--league", type=int, help="Filtrar por LeagueId (e.g. 1=Eurocopa, 8=LaLiga)")
    parser.add_argument("--dry-run", action="store_true", help="Show pending rounds without inserting")
    args = parser.parse_args()

    create_matches_table_if_not_exists()

    pending = get_pending_rounds(league_id=args.league)
    label = f"LeagueId={args.league}" if args.league else "todas las ligas"
    print(f"Jornadas pendientes de backfill ({label}): {len(pending)}")

    if args.dry_run:
        for league_id, season_id, round_id in pending:
            print(f"  LeagueId={league_id}  SeasonId={season_id}  Round={round_id}")
        return

    total_inserted = 0
    seasons_loaded = set()

    for league_id, season_id, round_id in pending:
        # Load seasons registry once per league
        if league_id not in seasons_loaded:
            get_seasons_dict_result(league_id)
            seasons_loaded.add(league_id)

        print(f"  LeagueId={league_id}  SeasonId={season_id}  Round={round_id} ...", end=" ")
        try:
            metadata = collect_round_metadata(
                my_league=league_id,
                my_season=season_id,
                my_round=round_id,
            )
            inserted = insert_match_metadata(metadata)
            total_inserted += inserted
            print(f"{len(metadata)} matches, {inserted} new inserted.")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(5)

    print(f"\nBackfill completado. Total insertado: {total_inserted} filas en Matches.")


if __name__ == "__main__":
    main()

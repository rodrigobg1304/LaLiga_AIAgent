"""
MySQL tool for the data-collection crew.

Reads credentials from MYSQL_* env vars (crew/.env) with fallback to DB_* vars
so this tool is compatible with either naming convention.
"""
import os
import mysql.connector
from crewai.tools import tool

# Leagues managed by this crew, with their fixed Sofascore season IDs.
LEAGUE_SEASONS = {
    8:  {"name": "LaLiga",         "season": 77559},
    17: {"name": "Premier League", "season": 76986},
    23: {"name": "Serie A",        "season": 76457},
}


def _get_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST") or os.environ["DB_HOST"],
        port=int(os.environ.get("MYSQL_PORT") or os.environ.get("DB_PORT", 3306)),
        database=os.environ.get("MYSQL_DATABASE") or os.environ["DB_DATABASE"],
        user=os.environ.get("MYSQL_USER") or os.environ["DB_USER"],
        password=os.environ.get("MYSQL_PASSWORD") or os.environ["DB_PASSWORD"],
    )


@tool("get_next_round")
def get_next_round(league_id: int) -> dict:
    """
    Query MySQL to find the highest round already stored for a given league_id
    in the Matches table. Returns the next round to collect (max + 1), or 1 if
    no data exists yet for this league.

    Input:  league_id (int) — e.g. 8 for LaLiga, 17 for Premier League, 23 for Serie A.
    Output: dict with keys: league_id, next_round, max_round_stored (None if no data).
    """
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(CAST(Round AS UNSIGNED)) FROM Matches WHERE LeagueId = %s",
            (league_id,),
        )
        row = cursor.fetchone()
        max_round = row[0] if row and row[0] is not None else None
        next_round = (max_round + 1) if max_round is not None else 1
        return {
            "league_id": league_id,
            "next_round": next_round,
            "max_round_stored": max_round,
        }
    finally:
        conn.close()


@tool("get_all_next_rounds")
def get_all_next_rounds() -> list:
    """
    Query MySQL for the next round to collect for all three configured leagues:
    LaLiga (8), Premier League (17), and Serie A (23).

    Returns a list of dicts — one per league — each containing:
      league_id, league_name, season_id, next_round, max_round_stored.
    """
    results = []
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        for league_id, info in LEAGUE_SEASONS.items():
            cursor.execute(
                "SELECT MAX(CAST(Round AS UNSIGNED)) FROM Matches WHERE LeagueId = %s",
                (league_id,),
            )
            row = cursor.fetchone()
            max_round = row[0] if row and row[0] is not None else None
            next_round = (max_round + 1) if max_round is not None else 1
            results.append({
                "league_id": league_id,
                "league_name": info["name"],
                "season_id": info["season"],
                "next_round": next_round,
                "max_round_stored": max_round,
            })
    finally:
        conn.close()
    return results

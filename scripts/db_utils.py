"""
Shared database utilities for all data collection scripts.

DB credentials are loaded from scripts/.env automatically when this module is imported.
Copy scripts/.env.example → scripts/.env and fill in your credentials.
"""
import os
import math
from pathlib import Path
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

# Load .env from the scripts/ directory, regardless of where the script is run from.
# override=False so that variables already set in the shell take precedence.
load_dotenv(Path(__file__).parent / ".env", override=False)

LEAGUES_TABLE = "Leagues"
MATCHES_TABLE = "Matches"


def get_connection():
    return mysql.connector.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_DATABASE"],
    )


def create_matches_table_if_not_exists():
    """
    Create the Matches table if it does not already exist.
    Also adds MatchDateLocal column if the table already exists without it (migration).
    """
    sql_create = f"""
    CREATE TABLE IF NOT EXISTS {MATCHES_TABLE} (
        MatchId        INT          NOT NULL,
        LeagueId       INT          NOT NULL,
        SeasonId       INT,
        Round          VARCHAR(50)  NOT NULL,
        homeTeam       VARCHAR(100) NOT NULL,
        awayTeam       VARCHAR(100) NOT NULL,
        MatchDate      DATETIME     NOT NULL,
        MatchDateLocal DATETIME,
        homeScore      INT,
        awayScore      INT,
        homeScoreET    INT,
        awayScoreET    INT,
        homeScorePen   INT,
        awayScorePen   INT,
        PRIMARY KEY (MatchId)
    );
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql_create)
    # Add MatchDateLocal if not present (migration for tables created before this column was added)
    cursor.execute(f"""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = '{MATCHES_TABLE}'
          AND COLUMN_NAME = 'MatchDateLocal'
    """)
    (col_exists,) = cursor.fetchone()
    if not col_exists:
        cursor.execute(f"ALTER TABLE {MATCHES_TABLE} ADD COLUMN MatchDateLocal DATETIME AFTER MatchDate;")
    conn.commit()
    conn.close()


def _format_value(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'{}'".format(value.replace("'", "\\'"))
    if isinstance(value, float) and math.isnan(value):
        return "NULL"
    return str(value)


def insert_statistics(df: pd.DataFrame) -> int:
    """
    Insert statistics rows into Leagues table.
    Uses INSERT IGNORE — deduplication via PRIMARY KEY (LeagueId, SeasonId, MatchId, Round, name, period).
    Returns number of new rows inserted.
    """
    if df.empty:
        return 0

    df = df.drop_duplicates()
    columns = ", ".join(f"`{c}`" for c in df.columns)
    value_rows = [
        "({})".format(", ".join(_format_value(v) for v in row))
        for _, row in df.iterrows()
    ]
    sql = f"INSERT IGNORE INTO {LEAGUES_TABLE} ({columns}) VALUES {', '.join(value_rows)};"

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def insert_match_metadata(matches: list[dict]) -> int:
    """
    Insert match metadata rows into Matches table.
    Uses INSERT ... ON DUPLICATE KEY UPDATE MatchDateLocal so existing rows
    (inserted before the timezone column was added) get the local time on re-run.
    Returns number of affected rows (2 per updated row, 1 per inserted row — MySQL convention).
    """
    if not matches:
        return 0

    columns = (
        "MatchId, LeagueId, SeasonId, Round, homeTeam, awayTeam, "
        "MatchDate, MatchDateLocal, homeScore, awayScore, homeScoreET, awayScoreET, homeScorePen, awayScorePen"
    )
    value_rows = [
        "({})".format(", ".join([
            _format_value(m["MatchId"]),
            _format_value(m["LeagueId"]),
            _format_value(m["SeasonId"]),
            _format_value(m["Round"]),
            _format_value(m["homeTeam"]),
            _format_value(m["awayTeam"]),
            _format_value(m["MatchDate"]),
            _format_value(m.get("MatchDateLocal")),
            _format_value(m["homeScore"]),
            _format_value(m["awayScore"]),
            _format_value(m["homeScoreET"]),
            _format_value(m["awayScoreET"]),
            _format_value(m["homeScorePen"]),
            _format_value(m["awayScorePen"]),
        ]))
        for m in matches
    ]
    # COALESCE keeps the existing value when the incoming value is NULL, so
    # fixture rows stored without scores get updated once real results arrive,
    # but valid scores are never overwritten with NULL on a re-run.
    sql = (
        f"INSERT INTO {MATCHES_TABLE} ({columns}) VALUES {', '.join(value_rows)} "
        f"ON DUPLICATE KEY UPDATE "
        f"MatchDateLocal  = VALUES(MatchDateLocal), "
        f"homeTeam        = VALUES(homeTeam), "
        f"awayTeam        = VALUES(awayTeam), "
        f"homeScore       = COALESCE(VALUES(homeScore),    homeScore), "
        f"awayScore       = COALESCE(VALUES(awayScore),    awayScore), "
        f"homeScoreET     = COALESCE(VALUES(homeScoreET),  homeScoreET), "
        f"awayScoreET     = COALESCE(VALUES(awayScoreET),  awayScoreET), "
        f"homeScorePen    = COALESCE(VALUES(homeScorePen), homeScorePen), "
        f"awayScorePen    = COALESCE(VALUES(awayScorePen), awayScorePen);"
    )

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


PREDICTIONS_TABLE = "DailyPredictions"


def save_prediction(match: dict, pred: dict, predicted_outcome: str,
                    scoreline: str, eg_home: float, eg_away: float,
                    ou_15: float, ou_25: float, is_value_bet: bool = False):
    """Insert a pre-match prediction into DailyPredictions. Skips if already stored today."""
    resultado = pred.get("resultado", {})
    probs = resultado.get("probabilities", {})
    odds  = resultado.get("odds", {})

    kick_off = match.get("MatchDateLocal")
    match_date = kick_off.date() if hasattr(kick_off, "date") else kick_off

    sql = f"""
        INSERT IGNORE INTO {PREDICTIONS_TABLE}
            (predicted_at, match_date, match_id, league_id, home_team, away_team,
             p_home, p_draw, p_away, predicted_outcome, scoreline,
             ou_15, ou_25, eg_home, eg_away,
             odds_home, odds_draw, odds_away, value_bet)
        VALUES (%s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s)
    """
    from datetime import datetime
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (
                datetime.now(), match_date,
                match.get("MatchId"), match.get("LeagueId"),
                match["homeTeam"], match["awayTeam"],
                probs.get("1"), probs.get("X"), probs.get("2"),
                predicted_outcome, scoreline,
                ou_15, ou_25, eg_home, eg_away,
                odds.get("1"), odds.get("X"), odds.get("2"),
                int(is_value_bet),
            ))
        conn.commit()
    finally:
        conn.close()


def update_prediction_results():
    """
    Fill in actual_outcome, actual_goals, and correctness flags for predictions
    where the match has already been played (Matches table has homeScore != NULL).
    """
    sql_update = f"""
        UPDATE {PREDICTIONS_TABLE} dp
        JOIN {MATCHES_TABLE} m ON dp.match_id = m.MatchId
        SET
            dp.actual_goals   = m.homeScore + m.awayScore,
            dp.actual_outcome = CASE
                WHEN m.homeScore > m.awayScore THEN '1'
                WHEN m.homeScore < m.awayScore THEN '2'
                ELSE 'X'
            END,
            dp.winner_correct = CASE
                WHEN dp.predicted_outcome = CASE
                    WHEN m.homeScore > m.awayScore THEN '1'
                    WHEN m.homeScore < m.awayScore THEN '2'
                    ELSE 'X'
                END THEN 1 ELSE 0
            END,
            dp.ou15_correct = CASE
                WHEN (m.homeScore + m.awayScore > 1.5 AND dp.ou_15 >= 50) THEN 1
                WHEN (m.homeScore + m.awayScore <= 1.5 AND dp.ou_15 < 50) THEN 1
                ELSE 0
            END,
            dp.ou25_correct = CASE
                WHEN (m.homeScore + m.awayScore > 2.5 AND dp.ou_25 >= 50) THEN 1
                WHEN (m.homeScore + m.awayScore <= 2.5 AND dp.ou_25 < 50) THEN 1
                ELSE 0
            END
        WHERE dp.actual_outcome IS NULL
          AND m.homeScore IS NOT NULL
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_update)
            updated = cur.rowcount
        conn.commit()
        return updated
    finally:
        conn.close()


def get_prediction_stats(last_n_days: int = 30) -> dict:
    """Returns accuracy stats for the last N days from stored predictions."""
    sql = f"""
        SELECT
            COUNT(*) as total,
            SUM(winner_correct) as winner_ok,
            SUM(ou15_correct)   as ou15_ok,
            SUM(ou25_correct)   as ou25_ok,
            SUM(value_bet)      as value_bets,
            SUM(CASE WHEN value_bet=1 AND winner_correct=1 THEN 1 ELSE 0 END) as value_ok
        FROM {PREDICTIONS_TABLE}
        WHERE actual_outcome IS NOT NULL
          AND predicted_at >= NOW() - INTERVAL %s DAY
    """
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (last_n_days,))
            return cur.fetchone() or {}
    finally:
        conn.close()

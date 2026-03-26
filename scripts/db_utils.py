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
    sql = (
        f"INSERT INTO {MATCHES_TABLE} ({columns}) VALUES {', '.join(value_rows)} "
        f"ON DUPLICATE KEY UPDATE MatchDateLocal = VALUES(MatchDateLocal);"
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

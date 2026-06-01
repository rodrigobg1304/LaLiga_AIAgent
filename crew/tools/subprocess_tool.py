"""
Subprocess tool for the data-collection crew.

Wraps scripts/leagues/collect_leagues.py as a CrewAI tool.
The subprocess inherits the current environment and additionally receives
DB_* vars mapped from MYSQL_* so collect_leagues.py / db_utils.py work
with either env-var naming convention.
"""
import os
import sys
import subprocess
from pathlib import Path
from crewai.tools import tool

# Absolute paths derived from this file's location (crew/tools/subprocess_tool.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_COLLECT_SCRIPT = _SCRIPTS_DIR / "leagues" / "collect_leagues.py"

# Fixed season IDs for the three automated leagues
LEAGUE_SEASONS = {
    8:  {"name": "LaLiga",         "season": 77559},
    17: {"name": "Premier League", "season": 76986},
    23: {"name": "Serie A",        "season": 76457},
}


def _subprocess_env() -> dict:
    """
    Build the environment for the subprocess.
    Maps MYSQL_* vars → DB_* so db_utils.py receives the credentials it expects.
    Existing DB_* vars in the environment are left untouched (they take precedence).
    """
    env = os.environ.copy()
    mapping = {
        "DB_HOST":     os.environ.get("MYSQL_HOST"),
        "DB_PORT":     os.environ.get("MYSQL_PORT"),
        "DB_USER":     os.environ.get("MYSQL_USER"),
        "DB_PASSWORD": os.environ.get("MYSQL_PASSWORD"),
        "DB_DATABASE": os.environ.get("MYSQL_DATABASE"),
    }
    for db_key, mysql_val in mapping.items():
        if mysql_val and db_key not in env:
            env[db_key] = mysql_val
    return env


@tool("collect_league_round")
def collect_league_round(league_id: int, round_id: int) -> dict:
    """
    Run scripts/leagues/collect_leagues.py to collect match statistics and
    metadata from Sofascore for the given league and round, then insert them
    into MySQL.

    Inputs:
      league_id — 8 (LaLiga), 17 (Premier League), or 23 (Serie A)
      round_id  — the round number to collect (integer)

    Returns a dict:
      success     — True if the script exited with code 0
      league_name — human-readable league name
      league_id   — as provided
      season_id   — fixed season ID for this league
      round_id    — as provided
      stdout      — script stdout (stats/counts logged by collect_leagues.py)
      stderr      — script stderr (errors if any)
      returncode  — process exit code
    """
    if league_id not in LEAGUE_SEASONS:
        return {
            "success": False, "league_name": "Unknown", "league_id": league_id,
            "season_id": None, "round_id": round_id,
            "stdout": "", "stderr": f"league_id {league_id} not configured.", "returncode": -1,
        }

    info = LEAGUE_SEASONS[league_id]
    season_id = info["season"]

    cmd = [
        sys.executable,
        str(_COLLECT_SCRIPT),
        "--league", str(league_id),
        "--season", str(season_id),
        "--round",  str(round_id),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_SCRIPTS_DIR),
            env=_subprocess_env(),
            timeout=600,  # 10-minute ceiling: 10s delay × up to 10 matches + overhead
        )
        return {
            "success": result.returncode == 0,
            "league_name": info["name"],
            "league_id": league_id,
            "season_id": season_id,
            "round_id": round_id,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False, "league_name": info["name"], "league_id": league_id,
            "season_id": season_id, "round_id": round_id,
            "stdout": "", "stderr": "Subprocess timed out after 600 seconds.", "returncode": -1,
        }
    except Exception as exc:
        return {
            "success": False, "league_name": info["name"], "league_id": league_id,
            "season_id": season_id, "round_id": round_id,
            "stdout": "", "stderr": str(exc), "returncode": -1,
        }

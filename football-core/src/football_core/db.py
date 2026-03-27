import os
import mysql.connector
from typing import Optional
import configparser

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE CONEXIÓN
# Prioridad: variables de entorno > mainconfig_secret.ini
# ─────────────────────────────────────────────

def _load_config() -> dict:
    """
    Carga la configuración de BD.
    Las variables de entorno tienen prioridad sobre el fichero .ini,
    lo que permite inyectar credenciales en contenedores sin tocar el código.
    """
    cfg = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT", "3306"),
        "database": os.getenv("DB_DATABASE") or os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD") or os.getenv("DB_PASS"),
        "table": os.getenv("DB_TABLE"),
    }

    # Si alguna variable falta, intentar leer del .ini
    if not all(cfg.values()):
        ini_path = os.path.join(os.path.dirname(__file__), "mainconfig_secret.ini")
        if os.path.exists(ini_path):
            parser = configparser.ConfigParser()
            parser.read(ini_path)
            section = parser["MySQL"]
            cfg["host"]     = cfg["host"]     or section.get("host")
            cfg["port"]     = cfg["port"]     or section.get("port") or "3306"
            cfg["database"] = cfg["database"] or section.get("database")
            cfg["user"]     = cfg["user"]     or section.get("user")
            cfg["password"] = cfg["password"] or section.get("password")
            cfg["table"]    = cfg["table"]    or section.get("table")

    return cfg


_cfg = _load_config()

DB_TYPE = os.getenv("DB_TYPE", "mysql")
DB_HOST = _cfg["host"]
DB_PORT = _cfg["port"]
DB_NAME = _cfg["database"]
DB_USER = _cfg["user"]
DB_PASS = _cfg["password"]
TABLE   = _cfg["table"]


def get_connection():
    return mysql.connector.connect(
        host=DB_HOST, port=int(DB_PORT), database=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


def run_query(sql: str, params: tuple = ()) -> list[dict]:
    """Ejecuta una query y devuelve lista de dicts."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True) if DB_TYPE == "mysql" else conn.cursor()
        cur.execute("SET SESSION sql_mode = ''")
        cur.execute(sql, params)
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()


# ─────────────────────────────────────────────
# QUERIES POR CASO DE USO
# ─────────────────────────────────────────────

def get_name() -> list[str]:
    return [row["name"] for row in run_query(f"SELECT DISTINCT name FROM {TABLE} ORDER BY name ASC")]


def get_teams() -> list[str]:
    return [row["homeTeam"] for row in run_query(f"SELECT DISTINCT homeTeam FROM {TABLE} ORDER BY homeTeam ASC")]


def get_teams_by_league(league_id: str, season: str = None) -> list[str]:
    if season:
        query = f"""
                SELECT DISTINCT team FROM (
                    SELECT homeTeam as team FROM {TABLE} WHERE LeagueId = %s AND Year = %s
                    UNION
                    SELECT awayTeam as team FROM {TABLE} WHERE LeagueId = %s AND Year = %s
                ) teams
                ORDER BY team
                """
        params = (league_id, season, league_id, season)
    else:
        query = f"""
                SELECT DISTINCT team FROM (
                    SELECT homeTeam as team FROM {TABLE} WHERE LeagueId = %s
                    UNION
                    SELECT awayTeam as team FROM {TABLE} WHERE LeagueId = %s
                ) teams
                ORDER BY team
                """
        params = (league_id, league_id)

    return [row["team"] for row in run_query(query, params)]


def get_years() -> list[str]:
    return [row["Year"] for row in run_query(f"SELECT DISTINCT Year FROM {TABLE} ORDER BY Year DESC")]


def get_years_by_league(league_id: str) -> list[str]:
    """Devuelve las temporadas disponibles para una liga concreta."""
    return [
        row["Year"]
        for row in run_query(
            f"SELECT DISTINCT Year FROM {TABLE} WHERE LeagueId = %s ORDER BY Year DESC",
            (league_id,)
        )
    ]


def get_stats() -> list[str]:
    return [row["name"] for row in run_query(f"SELECT DISTINCT name FROM {TABLE} ORDER BY name ASC")]


def get_team_results(team: str, year: Optional[str] = None, top_n: Optional[int] = None) -> list[dict]:
    sql = f"""
        SELECT CAST(Round AS SIGNED) as Round, homeTeam, awayTeam, LeagueId, Year,
               SUM(CAST(homeValue AS SIGNED)) as home_goals,
               SUM(CAST(awayValue AS SIGNED)) as away_goals,
               CASE WHEN SUM(homeValue) > SUM(awayValue) THEN '1'
                   WHEN SUM(homeValue) < SUM(awayValue) THEN '2'
                   ELSE 'X'
               END as result,
               CASE WHEN SUM(homeValue) > SUM(awayValue) AND homeTeam = %s THEN 3
                   WHEN SUM(homeValue) < SUM(awayValue) AND awayTeam = %s THEN 3
                   WHEN SUM(homeValue) = SUM(awayValue) AND (homeTeam = %s OR awayTeam = %s) THEN 1
                   ELSE 0
               END as points
        FROM {TABLE}
        WHERE (homeTeam = %s OR awayTeam = %s) and name = 'Goals'
    """
    params = [team, team, team, team, team, team]
    if year:
        sql += " AND year = %s"; params.append(year)
    sql += " GROUP BY CAST(Round AS SIGNED), homeTeam, awayTeam, LeagueId, Year ORDER BY CAST(Round AS SIGNED) Desc"
    if top_n:
        sql += " LIMIT %s"; params.append(top_n)
    return run_query(sql, tuple(params))


def get_goals_scored(team: str, year: Optional[str] = None) -> list[dict]:
    sql = f"""
        SELECT
            %s as team,
            year,
            SUM(CASE WHEN homeTeam = %s THEN CAST(homeValue AS SIGNED) ELSE 0 END) AS home_goals_scored,
            SUM(CASE WHEN homeTeam = %s THEN CAST(awayValue AS SIGNED) ELSE 0 END) AS home_goals_conceded,
            SUM(CASE WHEN awayTeam = %s THEN CAST(awayValue AS SIGNED) ELSE 0 END) AS away_goals_scored,
            SUM(CASE WHEN awayTeam = %s THEN CAST(homeValue AS SIGNED) ELSE 0 END) AS away_goals_conceded,
            SUM(CASE WHEN homeTeam = %s THEN CAST(homeValue AS SIGNED) ELSE 0 END) +
            SUM(CASE WHEN awayTeam = %s THEN CAST(awayValue AS SIGNED) ELSE 0 END) AS total_goals_scored,
            SUM(CASE WHEN homeTeam = %s THEN CAST(awayValue AS SIGNED) ELSE 0 END) +
            SUM(CASE WHEN awayTeam = %s THEN CAST(homeValue AS SIGNED) ELSE 0 END) AS total_goals_conceded
        FROM {TABLE}
        WHERE name='Goals'
          AND (homeTeam = %s OR awayTeam = %s)
    """
    params = [team, team, team, team, team, team, team, team, team, team, team]
    if year:
        sql += " AND year = %s"; params.append(year)
    sql += " GROUP BY year"
    return run_query(sql, tuple(params))


def get_team_stats(team: str, stat_name: str, year: Optional[str] = None) -> list[dict]:
    sql = f"""
        SELECT CAST(Round AS SIGNED) AS round, Year, homeTeam, awayTeam,
               SUM(CAST(homeValue AS SIGNED)) as homeValue,
               SUM(CAST(awayValue AS SIGNED)) as awayValue,
               name
        FROM {TABLE}
        WHERE name = %s AND (homeTeam = %s OR awayTeam = %s)
    """
    params = [stat_name, team, team]
    if year:
        sql += " AND Year = %s"; params.append(year)
    sql += " GROUP BY Year, CAST(Round AS SIGNED), homeTeam, awayTeam, name ORDER BY CAST(Round AS SIGNED) ASC"
    return run_query(sql, tuple(params))


def get_standings(leagueId: str, year: str) -> list[dict]:
    sql = f"""
        SELECT
            team,
            leagueId,
            Year as year,
            Count(*) AS played,
            SUM(gf) AS goals_for,
            SUM(ga) AS goals_against,
            SUM(gf) - SUM(ga) AS goal_diff,
            SUM(CASE WHEN gf > ga THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN gf = ga THEN 1 ELSE 0 END) AS draws,
            SUM(CASE WHEN gf < ga THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN gf > ga THEN 3 WHEN gf = ga THEN 1 ELSE 0 END) AS points
        FROM (
            SELECT
                homeTeam AS team,
                matchId,
                leagueId,
                Year,
                SUM(CAST(homeValue AS SIGNED)) AS gf,
                SUM(CAST(awayValue AS SIGNED)) AS ga
            FROM {TABLE}
            WHERE name = 'Goals' AND Year = %s AND leagueId = %s
            GROUP BY homeTeam, matchId, Year, leagueId

            UNION ALL

            SELECT
                awayTeam AS team,
                matchId,
                leagueId,
                Year,
                SUM(CAST(awayValue AS SIGNED)) AS gf,
                SUM(CAST(homeValue AS SIGNED)) AS ga
            FROM {TABLE}
            WHERE name = 'Goals' AND Year = %s AND leagueId = %s
            GROUP BY awayTeam, matchId, Year, leagueId
        ) AS match_totals
        GROUP BY team
        ORDER BY points DESC, goals_for - goals_against DESC
    """
    params = [year, leagueId, year, leagueId]
    results = run_query(sql, tuple(params))
    for i, row in enumerate(results, start=1):
        row['position'] = i
    return results


def get_top_stats(stat_name: str, year: str, top_n: int = 10) -> list[dict]:
    sql = f"""
        SELECT
            team,
            SUM(CASE WHEN location = 'H' THEN gf ELSE 0 END) AS sum_stat_home_for,
            SUM(CASE WHEN location = 'H' THEN ga ELSE 0 END) AS sum_stat_home_against,
            AVG(CASE WHEN location = 'H' THEN gf END)        AS avg_stat_home_for,
            AVG(CASE WHEN location = 'H' THEN ga END)        AS avg_stat_home_against,
            SUM(CASE WHEN location = 'A' THEN gf ELSE 0 END) AS sum_stat_away_for,
            SUM(CASE WHEN location = 'A' THEN ga ELSE 0 END) AS sum_stat_away_against,
            AVG(CASE WHEN location = 'A' THEN gf END)        AS avg_stat_away_for,
            AVG(CASE WHEN location = 'A' THEN ga END)        AS avg_stat_away_against
        FROM (
            SELECT
                homeTeam AS team,
                'H' as location,
                matchId,
                SUM(CAST(homeValue AS SIGNED)) AS gf,
                SUM(CAST(awayValue AS SIGNED)) AS ga
            FROM {TABLE}
            WHERE name = %s AND Year = %s
            GROUP BY homeTeam, matchId, location

            UNION ALL

            SELECT
                awayTeam AS team,
                'A' as location,
                matchId,
                SUM(CAST(awayValue AS SIGNED)) AS gf,
                SUM(CAST(homeValue AS SIGNED)) AS ga
            FROM {TABLE}
            WHERE name = %s AND Year = %s
            GROUP BY awayTeam, matchId, location
        ) AS match_totals
        GROUP BY team
        ORDER BY avg_stat_home_for DESC
        LIMIT %s
    """
    params = [stat_name, year, stat_name, year, top_n]
    return run_query(sql, params=tuple(params))


def get_all_matches_chronological(league_id: Optional[str] = None) -> list[dict]:
    sql = f"""
        SELECT homeTeam, awayTeam,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals,
               Year, CAST(Round AS SIGNED) AS round_num
        FROM {TABLE}
        WHERE name = 'Goals' AND period IN ('1ST', '2ND')
    """
    params = []
    if league_id:
        sql += " AND LeagueId = %s"
        params.append(league_id)
    sql += """
        GROUP BY matchId, homeTeam, awayTeam, Year, CAST(Round AS SIGNED)
        ORDER BY Year ASC, CAST(Round AS SIGNED) ASC
    """
    return run_query(sql, tuple(params))


def get_league_matches(league_id: str, year: str) -> list[dict]:
    sql = f"""
        SELECT homeTeam, awayTeam, Year,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND LeagueId = %s
          AND Year = %s
        GROUP BY MatchId, homeTeam, awayTeam
    """
    return run_query(sql, (league_id, year))


def get_team_season_count(team: str, years: list[str]) -> int:
    if not years:
        return 0
    placeholders_years = ",".join(["%s"] * len(years))
    sql = f"""
        SELECT COUNT(DISTINCT Year) AS season_count
        FROM {TABLE}
        WHERE (homeTeam = %s OR awayTeam = %s)
          AND Year IN ({placeholders_years})
    """
    params = [team, team] + years
    rows = run_query(sql, tuple(params))
    return int(rows[0]["season_count"]) if rows else 0


def get_team_recent_matches_goals(team: str, league_id: str, years: list[str],
                                  role: str, n: int = 5) -> list[dict]:
    team_col = "homeTeam" if role == "home" else "awayTeam"
    if years:
        placeholders_years = ",".join(["%s"] * len(years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [team, league_id] + years + [n]
    else:
        year_filter = ""
        params = [team, league_id, n]
    sql = f"""
        SELECT matchId,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
        FROM {TABLE}
        WHERE {team_col} = %s
          AND name = 'Goals'
          AND LeagueId = %s
          {year_filter}
        GROUP BY matchId
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
        LIMIT %s
    """
    return run_query(sql, tuple(params))


def get_h2h_matches(team1: str, team2: str, n: int = 10) -> list[dict]:
    sql = f"""
        SELECT matchId, homeTeam, awayTeam,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND ((homeTeam = %s AND awayTeam = %s) OR (homeTeam = %s AND awayTeam = %s))
        GROUP BY matchId, homeTeam, awayTeam
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
        LIMIT %s
    """
    return run_query(sql, (team1, team2, team2, team1, n))


def get_proxy_h2h_matches(home_neighbors: list[str], away_neighbors: list[str], n: int = 10) -> list[dict]:
    if not home_neighbors or not away_neighbors:
        return []
    placeholders_home = ",".join(["%s"] * len(home_neighbors))
    placeholders_away = ",".join(["%s"] * len(away_neighbors))
    sql = f"""
        SELECT matchId, homeTeam, awayTeam,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND homeTeam IN ({placeholders_home})
          AND awayTeam IN ({placeholders_away})
        GROUP BY matchId, homeTeam, awayTeam
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
        LIMIT %s
    """
    params = home_neighbors + away_neighbors + [n]
    return run_query(sql, tuple(params))


def get_team_stat_average(team: str, role: str, stat_name: str,
                          years: list[str], n: Optional[int] = None) -> float:
    team_col = "homeTeam" if role == "home" else "awayTeam"
    stat_col = "homeValue" if role == "home" else "awayValue"
    if years:
        placeholders_years = ",".join(["%s"] * len(years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [team, stat_name] + years
    else:
        year_filter = ""
        params = [team, stat_name]
    subquery = f"""
        SELECT matchId,
               SUM(CAST({stat_col} AS DECIMAL)) AS stat_value
        FROM {TABLE}
        WHERE {team_col} = %s
          AND name = %s
          AND period IN ('1ST', '2ND')
          {year_filter}
        GROUP BY matchId
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
    """
    if n is not None:
        subquery += " LIMIT %s"
        params.append(n)
    sql = f"""
        SELECT AVG(CAST(stat_value AS DECIMAL)) AS avg_stat
        FROM ({subquery}) AS matches
    """
    rows = run_query(sql, tuple(params))
    return float(rows[0]["avg_stat"]) if (rows and rows[0]["avg_stat"] is not None) else 0.0


def get_team_multiple_stats_average(team: str, role: str, years: list[str],
                                    n: Optional[int] = None) -> dict:
    team_col = "homeTeam" if role == "home" else "awayTeam"
    stat_col = "homeValue" if role == "home" else "awayValue"
    if not years:
        return {}
    placeholders_years = ",".join(["%s"] * len(years))
    year_filter = f"AND Year IN ({placeholders_years})"
    subquery = f"""
        SELECT matchId,
               MAX(CASE WHEN name='Goals' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS goals,
               MAX(CASE WHEN name='Ball possession' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS ball_possession,
               MAX(CASE WHEN name='Total shots' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS total_shots,
               MAX(CASE WHEN name='Shots on target' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS shots_on_target,
               MAX(CASE WHEN name='Goalkeeper saves' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS gk_saves,
               MAX(CASE WHEN name='Big chances' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS big_chances,
               MAX(CASE WHEN name='Accurate passes' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS accurate_passes,
               MAX(CASE WHEN name='Tackles won' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS tackles_won,
               MAX(CASE WHEN name='Interceptions' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS interceptions,
               MAX(CASE WHEN name='Blocked shots' AND period IN ('1ST','2ND')
                   THEN CAST({stat_col} AS DECIMAL) END) AS blocked_shots
        FROM {TABLE}
        WHERE {team_col} = %s
          AND period IN ('1ST', '2ND')
          {year_filter}
        GROUP BY matchId
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
    """
    params = [team] + years
    if n is not None:
        subquery += " LIMIT %s"
        params.append(n)
    sql = f"""
        SELECT
            AVG(goals) AS goals,
            AVG(ball_possession) AS ball_possession,
            AVG(total_shots) AS total_shots,
            AVG(shots_on_target) AS shots_on_target,
            AVG(gk_saves) AS gk_saves,
            AVG(big_chances) AS big_chances,
            AVG(accurate_passes) AS accurate_passes,
            AVG(tackles_won) AS tackles_won,
            AVG(interceptions) AS interceptions,
            AVG(blocked_shots) AS blocked_shots
        FROM ({subquery}) AS matches
    """
    rows = run_query(sql, tuple(params))
    if not rows or not rows[0]:
        return {}
    return {
        'goals':           float(rows[0]['goals'] or 0),
        'ball_possession': float(rows[0]['ball_possession'] or 0),
        'total_shots':     float(rows[0]['total_shots'] or 0),
        'shots_on_target': float(rows[0]['shots_on_target'] or 0),
        'gk_saves':        float(rows[0]['gk_saves'] or 0),
        'big_chances':     float(rows[0]['big_chances'] or 0),
        'accurate_passes': float(rows[0]['accurate_passes'] or 0),
        'tackles_won':     float(rows[0]['tackles_won'] or 0),
        'interceptions':   float(rows[0]['interceptions'] or 0),
        'blocked_shots':   float(rows[0]['blocked_shots'] or 0)
    }


def get_team_win_rates(team: str, role: str, league_id: str, years: list[str],
                       n: Optional[int] = None) -> dict:
    team_col = "homeTeam" if role == "home" else "awayTeam"
    if not years:
        return {"wins": 0, "draws": 0, "losses": 0, "total": 0}
    placeholders_years = ",".join(["%s"] * len(years))
    year_filter = f"AND Year IN ({placeholders_years})"
    sql = f"""
        SELECT
            SUM(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END) AS draws,
            SUM(CASE WHEN home_goals < away_goals THEN 1 ELSE 0 END) AS losses,
            COUNT(*) AS total
        FROM (
            SELECT matchId,
                   SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
                   SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
            FROM {TABLE}
            WHERE {team_col} = %s
              AND name = 'Goals'
              AND LeagueId = %s
              {year_filter}
            GROUP BY matchId
            ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
    """
    params = [team, league_id] + years
    if n is not None:
        sql += " LIMIT %s"
        params.append(n)
    sql += ") AS matches"
    rows = run_query(sql, tuple(params))
    if not rows or rows[0]["total"] == 0:
        return {"wins": 0, "draws": 0, "losses": 0, "total": 0}
    return {
        "wins":   int(rows[0]["wins"] or 0),
        "draws":  int(rows[0]["draws"] or 0),
        "losses": int(rows[0]["losses"] or 0),
        "total":  int(rows[0]["total"])
    }


def get_team_goals_conceded_average(team: str, role: str, years: list[str],
                                    n: Optional[int] = None) -> float:
    team_col = "homeTeam" if role == "home" else "awayTeam"
    goals_col = "awayValue" if role == "home" else "homeValue"
    if not years:
        return 0.0
    placeholders_years = ",".join(["%s"] * len(years))
    year_filter = f"AND Year IN ({placeholders_years})"
    subquery = f"""
        SELECT matchId,
               SUM(CAST({goals_col} AS DECIMAL)) AS goals_conceded
        FROM {TABLE}
        WHERE {team_col} = %s
          AND name = 'Goals'
          AND period IN ('1ST', '2ND')
          {year_filter}
        GROUP BY matchId
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
    """
    params = [team] + years
    if n is not None:
        subquery += " LIMIT %s"
        params.append(n)
    sql = f"""
        SELECT AVG(CAST(goals_conceded AS DECIMAL)) AS avg_conceded
        FROM ({subquery}) AS matches
    """
    rows = run_query(sql, tuple(params))
    return float(rows[0]["avg_conceded"]) if (rows and rows[0]["avg_conceded"] is not None) else 0.0


def get_team_matches_total_goals(team: str, years: list[str], n: Optional[int] = None) -> list[dict]:
    if not years:
        return []
    placeholders_years = ",".join(["%s"] * len(years))
    year_filter = f"AND Year IN ({placeholders_years})"
    sql = f"""
        SELECT
            matchId,
            SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) +
            SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS total_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND (homeTeam = %s OR awayTeam = %s)
          {year_filter}
        GROUP BY matchId
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
    """
    params = [team, team] + years
    if n is not None:
        sql += " LIMIT %s"
        params.append(n)
    return run_query(sql, tuple(params))


def get_league_all_stats(league_id: str) -> list[dict]:
    sql = f"""
        SELECT
            matchId,
            Year as season,
            CAST(Round AS SIGNED) as round,
            homeTeam as home_team,
            awayTeam as away_team,
            LeagueId as league_id,
            MAX(CASE WHEN name = 'Goals' THEN CAST(homeValue AS SIGNED) END) as home_goals,
            MAX(CASE WHEN name = 'Goals' THEN CAST(awayValue AS SIGNED) END) as away_goals,
            MAX(CASE WHEN name = 'Shots on target' THEN CAST(homeValue AS SIGNED) END) as home_shots_on_target,
            MAX(CASE WHEN name = 'Shots on target' THEN CAST(awayValue AS SIGNED) END) as away_shots_on_target,
            MAX(CASE WHEN name = 'Ball possession' THEN CAST(homeValue AS SIGNED) END) as home_possession,
            MAX(CASE WHEN name = 'Ball possession' THEN CAST(awayValue AS SIGNED) END) as away_possession,
            MAX(CASE WHEN name = 'Total shots' THEN CAST(homeValue AS SIGNED) END) as home_total_shots,
            MAX(CASE WHEN name = 'Total shots' THEN CAST(awayValue AS SIGNED) END) as away_total_shots,
            MAX(CASE WHEN name = 'Goalkeeper saves' THEN CAST(homeValue AS SIGNED) END) as home_gk_saves,
            MAX(CASE WHEN name = 'Goalkeeper saves' THEN CAST(awayValue AS SIGNED) END) as away_gk_saves,
            MAX(CASE WHEN name = 'Big chances' THEN CAST(homeValue AS SIGNED) END) as home_big_chances,
            MAX(CASE WHEN name = 'Big chances' THEN CAST(awayValue AS SIGNED) END) as away_big_chances,
            MAX(CASE WHEN name = 'Accurate passes' THEN CAST(homeValue AS SIGNED) END) as home_accurate_passes,
            MAX(CASE WHEN name = 'Accurate passes' THEN CAST(awayValue AS SIGNED) END) as away_accurate_passes,
            MAX(CASE WHEN name = 'Tackles won' THEN CAST(homeValue AS SIGNED) END) as home_tackles_won,
            MAX(CASE WHEN name = 'Tackles won' THEN CAST(awayValue AS SIGNED) END) as away_tackles_won,
            MAX(CASE WHEN name = 'Interceptions' THEN CAST(homeValue AS SIGNED) END) as home_interceptions,
            MAX(CASE WHEN name = 'Interceptions' THEN CAST(awayValue AS SIGNED) END) as away_interceptions,
            MAX(CASE WHEN name = 'Blocked shots' THEN CAST(homeValue AS SIGNED) END) as home_blocked_shots,
            MAX(CASE WHEN name = 'Blocked shots' THEN CAST(awayValue AS SIGNED) END) as away_blocked_shots,
            CASE
                WHEN MAX(CASE WHEN name = 'Goals' THEN CAST(homeValue AS SIGNED) END) >
                     MAX(CASE WHEN name = 'Goals' THEN CAST(awayValue AS SIGNED) END) THEN '1'
                WHEN MAX(CASE WHEN name = 'Goals' THEN CAST(homeValue AS SIGNED) END) <
                     MAX(CASE WHEN name = 'Goals' THEN CAST(awayValue AS SIGNED) END) THEN '2'
                ELSE 'X'
            END as result
        FROM {TABLE}
        WHERE LeagueId = %s
        GROUP BY matchId, Year, CAST(Round AS SIGNED), homeTeam, awayTeam, LeagueId
        ORDER BY Year ASC, CAST(Round AS SIGNED) ASC
    """
    return run_query(sql, (league_id,))


def get_matches_with_goals(league_id: str, years: list[str]) -> list[dict]:
    if not years:
        return []
    years_str = ",".join([f"'{y}'" for y in years])
    sql = f"""
        SELECT
            matchId,
            homeTeam,
            awayTeam,
            LeagueId,
            Year,
            SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND')
                THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
            SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND')
                THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
        FROM {TABLE}
        WHERE LeagueId = %s
          AND Year IN ({years_str})
          AND name = 'Goals'
        GROUP BY matchId, homeTeam, awayTeam, LeagueId, Year
        ORDER BY Year ASC, CAST(Round AS SIGNED) ASC
    """
    return run_query(sql, (league_id,))


def get_matches_with_saves(league_id: str, years: list) -> list:
    years_str = "', '".join(years)
    sql = f"""
    SELECT
        matchId,
        homeTeam,
        awayTeam,
        LeagueId,
        Year,
        SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND')
            THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
        SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND')
            THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals,
        SUM(CASE WHEN name='Goalkeeper saves' AND period IN ('1ST','2ND')
            THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_saves,
        SUM(CASE WHEN name='Goalkeeper saves' AND period IN ('1ST','2ND')
            THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_saves
    FROM {TABLE}
    WHERE LeagueId = %s
      AND Year IN ('{years_str}')
      AND name IN ('Goals', 'Goalkeeper saves')
    GROUP BY matchId, homeTeam, awayTeam, LeagueId, Year
    HAVING home_saves IS NOT NULL
       AND away_saves IS NOT NULL
    ORDER BY Year ASC, CAST(Round AS SIGNED) ASC
    """
    return run_query(sql, (league_id,))


def get_team_clustering_features(
    league_ids: list[str],
    min_matches: int = 0,
    years: Optional[list[str]] = None,
) -> list[dict]:
    """
    Returns one row per team with averaged stats across all matches in the given leagues.
    Combines home and away appearances. Used for clustering analysis.

    min_matches: exclude teams with fewer total appearances than this threshold.
    years: optional season filter (e.g. ['25/26']). None means all seasons.
    """
    if not league_ids:
        return []
    placeholders = ",".join(["%s"] * len(league_ids))
    year_filter = ""
    year_params: list = []
    if years:
        year_placeholders = ",".join(["%s"] * len(years))
        year_filter = f"AND Year IN ({year_placeholders})"
        year_params = list(years)

    sql = f"""
        SELECT
            team,
            COUNT(*) AS match_count,
            AVG(goals)           AS avg_goals,
            AVG(shots_on_target) AS avg_shots_on_target,
            AVG(total_shots)     AS avg_total_shots,
            AVG(ball_possession) AS avg_ball_possession,
            AVG(gk_saves)        AS avg_gk_saves,
            AVG(big_chances)     AS avg_big_chances,
            AVG(accurate_passes) AS avg_accurate_passes,
            AVG(tackles_won)     AS avg_tackles_won,
            AVG(interceptions)   AS avg_interceptions,
            AVG(blocked_shots)   AS avg_blocked_shots
        FROM (
            SELECT homeTeam AS team, matchId,
                MAX(CASE WHEN name='Goals'            AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS goals,
                MAX(CASE WHEN name='Shots on target'  AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS shots_on_target,
                MAX(CASE WHEN name='Total shots'      AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS total_shots,
                MAX(CASE WHEN name='Ball possession'  AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS ball_possession,
                MAX(CASE WHEN name='Goalkeeper saves' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS gk_saves,
                MAX(CASE WHEN name='Big chances'      AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS big_chances,
                MAX(CASE WHEN name='Accurate passes'  AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS accurate_passes,
                MAX(CASE WHEN name='Tackles won'      AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS tackles_won,
                MAX(CASE WHEN name='Interceptions'    AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS interceptions,
                MAX(CASE WHEN name='Blocked shots'    AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) END) AS blocked_shots
            FROM {TABLE}
            WHERE LeagueId IN ({placeholders}) {year_filter}
            GROUP BY matchId, homeTeam

            UNION ALL

            SELECT awayTeam AS team, matchId,
                MAX(CASE WHEN name='Goals'            AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS goals,
                MAX(CASE WHEN name='Shots on target'  AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS shots_on_target,
                MAX(CASE WHEN name='Total shots'      AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS total_shots,
                MAX(CASE WHEN name='Ball possession'  AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS ball_possession,
                MAX(CASE WHEN name='Goalkeeper saves' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS gk_saves,
                MAX(CASE WHEN name='Big chances'      AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS big_chances,
                MAX(CASE WHEN name='Accurate passes'  AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS accurate_passes,
                MAX(CASE WHEN name='Tackles won'      AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS tackles_won,
                MAX(CASE WHEN name='Interceptions'    AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS interceptions,
                MAX(CASE WHEN name='Blocked shots'    AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) END) AS blocked_shots
            FROM {TABLE}
            WHERE LeagueId IN ({placeholders}) {year_filter}
            GROUP BY matchId, awayTeam
        ) AS team_matches
        GROUP BY team
        HAVING match_count >= %s
        ORDER BY team ASC
    """
    params = list(league_ids) + year_params + list(league_ids) + year_params + [min_matches]
    return run_query(sql, tuple(params))


def get_matches_with_corners(league_id: str, years: list) -> list:
    years_str = "', '".join(years)
    sql = f"""
    SELECT
        matchId,
        homeTeam,
        awayTeam,
        LeagueId,
        Year,
        SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND')
            THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
        SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND')
            THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals,
        SUM(CASE WHEN name='Corner kicks' AND period IN ('1ST','2ND')
            THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_corners,
        SUM(CASE WHEN name='Corner kicks' AND period IN ('1ST','2ND')
            THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_corners
    FROM {TABLE}
    WHERE LeagueId = %s
      AND Year IN ('{years_str}')
      AND name IN ('Goals', 'Corner kicks')
    GROUP BY matchId, homeTeam, awayTeam, LeagueId, Year
    HAVING home_corners IS NOT NULL
       AND away_corners IS NOT NULL
    ORDER BY Year ASC, CAST(Round AS SIGNED) ASC
    """
    return run_query(sql, (league_id,))
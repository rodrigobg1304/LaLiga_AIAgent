import os
import mysql.connector
from typing import Optional
import configparser

leagues_dict = {
    8: "LaLiga"
}

# --- CONFIGURACIÓN ---
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'mainconfig_secret.ini'))
DB_TYPE = os.getenv("DB_TYPE", "mysql")  # "mysql" o "postgresql"
DB_HOST = config['MySQL']['host']
DB_PORT = config['MySQL']['port']
DB_NAME = config['MySQL']['database']
DB_USER = config['MySQL']['user']
DB_PASS = config['MySQL']['password']
TABLE   = config['MySQL']['table']


def get_connection():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


def run_query(sql: str, params: tuple = ()) -> list[dict]:
    """Ejecuta una query y devuelve lista de dicts."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True) if DB_TYPE == "mysql" else conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()


# ─────────────────────────────────────────────
# QUERIES POR CASO DE USO
# ─────────────────────────────────────────────

def get_teams() -> list[str]:
    """
    Get list of the teams available.
    :return: List of the teams
    """
    return [row["homeTeam"] for row in run_query(f"SELECT DISTINCT homeTeam FROM {TABLE} ORDER BY homeTeam ASC")]


def get_years() -> list[str]:
    """
    Get list of the years available.
    :return: List of the years
    """
    return [row["Year"] for row in run_query(f"SELECT DISTINCT Year FROM {TABLE} ORDER BY Year DESC")]


def get_stats() -> list[str]:
    """
    Get list of the stats available.
    :return: List of the stats
    """
    return [row["name"] for row in run_query(f"SELECT DISTINCT name FROM {TABLE} ORDER BY name ASC")]


def get_team_results(team: str, year: Optional[str] = None, top_n: Optional[int] = None) -> list[dict]:
    """
    Get the last team results for a team.
    :param team: Mandatory string for the club to search
    :param year: Optional string for the year to search
    :param top_n: Optional integer for the number of rounds to return
    :return: list with JSON
    """
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
    """Goles marcados y encajados totales."""
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
    """Estadística específica (possession, shots, xg...) de un equipo."""
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


def get_standings(year: str) -> list[dict]:
    """Clasificación calculada desde los partidos."""
    sql = f"""
        SELECT 
            team,
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
                SUM(CAST(homeValue AS SIGNED)) AS gf,
                SUM(CAST(awayValue AS SIGNED)) AS ga
            FROM {TABLE}
            WHERE name = 'Goals' AND Year = %s
            GROUP BY homeTeam, matchId
        
            UNION ALL
        
            SELECT 
                awayTeam AS team,
                matchId,
                SUM(CAST(awayValue AS SIGNED)) AS gf,
                SUM(CAST(homeValue AS SIGNED)) AS ga
            FROM {TABLE}
            WHERE name = 'Goals' AND Year = %s
            GROUP BY awayTeam, matchId
        ) AS match_totals
        GROUP BY team
        ORDER BY points DESC, goals_for - goals_against DESC
    """
    params = [year, year]
    return run_query(sql, tuple(params))


def get_top_stats(stat_name: str, year: str, top_n: int = 10) -> list[dict]:
    """Ranking de equipos por una estadística (ej: possession, shots...)."""
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

if __name__ == '__main__':
    team_test = "real-betis"
    matches_test = 5
    team_result = get_team_results(team=team_test, year="25/26", top_n=matches_test)
    print(team_result)
    goals_result = get_goals_scored(team=team_test)
    print(goals_result)
    import pandas as pd
    print(pd.DataFrame(goals_result))
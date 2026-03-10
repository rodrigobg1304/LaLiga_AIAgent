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
        cur.execute("SET SESSION sql_mode = ''")
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


def get_teams_by_league(league_id: str) -> list[str]:
    """
    Get list of the teams available filtered by league.
    :param league_id:
    :return:
    """
    return [row["homeTeam"] for row in run_query(
        f"SELECT DISTINCT homeTeam FROM {TABLE} WHERE LeagueId = %s ORDER BY homeTeam",
        (league_id,)
    )]


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


def get_standings(leagueId: str, year: str) -> list[dict]:
    """Clasificación calculada desde los partidos."""
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

    # Añadir posición manualmente
    for i, row in enumerate(results, start=1):
        row['position'] = i

    return results


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


def get_all_matches_chronological(league_id: Optional[str] = None) -> list[dict]:
    """
    Obtiene todos los partidos históricos ordenados cronológicamente para cálculo de Elo.
    Devuelve: homeTeam, awayTeam, home_goals, away_goals

    :param league_id: Opcional, filtrar por liga específica
    :return: Lista de dicts con partidos ordenados por Year ASC, Round ASC
    """
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
    """
    Obtiene todos los partidos de una liga en un año específico.
    Usado para calcular clasificaciones y estadísticas agregadas.

    :param league_id: ID de la liga (ej: '8' para LaLiga)
    :param year: Año/temporada (ej: '24/25')
    :return: Lista de dicts con homeTeam, awayTeam, home_goals, away_goals
    """
    print(f"[DB] 🔍 Ejecutando query para liga={league_id}, año={year}")  # 👈 Temporal

    sql = f"""
        SELECT homeTeam, awayTeam,
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
    """
    Cuenta cuántas temporadas distintas tiene un equipo en un rango de años.
    Usado para detectar equipos sin histórico (recién ascendidos).

    :param team: Nombre del equipo (ej: 'real-madrid')
    :param years: Lista de años a considerar (ej: ['24/25', '23/24', '22/23'])
    :return: Número de temporadas con datos (0 si no hay datos)
    """
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
    """
    Obtiene los últimos N partidos de un equipo con resultados de goles.
    Usado para calcular puntos de forma reciente.

    :param team: Nombre del equipo
    :param league_id: ID de la liga
    :param years: Lista de años a considerar
    :param role: 'home' o 'away' (filtro por localía)
    :param n: Número de partidos recientes a obtener
    :return: Lista de dicts con matchId, home_goals, away_goals
    """
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
    """
    Obtiene historial de enfrentamientos directos entre dos equipos.
    Devuelve partidos en cualquier orden (team1 home vs team2 away, o viceversa).

    :param team1: Primer equipo
    :param team2: Segundo equipo
    :param n: Número máximo de partidos H2H a obtener
    :return: Lista de dicts con matchId, homeTeam, awayTeam, home_goals, away_goals
    """
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
    """
    Obtiene partidos entre equipos vecinos (proxy cuando H2H directo es insuficiente).
    Usado cuando dos equipos tienen < 2 enfrentamientos históricos.

    :param home_neighbors: Lista de equipos vecinos del equipo local
    :param away_neighbors: Lista de equipos vecinos del equipo visitante
    :param n: Número máximo de partidos proxy a obtener
    :return: Lista de dicts con matchId, homeTeam, awayTeam, home_goals, away_goals
    """
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
    """
    Calcula promedio de una estadística específica para un equipo.

    :param team: Nombre del equipo
    :param role: 'home' o 'away' (localía)
    :param stat_name: Nombre de la estadística (ej: 'Goals', 'Expected goals', 'Possession')
    :param years: Lista de años a considerar
    :param n: Opcional - Si se especifica, limita a los últimos N partidos (forma reciente)
    :return: Promedio de la estadística (0.0 si no hay datos)
    """
    team_col = "homeTeam" if role == "home" else "awayTeam"
    stat_col = "homeValue" if role == "home" else "awayValue"

    if years:
        placeholders_years = ",".join(["%s"] * len(years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [team, stat_name] + years
    else:
        year_filter = ""
        params = [team, stat_name]

    # Subquery para agrupar por partido
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

    # Si n está especificado, limitar a últimos N partidos
    if n is not None:
        subquery += f" LIMIT %s"
        params.append(n)

    # Query principal: calcular promedio
    sql = f"""
        SELECT AVG(CAST(stat_value AS DECIMAL)) AS avg_stat
        FROM ({subquery}) AS matches
    """

    rows = run_query(sql, tuple(params))
    return float(rows[0]["avg_stat"]) if (rows and rows[0]["avg_stat"] is not None) else 0.0


def get_team_multiple_stats_average(team: str, role: str, years: list[str],
                                    n: Optional[int] = None) -> dict:
    """
    Obtiene promedios de TODAS las stats principales de un equipo en una sola query.
    Versión ultra-optimizada: 1 query en lugar de 10.

    :param team: Nombre del equipo
    :param role: 'home' o 'away' (localía)
    :param years: Lista de años a considerar
    :param n: Opcional - Si se especifica, limita a los últimos N partidos (forma reciente)
    :return: Dict con promedios de todas las stats
    """
    team_col = "homeTeam" if role == "home" else "awayTeam"
    stat_col = "homeValue" if role == "home" else "awayValue"

    if not years:
        # Sin años, retornar dict vacío
        return {}

    placeholders_years = ",".join(["%s"] * len(years))
    year_filter = f"AND Year IN ({placeholders_years})"

    # Subquery: agrupa todas las stats por partido
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

    # Si n está especificado, limitar a últimos N partidos
    if n is not None:
        subquery += " LIMIT %s"
        params.append(n)

    # Query principal: calcular promedios de todas las stats
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

    # Retornar dict con las stats (o dict vacío si no hay datos)
    if not rows or not rows[0]:
        return {}

    return {
        'goals': float(rows[0]['goals'] or 0),
        'ball_possession': float(rows[0]['ball_possession'] or 0),
        'total_shots': float(rows[0]['total_shots'] or 0),
        'shots_on_target': float(rows[0]['shots_on_target'] or 0),
        'gk_saves': float(rows[0]['gk_saves'] or 0),
        'big_chances': float(rows[0]['big_chances'] or 0),
        'accurate_passes': float(rows[0]['accurate_passes'] or 0),
        'tackles_won': float(rows[0]['tackles_won'] or 0),
        'interceptions': float(rows[0]['interceptions'] or 0),
        'blocked_shots': float(rows[0]['blocked_shots'] or 0)
    }


def get_team_win_rates(team: str, role: str, league_id: str, years: list[str],
                       n: Optional[int] = None) -> dict:
    """
    Calcula win/draw/loss rates de un equipo.

    :param team: Nombre del equipo
    :param role: 'home' o 'away' (para invertir lógica de victorias si es away)
    :param league_id: ID de la liga
    :param years: Lista de años a considerar
    :param n: Opcional - Si se especifica, limita a los últimos N partidos
    :return: Dict con wins, draws, losses, total
    """
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
        "wins": int(rows[0]["wins"] or 0),
        "draws": int(rows[0]["draws"] or 0),
        "losses": int(rows[0]["losses"] or 0),
        "total": int(rows[0]["total"])
    }


def get_team_goals_conceded_average(team: str, role: str, years: list[str],
                                    n: Optional[int] = None) -> float:
    """
    Calcula promedio de goles concedidos por un equipo.

    :param team: Nombre del equipo
    :param role: 'home' o 'away'
    :param years: Lista de años a considerar
    :param n: Opcional - Si se especifica, limita a los últimos N partidos
    :return: Promedio de goles concedidos
    """
    team_col = "homeTeam" if role == "home" else "awayTeam"
    goals_col = "awayValue" if role == "home" else "homeValue"  # Invertido

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
    """
    Obtiene partidos de un equipo con el total de goles (home + away).
    Usado para calcular over/under rates.

    :param team: Nombre del equipo
    :param years: Lista de años a considerar
    :param n: Opcional - Si se especifica, limita a los últimos N partidos
    :return: Lista de dicts con matchId y total_goals
    """
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


if __name__ == '__main__':
    team_test = "real-betis"
    matches_test = 5
    team_result = get_team_results(team=team_test, year="25/26", top_n=matches_test)
    print(team_result)
    goals_result = get_goals_scored(team=team_test)
    print(goals_result)
    import pandas as pd
    print(pd.DataFrame(goals_result))
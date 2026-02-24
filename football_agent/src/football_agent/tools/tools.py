from crewai.tools import tool
import requests
from football_agent.db import get_team_results, get_goals_scored, get_team_stats, get_standings

@tool("Resultados de equipo")
def team_results_tool(team: str, year: str = None, limit: int = None) -> list[dict]:
    """
    SIEMPRE usa esta tool para obtener resultados de partidos.
    Parámetros:
    - team: nombre del equipo en formato slug (ej: 'real-betis', 'real-madrid', 'barcelona')
    - year: temporada en formato '25/26', '24/25', '23/24' (opcional). Si no te especifican el año selecciona el último
    disponible y luego saca información general de todas las temporadas disponibles.
    - limit: número de partidos a devolver (opcional)
    NUNCA respondas sin llamar a esta tool primero cuando te pregunten por resultados o partidos.
    """
    return get_team_results(team, year, limit)


@tool("Goles de equipo")
def goals_tool(team: str, year: str = None) -> list[dict]:
    """
    Devuelve los goles marcados y encajados de un equipo por temporada.
    Usa esta tool cuando el usuario pregunte por goles, estadísticas ofensivas o defensivas.
    - team: nombre del equipo en formato slug (ej: 'real-betis', 'real-madrid', 'barcelona')
    - year: temporada en formato '25/26', '24/25', '23/24' (opcional). Si no te especifican el año selecciona el último
    disponible y luego saca información general de todas las temporadas disponibles.

    """
    return get_goals_scored(team, year)


@tool("Estadística específica")
def stat_tool(team: str, stat: str, year: str = None) -> list[dict]:
    """
    Devuelve una estadística concreta (possession, shots, xg...) de un equipo.
    Usa esta tool cuando el usuario pregunte por posesión, tiros, xG u otras métricas.
    - team: nombre del equipo en formato slug (ej: 'real-betis', 'real-madrid', 'barcelona')
    - stat: nombre del stat, tienes todas las opciones disponibles en la función get_stats() db.py, las stats vienen en
    inglés, si te preguntan en español busca la más aproximada.
    - year: temporada en formato '25/26', '24/25', '23/24' (opcional). Si no te especifican el año selecciona el último
    disponible y luego saca información general de todas las temporadas disponibles.

    """
    return get_team_stats(team, stat, year)


@tool("Clasificación de liga")
def standings_tool(league: str, year: str) -> list[dict]:
    """
    Devuelve la clasificación de una liga en una temporada.
    Usa esta tool cuando el usuario pregunte por la tabla, puntos o clasificación.
    """
    return get_standings(league, year)

@tool("Predicción de partido")
def prediction_tool(home_team: str, away_team: str, year: str = None) -> dict:
    """
    Predice el resultado y probabilidades de over/under goles de un partido.
    Usa esta tool cuando el usuario pregunte por predicciones, probabilidades,
    quién ganará, over/under goles o análisis de un partido futuro.
    Parámetros:
    - home_team: equipo local en formato slug (ej: 'real-betis', 'real-madrid')
    - away_team: equipo visitante en formato slug (ej: 'real-betis', 'real-madrid')
    - year: temporada en formato '25/26' (opcional)
    """
    try:
        response = requests.post(
            "http://localhost:8001/predict",
            json={"home_team": home_team, "away_team": away_team, "year": year}
        )
        return response.json()
    except Exception as e:
        return {"error": f"No se pudo conectar con el servicio de predicción: {e}"}
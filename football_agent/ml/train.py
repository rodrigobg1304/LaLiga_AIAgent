# train.py
import os, sys
import pandas as pd
import configparser
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, classification_report
import mysql.connector
import pickle
from datetime import datetime
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
# Importar funciones de tu db.py
from football_agent.db import TABLE, get_connection, run_query  # O las funciones que tengas


# ============================================================================
# CONFIGURACIÓN DE LOGGING CON COLORES
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Formatter personalizado con colores ANSI"""

    # Códigos de color ANSI
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[34m',  # Azul
        'WARNING': '\033[33m',  # Amarillo
        'ERROR': '\033[31m',  # Rojo
        'CRITICAL': '\033[35m',  # Magenta
        'SUCCESS': '\033[32m',  # Verde
    }
    RESET = '\033[0m'
    WHITE = '\033[37m'  # Blanco explícito

    def format(self, record):
        # Guardar el levelname original
        levelname_original = record.levelname

        # Obtener color
        log_color = self.COLORS.get(levelname_original, '')

        # Formatear el tiempo primero
        record.asctime = self.formatTime(record, self.datefmt)

        # Formatear con color SOLO en levelname y RESET + WHITE explícito en el mensaje
        formatted = (
            f"{self.WHITE}{record.asctime}{self.RESET} - "
            f"{log_color}{levelname_original}{self.RESET} - "
            f"{self.WHITE}{record.getMessage()}{self.RESET}"
        )

        return formatted


# Añadir nivel SUCCESS personalizado
logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, 'SUCCESS')


def success(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.SUCCESS):
        self._log(logging.SUCCESS, message, args, **kwargs)


logging.Logger.success = success

# Configurar logging
handler = logging.StreamHandler()
formatter = ColoredFormatter(datefmt='%Y-%m-%d %H:%M:%S')

handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Parámetros globales del modelo"""
    # Ligas a incluir
    LEAGUES = {
        '8': "LaLiga",
        '17': "Premier League",
        '23': "Serie A",
    }

    # Split temporal
    TEST_SEASON = '25/26'  # Última temporada para test

    # Features Elo
    ELO_K = 40
    ELO_SCALE = 600
    # ELO_HOME_ADVANTAGE = 50
    ELO_INITIAL = 1500

    # Features Form
    FORM_WINDOW = 10  # Últimos N partidos
    FORM_WEIGHT_RECENT = 0.65  # 65% peso a forma reciente

    # H2H
    MIN_H2H_MATCHES = 2
    H2H_LOOKBACK = 5  # Últimos 5 enfrentamientos

    # Modelo
    MIN_PROB = 0.04  # Floor de probabilidad mínima
    RANDOM_STATE = 42

    # Output (directorio base)
    MODELS_DIR = './models'


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def load_data(league_id: str):
    """
    Carga datos de una liga específica desde MySQL usando db.py

    Args:
        league_id: ID de la liga (ej: 8 para LaLiga)

    Returns:
        pd.DataFrame: DataFrame con partidos de esa liga ordenados cronológicamente
        Columnas: matchId, date, season, round, home_team, away_team,
                  home_goals, away_goals, result, league_id
    """

    logger.info(f"Cargando datos de liga {league_id}...")

    # Stats a incluir (las 10 seleccionadas)
    stats_to_load = [
        'Goals',
        'Shots on target',
        'Ball possession',
        'Total shots',
        'Goalkeeper saves',
        'Big chances',
        'Accurate passes',
        'Tackles won',
        'Interceptions',
        'Blocked shots'
    ]

    sql = f"""
            SELECT 
                matchId,
                Year as season,
                CAST(Round AS SIGNED) as round,
                homeTeam as home_team,
                awayTeam as away_team,
                LeagueId as league_id,

                -- Goals
                MAX(CASE WHEN name = 'Goals' THEN CAST(homeValue AS SIGNED) END) as home_goals,
                MAX(CASE WHEN name = 'Goals' THEN CAST(awayValue AS SIGNED) END) as away_goals,

                -- Shots on target
                MAX(CASE WHEN name = 'Shots on target' THEN CAST(homeValue AS SIGNED) END) as home_shots_on_target,
                MAX(CASE WHEN name = 'Shots on target' THEN CAST(awayValue AS SIGNED) END) as away_shots_on_target,

                -- Ball possession
                MAX(CASE WHEN name = 'Ball possession' THEN CAST(homeValue AS SIGNED) END) as home_possession,
                MAX(CASE WHEN name = 'Ball possession' THEN CAST(awayValue AS SIGNED) END) as away_possession,

                -- Total shots
                MAX(CASE WHEN name = 'Total shots' THEN CAST(homeValue AS SIGNED) END) as home_total_shots,
                MAX(CASE WHEN name = 'Total shots' THEN CAST(awayValue AS SIGNED) END) as away_total_shots,

                -- Goalkeeper saves
                MAX(CASE WHEN name = 'Goalkeeper saves' THEN CAST(homeValue AS SIGNED) END) as home_gk_saves,
                MAX(CASE WHEN name = 'Goalkeeper saves' THEN CAST(awayValue AS SIGNED) END) as away_gk_saves,

                -- Big chances
                MAX(CASE WHEN name = 'Big chances' THEN CAST(homeValue AS SIGNED) END) as home_big_chances,
                MAX(CASE WHEN name = 'Big chances' THEN CAST(awayValue AS SIGNED) END) as away_big_chances,

                -- Accurate passes
                MAX(CASE WHEN name = 'Accurate passes' THEN CAST(homeValue AS SIGNED) END) as home_accurate_passes,
                MAX(CASE WHEN name = 'Accurate passes' THEN CAST(awayValue AS SIGNED) END) as away_accurate_passes,

                -- Tackles won
                MAX(CASE WHEN name = 'Tackles won' THEN CAST(homeValue AS SIGNED) END) as home_tackles_won,
                MAX(CASE WHEN name = 'Tackles won' THEN CAST(awayValue AS SIGNED) END) as away_tackles_won,

                -- Interceptions
                MAX(CASE WHEN name = 'Interceptions' THEN CAST(homeValue AS SIGNED) END) as home_interceptions,
                MAX(CASE WHEN name = 'Interceptions' THEN CAST(awayValue AS SIGNED) END) as away_interceptions,

                -- Blocked shots
                MAX(CASE WHEN name = 'Blocked shots' THEN CAST(homeValue AS SIGNED) END) as home_blocked_shots,
                MAX(CASE WHEN name = 'Blocked shots' THEN CAST(awayValue AS SIGNED) END) as away_blocked_shots,

                -- Resultado (derivado de Goals)
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

    rows = run_query(sql, (league_id,))
    df = pd.DataFrame(rows)

    # Convertir todas las columnas numéricas a float
    numeric_columns = [
        'round', 'league_id',
        'home_goals', 'away_goals',
        'home_shots_on_target', 'away_shots_on_target',
        'home_possession', 'away_possession',
        'home_total_shots', 'away_total_shots',
        'home_gk_saves', 'away_gk_saves',
        'home_big_chances', 'away_big_chances',
        'home_accurate_passes', 'away_accurate_passes',
        'home_tackles_won', 'away_tackles_won',
        'home_interceptions', 'away_interceptions',
        'home_blocked_shots', 'away_blocked_shots'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info(f"  - {len(df)} partidos cargados")
    logger.info(f"  - Temporadas: {df['season'].unique()}")
    logger.info(f"  - Equipos únicos: {df['home_team'].nunique()}")

    return df


def calculate_elo(df):
    """
    Calcula Elo rating separado para home/away de cada equipo

    Args:
        df: DataFrame con partidos ordenados cronológicamente
            Debe tener: season, round, home_team, away_team, result

    Returns:
        pd.DataFrame: DataFrame original + columnas elo_home, elo_away, elo_diff
    """
    logger.info("Calculando Elo ratings separados home/away...")

    # Inicializar diccionario de Elo por equipo
    # Estructura: team -> {'home': rating, 'away': rating}
    elo_ratings = {}

    # Listas para almacenar Elo de cada partido
    elo_home_list = []
    elo_away_list = []
    elo_diff_list = []

    # Parámetros
    K = Config.ELO_K
    SCALE = Config.ELO_SCALE
    INITIAL = Config.ELO_INITIAL

    # Control de temporada para regresión
    current_season = None

    # Iterar sobre cada partido en orden cronológico
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        result = row['result']
        season = row['season']

        # Detectar cambio de temporada
        if current_season is None:
            current_season = season

        if season != current_season:
            # Nueva temporada: regresión a la media 65/35
            logger.info(f"  - Nueva temporada detectada: {season}")
            for team in elo_ratings:
                elo_ratings[team]['home'] = elo_ratings[team]['home'] * 0.75 + INITIAL * 0.25
                elo_ratings[team]['away'] = elo_ratings[team]['away'] * 0.75 + INITIAL * 0.25
            current_season = season

        # Inicializar equipos si no existen
        if home_team not in elo_ratings:
            elo_ratings[home_team] = {'home': INITIAL, 'away': INITIAL}
        if away_team not in elo_ratings:
            elo_ratings[away_team] = {'home': INITIAL, 'away': INITIAL}

        # Obtener Elo PRE-partido según contexto (home/away)
        elo_home = elo_ratings[home_team]['home']  # Elo del local jugando EN CASA
        elo_away = elo_ratings[away_team]['away']  # Elo del visitante jugando FUERA

        # Guardar Elo PRE-partido (esto es lo que usa el modelo)
        elo_home_list.append(elo_home)
        elo_away_list.append(elo_away)
        elo_diff_list.append(elo_home - elo_away)

        # Calcular probabilidad esperada (SIN home advantage fijo)
        expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / SCALE))

        # Resultado real (1 = local gana, 0.5 = empate, 0 = visitante gana)
        if result == '1':
            actual_home = 1.0
        elif result == 'X':
            actual_home = 0.5
        else:  # result == '2'
            actual_home = 0.0

        # Actualizar Elo POST-partido
        # El local actualiza su Elo HOME
        elo_ratings[home_team]['home'] = elo_home + K * (actual_home - expected_home)

        # El visitante actualiza su Elo AWAY
        elo_ratings[away_team]['away'] = elo_away + K * ((1 - actual_home) - (1 - expected_home))

    # Añadir columnas al DataFrame
    df = df.copy()
    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = elo_diff_list

    # Calcular estadísticas de spread home/away
    home_away_spreads = []
    for team, ratings in elo_ratings.items():
        spread = ratings['home'] - ratings['away']
        home_away_spreads.append(spread)

    avg_spread = np.mean(home_away_spreads)
    max_spread = max(home_away_spreads)
    min_spread = min(home_away_spreads)

    logger.success(f"  - Elo calculado para {len(elo_ratings)} equipos")
    logger.info(
        f"  - Rango Elo Home: {min([r['home'] for r in elo_ratings.values()]):.0f} - {max([r['home'] for r in elo_ratings.values()]):.0f}")
    logger.info(
        f"  - Rango Elo Away: {min([r['away'] for r in elo_ratings.values()]):.0f} - {max([r['away'] for r in elo_ratings.values()]):.0f}")
    logger.info(f"  - Spread Home-Away promedio: {avg_spread:.0f} (min={min_spread:.0f}, max={max_spread:.0f})")

    return df

def calculate_form_features(df):
    """
    Calcula métricas de forma reciente (weighted) incluyendo stats avanzadas

    Args:
        df: DataFrame con partidos ordenados cronológicamente

    Returns:
        pd.DataFrame: DataFrame con features de forma añadidas
    """
    logger.info("Calculando features de forma...")

    # Parámetros
    WINDOW = Config.FORM_WINDOW
    WEIGHT_RECENT = Config.FORM_WEIGHT_RECENT

    # Diccionarios para tracking por equipo
    # Estructura: team -> {'home': [resultados], 'away': [resultados]}
    team_history = {}

    # Listas para features
    home_form_features = []
    away_form_features = []

    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        result = row['result']

        # Inicializar histórico si no existe
        if home_team not in team_history:
            team_history[home_team] = {'home': [], 'away': []}
        if away_team not in team_history:
            team_history[away_team] = {'home': [], 'away': []}

        # Calcular features PRE-partido para equipo local
        home_features = calculate_team_form(
            team_history[home_team]['home'],
            WINDOW,
            WEIGHT_RECENT
        )

        # Calcular features PRE-partido para equipo visitante
        away_features = calculate_team_form(
            team_history[away_team]['away'],
            WINDOW,
            WEIGHT_RECENT
        )

        # Guardar features
        home_form_features.append(home_features)
        away_form_features.append(away_features)

        # Actualizar histórico POST-partido
        # Para el equipo local
        home_match_data = {
            'goals_for': row['home_goals'],
            'goals_against': row['away_goals'],
            'points': 3 if result == '1' else (1 if result == 'X' else 0),
            'win': 1 if result == '1' else 0,

            # Stats avanzadas - usar .get() con default NaN si no existe
            'shots_on_target': row.get('home_shots_on_target', np.nan),
            'possession': row.get('home_possession', np.nan),
            'total_shots': row.get('home_total_shots', np.nan),
            'gk_saves': row.get('home_gk_saves', np.nan),
            'big_chances': row.get('home_big_chances', np.nan),  # ← Puede ser NaN
            'accurate_passes': row.get('home_accurate_passes', np.nan),
            'tackles_won': row.get('home_tackles_won', np.nan),
            'interceptions': row.get('home_interceptions', np.nan),
            'blocked_shots': row.get('home_blocked_shots', np.nan)
        }
        team_history[home_team]['home'].append(home_match_data)

        # Para el equipo visitante
        away_match_data = {
            'goals_for': row['away_goals'],
            'goals_against': row['home_goals'],
            'points': 3 if result == '2' else (1 if result == 'X' else 0),
            'win': 1 if result == '2' else 0,

            # Stats avanzadas - usar .get() con default NaN si no existe
            'shots_on_target': row.get('away_shots_on_target', np.nan),
            'possession': row.get('away_possession', np.nan),
            'total_shots': row.get('away_total_shots', np.nan),
            'gk_saves': row.get('away_gk_saves', np.nan),
            'big_chances': row.get('away_big_chances', np.nan),  # ← Puede ser NaN
            'accurate_passes': row.get('away_accurate_passes', np.nan),
            'tackles_won': row.get('away_tackles_won', np.nan),
            'interceptions': row.get('away_interceptions', np.nan),
            'blocked_shots': row.get('away_blocked_shots', np.nan)
        }
        team_history[away_team]['away'].append(away_match_data)

    # Convertir a DataFrame y añadir al original
    df = df.copy()

    # Features básicas del equipo local
    df['home_win_rate'] = [f['win_rate'] for f in home_form_features]
    df['home_goals_for_avg'] = [f['goals_for_avg'] for f in home_form_features]
    df['home_goals_against_avg'] = [f['goals_against_avg'] for f in home_form_features]
    df['home_points_avg'] = [f['points_avg'] for f in home_form_features]

    # Features avanzadas del equipo local
    df['home_shots_on_target_avg'] = [f['shots_on_target_avg'] for f in home_form_features]
    df['home_possession_avg'] = [f['possession_avg'] for f in home_form_features]
    df['home_total_shots_avg'] = [f['total_shots_avg'] for f in home_form_features]
    df['home_gk_saves_avg'] = [f['gk_saves_avg'] for f in home_form_features]
    df['home_big_chances_avg'] = [f['big_chances_avg'] for f in home_form_features]
    df['home_accurate_passes_avg'] = [f['accurate_passes_avg'] for f in home_form_features]
    df['home_tackles_won_avg'] = [f['tackles_won_avg'] for f in home_form_features]
    df['home_interceptions_avg'] = [f['interceptions_avg'] for f in home_form_features]
    df['home_blocked_shots_avg'] = [f['blocked_shots_avg'] for f in home_form_features]

    # Features básicas del equipo visitante
    df['away_win_rate'] = [f['win_rate'] for f in away_form_features]
    df['away_goals_for_avg'] = [f['goals_for_avg'] for f in away_form_features]
    df['away_goals_against_avg'] = [f['goals_against_avg'] for f in away_form_features]
    df['away_points_avg'] = [f['points_avg'] for f in away_form_features]

    # Features avanzadas del equipo visitante
    df['away_shots_on_target_avg'] = [f['shots_on_target_avg'] for f in away_form_features]
    df['away_possession_avg'] = [f['possession_avg'] for f in away_form_features]
    df['away_total_shots_avg'] = [f['total_shots_avg'] for f in away_form_features]
    df['away_gk_saves_avg'] = [f['gk_saves_avg'] for f in away_form_features]
    df['away_big_chances_avg'] = [f['big_chances_avg'] for f in away_form_features]
    df['away_accurate_passes_avg'] = [f['accurate_passes_avg'] for f in away_form_features]
    df['away_tackles_won_avg'] = [f['tackles_won_avg'] for f in away_form_features]
    df['away_interceptions_avg'] = [f['interceptions_avg'] for f in away_form_features]
    df['away_blocked_shots_avg'] = [f['blocked_shots_avg'] for f in away_form_features]

    logger.success(f"  - Features de forma calculadas para {len(team_history)} equipos")

    return df


def add_premier_features(df, league_name):
    """
    Añade features específicas para Premier League

    Args:
        df: DataFrame con features base
        league_name: Nombre de la liga

    Returns:
        pd.DataFrame: DataFrame con features adicionales si es Premier
    """
    if league_name != 'premier_league':
        return df  # Solo aplicar a Premier

    logger.info("  - Añadiendo features específicas de Premier League...")

    df = df.copy()

    # 1. VARIABILIDAD/CAOS (Premier más impredecible)
    # Calcular desviación estándar de goles en últimos N partidos
    # Necesitamos acceso al histórico por equipo

    # 2. FASE DE TEMPORADA (early season más caótico)
    df['season_progress'] = df['round'] / 38.0  # % de temporada completado
    df['is_early_season'] = (df['round'] <= 10).astype(int)
    df['is_late_season'] = (df['round'] >= 30).astype(int)

    # 3. MOMENTUM/RACHA (últimos 3 partidos)
    # Esto requiere calcular en calculate_form_features()
    # Por ahora, usaremos proxy: diferencia entre form reciente vs histórica
    df['home_form_momentum'] = df['home_win_rate'] - 0.4  # Desviación de media
    df['away_form_momentum'] = df['away_win_rate'] - 0.3

    # 4. EQUILIBRIO EXTREMO (cuando diferencias son MUY pequeñas)
    df['ultra_balanced'] = (
            (df['elo_diff_abs'] < 30) &
            (df['form_balance'] < 0.08)
    ).astype(int)

    # 5. INTERACCIÓN: Equipos débiles en casa vs fuertes fuera
    df['underdog_home_vs_strong_away'] = (
            (df['elo_home'] < 1450) &
            (df['elo_away'] > 1550)
    ).astype(int)

    logger.success(f"  - {5} features específicas de Premier añadidas")

    return df


def safe_mean(values, default=np.nan):
    """
    Calcula mean ignorando NaN, retorna default si todos son NaN

    Args:
        values: Lista de valores
        default: Valor por defecto si no hay datos válidos

    Returns:
        float: Media o default
    """
    filtered = [v for v in values if not (np.isnan(v) if isinstance(v, float) else False)]
    return np.mean(filtered) if len(filtered) > 0 else default


def calculate_team_form(match_history, window, weight_recent):
    """
    Calcula métricas de forma para un equipo (incluyendo stats avanzadas)

    Args:
        match_history: Lista de dicts con info de partidos pasados
        window: Ventana de partidos recientes
        weight_recent: Peso para partidos recientes (0.65)

    Returns:
        dict: Features de forma
    """
    if len(match_history) == 0:
        # Sin histórico: valores neutros
        return {
            'win_rate': 0.33,
            'goals_for_avg': 1.5,
            'goals_against_avg': 1.5,
            'points_avg': 1.0,

            # Stats avanzadas - valores neutrales basados en EDA
            'shots_on_target_avg': 4.5,
            'possession_avg': 50.0,
            'total_shots_avg': 12.0,
            'gk_saves_avg': 3.0,
            'big_chances_avg': 2.5,
            'accurate_passes_avg': 300.0,
            'tackles_won_avg': 8.0,
            'interceptions_avg': 7.0,
            'blocked_shots_avg': 2.0
        }

    # Ajustar peso según cantidad de datos
    if len(match_history) < 5:
        # Pocos datos: 100% reciente (aprende rápido)
        effective_weight = 1.0
    else:
        # Suficiente histórico: usar peso configurado
        effective_weight = weight_recent

    # Obtener partidos recientes (últimos WINDOW)
    recent = match_history[-window:] if len(match_history) >= window else match_history

    # Calcular promedios recientes - BÁSICOS
    recent_win_rate = np.mean([m['win'] for m in recent])
    recent_gf_avg = np.mean([m['goals_for'] for m in recent])
    recent_ga_avg = np.mean([m['goals_against'] for m in recent])
    recent_points_avg = np.mean([m['points'] for m in recent])

    # Calcular promedios recientes - AVANZADOS
    # Filtrar NaN primero, luego calcular mean
    def safe_mean(values, default=np.nan):
        """Calcula mean ignorando NaN, retorna default si todos son NaN"""
        filtered = [v for v in values if not np.isnan(v)]
        return np.mean(filtered) if len(filtered) > 0 else default

    recent_shots_on_target = safe_mean([m['shots_on_target'] for m in recent], default=4.5)
    recent_possession = safe_mean([m['possession'] for m in recent], default=50.0)
    recent_total_shots = safe_mean([m['total_shots'] for m in recent], default=12.0)
    recent_gk_saves = safe_mean([m['gk_saves'] for m in recent], default=3.0)
    recent_big_chances = safe_mean([m['big_chances'] for m in recent], default=2.5)
    recent_accurate_passes = safe_mean([m['accurate_passes'] for m in recent], default=300.0)
    recent_tackles_won = safe_mean([m['tackles_won'] for m in recent], default=8.0)
    recent_interceptions = safe_mean([m['interceptions'] for m in recent], default=7.0)
    recent_blocked_shots = safe_mean([m['blocked_shots'] for m in recent], default=2.0)

    # Si hay suficiente histórico, calcular promedios totales
    if len(match_history) > window:
        # BÁSICOS
        total_win_rate = np.mean([m['win'] for m in match_history])
        total_gf_avg = np.mean([m['goals_for'] for m in match_history])
        total_ga_avg = np.mean([m['goals_against'] for m in match_history])
        total_points_avg = np.mean([m['points'] for m in match_history])

        # AVANZADOS - usar safe_mean
        total_shots_on_target = safe_mean([m['shots_on_target'] for m in match_history], default=4.5)
        total_possession = safe_mean([m['possession'] for m in match_history], default=50.0)
        total_total_shots = safe_mean([m['total_shots'] for m in match_history], default=12.0)
        total_gk_saves = safe_mean([m['gk_saves'] for m in match_history], default=3.0)
        total_big_chances = safe_mean([m['big_chances'] for m in match_history], default=2.5)
        total_accurate_passes = safe_mean([m['accurate_passes'] for m in match_history], default=300.0)
        total_tackles_won = safe_mean([m['tackles_won'] for m in match_history], default=8.0)
        total_interceptions = safe_mean([m['interceptions'] for m in match_history], default=7.0)
        total_blocked_shots = safe_mean([m['blocked_shots'] for m in match_history], default=2.0)

        # Weighted average con effective_weight ajustado - BÁSICOS
        win_rate = recent_win_rate * effective_weight + total_win_rate * (1 - effective_weight)
        gf_avg = recent_gf_avg * effective_weight + total_gf_avg * (1 - effective_weight)
        ga_avg = recent_ga_avg * effective_weight + total_ga_avg * (1 - effective_weight)
        points_avg = recent_points_avg * effective_weight + total_points_avg * (1 - effective_weight)

        # Weighted average - AVANZADOS
        shots_on_target_avg = recent_shots_on_target * effective_weight + total_shots_on_target * (1 - effective_weight)
        possession_avg = recent_possession * effective_weight + total_possession * (1 - effective_weight)
        total_shots_avg = recent_total_shots * effective_weight + total_total_shots * (1 - effective_weight)
        gk_saves_avg = recent_gk_saves * effective_weight + total_gk_saves * (1 - effective_weight)
        big_chances_avg = recent_big_chances * effective_weight + total_big_chances * (1 - effective_weight)
        accurate_passes_avg = recent_accurate_passes * effective_weight + total_accurate_passes * (1 - effective_weight)
        tackles_won_avg = recent_tackles_won * effective_weight + total_tackles_won * (1 - effective_weight)
        interceptions_avg = recent_interceptions * effective_weight + total_interceptions * (1 - effective_weight)
        blocked_shots_avg = recent_blocked_shots * effective_weight + total_blocked_shots * (1 - effective_weight)
    else:
        # Solo histórico reciente disponible
        win_rate = recent_win_rate
        gf_avg = recent_gf_avg
        ga_avg = recent_ga_avg
        points_avg = recent_points_avg

        shots_on_target_avg = recent_shots_on_target
        possession_avg = recent_possession
        total_shots_avg = recent_total_shots
        gk_saves_avg = recent_gk_saves
        big_chances_avg = recent_big_chances
        accurate_passes_avg = recent_accurate_passes
        tackles_won_avg = recent_tackles_won
        interceptions_avg = recent_interceptions
        blocked_shots_avg = recent_blocked_shots

    return {
        # Básicos
        'win_rate': win_rate,
        'goals_for_avg': gf_avg,
        'goals_against_avg': ga_avg,
        'points_avg': points_avg,

        # Avanzados
        'shots_on_target_avg': shots_on_target_avg,
        'possession_avg': possession_avg,
        'total_shots_avg': total_shots_avg,
        'gk_saves_avg': gk_saves_avg,
        'big_chances_avg': big_chances_avg,
        'accurate_passes_avg': accurate_passes_avg,
        'tackles_won_avg': tackles_won_avg,
        'interceptions_avg': interceptions_avg,
        'blocked_shots_avg': blocked_shots_avg
    }


def calculate_h2h_features(df):
    """
    Calcula features de enfrentamientos directos
    Usa proxy teams si H2H < MIN_H2H_MATCHES

    Args:
        df: DataFrame con partidos ordenados cronológicamente

    Returns:
        pd.DataFrame: DataFrame con features H2H añadidas
    """
    logger.info("Calculando features H2H...")

    # Parámetros
    MIN_H2H = Config.MIN_H2H_MATCHES
    H2H_LOOKBACK = Config.H2H_LOOKBACK

    # Diccionario para almacenar enfrentamientos
    # Estructura: (home_team, away_team) -> [list of match results]
    h2h_history = {}

    # Listas para features
    h2h_features_list = []

    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_goals = row['home_goals']
        away_goals = row['away_goals']
        result = row['result']
        season = row['season']

        # Crear clave para este enfrentamiento
        h2h_key = (home_team, away_team)

        # Obtener histórico H2H
        if h2h_key not in h2h_history:
            h2h_history[h2h_key] = []

        past_h2h = h2h_history[h2h_key][-H2H_LOOKBACK:]  # Últimos N enfrentamientos

        # Calcular features PRE-partido
        if len(past_h2h) >= MIN_H2H:
            # Suficientes enfrentamientos directos
            h2h_features = calculate_h2h_stats(past_h2h, use_proxy=False)
        else:
            # Usar proxy teams (equipos con posición similar)
            proxy_h2h = get_proxy_h2h(df, idx, home_team, away_team, season)
            h2h_features = calculate_h2h_stats(proxy_h2h, use_proxy=True)

        h2h_features_list.append(h2h_features)

        # Actualizar histórico POST-partido
        h2h_match_data = {
            'home_goals': home_goals,
            'away_goals': away_goals,
            'result': result,
            'home_win': 1 if result == '1' else 0
        }
        h2h_history[h2h_key].append(h2h_match_data)

    # Añadir features al DataFrame
    df = df.copy()
    df['h2h_home_win_rate'] = [f['home_win_rate'] for f in h2h_features_list]
    df['h2h_home_goals_avg'] = [f['home_goals_avg'] for f in h2h_features_list]
    df['h2h_away_goals_avg'] = [f['away_goals_avg'] for f in h2h_features_list]
    df['h2h_total_goals_avg'] = [f['total_goals_avg'] for f in h2h_features_list]
    df['h2h_used_proxy'] = [f['used_proxy'] for f in h2h_features_list]

    logger.success(f"  - Features H2H calculadas")
    proxy_count = sum(df['h2h_used_proxy'])
    logger.info(f"  - Proxies usados: {proxy_count}/{len(df)} ({proxy_count / len(df) * 100:.1f}%)")

    return df


def calculate_h2h_stats(h2h_matches, use_proxy):
    """
    Calcula estadísticas de enfrentamientos H2H

    Args:
        h2h_matches: Lista de dicts con partidos H2H pasados
        use_proxy: Boolean indicando si se usó proxy

    Returns:
        dict: Features H2H
    """
    if len(h2h_matches) == 0:
        # Sin datos: valores neutros
        return {
            'home_win_rate': 0.33,
            'home_goals_avg': 1.5,
            'away_goals_avg': 1.5,
            'total_goals_avg': 3.0,
            'used_proxy': int(use_proxy)
        }

    # Calcular promedios
    home_win_rate = np.mean([m['home_win'] for m in h2h_matches])
    home_goals_avg = np.mean([m['home_goals'] for m in h2h_matches])
    away_goals_avg = np.mean([m['away_goals'] for m in h2h_matches])
    total_goals_avg = np.mean([m['home_goals'] + m['away_goals'] for m in h2h_matches])

    return {
        'home_win_rate': home_win_rate,
        'home_goals_avg': home_goals_avg,
        'away_goals_avg': away_goals_avg,
        'total_goals_avg': total_goals_avg,
        'used_proxy': int(use_proxy)
    }


def get_proxy_h2h(df, current_idx, home_team, away_team, season):
    """
    Obtiene enfrentamientos de equipos proxy cuando no hay suficiente H2H directo

    Estrategia: Usar equipos con posición similar en la tabla

    Args:
        df: DataFrame completo
        current_idx: Índice del partido actual
        home_team: Equipo local
        away_team: Equipo visitante
        season: Temporada actual

    Returns:
        list: Lista de enfrentamientos proxy
    """
    # Filtrar partidos ANTES del actual en la MISMA temporada
    past_matches = df.iloc[:current_idx]
    same_season = past_matches[past_matches['season'] == season]

    if len(same_season) < 10:
        # Temporada muy temprana, no hay suficiente info para proxies
        return []

    # Calcular posiciones aproximadas en la tabla
    standings = calculate_mini_standings(same_season)

    if home_team not in standings or away_team not in standings:
        return []

    home_position = standings[home_team]['position']
    away_position = standings[away_team]['position']

    # Buscar equipos proxy (±2 posiciones, ±3 si están en los extremos)
    home_proxies = get_teams_near_position(standings, home_position, margin=2)
    away_proxies = get_teams_near_position(standings, away_position, margin=2)

    # Buscar partidos entre equipos proxy
    proxy_matches = []
    for _, match in same_season.iterrows():
        if (match['home_team'] in home_proxies and
                match['away_team'] in away_proxies and
                match['home_team'] != home_team and
                match['away_team'] != away_team):
            proxy_match = {
                'home_goals': match['home_goals'],
                'away_goals': match['away_goals'],
                'result': match['result'],
                'home_win': 1 if match['result'] == '1' else 0
            }
            proxy_matches.append(proxy_match)

    return proxy_matches[-Config.H2H_LOOKBACK:]  # Últimos N


def calculate_mini_standings(matches_df):
    """
    Calcula tabla de posiciones simplificada

    Args:
        matches_df: DataFrame con partidos

    Returns:
        dict: {team: {'position': int, 'points': int}}
    """
    standings = {}

    for _, match in matches_df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        result = match['result']

        # Inicializar equipos
        if home_team not in standings:
            standings[home_team] = {'points': 0, 'gd': 0}
        if away_team not in standings:
            standings[away_team] = {'points': 0, 'gd': 0}

        # Asignar puntos
        if result == '1':
            standings[home_team]['points'] += 3
        elif result == '2':
            standings[away_team]['points'] += 3
        else:
            standings[home_team]['points'] += 1
            standings[away_team]['points'] += 1

        # Goal difference
        standings[home_team]['gd'] += (match['home_goals'] - match['away_goals'])
        standings[away_team]['gd'] += (match['away_goals'] - match['home_goals'])

    # Ordenar por puntos y GD
    sorted_teams = sorted(
        standings.items(),
        key=lambda x: (x[1]['points'], x[1]['gd']),
        reverse=True
    )

    # Asignar posiciones
    for position, (team, stats) in enumerate(sorted_teams, start=1):
        standings[team]['position'] = position

    return standings


def get_teams_near_position(standings, target_position, margin=2):
    """
    Obtiene equipos con posiciones cercanas en la tabla

    Args:
        standings: Dict con posiciones
        target_position: Posición objetivo
        margin: Margen de posiciones (±2 por defecto, ±3 en extremos)

    Returns:
        list: Lista de equipos en rango
    """
    max_position = max(s['position'] for s in standings.values())

    # Ajustar margen en extremos de la tabla
    if target_position <= 3 or target_position >= max_position - 2:
        margin = 3

    nearby_teams = [
        team for team, stats in standings.items()
        if abs(stats['position'] - target_position) <= margin
    ]

    return nearby_teams


def prepare_features(df, league_name='laliga'):
    """
    Prepara X, y para entrenamiento

    Args:
        df: DataFrame con todas las features calculadas

    Returns:
        tuple: (X, y) arrays numpy para sklearn
    """
    logger.info("Preparando features para modelo...")

    # ✅ CALCULAR FEATURES DE BALANCE/EQUILIBRIO
    df = df.copy()

    # Diferencias absolutas (equilibrio = valores bajos)
    df['elo_diff_abs'] = np.abs(df['elo_diff'])
    df['form_balance'] = np.abs(df['home_win_rate'] - df['away_win_rate'])
    df['goals_balance'] = np.abs(df['home_goals_for_avg'] - df['away_goals_for_avg'])
    df['possession_balance'] = np.abs(df['home_possession_avg'] - df['away_possession_avg'])
    df['shots_balance'] = np.abs(df['home_total_shots_avg'] - df['away_total_shots_avg'])
    df['shots_on_target_balance'] = np.abs(df['home_shots_on_target_avg'] - df['away_shots_on_target_avg'])

    # Lista de columnas de features (EXTENDIDA)
    feature_columns = [
        # Elo (3)
        'elo_home',
        'elo_away',
        'elo_diff',

        # Form - Home BÁSICAS (4)
        'home_win_rate',
        'home_goals_for_avg',
        'home_goals_against_avg',
        'home_points_avg',

        # Form - Home AVANZADAS (9)
        'home_shots_on_target_avg',
        'home_possession_avg',
        'home_total_shots_avg',
        'home_gk_saves_avg',
        'home_big_chances_avg',
        'home_accurate_passes_avg',
        'home_tackles_won_avg',
        'home_interceptions_avg',
        'home_blocked_shots_avg',

        # Form - Away BÁSICAS (4)
        'away_win_rate',
        'away_goals_for_avg',
        'away_goals_against_avg',
        'away_points_avg',

        # Form - Away AVANZADAS (9)
        'away_shots_on_target_avg',
        'away_possession_avg',
        'away_total_shots_avg',
        'away_gk_saves_avg',
        'away_big_chances_avg',
        'away_accurate_passes_avg',
        'away_tackles_won_avg',
        'away_interceptions_avg',
        'away_blocked_shots_avg',

        # H2H (5)
        'h2h_home_win_rate',
        'h2h_home_goals_avg',
        'h2h_away_goals_avg',
        'h2h_total_goals_avg',
        'h2h_used_proxy',

        # ✅ FEATURES DE BALANCE (6)
        'elo_diff_abs',
        'form_balance',
        'goals_balance',
        'possession_balance',
        'shots_balance',
        'shots_on_target_balance'
    ]

    # Premier (balance ya calculado)
    df = add_premier_features(df, league_name)

    # AÑADIR FEATURES PREMIER SI APLICA
    if league_name == 'premier_league':
        premier_features = [
            'season_progress',
            'is_early_season',
            'is_late_season',
            'home_form_momentum',
            'away_form_momentum',
            'ultra_balanced',
            'underdog_home_vs_strong_away'
        ]
        feature_columns.extend(premier_features)
        logger.info(f"  - Total features: {len(feature_columns)} (40 base + 7 Premier)")
    else:
        logger.info(f"  - Total features: {len(feature_columns)} (40 base)")

    # Validar que todas las columnas existen
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing_cols}")

    # Convertir TODAS las features a float64 explícitamente
    df_features = df[feature_columns].copy()
    for col in feature_columns:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

    # Verificar tipos después de conversión
    non_numeric = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.error(f"Columnas que no se pudieron convertir a numérico: {non_numeric}")
        for col in non_numeric:
            logger.error(f"  {col}: tipos únicos = {df_features[col].apply(type).unique()}")
        raise ValueError(f"Features no numéricas detectadas: {non_numeric}")

    # Extraer features (X)
    X = df_features.values

    # Extraer target (y) y codificar
    result_mapping = {'1': 0, 'X': 1, '2': 2}
    y = df['result'].map(result_mapping).values

    # Verificar que no hay NaN
    if np.isnan(X).any():
        nan_counts = np.isnan(X).sum(axis=0)
        nan_features = [(feature_columns[i], count) for i, count in enumerate(nan_counts) if count > 0]
        logger.warning(f"Features con NaN detectados: {nan_features}")
        logger.info("Imputando NaN con media de la columna...")

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

    logger.success(f"  - Features preparadas: {X.shape}")
    logger.info(f"  - Total features: {len(feature_columns)} (34 originales + 6 balance)")
    logger.info(f"  - Distribución target: 1={np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%), "
                f"X={np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%), "
                f"2={np.sum(y == 2)} ({np.sum(y == 2) / len(y) * 100:.1f}%)")

    return X, y


def train_model(X_train, y_train, league_name):
    """
    Entrena Random Forest y aplica calibración isotónica

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        league_name: Nombre de la liga (para ajustes específicos)
    Returns:
        CalibratedClassifierCV: Modelo calibrado
    """
    logger.info("Entrenando modelo Random Forest...")

    # Calcular class weights para balancear clases
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    if league_name == 'premier_league':
        # Premier: Forzar MUCHO más empates
        class_weight_dict[1] *= 2.0  # x2.0 para clase X (empate)
        logger.info(f"  - Premier League: Aumentando peso de empates x2.0")

    elif league_name == 'serie_a':
        # Serie A: Ya funciona bien, ligero boost
        class_weight_dict[1] *= 1.0
        logger.info(f"  - Serie A: Aumentando peso de empates x1.1")

    elif league_name == 'laliga':
        # LaLiga: Boost moderado
        class_weight_dict[1] *= 1.3
        logger.info(f"  - LaLiga: Aumentando peso de empates x1.3")

    logger.info(f"  - Class weights: {class_weight_dict}")

    # Entrenar Random Forest base
    rf_model = RandomForestClassifier(
        n_estimators=200,  # Número de árboles
        max_depth=8,  # Profundidad máxima (regularización)
        min_samples_split=15,  # Mínimo para split
        min_samples_leaf=10,  # Mínimo por hoja
        max_features='sqrt',  # Features por split
        class_weight=class_weight_dict,  # Balanceo de clases
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,  # Usar todos los cores
        verbose=0
    )

    logger.info("  - Entrenando Random Forest...")
    rf_model.fit(X_train, y_train)

    # Accuracy en train (para detectar overfitting)
    train_acc = rf_model.score(X_train, y_train)
    logger.info(f"  - Accuracy en train: {train_acc:.3f}")

    # Calibración isotónica
    logger.info("  - Aplicando calibración isotónica...")
    calibrated_model = CalibratedClassifierCV(
        estimator=rf_model,
        method='isotonic',  # Isotonic calibration
        cv=5,  # Cross-validation
        n_jobs=-1
    )

    # Re-entrenar calibrador en datos SIN balancear (importante)
    calibrated_model.fit(X_train, y_train)

    logger.success("  - Modelo entrenado y calibrado exitosamente")

    return calibrated_model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa modelo y genera métricas

    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Target de test

    Returns:
        dict: Métricas de evaluación
    """
    logger.info("Evaluando modelo...")

    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Aplicar MIN_PROB floor y re-normalizar
    y_pred_proba = np.maximum(y_pred_proba, Config.MIN_PROB)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    # Métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    # Classification report
    target_names = ['1 (Home)', 'X (Draw)', '2 (Away)']
    class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Distribución de predicciones
    pred_dist = {
        '1': np.sum(y_pred == 0),
        'X': np.sum(y_pred == 1),
        '2': np.sum(y_pred == 2)
    }

    # Distribución real
    real_dist = {
        '1': np.sum(y_test == 0),
        'X': np.sum(y_test == 1),
        '2': np.sum(y_test == 2)
    }

    # Logging
    logger.success(f"  - Accuracy: {accuracy:.3f}")
    logger.info(f"  - Log Loss: {logloss:.3f}")
    logger.info(f"  - Distribución real: 1={real_dist['1']}, X={real_dist['X']}, 2={real_dist['2']}")
    logger.info(f"  - Distribución predicha: 1={pred_dist['1']}, X={pred_dist['X']}, 2={pred_dist['2']}")

    # Precision/Recall por clase
    logger.info("  - Métricas por clase:")
    for class_name in target_names:
        precision = class_report[class_name]['precision']
        recall = class_report[class_name]['recall']
        f1 = class_report[class_name]['f1-score']
        logger.info(f"    {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # Preparar dict de métricas para guardar
    metrics = {
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'classification_report': class_report,
        'predictions_distribution': pred_dist,
        'real_distribution': real_dist,
        'test_size': len(y_test)
    }

    # Feature importance (top 15)
    if hasattr(model, 'estimators_'):
        # Model is RandomForest (acceso directo)
        feature_importances = model.feature_importances_
    elif hasattr(model, 'calibrated_classifiers_'):
        # Model is CalibratedClassifierCV
        # Promediar importances de todos los calibradores
        importances_list = []
        for calibrated_clf in model.calibrated_classifiers_:
            if hasattr(calibrated_clf.estimator, 'feature_importances_'):
                importances_list.append(calibrated_clf.estimator.feature_importances_)
        if importances_list:
            feature_importances = np.mean(importances_list, axis=0)
        else:
            feature_importances = None
    else:
        feature_importances = None

    if feature_importances is not None:
        # Asumiendo que tienes las 47 features en orden
        feature_names = [
            'elo_home', 'elo_away', 'elo_diff',
            'home_win_rate', 'home_goals_for_avg', 'home_goals_against_avg', 'home_points_avg',
            'home_shots_on_target_avg', 'home_possession_avg', 'home_total_shots_avg',
            'home_gk_saves_avg', 'home_big_chances_avg', 'home_accurate_passes_avg',
            'home_tackles_won_avg', 'home_interceptions_avg', 'home_blocked_shots_avg',
            'away_win_rate', 'away_goals_for_avg', 'away_goals_against_avg', 'away_points_avg',
            'away_shots_on_target_avg', 'away_possession_avg', 'away_total_shots_avg',
            'away_gk_saves_avg', 'away_big_chances_avg', 'away_accurate_passes_avg',
            'away_tackles_won_avg', 'away_interceptions_avg', 'away_blocked_shots_avg',
            'h2h_home_win_rate', 'h2h_home_goals_avg', 'h2h_away_goals_avg',
            'h2h_total_goals_avg', 'h2h_used_proxy',
            'elo_diff_abs', 'form_balance', 'goals_balance',
            'possession_balance', 'shots_balance', 'shots_on_target_balance'
        ]

        # Si es Premier, añadir features adicionales
        if len(feature_importances) == 47:
            feature_names.extend([
                'season_progress', 'is_early_season', 'is_late_season',
                'home_form_momentum', 'away_form_momentum',
                'ultra_balanced', 'underdog_home_vs_strong_away'
            ])

        # Top 15 features
        indices = np.argsort(feature_importances)[::-1][:15]
        logger.info("  - Top 15 features más importantes:")
        for i, idx in enumerate(indices, 1):
            logger.info(f"    {i}. {feature_names[idx]}: {feature_importances[idx]:.4f}")

        # Guardar en métricas
        metrics['feature_importances'] = {
            feature_names[i]: float(feature_importances[i])
            for i in range(len(feature_importances))
        }

    return metrics


def save_model(model, metrics, league):
    """
    Guarda modelo y métricas para una liga específica

    Args:
        model: Modelo entrenado
        metrics: Dict con métricas
        league: Nombre de la liga (ej: 'laliga')
    """
    import json

    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    model_path = f"{Config.MODELS_DIR}/model_1x2_{league}.pkl"
    metrics_path = f"{Config.MODELS_DIR}/training_metrics_{league}.json"

    # Guardar modelo
    logger.info(f"  - Guardando modelo en {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Guardar métricas
    logger.info(f"  - Guardando métricas en {metrics_path}...")

    # Añadir metadata
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['league'] = league
    metrics['model_type'] = 'RandomForest_Calibrated'
    metrics['config'] = {
        'elo_k': Config.ELO_K,
        'elo_scale': Config.ELO_SCALE,
        'elo_system': 'dual_home_away',  # ✅ Indicar que usa Elo dual
        'elo_initial': Config.ELO_INITIAL,  # ✅ Añadir valor inicial
        'form_window': Config.FORM_WINDOW,
        'form_weight_recent': Config.FORM_WEIGHT_RECENT,
        'h2h_min_matches': Config.MIN_H2H_MATCHES,
        'min_prob': Config.MIN_PROB,
        'test_season': Config.TEST_SEASON
    }

    # ✅ Convertir tipos numpy a tipos nativos de Python
    def convert_to_native(obj):
        """Convierte tipos numpy a tipos nativos de Python para JSON"""
        if isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    metrics_native = convert_to_native(metrics)

    with open(metrics_path, 'w') as f:
        json.dump(metrics_native, f, indent=2)

    logger.success(f"  - Modelo y métricas guardados exitosamente")


def train_league(league_id, league_name):
    """
    Pipeline de entrenamiento para una liga específica

    Args:
        league_id: ID numérico de la liga
        league_name: Nombre legible de la liga
    """
    logger.info("=" * 60)
    logger.info(f"ENTRENANDO MODELO PARA: {league_name} (ID: {league_id})")
    logger.info("=" * 60)

    # 1. Cargar datos
    df = load_data(league_id)
    logger.success(f"Datos cargados correctamente: {len(df)} partidos")

    # 2. Feature engineering
    df = calculate_elo(df)
    df = calculate_form_features(df)
    df = calculate_h2h_features(df)

    # 3. Split temporal
    logger.info(f"Split temporal: test season = {Config.TEST_SEASON}")
    train_df = df[df['season'] != Config.TEST_SEASON].copy()
    test_df = df[df['season'] == Config.TEST_SEASON].copy()

    logger.info(f"Train: {len(train_df)} partidos")
    logger.info(f"Test: {len(test_df)} partidos")

    # 4. Preparar features
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    # 5. Entrenar
    league_name_normalized = league_name.lower().replace(" ", "_")
    model = train_model(X_train, y_train, league_name_normalized)

    # 6. Evaluar
    metrics = evaluate_model(model, X_test, y_test)

    # 7. Guardar
    save_model(model, metrics, league_name.lower().replace(" ", "_"))
    logger.success(f"Modelo {league_name} guardado exitosamente")

#    return metrics
    return []


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Pipeline completo - entrena un modelo por liga"""

    logger.info("=" * 60)
    logger.info("INICIO ENTRENAMIENTO MODELO 1X2")
    logger.info("=" * 60)

    all_metrics = {}

    # Entrenar un modelo por cada liga
    for league_id, league_name in Config.LEAGUES.items():
        try:
            metrics = train_league(league_id, league_name)
            all_metrics[league_name] = metrics
        except Exception as e:
            logger.error(f"Error entrenando {league_name}: {str(e)}")
            continue

    logger.info("=" * 60)
    logger.success("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")  # VERDE
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
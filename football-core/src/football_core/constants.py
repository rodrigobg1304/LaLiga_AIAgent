"""
Constantes compartidas entre scripts de entrenamiento y predicción.

Este archivo centraliza todas las configuraciones y parámetros del sistema
para evitar duplicación y facilitar mantenimiento.
"""
import os

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LIGAS - Para scripts genéricos
# ═══════════════════════════════════════════════════════════

LEAGUE_CONFIG = {
    'laliga': {
        'id': '8',
        'name': 'LaLiga',
        'slug': 'laliga',
        'country': 'Spain'
    },
    'premier': {
        'id': '17',
        'name': 'Premier League',
        'slug': 'premier_league',
        'country': 'England'
    },
    'serie_a': {
        'id': '23',
        'name': 'Serie A',
        'slug': 'serie_a',
        'country': 'Italy'
    }
}

# Temporadas para entrenamiento y test
TRAIN_SEASONS = ['19/20', '20/21', '21/22', '22/23', '23/24', '24/25']
TEST_SEASONS = ['25/26']

# Umbrales Over/Under
OVER_UNDER_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]

# Directorio base de modelos — se puede sobreescribir con la variable de entorno MODELS_DIR
MODEL_BASE_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# ═══════════════════════════════════════════════════════════
# FEATURES - Stats utilizadas en modelos
# ═══════════════════════════════════════════════════════════

FEATURES = [
    'Goals',
    'Ball possession',
    'Total shots',
    'Shots on target',
    'Goalkeeper saves',
    'Big chances',
    'Accurate passes',
    'Tackles won',
    'Interceptions',
    'Blocked shots'
]

# ═══════════════════════════════════════════════════════════
# PESOS DE PONDERACIÓN - Features
# ═══════════════════════════════════════════════════════════

# Para stats generales (goals, shots, possession, etc.)
# 65% forma reciente + 35% histórico completo
RECENT_WEIGHT = 0.65
HISTORICAL_WEIGHT = 0.35

# Para win rates (ponderación INVERTIDA)
# 35% forma reciente + 65% histórico completo
# Razón: Win rates son más volátiles, priorizamos histórico
WIN_RATE_RECENT_WEIGHT = 0.35
WIN_RATE_HISTORICAL_WEIGHT = 0.65

# Current tournament form blending weights for international leagues
# Credibility formula: w_current = n / (n + TOURN_CREDIBILITY_PSEUDO_COUNT)
# 0 matches→0%, 1→25%, 2→40%, 3→50%, 5→62%
TOURN_CREDIBILITY_PSEUDO_COUNT = 3

# ═══════════════════════════════════════════════════════════
# ELO RATING - Configuración
# ═══════════════════════════════════════════════════════════

ELO_K = 40                  # Factor K (sensibilidad a cambios)
ELO_SCALE = 600             # Escala de diferencia de rating
ELO_HOME_ADVANTAGE = 0      # Sin ventaja home fija (usamos Elo dual home/away)
ELO_INITIAL = 1500          # Rating inicial para equipos nuevos

# Regresión a la media entre temporadas (75% rating anterior + 25% inicial)
ELO_SEASON_REGRESSION = 0.75

# ═══════════════════════════════════════════════════════════
# FORM - Ventanas temporales
# ═══════════════════════════════════════════════════════════

FORM_WINDOW = 10            # Últimos N partidos para forma reciente
H2H_LOOKBACK = 5            # Últimos N enfrentamientos H2H
MIN_H2H_MATCHES = 1         # Mínimo de H2H antes de usar proxy

# ═══════════════════════════════════════════════════════════
# MODELO - Configuración general
# ═══════════════════════════════════════════════════════════

MIN_PROB = 0.05             # Floor de probabilidad mínima (evita 0% exacto)
RANDOM_STATE = 42           # Seed para reproducibilidad

# ═══════════════════════════════════════════════════════════
# LIGAS - IDs y nombres
# ═══════════════════════════════════════════════════════════

LEAGUES = {
    '8': 'LaLiga',
    '17': 'Premier League',
    '23': 'Serie A',
    '11': 'Qualy WC Europe',
    '16': 'World Cup',
    '295': 'Qualy WC Conmebol',
    '140': 'Qualy WC Concacaf',
}

# ── Ligas domésticas ──────────────────────────────────────────────────────────
SOFASCORE_LEAGUES = {
    '8': 'LaLiga',
    '17': 'PremierLeague',
    '23': 'SerieA',
}

# ── Torneos regulares (histórico continuo: Champions, EL, Conference) ─────────
# IDs pendientes de confirmar en Sofascore
SOFASCORE_TOURNAMENTS_REGULAR: dict[str, str] = {
    # '7':   'ChampionsLeague',
    # '679': 'EuropaLeague',
    # 'XXX': 'ConferenceLeague',
}

# ── Torneos/Clasificatorias internacionales (cada 4 años) ────────────────────
SOFASCORE_TOURNAMENTS_QUALIFIER: dict[str, str] = {
    '1'  : 'Eurocopa',
    '27' : 'Qualy_Euro',
    '16' : 'WorldCup',
    '11' : 'Qualy_WorldCup_Europe',
    '295': 'Qualy_WorldCup_Conmebol',
    '13' : 'Qualy_WorldCup_CAF',
    '140': 'Qualy_WorldCup_Concacaf',
}

# ── Combinado — usado por sofascore_client.py como league_dict ────────────────
SOFASCORE_ALL: dict[str, str] = {
    **SOFASCORE_LEAGUES,
    **SOFASCORE_TOURNAMENTS_REGULAR,
    **SOFASCORE_TOURNAMENTS_QUALIFIER,
}

# ── Zonas horarias por LeagueId — para almacenar hora local del partido ───────
# Usadas para calcular MatchDateLocal a partir del timestamp UTC de Sofascore.
# Los torneos usan la zona horaria del país sede del evento.
LEAGUE_TIMEZONES: dict[str, str] = {
    '1'  : 'Europe/Berlin',       # Eurocopa (sede Germany 2024)
    '8'  : 'Europe/Madrid',       # LaLiga
    '11' : 'Europe/Madrid',       # Clasificación WorldCup Euro (Elegimos España)
    '13' : 'Africa/Cairo',        # Clasificación WorldCup Africa (Elegimos El Cairo)
    '16' : 'Europe/Madrid',       # WorldCup (sede cambia elegimos España)
    '17' : 'Europe/London',       # Premier League
    '23' : 'Europe/Rome',         # Serie A
    '27' : 'Europe/Madrid',       # Clasificación Euro (partidos en España principalmente)
    '140': 'America/Mexico_City', # Clasificación WorldCup Concacaf (Elegimos Mexico)
    '295': 'America/Bogota',      # Clasificación WorldCup Sudamérica (Elegimos Bogotá)
}

# Normalización de nombres (para rutas de archivos)
LEAGUE_NAMES_NORMALIZED = {
    'LaLiga': 'laliga',
    'Premier League': 'premier_league',
    'Serie A': 'serie_a',
    'World Cup': 'worldcup',
    'Qualy WC Europe': 'worldcup_europe',
    'Qualy WC Conmebol': 'worldcup_conmebol',
    'Qualy WC CAF': 'worldcup_caf',
}

# ═══════════════════════════════════════════════════════════
# TEMPORADAS - Splits train/test
# ═══════════════════════════════════════════════════════════

TEST_SEASON = '25/26'       # Temporada actual para testing

# ═══════════════════════════════════════════════════════════
# MODELOS - Rutas relativas al MODEL_BASE_DIR
# ═══════════════════════════════════════════════════════════

MODEL_1X2_PREMIER = os.path.join(MODEL_BASE_DIR, "models", "1x2", "production", "premier_league")

# ═══════════════════════════════════════════════════════════
# THRESHOLDS - Saves y Corners
# ═══════════════════════════════════════════════════════════

SAVES_THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
CORNERS_THRESHOLDS = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

# ═══════════════════════════════════════════════════════════
# LIGAS INTERNACIONALES — IDs de selecciones nacionales
# ═══════════════════════════════════════════════════════════

# Todos los IDs de torneos internacionales/clasificatorias.
# Para equipos nacionales, get_team_win_rates debe consultar TODOS estos
# en lugar de uno solo, porque una selección puede tener datos en varias
# competiciones (ej. Spain: liga 11 qualifying + liga 16 World Cup).
INTERNATIONAL_LEAGUE_IDS = {'11', '16', '295', '140', '1', '27', '13'}

# ═══════════════════════════════════════════════════════════
# NEUTRAL FEATURES — Valores de referencia para selecciones sin datos
# ═══════════════════════════════════════════════════════════

# Coinciden exactamente con los valores NEUTRAL del training script (train_qualy.py).
# Se usan como fallback cuando la BD no devuelve stats para un equipo nacional,
# evitando que el modelo reciba ceros (fuera de la distribución de entrenamiento).
NEUTRAL_TEAM_STATS = {
    'win_rate': 0.33,
    'goals_for_avg': 1.5,
    'goals_against_avg': 1.5,
    'points_avg': 1.0,          # form_pts / 5.0 → form_pts = 5
    'shots_on_target_avg': 4.5,
    'possession_avg': 50.0,
    'total_shots_avg': 12.0,
    'gk_saves_avg': 3.0,
    'big_chances_avg': 2.5,
    'accurate_passes_avg': 300.0,
    'tackles_won_avg': 8.0,
    'interceptions_avg': 7.0,
    'blocked_shots_avg': 2.0,
}
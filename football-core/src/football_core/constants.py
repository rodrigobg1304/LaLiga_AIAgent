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
MIN_H2H_MATCHES = 2         # Mínimo de H2H antes de usar proxy

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
    '1':  'Eurocopa',
    '27': 'Qualy_Euro',
    '16':  'WorldCup',
    '11': 'Qualy_WorldCup_Europe',
    # 'XXX': 'CopaAmerica',
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
    '8':  'Europe/Madrid',   # LaLiga
    '17': 'Europe/London',   # Premier League
    '23': 'Europe/Rome',     # Serie A
    '1':  'Europe/Berlin',   # Eurocopa (sede Germany 2024)
    '27': 'Europe/Madrid',   # Qualy Euro (partidos en España principalmente)
    '11': 'Europe/Madrid',   # Qualy WorldCup Euro (partidos en España principalmente)
    '16': 'Europe/Madrid',    # WorldCup (sede cambia elegimos España)
}

# Normalización de nombres (para rutas de archivos)
LEAGUE_NAMES_NORMALIZED = {
    'LaLiga': 'laliga',
    'Premier League': 'premier_league',
    'Serie A': 'serie_a',
    'Qualy WC Europe': 'worldcup_europe',
    'World Cup': 'worldcup_europe',
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
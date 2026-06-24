"""
Entrenamiento de modelos para clasificatorias internacionales (World Cup, Euro qualifying).

Soporta combinar una fase clasificatoria con el torneo principal para enriquecer el dataset.
Los datos se ordenan cronológicamente: clasificatoria primero, torneo después (mismo año).
La feature 'is_qualifier' permite al modelo distinguir el contexto competitivo.

Diferencias clave frente a los scripts de ligas domésticas:
  - Detecta temporadas automáticamente desde los datos (sin hardcodear '25/26')
  - RF más conservador (max_depth=5, min_samples_leaf=15) para datasets pequeños
  - Evaluación con CV-5 + holdout temporal (última campaña)
  - Entrena los 4 tipos de modelos en un solo script (1x2, goals, saves, corners)
  - Sin features específicas de Premier League
  - 45 features: 40 estándar + is_qualifier + home/away xG avg + home/away tournament points

Uso:
    cd training
    python train/train_qualy.py --league 11              # Solo clasificatoria WC Europa
    python train/train_qualy.py --league 11 16 --slug worldcup_europe
        # Clasificatoria (11) + Mundial (16) combinados, salida en worldcup_europe/
    python train/train_qualy.py --league 27 1 --qualifier 27 --slug euro
        # Qualy Euro (27) + Eurocopa (1); --qualifier marca explícitamente cuáles son clasificatoria

    Por defecto, los IDs en SOFASCORE_TOURNAMENTS_QUALIFIER se marcan como is_qualifier=1
    y el resto como is_qualifier=0 (torneo principal).
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Cargar credenciales de BD antes de importar football_core.db,
# ya que _load_config() se ejecuta en el momento del import.
# Busca training/.env primero; si no existe, sube a scripts/.env como fallback.
from dotenv import load_dotenv
_env_candidates = [
    Path(__file__).parent.parent / ".env",          # training/.env
    Path(__file__).parent.parent.parent / "scripts" / ".env",  # scripts/.env
]
for _env_path in _env_candidates:
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
        break

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, cross_val_score

from football_core.db import get_league_all_stats, get_matches_with_corners
from football_core.constants import (
    SOFASCORE_TOURNAMENTS_QUALIFIER,
    CORNERS_THRESHOLDS,
    SAVES_THRESHOLDS,
    OVER_UNDER_THRESHOLDS,
    ELO_K, ELO_SCALE, ELO_INITIAL, ELO_SEASON_REGRESSION,
    FORM_WINDOW, H2H_LOOKBACK, MIN_H2H_MATCHES,
    RECENT_WEIGHT, RANDOM_STATE,
    MODEL_BASE_DIR,
)


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

# Parámetros RF más conservadores que en ligas domésticas (dataset pequeño)
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=15,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Calibración con cv=3 por dataset pequeño (en lugar de cv=5)
CALIBRATION_CV = 3

# Source-based match weights for form calculation
WEIGHT_CURRENT_WC   = 3.0   # Matches in the same WC tournament being predicted
WEIGHT_QUALIFIER    = 1.0   # WC qualifier matches
WEIGHT_PREV_WC      = 0.5   # Previous WC editions

# 45 features: 40 estándar + is_qualifier + home_xg_avg + away_xg_avg + home_tournament_points + away_tournament_points
FEATURE_COLUMNS = [
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
    'possession_balance', 'shots_balance', 'shots_on_target_balance',
    'is_qualifier',          # 1 = partido de clasificatoria, 0 = partido del torneo principal
    'home_xg_avg',           # Expected goals promedio del equipo local (forma reciente)
    'away_xg_avg',           # Expected goals promedio del equipo visitante (forma reciente)
    'home_tournament_points', # Puntos acumulados en el torneo actual (antes de este partido)
    'away_tournament_points', # Puntos acumulados en el torneo actual (antes de este partido)
]


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────

_NUMERIC_COLS = [
    'round', 'home_goals', 'away_goals',
    'home_shots_on_target', 'away_shots_on_target',
    'home_possession', 'away_possession',
    'home_total_shots', 'away_total_shots',
    'home_gk_saves', 'away_gk_saves',
    'home_big_chances', 'away_big_chances',
    'home_accurate_passes', 'away_accurate_passes',
    'home_tackles_won', 'away_tackles_won',
    'home_interceptions', 'away_interceptions',
    'home_blocked_shots', 'away_blocked_shots',
    'home_xg', 'away_xg',
]


def load_data(league_id: str) -> pd.DataFrame:
    """Carga una sola liga. Añade is_qualifier=1 si está en SOFASCORE_TOURNAMENTS_QUALIFIER."""
    return load_data_combined([league_id], set(SOFASCORE_TOURNAMENTS_QUALIFIER.keys()))


def load_data_combined(league_ids: list, qualifier_ids: set) -> pd.DataFrame:
    """
    Carga y combina datos de múltiples ligas/torneos.

    - qualifier_ids: qué LeagueIds marcar como is_qualifier=1.
    - competition_order: 0 para clasificatorias, 1 para torneo principal.
      Permite ordenar cronológicamente dentro del mismo año (clasificatoria → torneo).
    """
    frames = []
    for lid in league_ids:
        rows = get_league_all_stats(lid)
        if not rows:
            logger.warning(f"Sin datos para LeagueId={lid} — omitido.")
            continue
        frame = pd.DataFrame(rows)
        is_qualy = 1 if lid in qualifier_ids else 0
        frame['is_qualifier'] = is_qualy
        frame['competition_order'] = 0 if is_qualy else 1
        frames.append(frame)

    if not frames:
        raise ValueError("No hay datos para ninguno de los leagues especificados.")

    df = pd.concat(frames, ignore_index=True)

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ordenar: temporada ASC, clasificatoria antes del torneo principal, ronda ASC
    df = df.sort_values(
        ['season', 'competition_order', 'round'], ascending=True, na_position='first'
    ).reset_index(drop=True)

    seasons = sorted(df['season'].unique())
    qualy_count = int(df['is_qualifier'].sum())
    logger.info(
        f"Partidos cargados: {len(df)} (clasificatoria={qualy_count}, torneo={len(df)-qualy_count}) "
        f"| Temporadas: {seasons} | Equipos únicos: {df['home_team'].nunique()}"
    )
    return df


def detect_test_season(df: pd.DataFrame):
    """
    Determina la estrategia de train/test split:
    - Si hay más de una temporada: test = la más reciente (split por temporada).
    - Si solo hay una temporada: test = None → split temporal 80/20 por índice de partido.

    Devuelve el nombre de la temporada de test o None.
    """
    seasons = sorted(df['season'].unique())
    logger.info(f"Temporadas disponibles: {seasons}")
    if len(seasons) == 1:
        logger.info("Solo una temporada disponible — usando split temporal 80/20 por índice de partido.")
        return None
    test = seasons[-1]
    logger.info(f"Temporada de test: {test}")
    return test


def _get_train_test_masks(df: pd.DataFrame, test_season) -> tuple:
    """
    Devuelve (train_mask, test_mask) como arrays booleanos de longitud len(df).
    Si test_season es None: split temporal 80/20 por índice.
    Si test_season es str: split por temporada.
    """
    n = len(df)
    if test_season is None:
        split_idx = int(n * 0.8)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:split_idx] = True
        test_mask = ~train_mask
    else:
        train_mask = (df['season'] != test_season).values
        test_mask  = (df['season'] == test_season).values
    return train_mask, test_mask


# ─────────────────────────────────────────────
# ELO (dual home/away, igual que train_1x2.py)
# ─────────────────────────────────────────────

def calculate_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Elo dual (home/away) con tratamiento diferenciado para terreno neutral.

    - Clasificatorias (is_qualifier=1): Elo rol-específico (home juega en casa, away fuera).
    - Torneos en terreno neutral (is_qualifier=0, ej: Mundial, Eurocopa): se usa el
      promedio de los dos ratings de cada equipo, y la actualización se aplica por igual
      a ambos ratings para no distorsionar el Elo con un rol que no tiene significado.
    """
    elo_ratings: dict = {}
    elo_home_list, elo_away_list, elo_diff_list = [], [], []
    current_season = None

    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        season, result = row['season'], row['result']
        is_neutral = (int(row.get('is_qualifier', 1)) == 0)

        if current_season is None:
            current_season = season
        if season != current_season:
            for team in elo_ratings:
                elo_ratings[team]['home'] = elo_ratings[team]['home'] * ELO_SEASON_REGRESSION + ELO_INITIAL * (1 - ELO_SEASON_REGRESSION)
                elo_ratings[team]['away'] = elo_ratings[team]['away'] * ELO_SEASON_REGRESSION + ELO_INITIAL * (1 - ELO_SEASON_REGRESSION)
            current_season = season

        for team in (home, away):
            if team not in elo_ratings:
                elo_ratings[team] = {'home': ELO_INITIAL, 'away': ELO_INITIAL}

        if is_neutral:
            # Terreno neutral: usar rating promedio de cada selección
            elo_h = (elo_ratings[home]['home'] + elo_ratings[home]['away']) / 2
            elo_a = (elo_ratings[away]['home'] + elo_ratings[away]['away']) / 2
        else:
            # Clasificatoria: rol importa (juegan en su estadio o fuera)
            elo_h = elo_ratings[home]['home']
            elo_a = elo_ratings[away]['away']

        elo_home_list.append(elo_h)
        elo_away_list.append(elo_a)
        elo_diff_list.append(elo_h - elo_a)

        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / ELO_SCALE))
        actual_h = 1.0 if result == '1' else (0.5 if result == 'X' else 0.0)
        delta_h = ELO_K * (actual_h - exp_h)
        delta_a = ELO_K * ((1 - actual_h) - (1 - exp_h))

        if is_neutral:
            # Actualizar ambos ratings por igual (no hay diferenciación de rol)
            elo_ratings[home]['home'] += delta_h
            elo_ratings[home]['away'] += delta_h
            elo_ratings[away]['home'] += delta_a
            elo_ratings[away]['away'] += delta_a
        else:
            elo_ratings[home]['home'] += delta_h
            elo_ratings[away]['away'] += delta_a

    df = df.copy()
    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = elo_diff_list
    logger.info(f"ELO calculado para {len(elo_ratings)} equipos")
    return df


# ─────────────────────────────────────────────
# FORMA (igual que train_1x2.py)
# ─────────────────────────────────────────────

def _safe_mean(values, default=np.nan):
    filtered = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    return np.mean(filtered) if filtered else default


NEUTRAL = {
    'win_rate': 0.33, 'goals_for_avg': 1.5, 'goals_against_avg': 1.5, 'points_avg': 1.0,
    'shots_on_target_avg': 4.5, 'possession_avg': 50.0, 'total_shots_avg': 12.0,
    'gk_saves_avg': 3.0, 'big_chances_avg': 2.5, 'accurate_passes_avg': 300.0,
    'tackles_won_avg': 8.0, 'interceptions_avg': 7.0, 'blocked_shots_avg': 2.0,
    'xg_avg': 1.2,
}


def _calc_team_form(history: list, window: int = FORM_WINDOW) -> dict:
    if not history:
        return NEUTRAL

    recent = history[-window:] if len(history) >= window else history

    def weighted_mean(key, default):
        pairs = [(m[key], m.get('src_w', 1.0))
                 for m in recent
                 if m.get(key) is not None and not (isinstance(m[key], float) and np.isnan(m[key]))]
        if not pairs:
            return default
        total_w = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / total_w

    return {
        'win_rate':              weighted_mean('win', 0.33),
        'goals_for_avg':         weighted_mean('goals_for', 1.5),
        'goals_against_avg':     weighted_mean('goals_against', 1.5),
        'points_avg':            weighted_mean('points', 1.0),
        'shots_on_target_avg':   weighted_mean('shots_on_target', 4.5),
        'possession_avg':        weighted_mean('possession', 50.0),
        'total_shots_avg':       weighted_mean('total_shots', 12.0),
        'gk_saves_avg':          weighted_mean('gk_saves', 3.0),
        'big_chances_avg':       weighted_mean('big_chances', 2.5),
        'accurate_passes_avg':   weighted_mean('accurate_passes', 300.0),
        'tackles_won_avg':       weighted_mean('tackles_won', 8.0),
        'interceptions_avg':     weighted_mean('interceptions', 7.0),
        'blocked_shots_avg':     weighted_mean('blocked_shots', 2.0),
        'xg_avg':                weighted_mean('xg', 1.2),
    }


def calculate_form_features(df: pd.DataFrame) -> pd.DataFrame:
    team_history: dict = {}
    home_forms, away_forms = [], []

    # Precompute the most recent season to determine source weights
    current_season = df['season'].max()

    # Tournament points tracker: (team, season) -> cumulative points
    tournament_points: dict = {}

    home_tourn_pts_list: list = []
    away_tourn_pts_list: list = []

    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        result = row['result']
        season = row['season']
        is_wc = (int(row.get('is_qualifier', 1)) == 0)
        is_current = (season == current_season and is_wc)
        is_prev_wc = (season != current_season and is_wc)

        if is_current:
            src_w = WEIGHT_CURRENT_WC
        elif is_prev_wc:
            src_w = WEIGHT_PREV_WC
        else:
            src_w = WEIGHT_QUALIFIER

        for t in (home, away):
            if t not in team_history:
                team_history[t] = {'home': [], 'away': []}

        home_forms.append(_calc_team_form(team_history[home]['home']))
        away_forms.append(_calc_team_form(team_history[away]['away']))

        # Record tournament points BEFORE this match is processed
        home_tourn_pts_list.append(tournament_points.get((home, season), 0))
        away_tourn_pts_list.append(tournament_points.get((away, season), 0))

        for role, team, side_key, goals_for, goals_against, res_win in [
            ('home', home, 'home', 'home_goals', 'away_goals', '1'),
            ('away', away, 'away', 'away_goals', 'home_goals', '2'),
        ]:
            team_history[team][side_key].append({
                'goals_for':       row[goals_for],
                'goals_against':   row[goals_against],
                'points':          3 if result == res_win else (1 if result == 'X' else 0),
                'win':             1 if result == res_win else 0,
                'shots_on_target': row.get(f'{role}_shots_on_target', np.nan),
                'possession':      row.get(f'{role}_possession', np.nan),
                'total_shots':     row.get(f'{role}_total_shots', np.nan),
                'gk_saves':        row.get(f'{role}_gk_saves', np.nan),
                'big_chances':     row.get(f'{role}_big_chances', np.nan),
                'accurate_passes': row.get(f'{role}_accurate_passes', np.nan),
                'tackles_won':     row.get(f'{role}_tackles_won', np.nan),
                'interceptions':   row.get(f'{role}_interceptions', np.nan),
                'blocked_shots':   row.get(f'{role}_blocked_shots', np.nan),
                'xg':              row.get(f'{role}_xg', np.nan),
                'src_w':           src_w,
            })

        # Update tournament points AFTER appending to history (cumulative)
        home_goals = int(row['home_goals']) if row['home_goals'] is not None else 0
        away_goals = int(row['away_goals']) if row['away_goals'] is not None else 0
        if home_goals > away_goals:
            tournament_points[(home, season)] = tournament_points.get((home, season), 0) + 3
        elif home_goals == away_goals:
            tournament_points[(home, season)] = tournament_points.get((home, season), 0) + 1
            tournament_points[(away, season)] = tournament_points.get((away, season), 0) + 1
        else:
            tournament_points[(away, season)] = tournament_points.get((away, season), 0) + 3

    df = df.copy()
    for prefix, forms in [('home', home_forms), ('away', away_forms)]:
        for key in forms[0]:
            df[f'{prefix}_{key}'] = [f[key] for f in forms]

    # Add tournament points columns (accumulated before each match, chronologically)
    df['home_tournament_points'] = home_tourn_pts_list
    df['away_tournament_points'] = away_tourn_pts_list

    logger.info(f"Form features calculadas para {len(team_history)} equipos")
    return df


# ─────────────────────────────────────────────
# H2H (igual que train_1x2.py pero simplificado)
# ─────────────────────────────────────────────

def calculate_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    h2h_history: dict = {}
    h2h_features_list = []

    for idx, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        key = (home, away)
        if key not in h2h_history:
            h2h_history[key] = []
        past = h2h_history[key][-H2H_LOOKBACK:]

        if len(past) >= MIN_H2H_MATCHES:
            feats = {
                'h2h_home_win_rate':   np.mean([m['home_win'] for m in past]),
                'h2h_home_goals_avg':  np.mean([m['home_goals'] for m in past]),
                'h2h_away_goals_avg':  np.mean([m['away_goals'] for m in past]),
                'h2h_total_goals_avg': np.mean([m['home_goals'] + m['away_goals'] for m in past]),
                'h2h_used_proxy': 0,
            }
        else:
            # Proxy: partidos de equipos con puntos similares en la misma temporada
            feats = _proxy_h2h(df, idx, home, away, row['season'])

        h2h_features_list.append(feats)
        h2h_history[key].append({
            'home_goals': row['home_goals'],
            'away_goals': row['away_goals'],
            'home_win':   1 if row['result'] == '1' else 0,
        })

    df = df.copy()
    for key in h2h_features_list[0]:
        df[key] = [f[key] for f in h2h_features_list]

    proxy_count = df['h2h_used_proxy'].sum()
    logger.info(f"H2H calculado — proxies usados: {proxy_count}/{len(df)} ({proxy_count/len(df)*100:.1f}%)")
    return df


def _proxy_h2h(df, current_idx, home, away, season) -> dict:
    NEUTRAL = {
        'h2h_home_win_rate': 0.33, 'h2h_home_goals_avg': 1.5,
        'h2h_away_goals_avg': 1.5, 'h2h_total_goals_avg': 3.0, 'h2h_used_proxy': 1,
    }
    past = df.iloc[:current_idx]
    same_season = past[past['season'] == season]
    if len(same_season) < 10:
        return NEUTRAL

    # Tabla simplificada de puntos
    points: dict = {}
    for _, r in same_season.iterrows():
        for team, res_key in [(r['home_team'], '1'), (r['away_team'], '2')]:
            pts = 3 if r['result'] == res_key else (1 if r['result'] == 'X' else 0)
            points[team] = points.get(team, 0) + pts

    home_pts = points.get(home, 0)
    away_pts = points.get(away, 0)
    margin = max(3, int(len(points) * 0.15))  # ±15% del campo

    home_proxies = {t for t, p in points.items() if abs(p - home_pts) <= margin and t != home}
    away_proxies = {t for t, p in points.items() if abs(p - away_pts) <= margin and t != away}

    proxy_matches = same_season[
        same_season['home_team'].isin(home_proxies) &
        same_season['away_team'].isin(away_proxies)
    ]
    if proxy_matches.empty:
        return NEUTRAL

    proxy_list = proxy_matches.tail(H2H_LOOKBACK).to_dict('records')
    return {
        'h2h_home_win_rate':   np.mean([1 if m['result'] == '1' else 0 for m in proxy_list]),
        'h2h_home_goals_avg':  np.mean([m['home_goals'] for m in proxy_list]),
        'h2h_away_goals_avg':  np.mean([m['away_goals'] for m in proxy_list]),
        'h2h_total_goals_avg': np.mean([m['home_goals'] + m['away_goals'] for m in proxy_list]),
        'h2h_used_proxy': 1,
    }


# ─────────────────────────────────────────────
# PREPARACIÓN DE FEATURES
# ─────────────────────────────────────────────

def prepare_X(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    df['elo_diff_abs']          = np.abs(df['elo_diff'])
    df['form_balance']          = np.abs(df['home_win_rate'] - df['away_win_rate'])
    df['goals_balance']         = np.abs(df['home_goals_for_avg'] - df['away_goals_for_avg'])
    df['possession_balance']    = np.abs(df['home_possession_avg'] - df['away_possession_avg'])
    df['shots_balance']         = np.abs(df['home_total_shots_avg'] - df['away_total_shots_avg'])
    df['shots_on_target_balance'] = np.abs(df['home_shots_on_target_avg'] - df['away_shots_on_target_avg'])

    X = df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    if X.isna().any().any():
        imputer = SimpleImputer(strategy='mean')
        return imputer.fit_transform(X.values)
    return X.values


# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────

def _train_rf(X_train, y_train, class_weight_override: float = 1.0) -> CalibratedClassifierCV:
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    cw = dict(zip(classes, weights))
    # Boost ligero a empates (clase 1) o "over" cuando corresponda
    if 1 in cw:
        cw[1] *= class_weight_override

    rf = RandomForestClassifier(**RF_PARAMS, class_weight=cw)
    calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=CALIBRATION_CV)
    calibrated.fit(X_train, y_train)
    return calibrated


def _cv_score(X, y, n_splits=5) -> dict:
    rf = RandomForestClassifier(**RF_PARAMS, class_weight='balanced')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    accs = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
    return {'cv_accuracy_mean': float(accs.mean()), 'cv_accuracy_std': float(accs.std()), 'cv_folds': n_splits}


def _save(model, metrics: dict, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename + '.pkl'), 'wb') as f:
        pickle.dump(model, f)
    metrics['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(out_dir, filename + '_metrics.json'), 'w') as f:
        json.dump(_to_native(metrics), f, indent=2)
    logger.info(f"  Guardado → {out_dir}/{filename}.pkl")


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ─────────────────────────────────────────────
# PIPELINE 1X2
# ─────────────────────────────────────────────

def train_1x2(df: pd.DataFrame, test_season, out_dir: str):
    logger.info("\n── Modelo 1X2 ──────────────────────────────")
    X = prepare_X(df)
    result_map = {'1': 0, 'X': 1, '2': 2}
    y = df['result'].map(result_map).values

    train_mask, test_mask = _get_train_test_masks(df, test_season)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    logger.info(f"Train: {len(X_train)} partidos | Test: {len(X_test)} partidos")

    # CV score en train para estimación robusta
    cv = _cv_score(X_train, y_train)
    logger.info(f"CV accuracy (train): {cv['cv_accuracy_mean']:.3f} ± {cv['cv_accuracy_std']:.3f}")

    model = _train_rf(X_train, y_train, class_weight_override=1.3)

    metrics = dict(**cv)
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        metrics['test_accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['test_f1_macro'] = float(f1_score(y_test, y_pred, average='macro'))
        metrics['test_report']   = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Test accuracy: {metrics['test_accuracy']:.3f} | F1 macro: {metrics['test_f1_macro']:.3f}")
    else:
        logger.warning("La temporada de test no tiene partidos — sin evaluación holdout.")

    _save(model, metrics, out_dir, 'model_1x2_qualy')


# ─────────────────────────────────────────────
# PIPELINE OVER/UNDER GOALS
# ─────────────────────────────────────────────

def train_over_under_goals(df: pd.DataFrame, test_season, out_dir: str):
    logger.info("\n── Over/Under Goals ────────────────────────")
    df = df.copy()
    df['total_goals'] = df['home_goals'] + df['away_goals']

    X = prepare_X(df)
    train_mask, test_mask = _get_train_test_masks(df, test_season)

    for threshold in OVER_UNDER_THRESHOLDS:
        label = str(threshold).replace('.', '_')
        y = (df['total_goals'] > threshold).astype(int).values
        y_train, y_test = y[train_mask], y[test_mask]
        X_train, X_test = X[train_mask], X[test_mask]

        model = _train_rf(X_train, y_train)
        metrics = _cv_score(X_train, y_train)
        if len(X_test) > 0:
            metrics['test_accuracy'] = float(accuracy_score(y_test, model.predict(X_test)))
            try:
                metrics['test_auc'] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            except Exception:
                pass

        _save(model, metrics, out_dir, f'model_over_{label}_qualy')
        logger.info(f"  Over {threshold}: CV={metrics['cv_accuracy_mean']:.3f} | over_rate={y.mean():.2f}")


# ─────────────────────────────────────────────
# PIPELINE OVER/UNDER SAVES
# ─────────────────────────────────────────────

def train_over_under_saves(df: pd.DataFrame, test_season, out_dir: str):
    logger.info("\n── Over/Under Saves ────────────────────────")
    # get_league_all_stats ya incluye home_gk_saves / away_gk_saves
    # Expandir a 2 filas por partido (portero local + portero visitante)
    home_side = df[df['home_gk_saves'].notna()].copy()
    home_side['team_saves'] = home_side['home_gk_saves']
    home_side['is_home'] = 1

    away_side = df[df['away_gk_saves'].notna()].copy()
    away_side['team_saves'] = away_side['away_gk_saves']
    away_side['is_home'] = 0

    saves_df = pd.concat([home_side, away_side], ignore_index=True)

    if saves_df.empty:
        logger.warning("Sin datos de saves — omitiendo modelos de paradas.")
        return

    X_base  = prepare_X(saves_df)
    is_home = saves_df['is_home'].values.reshape(-1, 1)
    X       = np.hstack([X_base, is_home])

    train_mask, test_mask = _get_train_test_masks(saves_df, test_season)

    for threshold in SAVES_THRESHOLDS:
        label = str(threshold).replace('.', '_')
        y = (saves_df['team_saves'] > threshold).astype(int).values
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        model = _train_rf(X_train, y_train)
        metrics = _cv_score(X_train, y_train)
        if len(X_test) > 0:
            metrics['test_accuracy'] = float(accuracy_score(y_test, model.predict(X_test)))
            try:
                metrics['test_auc'] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            except Exception:
                pass

        _save(model, metrics, out_dir, f'model_over_{label}_saves_qualy')
        logger.info(f"  Over {threshold} saves: CV={metrics['cv_accuracy_mean']:.3f}")


# ─────────────────────────────────────────────
# PIPELINE OVER/UNDER CORNERS
# ─────────────────────────────────────────────

def train_over_under_corners(df: pd.DataFrame, test_season, out_dir: str, league_ids: list):
    logger.info("\n── Over/Under Corners ──────────────────────")
    seasons = list(df['season'].unique())
    rows = []
    for lid in league_ids:
        rows.extend(get_matches_with_corners(lid, seasons))
    if not rows:
        logger.warning("Sin datos de corners — omitiendo modelos de córners.")
        return

    # Extraer solo home_corners / away_corners y unir con el df enriquecido (ELO + form + H2H)
    corners_raw = pd.DataFrame(rows)[['matchId', 'home_corners', 'away_corners']]
    merged = df.merge(corners_raw, on='matchId', how='inner')
    if merged.empty:
        logger.warning("No se pudo hacer merge de corners con features — omitiendo.")
        return

    # Expandir a 2 filas por partido (home team + away team)
    home_side = merged.copy()
    home_side['team_corners'] = home_side['home_corners']
    home_side['is_home'] = 1

    away_side = merged.copy()
    away_side['team_corners'] = away_side['away_corners']
    away_side['is_home'] = 0

    exp_df = pd.concat([home_side, away_side], ignore_index=True)
    exp_df = exp_df.dropna(subset=['team_corners'])

    X_base  = prepare_X(exp_df)
    is_home = exp_df['is_home'].values.reshape(-1, 1)
    X       = np.hstack([X_base, is_home])

    train_mask, test_mask = _get_train_test_masks(exp_df, test_season)

    for threshold in CORNERS_THRESHOLDS:
        label = str(threshold).replace('.', '_')
        y = (exp_df['team_corners'] > threshold).astype(int).values
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(np.unique(y_train)) < 2:
            logger.warning(f"  Umbral {threshold}: solo 1 clase en train — omitido")
            continue

        model = _train_rf(X_train, y_train)
        metrics = _cv_score(X_train, y_train)
        if len(X_test) > 0:
            metrics['test_accuracy'] = float(accuracy_score(y_test, model.predict(X_test)))
            try:
                metrics['test_auc'] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            except Exception:
                pass

        _save(model, metrics, out_dir, f'model_over_{label}_corners_qualy')
        logger.info(f"  Over {threshold} corners: CV={metrics['cv_accuracy_mean']:.3f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

_SLUG_NAMES = {
    '11': 'qualy_worldcup_europe',
    '27': 'qualy_euro',
    '1':  'eurocopa',
    '16': 'worldcup',
}


def resolve_league_slug(league_ids: list, slug_override: str = None) -> str:
    """
    Devuelve el slug del directorio de salida.

    Si se pasa --slug, se usa directamente.
    Con un solo ID se mapea por nombre conocido.
    Con varios IDs se usa el primer clasificatorio (qualifier) como base.
    """
    if slug_override:
        return slug_override
    if len(league_ids) == 1:
        lid = league_ids[0]
        raw = SOFASCORE_TOURNAMENTS_QUALIFIER.get(lid, f'qualy_{lid}')
        return _SLUG_NAMES.get(lid, raw.lower().replace(' ', '_'))
    # Multi-league: prioriza clasificatorias para nombrar el directorio
    for lid in league_ids:
        if lid in SOFASCORE_TOURNAMENTS_QUALIFIER and lid in _SLUG_NAMES:
            return _SLUG_NAMES[lid]
    return '_'.join(league_ids)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos para clasificatorias.")
    parser.add_argument(
        '--league', type=str, nargs='+', required=True,
        help="LeagueId(s) de Sofascore. Ej: --league 11 (solo Qualy WC), --league 11 16 (Qualy+Mundial)",
    )
    parser.add_argument(
        '--qualifier', type=str, nargs='*', default=None,
        help=(
            "LeagueIds que se marcan como is_qualifier=1. "
            "Por defecto: cualquiera que esté en SOFASCORE_TOURNAMENTS_QUALIFIER. "
            "Ej: --qualifier 27 (marca solo la Qualy Euro como clasificatoria)"
        ),
    )
    parser.add_argument(
        '--slug', type=str, default=None,
        help="Nombre del directorio de salida. Por defecto se infiere del primer --league.",
    )
    args = parser.parse_args()

    league_ids = args.league
    # qualifier_ids: los IDs que el usuario pasó explícitamente, o todos los que ya son
    # clasificatorias según constants.py, intersectados con los leagues cargados.
    if args.qualifier is not None:
        qualifier_ids = set(args.qualifier)
    else:
        qualifier_ids = set(SOFASCORE_TOURNAMENTS_QUALIFIER.keys())
    qualifier_ids = qualifier_ids.intersection(set(league_ids))

    slug = resolve_league_slug(league_ids, args.slug)
    base = MODEL_BASE_DIR

    out_1x2     = os.path.join(base, 'models', '1x2', 'production', slug)
    out_goals   = os.path.join(base, 'models', 'over_under', 'goals', 'production', slug)
    out_saves   = os.path.join(base, 'models', 'over_under', 'saves', 'production', slug)
    out_corners = os.path.join(base, 'models', 'over_under', 'corners', 'production', slug)

    logger.info("=" * 60)
    logger.info(f"ENTRENAMIENTO CLASIFICATORIA — Leagues={league_ids} | Qualifiers={qualifier_ids} | slug={slug}")
    logger.info("=" * 60)

    # 1. Cargar y procesar datos
    df = load_data_combined(league_ids, qualifier_ids)
    test_season = detect_test_season(df)

    df = calculate_elo(df)
    df = calculate_form_features(df)
    df = calculate_h2h_features(df)

    train_mask, test_mask = _get_train_test_masks(df, test_season)
    split_desc = f"temporada {test_season}" if test_season else "último 20% (split temporal)"
    logger.info(f"\nSplit: {split_desc} | Train: {train_mask.sum()} | Test: {test_mask.sum()}")

    # 2. Entrenar los 4 tipos de modelos
    train_1x2(df, test_season, out_1x2)
    train_over_under_goals(df, test_season, out_goals)
    train_over_under_saves(df, test_season, out_saves)
    train_over_under_corners(df, test_season, out_corners, league_ids)

    logger.info("\n" + "=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info(f"  1X2    → {out_1x2}")
    logger.info(f"  Goals  → {out_goals}")
    logger.info(f"  Saves  → {out_saves}")
    logger.info(f"  Corners→ {out_corners}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

import pickle
import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from football_agent.db import run_query, TABLE

try:
    # Necesario para que pickle pueda deserializar HybridCalibratedModel
    from ml.train import HybridCalibratedModel  # noqa: F401 — cuando se importa desde raíz
    from ml.train import FEATURES
except ModuleNotFoundError:
    from train import HybridCalibratedModel  # noqa: F401 — cuando se ejecuta desde ml/  # noqa: F401
    from train import FEATURES

app = FastAPI(title="Football Prediction API", version="1.0")

# ───────────────────────────────────────────────
# REENTRENAMIENTO DE MODELOS AL ACTUALIZAR DATOS
# ───────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_TIMESTAMPS = {}

def get_model_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0

for _league in ["laliga", "premier"]:
    _rpath = os.path.join(MODELS_DIR, f"model_result_{_league}.pkl")
    _opath = os.path.join(MODELS_DIR, f"models_over_under_{_league}.pkl")
    MODEL_TIMESTAMPS[f"result_{_league}"]    = get_model_mtime(_rpath)
    MODEL_TIMESTAMPS[f"over_under_{_league}"] = get_model_mtime(_opath)

def reload_models_if_updated():
    """Recarga los modelos si los ficheros .pkl han cambiado."""
    global MODEL_TIMESTAMPS
    for league_name in ["laliga", "premier"]:
        result_path = os.path.join(MODELS_DIR, f"model_result_{league_name}.pkl")
        over_path   = os.path.join(MODELS_DIR, f"models_over_under_{league_name}.pkl")

        result_mtime = get_model_mtime(result_path)
        over_mtime   = get_model_mtime(over_path)

        if (MODEL_TIMESTAMPS.get(f"result_{league_name}") != result_mtime or
                MODEL_TIMESTAMPS.get(f"over_under_{league_name}") != over_mtime):
            print(f"🔄 Recargando modelos {league_name}: {datetime.now()}")
            MODEL_TIMESTAMPS[f"result_{league_name}"]    = result_mtime
            MODEL_TIMESTAMPS[f"over_under_{league_name}"] = over_mtime


def calculate_odds(probabilities: dict, margin: float = 0.07) -> dict:
    """Convierte probabilidades del modelo en cuotas con margen del 7%."""
    return {
        result: round(float((1 / (prob / 100)) * (1 + margin)), 2)
        for result, prob in probabilities.items()
        if prob > 0
    }


# ─────────────────────────────────────────────
# CARGAR MODELOS
# ─────────────────────────────────────────────

LEAGUE_MODEL_MAP = {
    "8":  "laliga",
    "17": "premier",
    "23": "seriea"
}

def load_models(league_id: str) -> tuple:
    league_name = LEAGUE_MODEL_MAP.get(str(league_id), "laliga")

    result_path = os.path.join(MODELS_DIR, f"model_result_{league_name}.pkl")
    over_path   = os.path.join(MODELS_DIR, f"models_over_under_{league_name}.pkl")

    with open(result_path, "rb") as f:
        bundle = pickle.load(f)

    with open(over_path, "rb") as f:
        over_models = pickle.load(f)

    return (
        bundle["model"],
        bundle["calibrated_model"],
        bundle["encoder"],
        bundle["features"],
        bundle.get("elo_diff_threshold", 20),  # retrocompatible con modelos anteriores
        over_models
    )

OVER_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]


# ─────────────────────────────────────────────
# CONSTRUIR FEATURES PARA UN PARTIDO NUEVO
# ─────────────────────────────────────────────

def get_recent_years(year: str, n: int = 2) -> list:
    """Devuelve las últimas N temporadas incluyendo la actual.
    Ej: '25/26' → ['25/26', '24/25']"""
    start = int(year.split("/")[0])
    return [f"{start - i}/{str((start - i + 1) % 100).zfill(2)}" for i in range(n)]


def get_team_features(team: str, role: str, year: Optional[str], n: int = 10) -> dict:
    """
    Calcula las stats rolling medias de un equipo (Goals, xG, shots, etc.)
    usando los últimos N partidos jugados en ese rol (home o away).
    El filtro year actúa como tope superior (<=), no como filtro exacto.
    """
    team_col = "homeTeam" if role == "home" else "awayTeam"
    stat_col = "homeValue" if role == "home" else "awayValue"

    placeholders = ",".join(["%s"] * len(FEATURES))

    recent_years = get_recent_years(year) if year else []
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [team] + FEATURES + recent_years + [n * len(FEATURES)]
    else:
        year_filter = ""
        params = [team] + FEATURES + [n * len(FEATURES)]

    sql = f"""
        SELECT name,
               AVG(CAST({stat_col} AS DECIMAL)) AS stat_value
        FROM (
            SELECT matchId, name,
                   SUM(CAST({stat_col} AS DECIMAL)) AS {stat_col}
            FROM {TABLE}
            WHERE {team_col} = %s
              AND name IN ({placeholders})
              AND period IN ('1ST', '2ND')
              {year_filter}
            GROUP BY matchId, name
            ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
            LIMIT %s
        ) AS last_matches
        GROUP BY name
    """
    rows = run_query(sql, tuple(params))

    if not rows:
        return {}

    # Mapa: nombre en BD → nombre propio que usa el modelo
    STAT_NAME_MAP = {
        "Expected goals": f"{role}_avg_xG",
        "Total shots":    f"{role}_avg_shots",
        "Big chances":    f"{role}_avg_big_chances",
    }

    features = {}
    stat_values = {}

    for row in rows:
        feat = row["name"]
        val  = float(row["stat_value"] or 0)
        stat_values[feat] = val
        features[f"{role}_avg_{feat}"] = val
        if feat in STAT_NAME_MAP:
            features[STAT_NAME_MAP[feat]] = val

    # Goals scored en rol específico
    features[f"{role}_avg_goals_scored"] = stat_values.get("Goals", 0)

    return features


def get_form_features(home_team: str, away_team: str, year: Optional[str], n: int = 10) -> dict:
    """
    Calcula win rates, form points y goles globales para ambos equipos.
    Estas features las calcula train.py en memoria; aquí las replicamos desde la BD.
    """
    limit = n * 4
    recent_years = get_recent_years(year) if year else []
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [home_team, home_team, away_team, away_team] + recent_years + [limit]
    else:
        year_filter = ""
        params = [home_team, home_team, away_team, away_team] + [limit]

    # Recuperar últimos N partidos de cada equipo (en cualquier rol)
    sql_matches = f"""
        SELECT matchId, homeTeam, awayTeam,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND (homeTeam = %s OR awayTeam = %s OR homeTeam = %s OR awayTeam = %s)
          {year_filter}
        GROUP BY matchId, homeTeam, awayTeam
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
        LIMIT %s
    """
    all_rows = run_query(sql_matches, tuple(params))

    if not all_rows:
        return {}

    # Separar partidos por equipo
    home_matches = [r for r in all_rows if r["homeTeam"] == home_team or r["awayTeam"] == home_team][:n]
    away_matches = [r for r in all_rows if r["homeTeam"] == away_team or r["awayTeam"] == away_team][:n]

    # Últimos 5 como local (home_team jugando en casa) para form
    home_as_home = [r for r in all_rows if r["homeTeam"] == home_team][:5]
    # Últimos 5 como visitante (away_team jugando fuera) para form
    away_as_away = [r for r in all_rows if r["awayTeam"] == away_team][:5]

    features = {}

    # ── Win rate como local (últimos N jugando en casa) ──
    home_home_matches = [r for r in all_rows if r["homeTeam"] == home_team][:n]
    if home_home_matches:
        wins   = sum(1 for r in home_home_matches if float(r["home_goals"] or 0) > float(r["away_goals"] or 0))
        draws  = sum(1 for r in home_home_matches if float(r["home_goals"] or 0) == float(r["away_goals"] or 0))
        losses = sum(1 for r in home_home_matches if float(r["home_goals"] or 0) < float(r["away_goals"] or 0))
        total  = len(home_home_matches)
        features["home_win_rate_home"]  = wins   / total
        features["home_draw_rate_home"] = draws  / total
        features["home_loss_rate_home"] = losses / total

    # ── Win rate como visitante (últimos N jugando fuera) ──
    away_away_matches = [r for r in all_rows if r["awayTeam"] == away_team][:n]
    if away_away_matches:
        wins   = sum(1 for r in away_away_matches if float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
        draws  = sum(1 for r in away_away_matches if float(r["away_goals"] or 0) == float(r["home_goals"] or 0))
        losses = sum(1 for r in away_away_matches if float(r["away_goals"] or 0) < float(r["home_goals"] or 0))
        total  = len(away_away_matches)
        features["away_win_rate_away"]  = wins   / total
        features["away_draw_rate_away"] = draws  / total
        features["away_loss_rate_away"] = losses / total

    # ── Win rate global + goles globales (cualquier rol) ──
    if home_matches:
        home_wins_global = sum(
            1 for r in home_matches
            if (r["homeTeam"] == home_team and float(r["home_goals"] or 0) > float(r["away_goals"] or 0)) or
               (r["awayTeam"] == home_team and float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
        )
        features["home_win_rate_global"] = home_wins_global / len(home_matches)
        features["home_avg_goals_scored_global"] = sum(
            float(r["home_goals"] or 0) if r["homeTeam"] == home_team else float(r["away_goals"] or 0)
            for r in home_matches
        ) / len(home_matches)
        features["home_avg_goals_conceded_global"] = sum(
            float(r["away_goals"] or 0) if r["homeTeam"] == home_team else float(r["home_goals"] or 0)
            for r in home_matches
        ) / len(home_matches)

    if away_matches:
        away_wins_global = sum(
            1 for r in away_matches
            if (r["homeTeam"] == away_team and float(r["home_goals"] or 0) > float(r["away_goals"] or 0)) or
               (r["awayTeam"] == away_team and float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
        )
        features["away_win_rate_global"] = away_wins_global / len(away_matches)
        features["away_avg_goals_scored_global"] = sum(
            float(r["home_goals"] or 0) if r["homeTeam"] == away_team else float(r["away_goals"] or 0)
            for r in away_matches
        ) / len(away_matches)
        features["away_avg_goals_conceded_global"] = sum(
            float(r["away_goals"] or 0) if r["homeTeam"] == away_team else float(r["home_goals"] or 0)
            for r in away_matches
        ) / len(away_matches)

    # ── Form points últimos 5 (local jugando en casa, visitante jugando fuera) ──
    if home_as_home:
        features["home_form_pts"] = sum(
            3 if float(r["home_goals"] or 0) > float(r["away_goals"] or 0) else
            (1 if float(r["home_goals"] or 0) == float(r["away_goals"] or 0) else 0)
            for r in home_as_home
        )
    if away_as_away:
        features["away_form_pts"] = sum(
            3 if float(r["away_goals"] or 0) > float(r["home_goals"] or 0) else
            (1 if float(r["away_goals"] or 0) == float(r["home_goals"] or 0) else 0)
            for r in away_as_away
        )

    return features


def get_h2h_features(home_team: str, away_team: str, year: Optional[str], n: int = 5) -> dict:
    """
    Calcula features head-to-head entre los dos equipos.
    Busca en ambas direcciones (A en casa vs B y B en casa vs A).
    Los resultados se normalizan desde la perspectiva del home_team del partido predicho.
    Sin filtro de temporada — el LIMIT controla la cantidad y el H2H necesita historial largo.
    """
    params = [home_team, away_team, away_team, home_team, n]

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
    rows = run_query(sql, tuple(params))

    if not rows:
        return {
            "h2h_home_wins": 0, "h2h_draws": 0,
            "h2h_away_wins": 0, "h2h_avg_goals": 0
        }

    total = len(rows)

    # Normalizar desde la perspectiva del home_team del partido predicho
    home_wins = sum(
        1 for r in rows if (
            (r["homeTeam"] == home_team and float(r["home_goals"] or 0) > float(r["away_goals"] or 0)) or
            (r["awayTeam"] == home_team and float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
        )
    )
    draws = sum(
        1 for r in rows if float(r["home_goals"] or 0) == float(r["away_goals"] or 0)
    )
    away_wins = sum(
        1 for r in rows if (
            (r["homeTeam"] == away_team and float(r["home_goals"] or 0) > float(r["away_goals"] or 0)) or
            (r["awayTeam"] == away_team and float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
        )
    )
    avg_goals = sum(
        float(r["home_goals"] or 0) + float(r["away_goals"] or 0) for r in rows
    ) / total

    return {
        "h2h_home_wins": home_wins / total,
        "h2h_draws":     draws     / total,
        "h2h_away_wins": away_wins / total,
        "h2h_avg_goals": round(avg_goals, 2)
    }


def get_over_rates(home_team: str, away_team: str, year: Optional[str], n: int = 10) -> dict:
    """Calcula las over rates históricas para cada threshold."""
    OVER_THRESHOLDS_LOCAL = [0.5, 1.5, 2.5, 3.5]
    recent_years = get_recent_years(year) if year else []
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [home_team, away_team, home_team, away_team] + recent_years + [n * 2]
    else:
        year_filter = ""
        params = [home_team, away_team, home_team, away_team] + [n * 2]

    sql = f"""
        SELECT matchId,
               homeTeam, awayTeam,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) +
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS total_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND (homeTeam IN (%s, %s) OR awayTeam IN (%s, %s))
          {year_filter}
        GROUP BY matchId, homeTeam, awayTeam
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
        LIMIT %s
    """
    rows = run_query(sql, tuple(params))

    if not rows:
        return {}

    home_rows = [r for r in rows if r["homeTeam"] == home_team or r["awayTeam"] == home_team][:n]
    away_rows = [r for r in rows if r["homeTeam"] == away_team or r["awayTeam"] == away_team][:n]

    features = {}
    for t in OVER_THRESHOLDS_LOCAL:
        col = f"over_{str(t).replace('.', '_')}"
        if home_rows:
            features[f"home_{col}_rate"] = sum(1 for r in home_rows if float(r["total_goals"] or 0) > t) / len(home_rows)
        if away_rows:
            features[f"away_{col}_rate"] = sum(1 for r in away_rows if float(r["total_goals"] or 0) > t) / len(away_rows)
        h_rate = features.get(f"home_{col}_rate", 0)
        a_rate = features.get(f"away_{col}_rate", 0)
        features[f"combined_{col}_rate"] = (h_rate + a_rate) / 2

    return features


def get_elo(home_team: str, away_team: str) -> dict:
    """Recupera el Elo actual de ambos equipos calculándolo desde el histórico."""
    sql = f"""
        SELECT homeTeam, awayTeam,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS hg,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS ag
        FROM {TABLE}
        WHERE name = 'Goals' AND period IN ('1ST', '2ND')
        GROUP BY matchId, homeTeam, awayTeam
        ORDER BY Year ASC, CAST(Round AS SIGNED) ASC
    """
    rows = run_query(sql, ())
    elo = {}
    for row in rows:
        h, a = row["homeTeam"], row["awayTeam"]
        elo.setdefault(h, 1500)
        elo.setdefault(a, 1500)

        exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        hg, ag = float(row["hg"] or 0), float(row["ag"] or 0)

        if hg > ag:   sh, sa = 1, 0
        elif hg < ag: sh, sa = 0, 1
        else:         sh, sa = 0.5, 0.5

        elo[h] += 20 * (sh - exp_h)
        elo[a] += 20 * (sa - (1 - exp_h))

    return {
        "elo_home": elo.get(home_team, 1500),
        "elo_away": elo.get(away_team, 1500),
        "elo_diff": elo.get(home_team, 1500) - elo.get(away_team, 1500)
    }



def get_team_season_count(team: str, year: Optional[str]) -> int:
    """
    Devuelve el número de temporadas distintas con partidos en Primera
    disponibles para este equipo en las últimas 2 temporadas.
    Usado para detectar equipos recién ascendidos sin histórico suficiente.
    """
    recent_years = get_recent_years(year) if year else []
    if not recent_years:
        return 0
    placeholders_years = ",".join(["%s"] * len(recent_years))
    sql = f"""
        SELECT COUNT(DISTINCT Year) AS season_count
        FROM {TABLE}
        WHERE (homeTeam = %s OR awayTeam = %s)
          AND Year IN ({placeholders_years})
    """
    rows = run_query(sql, tuple([team, team] + recent_years))
    return int(rows[0]["season_count"]) if rows else 0


def get_league_standings(league_id: str, year: Optional[str]) -> list:
    """
    Calcula la clasificación actual de la liga desde los resultados de partidos.
    Devuelve lista de dicts ordenada por puntos desc:
      [{"team": "barcelona", "pts": 60, "gf": 67, "ga": 20, "gd": 47}, ...]
    Solo incluye equipos con histórico suficiente (>= 2 temporadas).
    """
    recent_years = get_recent_years(year) if year else []
    if not recent_years:
        return []

    placeholders_years = ",".join(["%s"] * len(recent_years))
    sql = f"""
        SELECT homeTeam, awayTeam,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS hg,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS ag
        FROM {TABLE}
        WHERE name = 'Goals'
          AND LeagueId = %s
          AND Year = %s
        GROUP BY MatchId, homeTeam, awayTeam
    """
    rows = run_query(sql, (league_id, year))
    if not rows:
        return []

    table = {}
    for r in rows:
        h, a = r["homeTeam"], r["awayTeam"]
        hg, ag = float(r["hg"] or 0), float(r["ag"] or 0)
        for team in [h, a]:
            if team not in table:
                table[team] = {"team": team, "pts": 0, "gf": 0, "ga": 0}

        if hg > ag:
            table[h]["pts"] += 3
        elif hg < ag:
            table[a]["pts"] += 3
        else:
            table[h]["pts"] += 1
            table[a]["pts"] += 1

        table[h]["gf"] += hg
        table[h]["ga"] += ag
        table[a]["gf"] += ag
        table[a]["ga"] += hg

    for t in table.values():
        t["gd"] = t["gf"] - t["ga"]

    return sorted(table.values(), key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)


def get_features_from_neighbors(team: str, role: str, year: Optional[str],
                                  league_id: str, standings: list) -> dict:
    """
    Para equipos sin histórico suficiente (recién ascendidos), estima sus features
    usando la media de los 3 equipos más cercanos en tabla que sí tienen histórico.
    Esto evita que el modelo trate al equipo como uno de nivel medio.
    """
    # Posición del equipo en la tabla (0-indexed)
    pos = next((i for i, t in enumerate(standings) if t["team"] == team), None)
    if pos is None:
        return {}

    # Buscar vecinos con histórico suficiente (al menos 2 temporadas)
    neighbors = []
    for i, t in enumerate(standings):
        if t["team"] == team:
            continue
        if get_team_season_count(t["team"], year) >= 2:
            neighbors.append((abs(i - pos), t["team"]))

    # Los 3 más cercanos en tabla
    neighbors.sort(key=lambda x: x[0])
    closest = [name for _, name in neighbors[:3]]

    if not closest:
        return {}

    # Media de sus features
    all_features = [get_team_features(n, role, year) for n in closest]
    all_features = [f for f in all_features if f]  # filtrar vacíos

    if not all_features:
        return {}

    merged = {}
    for key in all_features[0]:
        vals = [f[key] for f in all_features if key in f]
        if vals:
            merged[key] = sum(vals) / len(vals)

    return merged

def build_features(home_team: str, away_team: str, year: Optional[str],
                   feature_cols: list, league_id: str = "8") -> pd.DataFrame:
    """Construye el vector de features completo para el partido."""
    features = {}

    print(f"DEBUG build_features: year={year!r}, league_id={league_id!r}")

    # ── Detectar equipos sin histórico suficiente y usar proxy de clasificación ──
    standings = get_league_standings(league_id, year) if year else []

    for team, role in [(home_team, "home"), (away_team, "away")]:
        season_count = get_team_season_count(team, year) if year else 2
        print(f"DEBUG {team}: season_count={season_count}")
        if season_count < 2 and standings:
            print(f"⚠️  {team} sin histórico suficiente ({season_count} temporadas) — usando proxy de clasificación")
            team_features = get_features_from_neighbors(team, role, year, league_id, standings)
        else:
            team_features = get_team_features(team, role, year)
        features.update(team_features)
    features.update(get_form_features(home_team, away_team, year))  # win rates, form pts, goles globales

    # Mapear goals_conceded desde los globales (única fuente disponible en predict)
    features["home_avg_goals_conceded"] = features.get("home_avg_goals_conceded_global", 0)
    features["away_avg_goals_conceded"] = features.get("away_avg_goals_conceded_global", 0)
    features.update(get_h2h_features(home_team, away_team, year))   # head to head
    features.update(get_over_rates(home_team, away_team, year))     # over rates por threshold
    features.update(get_elo(home_team, away_team))                  # elo ratings

    # ── Features combinadas que train.py calcula cruzando ambos equipos ──
    h_scored   = features.get("home_avg_goals_scored",  0)
    h_conceded = features.get("home_avg_goals_conceded", 0)
    a_scored   = features.get("away_avg_goals_scored",  0)
    a_conceded = features.get("away_avg_goals_conceded", 0)
    features["xG_match"] = (h_scored + a_conceded + a_scored + h_conceded) / 2

    h_xg = features.get("home_avg_xG", 0)
    a_xg = features.get("away_avg_xG", 0)
    features["expected_xG_match"] = h_xg + a_xg

    row = {col: features.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row]).astype(float)


# ─────────────────────────────────────────────
# LÓGICA HÍBRIDA: confidence basado en elo_diff
# ─────────────────────────────────────────────

def get_match_confidence(elo_diff: float, threshold: float) -> dict:
    """
    Determina si el partido es equilibrado o tiene un favorito claro
    basándose en la diferencia de Elo entre los equipos.

    - |elo_diff| >= threshold → favorito claro → "high"
    - |elo_diff| < threshold  → partido equilibrado → "balanced"

    También devuelve el valor numérico normalizado (0-100) para uso en el dashboard.
    """
    abs_diff = abs(elo_diff)
    is_clear_favorite = abs_diff >= threshold

    # Normalizar la confianza a escala 0-100 (cap en 200 puntos de elo)
    confidence_score = min(round((abs_diff / 200) * 100, 1), 100.0)

    return {
        "level":       "high" if is_clear_favorite else "balanced",
        "score":       confidence_score,
        "elo_diff":    round(elo_diff, 1),
        "description": "Favorito claro" if is_clear_favorite else "Partido equilibrado"
    }


# ─────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    year: Optional[str] = None
    league_id: str = "8"


@app.post("/predict")
def predict(req: PredictionRequest):
    reload_models_if_updated()

    model_result, calibrated_model, label_encoder, feature_cols, elo_threshold, over_models = load_models(req.league_id)

    try:
        X = build_features(req.home_team, req.away_team, req.year, feature_cols, req.league_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error construyendo features: {e}")

    # ── Resultado: siempre usar el modelo calibrado ──
    # El modelo calibrado ajusta mejor las probabilidades en todos los partidos,
    # tanto para favoritos claros como para partidos equilibrados.
    proba_cal = calibrated_model.predict_proba(X)[0]
    classes   = label_encoder.classes_
    result_proba = {cls: round(float(p) * 100, 4) for cls, p in zip(classes, proba_cal)}

    # predicted se calcula sobre las probabilidades calibradas (no las raw)
    predicted = classes[np.argmax(proba_cal)]

    # ── Confidence basado en elo_diff ──
    elo_diff   = float(X["elo_diff"].iloc[0])
    confidence = get_match_confidence(elo_diff, elo_threshold)

    # ── Cuotas estimadas ──
    odds = calculate_odds(result_proba, margin=0.07)

    # ── Over/Under ──
    over_under = {}
    for t in OVER_THRESHOLDS:
        key   = str(t)
        model = over_models[key]["model"]
        proba = model.predict_proba(X)[0]
        over_pct  = round(float(proba[1]) * 100, 4)
        under_pct = round(float(proba[0]) * 100, 4)
        over_under[f"over_{str(t).replace('.', '_')}"] = {
            "over":       over_pct,
            "under":      under_pct,
            "odds_over":  round((1 / (over_pct  / 100)) * (1 + 0.07), 2),
            "odds_under": round((1 / (under_pct / 100)) * (1 + 0.07), 2),
        }

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "year":      req.year,
        "resultado": {
            "predicted":     predicted,
            "probabilities": result_proba,
            "odds":          odds,
            "confidence":    confidence,   # {"level": "high"|"balanced", "score": 0-100, "elo_diff": X, "description": "..."}
        },
        "over_under": over_under
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
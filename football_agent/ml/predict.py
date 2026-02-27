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
    from ml.train import HybridCalibratedModel  # noqa: F401
    from ml.train import FEATURES
except ModuleNotFoundError:
    from train import HybridCalibratedModel  # noqa: F401
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
    MODEL_TIMESTAMPS[f"result_{_league}"]     = get_model_mtime(_rpath)
    MODEL_TIMESTAMPS[f"over_under_{_league}"] = get_model_mtime(_opath)

def reload_models_if_updated():
    global MODEL_TIMESTAMPS
    for league_name in ["laliga", "premier"]:
        result_path = os.path.join(MODELS_DIR, f"model_result_{league_name}.pkl")
        over_path   = os.path.join(MODELS_DIR, f"models_over_under_{league_name}.pkl")
        result_mtime = get_model_mtime(result_path)
        over_mtime   = get_model_mtime(over_path)
        if (MODEL_TIMESTAMPS.get(f"result_{league_name}") != result_mtime or
                MODEL_TIMESTAMPS.get(f"over_under_{league_name}") != over_mtime):
            print(f"🔄 Recargando modelos {league_name}: {datetime.now()}")
            MODEL_TIMESTAMPS[f"result_{league_name}"]     = result_mtime
            MODEL_TIMESTAMPS[f"over_under_{league_name}"] = over_mtime


def calculate_odds(probabilities: dict, margin: float = 0.07) -> dict:
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
        bundle.get("elo_diff_threshold", 20),
        over_models
    )

OVER_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]


# ─────────────────────────────────────────────
# HELPERS DE TEMPORADA
# ─────────────────────────────────────────────

def get_recent_years(year: str, n: int = 2) -> list:
    """Devuelve las últimas N temporadas incluyendo la actual.
    Ej: '25/26' -> ['25/26', '24/25']"""
    start = int(year.split("/")[0])
    return [f"{start - i}/{str((start - i + 1) % 100).zfill(2)}" for i in range(n)]


def get_prediction_year(current_year: str) -> str:
    """Devuelve la temporada anterior completa para usar como referencia historica.
    Ej: '25/26' -> '24/25'"""
    try:
        start = int(current_year.split("/")[0])
        prev = start - 1
        return f"{prev}/{str(prev + 1)[-2:]}"
    except Exception:
        return current_year


# ─────────────────────────────────────────────
# FEATURES DE EQUIPO
# ─────────────────────────────────────────────

def get_team_features(team: str, role: str, year: Optional[str], n: int = 10) -> dict:
    """Stats rolling medias de un equipo en los ultimos N partidos."""
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

    features[f"{role}_avg_goals_scored"] = stat_values.get("Goals", 0)
    return features


def get_team_season_count(team: str, year: Optional[str]) -> int:
    """Numero de temporadas distintas disponibles para este equipo en las ultimas 2 temporadas."""
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
    """Clasificacion actual de la liga calculada desde los resultados de la temporada en curso."""
    if not year:
        return []
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


def get_neighbors(team: str, year: Optional[str], standings: list, n_neighbors: int = 3) -> list:
    """Devuelve los N equipos mas cercanos en tabla con historico suficiente (>= 2 temporadas)."""
    pos = next((i for i, t in enumerate(standings) if t["team"] == team), None)
    if pos is None:
        return []
    candidates = []
    for i, t in enumerate(standings):
        if t["team"] == team:
            continue
        if get_team_season_count(t["team"], year) >= 2:
            candidates.append((abs(i - pos), t["team"]))
    candidates.sort(key=lambda x: x[0])
    return [name for _, name in candidates[:n_neighbors]]


def get_features_from_neighbors(team: str, role: str, year: Optional[str],
                                 league_id: str, standings: list) -> dict:
    """Estima features de un equipo sin historico usando la media de sus vecinos en tabla."""
    closest = get_neighbors(team, year, standings)
    if not closest:
        return {}
    all_features = [f for f in [get_team_features(n, role, year) for n in closest] if f]
    if not all_features:
        return {}
    merged = {}
    for key in all_features[0]:
        vals = [f[key] for f in all_features if key in f]
        if vals:
            merged[key] = sum(vals) / len(vals)
    print(f"WARNING {team} sin historico — proxy de vecinos: {closest}")
    return merged


# ─────────────────────────────────────────────
# FORM FEATURES
# ─────────────────────────────────────────────

def _compute_team_form(team: str, role: str, matches_global: list, matches_as_role: list) -> dict:
    """Calcula win rates, form points y goles globales de un equipo."""
    result = {}
    if matches_as_role:
        total = len(matches_as_role)
        if role == "home":
            wins   = sum(1 for r in matches_as_role if float(r["home_goals"] or 0) > float(r["away_goals"] or 0))
            draws  = sum(1 for r in matches_as_role if float(r["home_goals"] or 0) == float(r["away_goals"] or 0))
            losses = sum(1 for r in matches_as_role if float(r["home_goals"] or 0) < float(r["away_goals"] or 0))
            result[f"{role}_form_pts"] = sum(
                3 if float(r["home_goals"] or 0) > float(r["away_goals"] or 0) else
                (1 if float(r["home_goals"] or 0) == float(r["away_goals"] or 0) else 0)
                for r in matches_as_role
            )
        else:
            wins   = sum(1 for r in matches_as_role if float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
            draws  = sum(1 for r in matches_as_role if float(r["away_goals"] or 0) == float(r["home_goals"] or 0))
            losses = sum(1 for r in matches_as_role if float(r["away_goals"] or 0) < float(r["home_goals"] or 0))
            result[f"{role}_form_pts"] = sum(
                3 if float(r["away_goals"] or 0) > float(r["home_goals"] or 0) else
                (1 if float(r["away_goals"] or 0) == float(r["home_goals"] or 0) else 0)
                for r in matches_as_role
            )
        result[f"{role}_win_rate_{role}"]  = wins   / total
        result[f"{role}_draw_rate_{role}"] = draws  / total
        result[f"{role}_loss_rate_{role}"] = losses / total

    if matches_global:
        total_g = len(matches_global)
        global_wins = sum(
            1 for r in matches_global
            if (r["homeTeam"] == team and float(r["home_goals"] or 0) > float(r["away_goals"] or 0)) or
               (r["awayTeam"] == team and float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
        )
        result[f"{role}_win_rate_global"] = global_wins / total_g
        result[f"{role}_avg_goals_scored_global"] = sum(
            float(r["home_goals"] or 0) if r["homeTeam"] == team else float(r["away_goals"] or 0)
            for r in matches_global
        ) / total_g
        result[f"{role}_avg_goals_conceded_global"] = sum(
            float(r["away_goals"] or 0) if r["homeTeam"] == team else float(r["home_goals"] or 0)
            for r in matches_global
        ) / total_g
    return result


def get_form_features(home_team: str, away_team: str, league_id: str,
                      year: Optional[str], n: int = 10) -> dict:
    """Win rates, form points y goles globales para ambos equipos."""
    limit = n * 4
    recent_years = get_recent_years(year) if year else []
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [home_team, home_team, away_team, away_team] + recent_years + [league_id, limit]
    else:
        year_filter = ""
        params = [home_team, home_team, away_team, away_team, league_id, limit]

    sql = f"""
        SELECT matchId, homeTeam, awayTeam,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
               SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND (homeTeam = %s OR awayTeam = %s OR homeTeam = %s OR awayTeam = %s)
          {year_filter}
          AND LeagueId = %s
        GROUP BY matchId, homeTeam, awayTeam
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
        LIMIT %s
    """
    all_rows = run_query(sql, tuple(params))
    if not all_rows:
        return {}

    home_matches = [r for r in all_rows if r["homeTeam"] == home_team or r["awayTeam"] == home_team][:n]
    away_matches = [r for r in all_rows if r["homeTeam"] == away_team or r["awayTeam"] == away_team][:n]
    home_as_home = [r for r in all_rows if r["homeTeam"] == home_team][:5]
    away_as_away = [r for r in all_rows if r["awayTeam"] == away_team][:5]

    features = {}

    if home_matches:
        features.update(_compute_team_form(home_team, "home", home_matches, home_as_home))
    else:
        standings = get_league_standings(league_id, year) if year else []
        closest = get_neighbors(home_team, year, standings)
        neighbor_forms = []
        for neighbor in closest:
            nf = get_form_features(neighbor, neighbor, league_id, year, n)
            home_part = {k: v for k, v in nf.items() if k.startswith("home_")}
            if home_part:
                neighbor_forms.append(home_part)
        if neighbor_forms:
            for key in neighbor_forms[0]:
                vals = [f[key] for f in neighbor_forms if key in f]
                if vals:
                    features[key] = sum(vals) / len(vals)

    if away_matches:
        features.update(_compute_team_form(away_team, "away", away_matches, away_as_away))
    else:
        standings = get_league_standings(league_id, year) if year else []
        closest = get_neighbors(away_team, year, standings)
        neighbor_forms = []
        for neighbor in closest:
            nf = get_form_features(neighbor, neighbor, league_id, year, n)
            away_part = {k: v for k, v in nf.items() if k.startswith("away_")}
            if away_part:
                neighbor_forms.append(away_part)
        if neighbor_forms:
            for key in neighbor_forms[0]:
                vals = [f[key] for f in neighbor_forms if key in f]
                if vals:
                    features[key] = sum(vals) / len(vals)

    return features


# ─────────────────────────────────────────────
# H2H, OVER RATES, ELO
# ─────────────────────────────────────────────

def get_h2h_features(home_team: str, away_team: str, year: Optional[str], n: int = 5) -> dict:
    """
    H2H en ambas direcciones, normalizado desde perspectiva del home_team predicho.
    Con menos de 3 partidos devuelve distribucion neutral (muestra insuficiente).
    Sin filtro de temporada — LIMIT controla la cantidad.
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
    rows = run_query(sql, (home_team, away_team, away_team, home_team, n))

    if not rows:
        return {"h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0, "h2h_avg_goals": 0}

    total     = len(rows)
    avg_goals = sum(float(r["home_goals"] or 0) + float(r["away_goals"] or 0) for r in rows) / total

    # Con menos de 3 partidos la muestra es insignificante -> distribucion neutral
    if total < 3:
        return {
            "h2h_home_wins": 0.33,
            "h2h_draws":     0.33,
            "h2h_away_wins": 0.33,
            "h2h_avg_goals": round(avg_goals, 2)
        }

    home_wins = sum(
        1 for r in rows if
        (r["homeTeam"] == home_team and float(r["home_goals"] or 0) > float(r["away_goals"] or 0)) or
        (r["awayTeam"] == home_team and float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
    )
    draws = sum(1 for r in rows if float(r["home_goals"] or 0) == float(r["away_goals"] or 0))
    away_wins = sum(
        1 for r in rows if
        (r["homeTeam"] == away_team and float(r["home_goals"] or 0) > float(r["away_goals"] or 0)) or
        (r["awayTeam"] == away_team and float(r["away_goals"] or 0) > float(r["home_goals"] or 0))
    )

    return {
        "h2h_home_wins": home_wins / total,
        "h2h_draws":     draws     / total,
        "h2h_away_wins": away_wins / total,
        "h2h_avg_goals": round(avg_goals, 2)
    }


def get_over_rates(home_team: str, away_team: str, year: Optional[str], n: int = 10) -> dict:
    """Over rates historicas para cada threshold."""
    OVER_THRESHOLDS_LOCAL = [0.5, 1.5, 2.5, 3.5]
    recent_years = get_recent_years(year) if year else []
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [home_team, away_team, home_team, away_team] + recent_years + [n * 2]
    else:
        year_filter = ""
        params = [home_team, away_team, home_team, away_team, n * 2]

    sql = f"""
        SELECT matchId, homeTeam, awayTeam,
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
    """Elo actual de ambos equipos calculado desde todo el historico."""
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


# ─────────────────────────────────────────────
# BUILD FEATURES
# ─────────────────────────────────────────────

def build_features(home_team: str, away_team: str, year: Optional[str],
                   feature_cols: list, league_id: str = "8") -> pd.DataFrame:
    """Construye el vector de features completo para el partido."""
    features = {}

    standings = get_league_standings(league_id, year) if year else []

    for team, role in [(home_team, "home"), (away_team, "away")]:
        season_count = get_team_season_count(team, year) if year else 2
        if season_count < 2 and standings:
            team_features = get_features_from_neighbors(team, role, year, league_id, standings)
        else:
            team_features = get_team_features(team, role, year)
        features.update(team_features)

    features.update(get_form_features(home_team, away_team, league_id, year))

    features["home_avg_goals_conceded"] = features.get("home_avg_goals_conceded_global", 0)
    features["away_avg_goals_conceded"] = features.get("away_avg_goals_conceded_global", 0)

    features.update(get_h2h_features(home_team, away_team, year))
    features.update(get_over_rates(home_team, away_team, year))
    features.update(get_elo(home_team, away_team))

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
# CONFIDENCE
# ─────────────────────────────────────────────

def get_match_confidence(elo_diff: float, threshold: float) -> dict:
    abs_diff = abs(elo_diff)
    is_clear_favorite = abs_diff >= threshold
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

    proba_cal = calibrated_model.predict_proba(X)[0]

    # Floor minimo del 5% por resultado para evitar probabilidades cero
    abs_elo = abs(float(X["elo_diff"].iloc[0]))
    MIN_PROB = 0.13 if abs_elo > 150 else 0.05
    proba_cal = np.maximum(proba_cal, MIN_PROB)
    proba_cal = proba_cal / proba_cal.sum()

    classes      = label_encoder.classes_
    result_proba = {cls: round(float(p) * 100, 4) for cls, p in zip(classes, proba_cal)}
    predicted    = classes[np.argmax(proba_cal)]

    elo_diff   = float(X["elo_diff"].iloc[0])
    confidence = get_match_confidence(elo_diff, elo_threshold)
    odds       = calculate_odds(result_proba, margin=0.07)

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
            "confidence":    confidence,
        },
        "over_under": over_under
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
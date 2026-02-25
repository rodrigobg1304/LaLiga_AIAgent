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

app = FastAPI(title="Football Prediction API", version="1.0")

# ───────────────────────────────────────────────
# REENTRENAMIENTO DE MODELOS AL ACTUALIZAR DATOS
# ───────────────────────────────────────────────
MODEL_TIMESTAMPS = {}
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def get_model_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0

for _league in ["laliga", "premier"]:
    _rpath = os.path.join(MODELS_DIR, f"model_result_{_league}.pkl")
    _opath = os.path.join(MODELS_DIR, f"models_over_under_{_league}.pkl")
    MODEL_TIMESTAMPS[f"result_{_league}"] = get_model_mtime(_rpath)
    MODEL_TIMESTAMPS[f"over_under_{_league}"]   = get_model_mtime(_opath)

def reload_models_if_updated():
    """Recarga los modelos si los ficheros .pkl han cambiado."""
    global MODEL_TIMESTAMPS

    for league_name in ["laliga", "premier"]:
        result_path = os.path.join(MODELS_DIR, f"model_result_{league_name}.pkl")
        over_path = os.path.join(MODELS_DIR, f"models_over_under_{league_name}.pkl")

        result_mtime = get_model_mtime(result_path)
        over_mtime = get_model_mtime(over_path)

        if (MODEL_TIMESTAMPS.get(f"result_{league_name}") != result_mtime or
                MODEL_TIMESTAMPS.get(f"over_{league_name}") != over_mtime):
            print(f"🔄 Recargando modelos {league_name}: {datetime.now()}")
            MODEL_TIMESTAMPS[f"result_{league_name}"] = result_mtime
            MODEL_TIMESTAMPS[f"over_{league_name}"] = over_mtime


def calculate_odds(probabilities: dict, margin: float = 0.07) -> dict:
    """
    Convierte probabilidades del modelo en cuotas con margen.
    Probabilities: {"1": 67.88, "2": 7.02, "X": 25.09} (en %)
    margin: margen de la casa (default 7%)
    """
    return {
        result: round((1 / (prob / 100)) * (1 + margin), 2)
        for result, prob in probabilities.items()
        if prob > 0
    }

def is_balanced_match(probabilities: dict, threshold: float = 20.0) -> bool:
    """
    Devuelve True si el partido es equilibrado.
    Un partido es equilibrado si la diferencia entre la prob más alta
    y la más baja es menor que el threshold.
    """
    values = list(probabilities.values())
    return (max(values) - min(values)) < threshold

# ─────────────────────────────────────────────
# CARGAR MODELOS
# ─────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LEAGUE_MODEL_MAP = {
    "17": "premier",
    "8": "laliga"
}

def load_models(league_id: str) -> tuple:
    league_name = LEAGUE_MODEL_MAP.get(str(league_id), "laliga")

    result_path = os.path.join(MODELS_DIR, f"model_result_{league_name}.pkl")
    over_path   = os.path.join(MODELS_DIR, f"models_over_under_{league_name}.pkl")

    with open(result_path, "rb") as f:
        bundle = pickle.load(f)

    with open(over_path, "rb") as f:
        over_models = pickle.load(f)

    return bundle["model"], bundle["encoder"], bundle["features"], over_models

OVER_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]


# ─────────────────────────────────────────────
# CONSTRUIR FEATURES PARA UN PARTIDO NUEVO
# ─────────────────────────────────────────────

def get_team_features(team: str, role: str, year: Optional[str], n: int = 10) -> dict:
    """
    Calcula las features rolling de un equipo para un partido futuro.
    role: 'home' o 'away'
    """
    team_col = "homeTeam" if role == "home" else "awayTeam"
    stat_col = "homeValue" if role == "home" else "awayValue"

    from train import FEATURES
    placeholders = ",".join(["%s"] * len(FEATURES))

    sql = f"""
        SELECT name,
               AVG(CAST({stat_col} AS DECIMAL)) AS stat_value
        FROM (
            SELECT matchId, name, 
                   SUM(CAST({stat_col} AS DECIMAL)) AS {stat_col},
                   MAX(Year) AS Year,
                   MAX(CAST(Round AS SIGNED)) AS Round
            FROM {TABLE}
            WHERE {team_col} = %s
              AND name IN ({placeholders})
              AND period IN ('1ST', '2ND')
              {"AND Year = %s" if year else ""}
            GROUP BY matchId, name
            ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
            LIMIT %s
        ) AS last_matches
        GROUP BY name
    """
    params = [team] + FEATURES + ([year] if year else []) + [n * len(FEATURES)]
    rows = run_query(sql, tuple(params))

    if not rows:
        return {}

    features = {}
    for row in rows:
        feat = row["name"]
        features[f"{role}_avg_{feat}"] = float(row["stat_value"] or 0)

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


def build_features(home_team: str, away_team: str, year: Optional[str], feature_cols: list) -> pd.DataFrame:
    """Construye el vector de features para el partido."""
    features = {}
    features.update(get_team_features(home_team, "home", year))
    features.update(get_team_features(away_team, "away", year))
    features.update(get_elo(home_team, away_team))

    # Rellenar features que falten con 0
    row = {col: features.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row]).astype(float)


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
    model_result, label_encoder, feature_cols, over_models = load_models(req.league_id)
    try:
        X = build_features(req.home_team, req.away_team, req.year, feature_cols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error construyendo features: {e}")

    # ── Resultado ──
    proba_result = model_result.predict_proba(X)[0]
    classes      = label_encoder.classes_
    result_proba = {cls: round(float(p)*100, 4) for cls, p in zip(classes, proba_result)}
    predicted    = classes[np.argmax(proba_result)]
    # ── Añadir cuotas estimadas ──
    odds = calculate_odds(result_proba, margin=0.07)

    # ── Over/Under ──
    over_under = {}
    for t in OVER_THRESHOLDS:
        key   = str(t)
        model = over_models[key]["model"]
        proba = model.predict_proba(X)[0]
        over_pct = round(float(proba[1]) * 100, 4)
        under_pct = round(float(proba[0]) * 100, 4)
        over_under[f"over_{str(t).replace('.', '_')}"] = {
            "over": over_pct,
            "under": under_pct,
            "odds_over":  round((1 / (over_pct  / 100)) * (1 + 0.07), 2),
            "odds_under": round((1 / (under_pct / 100)) * (1 + 0.07), 2),
        }

    return {
        "home_team":   req.home_team,
        "away_team":   req.away_team,
        "year":        req.year,
        "resultado": {
            "predicted":     predicted,
            "probabilities": result_proba,
            "odds": odds
        },
        "over_under": over_under
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
import pickle
import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from football_agent.db import run_query, TABLE

app = FastAPI(title="Football Prediction API", version="1.0")

# ─────────────────────────────────────────────
# CARGAR MODELOS
# ─────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

with open(os.path.join(MODELS_DIR, "model_result.pkl"), "rb") as f:
    result_bundle = pickle.load(f)
    model_result  = result_bundle["model"]
    label_encoder = result_bundle["encoder"]
    feature_cols  = result_bundle["features"]

with open(os.path.join(MODELS_DIR, "model_over_under.pkl"), "rb") as f:
    over_models = pickle.load(f)

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


def build_features(home_team: str, away_team: str, year: Optional[str]) -> pd.DataFrame:
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


@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        X = build_features(req.home_team, req.away_team, req.year)
        print("\nDEBUG Features:")
        print(X[["home_avg_Goals", "away_avg_Goals",
                 "home_avg_Expected goals", "away_avg_Expected goals",
                 "elo_home", "elo_away", "elo_diff"]].to_string())

        print(f"DEBUG Elo - home: {req.home_team}, away: {req.away_team}")
        elo = get_elo(req.home_team, req.away_team)
        print(f"DEBUG Elo result: {elo}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error construyendo features: {e}")

    # ── Resultado ──
    proba_result = model_result.predict_proba(X)[0]
    classes      = label_encoder.classes_
    result_proba = {cls: round(float(p)*100, 4) for cls, p in zip(classes, proba_result)}
    predicted    = classes[np.argmax(proba_result)]

    # ── Over/Under ──
    over_under = {}
    for t in OVER_THRESHOLDS:
        key   = str(t)
        model = over_models[key]["model"]
        proba = model.predict_proba(X)[0]
        over_under[f"over_{str(t).replace('.', '_')}"] = {
            "over":  round(float(proba[1])*100, 4),
            "under": round(float(proba[0])*100, 4)
        }

    return {
        "home_team":   req.home_team,
        "away_team":   req.away_team,
        "year":        req.year,
        "resultado": {
            "predicted":     predicted,
            "probabilities": result_proba
        },
        "over_under": over_under
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
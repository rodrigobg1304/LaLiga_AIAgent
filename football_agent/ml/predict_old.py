import pickle
import os
import sys
import time
import threading
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
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
    from train_old import HybridCalibratedModel  # noqa: F401
    from train_old import FEATURES


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precarga Elos y modelos al arrancar para que la primera request sea rápida."""
    print("🚀 Precargando caché de Elo...")
    try:
        global _elo_cache, _elo_cache_time
        _elo_cache = _build_elo_cache()
        _elo_cache_time = time.time()
    except Exception as e:
        print(f"⚠️  Error precargando Elo: {e}")

    print("🚀 Precargando modelos...")
    for league_id in ["8", "17", "23"]:
        try:
            load_models(league_id)
        except Exception as e:
            print(f"⚠️  No se pudo precargar modelo {league_id}: {e}")

    print("✅ Precarga completada. API lista.")
    yield


app = FastAPI(title="Football Prediction API", version="1.0", lifespan=lifespan)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

OVER_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]

LEAGUE_MODEL_MAP = {
    "8": "laliga",
    "17": "premier",
    "23": "seriea"
}

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS ELO (DEBEN COINCIDIR CON train_old.py)
# ══════════════════════════════════════════════════════════════════════════════
ELO_K = 32  # Factor K (velocidad de ajuste)
ELO_SCALE = 600  # Escala (400=ajedrez, 600=fútbol)
ELO_HOME_ADVANTAGE = 100  # Ventaja de jugar en casa

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE PONDERACIÓN (DEBEN COINCIDIR CON train_old.py)
# ══════════════════════════════════════════════════════════════════════════════
RECENT_WEIGHT = 0.65  # Peso de forma reciente (últimos N partidos)
HISTORICAL_WEIGHT = 0.35  # Peso de histórico completo
WIN_RATE_RECENT_WEIGHT = 0.35  # Peso reciente para win rates
WIN_RATE_HISTORICAL_WEIGHT = 0.65  # Peso histórico para win rates (invertido)

# ─────────────────────────────────────────────
# CACHÉ GLOBAL DE MODELOS (en memoria)
# ─────────────────────────────────────────────

_models_cache: dict = {}
_models_mtime: dict = {}
_models_lock = threading.Lock()


def load_models(league_id: str) -> tuple:
    league_name = LEAGUE_MODEL_MAP.get(str(league_id), "laliga")
    result_path = os.path.join(MODELS_DIR, f"model_result_{league_name}.pkl")
    over_path = os.path.join(MODELS_DIR, f"models_over_under_{league_name}.pkl")
    result_mtime = os.path.getmtime(result_path) if os.path.exists(result_path) else 0
    over_mtime = os.path.getmtime(over_path) if os.path.exists(over_path) else 0
    cache_key = league_name

    with _models_lock:
        cached = _models_cache.get(cache_key)
        if (
                cached is None
                or _models_mtime.get(f"result_{cache_key}") != result_mtime
                or _models_mtime.get(f"over_{cache_key}") != over_mtime
        ):
            print(f"🔄 Cargando modelos {league_name} desde disco: {datetime.now()}")
            with open(result_path, "rb") as f:
                bundle = pickle.load(f)
            with open(over_path, "rb") as f:
                over_models = pickle.load(f)
            cached = (
                bundle["model"],
                bundle["calibrated_model"],
                bundle["encoder"],
                bundle["features"],
                bundle.get("elo_diff_threshold", 20),
                over_models,
            )
            _models_cache[cache_key] = cached
            _models_mtime[f"result_{cache_key}"] = result_mtime
            _models_mtime[f"over_{cache_key}"] = over_mtime

    return cached


# ─────────────────────────────────────────────
# CACHÉ DE ELO
# ─────────────────────────────────────────────

_elo_cache: dict = {}
_elo_cache_time: float = 0.0
_ELO_TTL_SECONDS = 3600


def _build_elo_cache() -> dict:
    """Recalcula el Elo de todos los equipos desde el histórico completo.
    USA: k=32, scale=600, home_advantage=100 (igual que train_old.py)"""
    print(f"🔄 Recalculando caché de Elo global: {datetime.now()}")
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
    elo: dict = {}

    for row in rows:
        h, a = row["homeTeam"], row["awayTeam"]
        elo.setdefault(h, 1500)
        elo.setdefault(a, 1500)

        elo_home_adjusted = elo[h] + ELO_HOME_ADVANTAGE
        exp_h = 1 / (1 + 10 ** ((elo[a] - elo_home_adjusted) / ELO_SCALE))

        hg, ag = float(row["hg"] or 0), float(row["ag"] or 0)
        if hg > ag:
            sh, sa = 1, 0
        elif hg < ag:
            sh, sa = 0, 1
        else:
            sh, sa = 0.5, 0.5

        elo[h] += ELO_K * (sh - exp_h)
        elo[a] += ELO_K * (sa - (1 - exp_h))

    return elo


def get_elo(home_team: str, away_team: str) -> dict:
    """Devuelve los Elos actuales desde caché."""
    global _elo_cache, _elo_cache_time

    if not _elo_cache or (time.time() - _elo_cache_time) > _ELO_TTL_SECONDS:
        _elo_cache = _build_elo_cache()
        _elo_cache_time = time.time()

    return {
        "elo_home": _elo_cache.get(home_team, 1500),
        "elo_away": _elo_cache.get(away_team, 1500),
        "elo_diff": _elo_cache.get(home_team, 1500) - _elo_cache.get(away_team, 1500),
    }


# ─────────────────────────────────────────────
# CACHÉ DE STANDINGS
# ─────────────────────────────────────────────

_standings_cache: dict = {}
_standings_mtime: dict = {}
_STANDINGS_TTL = 900


def get_league_standings(league_id: str, year: Optional[str]) -> list:
    if not year:
        return []

    cache_key = f"{league_id}_{year}"
    cache_age = time.time() - _standings_mtime.get(cache_key, 0)

    if cache_key in _standings_cache and cache_age < _STANDINGS_TTL:
        return _standings_cache[cache_key]

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
        _standings_cache[cache_key] = []
        _standings_mtime[cache_key] = time.time()
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

    result = sorted(table.values(), key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    _standings_cache[cache_key] = result
    _standings_mtime[cache_key] = time.time()
    return result


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def calculate_odds(probabilities: dict, margin: float = 0.07) -> dict:
    return {
        result: round(float((1 / (prob / 100)) * (1 + margin)), 2)
        for result, prob in probabilities.items()
        if prob > 0
    }


def get_recent_years(year: str, n: int = 2) -> list:
    """Devuelve las últimas N temporadas incluyendo la actual."""
    start = int(year.split("/")[0])
    return [f"{start - i}/{str((start - i + 1) % 100).zfill(2)}" for i in range(n)]


# ─────────────────────────────────────────────
# FEATURES DE EQUIPO CON PONDERACIÓN 65/35
# ─────────────────────────────────────────────

def get_team_stats_weighted(team: str, role: str, year: Optional[str],
                            stat_name: str, n: int = 5) -> float:
    """
    Calcula estadística ponderada: 65% forma reciente + 35% histórico completo.

    Args:
        team: Nombre del equipo
        role: 'home' o 'away'
        year: Temporada actual
        stat_name: Nombre de la estadística (ej: 'Goals', 'Expected goals')
        n: Número de partidos para forma reciente

    Returns:
        Valor ponderado de la estadística
    """
    team_col = "homeTeam" if role == "home" else "awayTeam"
    stat_col = "homeValue" if role == "home" else "awayValue"

    recent_years = get_recent_years(year) if year else []

    # ── HISTÓRICO COMPLETO ──
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter_hist = f"AND Year IN ({placeholders_years})"
        params_hist = [team, stat_name] + recent_years
    else:
        year_filter_hist = ""
        params_hist = [team, stat_name]

    sql_hist = f"""
        SELECT AVG(CAST({stat_col} AS DECIMAL)) AS stat_value
        FROM (
            SELECT matchId,
                   SUM(CAST({stat_col} AS DECIMAL)) AS {stat_col}
            FROM {TABLE}
            WHERE {team_col} = %s
              AND name = %s
              AND period IN ('1ST', '2ND')
              {year_filter_hist}
            GROUP BY matchId
        ) AS all_matches
    """
    rows_hist = run_query(sql_hist, tuple(params_hist))
    historical_avg = float(rows_hist[0]["stat_value"]) if (
                rows_hist and rows_hist[0]["stat_value"] is not None) else 0.0

    # ── FORMA RECIENTE (últimos n partidos) ──
    if recent_years:
        params_recent = [team, stat_name] + recent_years + [n]
    else:
        params_recent = [team, stat_name, n]

    sql_recent = f"""
        SELECT AVG(CAST({stat_col} AS DECIMAL)) AS stat_value
        FROM (
            SELECT matchId,
                   SUM(CAST({stat_col} AS DECIMAL)) AS {stat_col}
            FROM {TABLE}
            WHERE {team_col} = %s
              AND name = %s
              AND period IN ('1ST', '2ND')
              {year_filter_hist}
            GROUP BY matchId
            ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
            LIMIT %s
        ) AS recent_matches
    """
    rows_recent = run_query(sql_recent, tuple(params_recent))
    recent_avg = float(rows_recent[0]["stat_value"]) if (
                rows_recent and rows_recent[0]["stat_value"] is not None) else 0.0

    # ── PONDERACIÓN 65% reciente + 35% histórico ──
    weighted_avg = float(RECENT_WEIGHT) * recent_avg + float(HISTORICAL_WEIGHT) * historical_avg

    return weighted_avg


def get_team_features(team: str, role: str, year: Optional[str], n: int = 5) -> dict:
    """
    Calcula features de equipo con ponderación 65% reciente + 35% histórico.
    Igual que train_old.py líneas 157-170.
    """
    features = {}

    STAT_NAME_MAP = {
        "Expected goals": f"{role}_avg_xG",
        "Total shots": f"{role}_avg_shots",
        "Big chances": f"{role}_avg_big_chances",
    }

    # Calcular cada feature con ponderación
    for feat in FEATURES:
        weighted_val = get_team_stats_weighted(team, role, year, feat, n)
        features[f"{role}_avg_{feat}"] = weighted_val

        if feat in STAT_NAME_MAP:
            features[STAT_NAME_MAP[feat]] = weighted_val

    # Goals es especial (se usa mucho)
    features[f"{role}_avg_goals_scored"] = get_team_stats_weighted(team, role, year, "Goals", n)

    return features


def get_team_season_count(team: str, year: Optional[str]) -> int:
    """Número de temporadas disponibles para este equipo."""
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


def get_neighbors(team: str, year: Optional[str], standings: list, n_neighbors: int = 3) -> list:
    """Devuelve equipos vecinos en tabla con histórico >= 2 temporadas."""
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
    """Estima features usando vecinos cuando el equipo no tiene histórico."""
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
    print(f"⚠️  {team} sin histórico — proxy de vecinos: {closest}")
    return merged


# ─────────────────────────────────────────────
# FORM FEATURES CON PONDERACIÓN
# ─────────────────────────────────────────────

def get_win_rates_weighted(team: str, role: str, year: Optional[str],
                           league_id: str, n: int = 5) -> dict:
    """
    Calcula win rates con ponderación INVERTIDA: 35% reciente + 65% histórico.
    Igual que train_old.py líneas 291-313.
    """
    team_col = "homeTeam" if role == "home" else "awayTeam"
    result_win = "1" if role == "home" else "2"

    recent_years = get_recent_years(year) if year else []

    # ── HISTÓRICO COMPLETO ──
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params_hist = [team, league_id] + recent_years
    else:
        year_filter = ""
        params_hist = [team, league_id]

    sql_hist = f"""
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
        ) AS matches
    """
    rows_hist = run_query(sql_hist, tuple(params_hist))

    if not rows_hist or rows_hist[0]["total"] == 0:
        hist_win_rate = hist_draw_rate = hist_loss_rate = 0
    else:
        total_hist = float(rows_hist[0]["total"])
        if role == "home":
            hist_win_rate = float(rows_hist[0]["wins"]) / total_hist
            hist_draw_rate = float(rows_hist[0]["draws"]) / total_hist
            hist_loss_rate = float(rows_hist[0]["losses"]) / total_hist
        else:  # away
            hist_win_rate = float(rows_hist[0]["losses"]) / total_hist  # away gana cuando home pierde
            hist_draw_rate = float(rows_hist[0]["draws"]) / total_hist
            hist_loss_rate = float(rows_hist[0]["wins"]) / total_hist

    # ── FORMA RECIENTE ──
    if recent_years:
        params_recent = [team, league_id] + recent_years + [n]
    else:
        params_recent = [team, league_id, n]

    sql_recent = f"""
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
            LIMIT %s
        ) AS recent
    """
    rows_recent = run_query(sql_recent, tuple(params_recent))

    if not rows_recent or rows_recent[0]["total"] == 0:
        recent_win_rate = recent_draw_rate = recent_loss_rate = 0
    else:
        total_recent = float(rows_recent[0]["total"])
        if role == "home":
            recent_win_rate = float(rows_recent[0]["wins"]) / total_recent
            recent_draw_rate = float(rows_recent[0]["draws"]) / total_recent
            recent_loss_rate = float(rows_recent[0]["losses"]) / total_recent
        else:
            recent_win_rate = float(rows_recent[0]["losses"]) / total_recent
            recent_draw_rate = float(rows_recent[0]["draws"]) / total_recent
            recent_loss_rate = float(rows_recent[0]["wins"]) / total_recent

    # ── PONDERACIÓN INVERTIDA: 35% reciente + 65% histórico ──
    return {
        f"{role}_win_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_win_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_win_rate,
        f"{role}_draw_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_draw_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_draw_rate,
        f"{role}_loss_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_loss_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_loss_rate,
    }


def get_form_points(team: str, role: str, year: Optional[str],
                    league_id: str, n: int = 5) -> int:
    """Calcula puntos de forma de últimos N partidos."""
    team_col = "homeTeam" if role == "home" else "awayTeam"

    recent_years = get_recent_years(year) if year else []
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params = [team, league_id] + recent_years + [n]
    else:
        year_filter = ""
        params = [team, league_id, n]

    sql = f"""
        SELECT 
            SUM(CASE 
                WHEN home_goals > away_goals THEN 3
                WHEN home_goals = away_goals THEN 1
                ELSE 0
            END) AS points
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
            LIMIT %s
        ) AS recent
    """
    rows = run_query(sql, tuple(params))

    if not rows:
        return 0

    points = int(rows[0]["points"] or 0)

    # Para away, necesitamos invertir la lógica
    if role == "away":
        sql_away = f"""
            SELECT 
                SUM(CASE 
                    WHEN away_goals > home_goals THEN 3
                    WHEN away_goals = home_goals THEN 1
                    ELSE 0
                END) AS points
            FROM (
                SELECT matchId,
                       SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
                       SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
                FROM {TABLE}
                WHERE awayTeam = %s
                  AND name = 'Goals'
                  AND LeagueId = %s
                  {year_filter}
                GROUP BY matchId
                ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
                LIMIT %s
            ) AS recent
        """
        rows_away = run_query(sql_away, tuple(params))
        points = int(rows_away[0]["points"] or 0) if rows_away else 0

    return points


def get_goals_conceded_weighted(team: str, role: str, year: Optional[str], n: int = 5) -> float:
    """Calcula goles concedidos ponderados."""
    # Para home: concede los away_goals cuando juega como local
    # Para away: concede los home_goals cuando juega como visitante

    if role == "home":
        team_col = "homeTeam"
        goals_col = "awayValue"  # Concede goles del visitante
    else:
        team_col = "awayTeam"
        goals_col = "homeValue"  # Concede goles del local

    recent_years = get_recent_years(year) if year else []

    # Histórico
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params_hist = [team] + recent_years
    else:
        year_filter = ""
        params_hist = [team]

    sql_hist = f"""
        SELECT AVG(CAST({goals_col} AS DECIMAL)) AS avg_conceded
        FROM (
            SELECT matchId,
                   SUM(CAST({goals_col} AS DECIMAL)) AS {goals_col}
            FROM {TABLE}
            WHERE {team_col} = %s
              AND name = 'Goals'
              AND period IN ('1ST', '2ND')
              {year_filter}
            GROUP BY matchId
        ) AS all_matches
    """
    rows_hist = run_query(sql_hist, tuple(params_hist))
    hist_avg = float(rows_hist[0]["avg_conceded"]) if (rows_hist and rows_hist[0]["avg_conceded"] is not None) else 0.0

    # Reciente
    if recent_years:
        params_recent = [team] + recent_years + [n]
    else:
        params_recent = [team, n]

    sql_recent = f"""
        SELECT AVG(CAST({goals_col} AS DECIMAL)) AS avg_conceded
        FROM (
            SELECT matchId,
                   SUM(CAST({goals_col} AS DECIMAL)) AS {goals_col}
            FROM {TABLE}
            WHERE {team_col} = %s
              AND name = 'Goals'
              AND period IN ('1ST', '2ND')
              {year_filter}
            GROUP BY matchId
            ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
            LIMIT %s
        ) AS recent_matches
    """
    rows_recent = run_query(sql_recent, tuple(params_recent))
    recent_avg = float(rows_recent[0]["avg_conceded"]) if (
                rows_recent and rows_recent[0]["avg_conceded"] is not None) else 0.0

    return float(RECENT_WEIGHT) * recent_avg + float(HISTORICAL_WEIGHT) * hist_avg


def get_form_features(home_team: str, away_team: str, league_id: str, year: Optional[str]) -> dict:
    """Calcula win rates y form points con ponderación."""
    features = {}

    # Win rates con ponderación 35% reciente + 65% histórico
    features.update(get_win_rates_weighted(home_team, "home", year, league_id))
    features.update(get_win_rates_weighted(away_team, "away", year, league_id))

    # Form points (últimos 5 partidos)
    features["home_form_pts"] = get_form_points(home_team, "home", year, league_id, n=5)
    features["away_form_pts"] = get_form_points(away_team, "away", year, league_id, n=5)

    # Goles concedidos globales (usados para xG_match)
    features["home_avg_goals_conceded_global"] = get_goals_conceded_weighted(home_team, "home", year)
    features["away_avg_goals_conceded_global"] = get_goals_conceded_weighted(away_team, "away", year)

    # Win rate global (todo tipo de partidos)
    # Para simplificar, usamos el mismo que el específico de localía
    features["home_win_rate_global"] = features.get("home_win_rate_home", 0)
    features["away_win_rate_global"] = features.get("away_win_rate_away", 0)

    return features


# ─────────────────────────────────────────────
# H2H Y OVER RATES CON PONDERACIÓN
# ─────────────────────────────────────────────

def get_h2h_features(home_team: str, away_team: str, league_id: str,
                     year: Optional[str], n: int = 5) -> dict:
    """H2H con proxy de vecinos si < 2 enfrentamientos directos."""
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

    # Proxy con vecinos si < 2 enfrentamientos
    if len(rows) < 2:
        standings = get_league_standings(league_id, year) if year else []
        if standings:
            home_neighbors = get_neighbors(home_team, year, standings, n_neighbors=3)
            away_neighbors = get_neighbors(away_team, year, standings, n_neighbors=3)

            if home_neighbors and away_neighbors:
                neighbors_placeholders = ",".join(["%s"] * len(home_neighbors))
                neighbors_placeholders2 = ",".join(["%s"] * len(away_neighbors))

                sql_proxy = f"""
                    SELECT matchId, homeTeam, awayTeam,
                           SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
                           SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
                    FROM {TABLE}
                    WHERE name = 'Goals'
                      AND homeTeam IN ({neighbors_placeholders})
                      AND awayTeam IN ({neighbors_placeholders2})
                    GROUP BY matchId, homeTeam, awayTeam
                    ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
                    LIMIT %s
                """
                rows = run_query(sql_proxy, tuple(home_neighbors + away_neighbors + [n]))

                if rows:
                    print(f"⚠️  H2H {home_team} vs {away_team}: usando proxy de vecinos")

    if not rows:
        return {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33, "h2h_avg_goals": 0}

    total = len(rows)
    avg_goals = sum(float(r["home_goals"] or 0) + float(r["away_goals"] or 0) for r in rows) / total

    if total < 3:
        return {
            "h2h_home_wins": 0.33,
            "h2h_draws": 0.33,
            "h2h_away_wins": 0.33,
            "h2h_avg_goals": round(avg_goals, 2),
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
        "h2h_draws": draws / total,
        "h2h_away_wins": away_wins / total,
        "h2h_avg_goals": round(avg_goals, 2),
    }


def get_over_rates_weighted(team: str, year: Optional[str], n: int = 5) -> dict:
    """
    Calcula over rates con ponderación 65% reciente + 35% histórico.
    Igual que train_old.py líneas 183-200.
    """
    recent_years = get_recent_years(year) if year else []

    # ── HISTÓRICO COMPLETO ──
    if recent_years:
        placeholders_years = ",".join(["%s"] * len(recent_years))
        year_filter = f"AND Year IN ({placeholders_years})"
        params_hist = [team, team] + recent_years
    else:
        year_filter = ""
        params_hist = [team, team]

    sql_hist = f"""
        SELECT 
            SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) +
            SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS total_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND (homeTeam = %s OR awayTeam = %s)
          {year_filter}
        GROUP BY matchId
    """
    rows_hist = run_query(sql_hist, tuple(params_hist))

    # ── FORMA RECIENTE ──
    if recent_years:
        params_recent = [team, team] + recent_years + [n]
    else:
        params_recent = [team, team, n]

    sql_recent = f"""
        SELECT 
            SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(homeValue AS DECIMAL) ELSE 0 END) +
            SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS total_goals
        FROM {TABLE}
        WHERE name = 'Goals'
          AND (homeTeam = %s OR awayTeam = %s)
          {year_filter}
        GROUP BY matchId
        ORDER BY MAX(Year) DESC, MAX(CAST(Round AS SIGNED)) DESC
        LIMIT %s
    """
    rows_recent = run_query(sql_recent, tuple(params_recent))

    features = {}
    for t in [0.5, 1.5, 2.5, 3.5]:
        col = f"over_{str(t).replace('.', '_')}_rate"

        # Histórico
        hist_over_count = sum(1 for r in rows_hist if float(r["total_goals"] or 0) > t)
        hist_over_rate = float(hist_over_count) / float(len(rows_hist)) if rows_hist else 0.0

        # Reciente
        recent_over_count = sum(1 for r in rows_recent if float(r["total_goals"] or 0) > t)
        recent_over_rate = float(recent_over_count) / float(len(rows_recent)) if rows_recent else 0.0

        # Ponderación 65% reciente + 35% histórico
        features[col] = float(RECENT_WEIGHT) * recent_over_rate + float(HISTORICAL_WEIGHT) * hist_over_rate

    return features


def get_over_rates(home_team: str, away_team: str, year: Optional[str]) -> dict:
    """Over rates para home y away con ponderación."""
    home_rates = get_over_rates_weighted(home_team, year)
    away_rates = get_over_rates_weighted(away_team, year)

    features = {}
    for t in [0.5, 1.5, 2.5, 3.5]:
        col = f"over_{str(t).replace('.', '_')}_rate"
        features[f"home_{col}"] = home_rates.get(col, 0)
        features[f"away_{col}"] = away_rates.get(col, 0)
        features[f"combined_{col}"] = (home_rates.get(col, 0) + away_rates.get(col, 0)) / 2

    return features


# ─────────────────────────────────────────────
# BUILD FEATURES
# ─────────────────────────────────────────────

def build_features(home_team: str, away_team: str, year: Optional[str],
                   feature_cols: list, league_id: str = "8") -> pd.DataFrame:
    """Construye el vector de features con ponderación 65/35."""
    features = {}
    standings = get_league_standings(league_id, year) if year else []

    # Features de equipo con ponderación
    for team, role in [(home_team, "home"), (away_team, "away")]:
        season_count = get_team_season_count(team, year) if year else 2
        if season_count < 2 and standings:
            team_features = get_features_from_neighbors(team, role, year, league_id, standings)
        else:
            team_features = get_team_features(team, role, year, n=5)
        features.update(team_features)

    # Form features con ponderación
    features.update(get_form_features(home_team, away_team, league_id, year))

    # Goles concedidos
    features["home_avg_goals_conceded"] = features.get("home_avg_goals_conceded_global", 0)
    features["away_avg_goals_conceded"] = features.get("away_avg_goals_conceded_global", 0)

    # H2H con proxy
    features.update(get_h2h_features(home_team, away_team, league_id, year))

    # Over rates con ponderación
    features.update(get_over_rates(home_team, away_team, year))

    # Elo
    features.update(get_elo(home_team, away_team))

    # xG calculado
    h_scored = features.get("home_avg_goals_scored", 0)
    h_conceded = features.get("home_avg_goals_conceded", 0)
    a_scored = features.get("away_avg_goals_scored", 0)
    a_conceded = features.get("away_avg_goals_conceded", 0)
    features["xG_match"] = (h_scored + a_conceded + a_scored + h_conceded) / 2
    features["expected_xG_match"] = features.get("home_avg_xG", 0) + features.get("away_avg_xG", 0)

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
        "level": "high" if is_clear_favorite else "balanced",
        "score": confidence_score,
        "elo_diff": round(elo_diff, 1),
        "description": "Favorito claro" if is_clear_favorite else "Partido equilibrado",
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
    model_result, calibrated_model, label_encoder, feature_cols, elo_threshold, over_models = load_models(req.league_id)

    try:
        X = build_features(req.home_team, req.away_team, req.year, feature_cols, req.league_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error construyendo features: {e}")

    proba_cal = calibrated_model.predict_proba(X)[0]

    # Floor mínimo del 5% por resultado
    MIN_PROB = 0.05
    proba_cal = np.maximum(proba_cal, MIN_PROB)
    proba_cal = proba_cal / proba_cal.sum()

    classes = label_encoder.classes_
    result_proba = {cls: round(float(p) * 100, 4) for cls, p in zip(classes, proba_cal)}
    predicted = classes[np.argmax(proba_cal)]

    elo_diff = float(X["elo_diff"].iloc[0])
    confidence = get_match_confidence(elo_diff, elo_threshold)
    odds = calculate_odds(result_proba, margin=0.07)

    # Over/Under con monotonicidad
    raw_over_probs = {}
    for t in OVER_THRESHOLDS:
        key = str(t)
        model_o = over_models[key]["model"]
        calibrator = over_models[key].get("calibrator")
        raw_proba = model_o.predict_proba(X)[0][1]

        if calibrator is not None:
            raw_over_probs[t] = float(np.clip(calibrator.predict([raw_proba])[0], 0.01, 0.99))
        else:
            raw_over_probs[t] = float(np.clip(raw_proba, 0.01, 0.99))

    # MONOTONICIDAD DESACTIVADA PARA DEBUG - VER VALORES RAW
    over_under = {}
    for t in OVER_THRESHOLDS:  # [0.5, 1.5, 2.5, 3.5] sin ordenar
        over_prob = raw_over_probs[t]
        over_pct = round(over_prob * 100, 4)
        under_pct = round((1 - over_prob) * 100, 4)
        over_under[f"over_{str(t).replace('.', '_')}"] = {
            "over": over_pct,
            "under": under_pct,
            "odds_over": round((1 / (over_pct / 100)) * (1 + 0.07), 2) if over_pct > 0 else 999,
            "odds_under": round((1 / (under_pct / 100)) * (1 + 0.07), 2) if under_pct > 0 else 999,
            "raw_prob_debug": over_prob  # DEBUG: ver valor raw
        }

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "year": req.year,
        "resultado": {
            "predicted": predicted,
            "probabilities": result_proba,
            "odds": odds,
            "confidence": confidence,
        },
        "over_under": over_under,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
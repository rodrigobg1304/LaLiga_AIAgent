"""
Feature engineering compartido entre el servidor de predicción y los scripts de entrenamiento.

Extraído de ml/predict.py para romper la dependencia circular:
  train_over_under_*.py → predict.py → (carga de modelos)

Ahora el flujo es:
  train_over_under_*.py → football_core.feature_engineering  (sin modelos)
  predict.py            → football_core.feature_engineering  (solo features)
"""

import time
import numpy as np
import pandas as pd
from typing import Optional

from football_core.db import (
    get_all_matches_chronological,
    get_league_matches,
    get_team_season_count as db_get_team_season_count,
    get_team_recent_matches_goals,
    get_h2h_matches,
    get_proxy_h2h_matches,
    get_team_stat_average,
    get_team_multiple_stats_average,
    get_team_win_rates,
    get_team_goals_conceded_average,
    get_team_matches_total_goals,
    get_current_tournament_averages,
    get_team_xg_average,
    get_team_current_tournament_points,
)

from football_core.constants import (
    FEATURES,
    RECENT_WEIGHT,
    HISTORICAL_WEIGHT,
    WIN_RATE_RECENT_WEIGHT,
    WIN_RATE_HISTORICAL_WEIGHT,
    ELO_K,
    ELO_SCALE,
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    FORM_WINDOW,
    H2H_LOOKBACK,
    MIN_H2H_MATCHES,
    INTERNATIONAL_LEAGUE_IDS,
    NEUTRAL_TEAM_STATS,
    TOURN_CREDIBILITY_PSEUDO_COUNT,
)


# ─────────────────────────────────────────────
# CACHÉ DE ELO
# ─────────────────────────────────────────────

_elo_cache: dict = {}
_elo_cache_time: float = 0.0
_ELO_TTL_SECONDS = 3600


def _build_elo_cache() -> dict:
    """
    Recalcula el Elo de todos los equipos desde el histórico completo.
    """
    print(f"🔄 Recalculando caché de Elo global...")

    matches = get_all_matches_chronological()
    elo: dict = {}

    for match in matches:
        h, a = match["homeTeam"], match["awayTeam"]
        elo.setdefault(h, 1500)
        elo.setdefault(a, 1500)

        elo_home_adjusted = elo[h] + ELO_HOME_ADVANTAGE
        exp_h = 1 / (1 + 10 ** ((elo[a] - elo_home_adjusted) / ELO_SCALE))

        hg, ag = float(match["home_goals"] or 0), float(match["away_goals"] or 0)
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
    """Devuelve los Elos actuales desde caché.

    El caché usa las claves exactas de la BD (slugs en minúsculas).
    Se normaliza a minúsculas para evitar mismatches por capitalización.
    """
    global _elo_cache, _elo_cache_time

    if not _elo_cache or (time.time() - _elo_cache_time) > _ELO_TTL_SECONDS:
        _elo_cache = _build_elo_cache()
        _elo_cache_time = time.time()

    h = home_team.lower()
    a = away_team.lower()

    return {
        "elo_home": _elo_cache.get(h, 1500),
        "elo_away": _elo_cache.get(a, 1500),
        "elo_diff": _elo_cache.get(h, 1500) - _elo_cache.get(a, 1500),
    }


def warm_cache() -> None:
    """
    Precarga el caché de Elo en el arranque del servidor.
    Llamar desde el lifespan de FastAPI para que la primera request sea rápida.
    """
    global _elo_cache, _elo_cache_time
    _elo_cache = _build_elo_cache()
    _elo_cache_time = time.time()


# ─────────────────────────────────────────────
# CACHÉ DE STANDINGS
# ─────────────────────────────────────────────

_standings_cache: dict = {}
_standings_mtime: dict = {}
_STANDINGS_TTL = 900


def get_league_standings(league_id: str, year: Optional[str]) -> list:
    """Calcula la clasificación de una liga/año con caché."""
    if not year:
        return []

    cache_key = f"{league_id}_{year}"
    cache_age = time.time() - _standings_mtime.get(cache_key, 0)

    if cache_key in _standings_cache and cache_age < _STANDINGS_TTL:
        return _standings_cache[cache_key]

    matches = get_league_matches(league_id, year)

    if not matches:
        _standings_cache[cache_key] = []
        _standings_mtime[cache_key] = time.time()
        return []

    table = {}
    for match in matches:
        h, a = match["homeTeam"], match["awayTeam"]
        hg, ag = float(match["home_goals"] or 0), float(match["away_goals"] or 0)

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
# CACHÉ DE FEATURES POR EQUIPO
# ─────────────────────────────────────────────

_team_features_cache: dict = {}
_team_features_mtime: dict = {}
_TEAM_FEATURES_TTL = 1800  # 30 minutos
_neighbors_cache = {}  # Cache: {(team, year, standings_tuple): [vecinos]}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_recent_years(year: str, n: int = 2) -> list:
    """Devuelve las últimas N temporadas incluyendo la actual.

    Soporta dos formatos:
    - Doméstico: "25/26" → ["25/26", "24/25"]
    - Internacional: "2026" → ["2026", "2022"] (ciclos de 4 años: WC, Euros)
    """
    if "/" in year:
        start = int(year.split("/")[0])
        return [f"{start - i}/{str((start - i + 1) % 100).zfill(2)}" for i in range(n)]
    else:
        # Formato de año completo (torneos internacionales: WC, Euros, etc.)
        # Los ciclos son cada 4 años: 2026 → [2026, 2022, 2018]
        start = int(year)
        return [str(start - i * 4) for i in range(n)]


# ─────────────────────────────────────────────
# FEATURES DE EQUIPO CON PONDERACIÓN 65/35
# ─────────────────────────────────────────────

def get_team_stats_weighted(team: str, role: str, year: Optional[str],
                            stat_name: str, n: int = 5) -> float:
    """Calcula estadística ponderada: 65% forma reciente + 35% histórico completo."""
    recent_years = get_recent_years(year) if year else []

    historical_avg = get_team_stat_average(team, role, stat_name, recent_years, n=None)
    recent_avg = get_team_stat_average(team, role, stat_name, recent_years, n=n)

    return float(RECENT_WEIGHT) * recent_avg + float(HISTORICAL_WEIGHT) * historical_avg


def _neutral_team_features(role: str) -> dict:
    """
    Devuelve el vector de features neutras para un equipo sin datos históricos.
    Coincide con los valores NEUTRAL del training script (train_qualy.py) para
    que el modelo reciba inputs dentro de su distribución de entrenamiento.
    """
    n = NEUTRAL_TEAM_STATS
    return {
        f"{role}_avg_Goals":              n['goals_for_avg'],
        f"{role}_avg_Ball possession":    n['possession_avg'],
        f"{role}_avg_Total shots":        n['total_shots_avg'],
        f"{role}_avg_Shots on target":    n['shots_on_target_avg'],
        f"{role}_avg_Goalkeeper saves":   n['gk_saves_avg'],
        f"{role}_avg_Big chances":        n['big_chances_avg'],
        f"{role}_avg_Accurate passes":    n['accurate_passes_avg'],
        f"{role}_avg_Tackles won":        n['tackles_won_avg'],
        f"{role}_avg_Interceptions":      n['interceptions_avg'],
        f"{role}_avg_Blocked shots":      n['blocked_shots_avg'],
        f"{role}_avg_goals_scored":       n['goals_for_avg'],
        f"{role}_avg_shots":              n['total_shots_avg'],
        f"{role}_avg_big_chances":        n['big_chances_avg'],
        f"{role}_avg_xG":                 0,
        f"{role}_win_rate_{role}":        n['win_rate'],
        f"{role}_draw_rate_{role}":       0.27,
        f"{role}_loss_rate_{role}":       0.40,
        f"{role}_form_pts":               n['points_avg'] * 5,
        f"{role}_avg_goals_conceded_global": n['goals_against_avg'],
    }


def get_team_features(team: str, role: str, year: Optional[str], league_id: str, n: int = 5) -> dict:
    """Versión OPTIMIZADA: 2 queries en lugar de 18."""
    recent_years = get_recent_years(year) if year else []

    if not recent_years:
        return _neutral_team_features(role)

    stats_hist = get_team_multiple_stats_average(team, role, recent_years, n=None)
    stats_recent = get_team_multiple_stats_average(team, role, recent_years, n=n)

    # Sin datos en BD → fallback a valores neutros (evita que el modelo reciba ceros)
    if not stats_hist and not stats_recent:
        return _neutral_team_features(role)

    features = {}

    stat_mapping = {
        'goals': 'Goals',
        'ball_possession': 'Ball possession',
        'total_shots': 'Total shots',
        'shots_on_target': 'Shots on target',
        'gk_saves': 'Goalkeeper saves',
        'big_chances': 'Big chances',
        'accurate_passes': 'Accurate passes',
        'tackles_won': 'Tackles won',
        'interceptions': 'Interceptions',
        'blocked_shots': 'Blocked shots'
    }

    for sql_col, feat_name in stat_mapping.items():
        hist_avg = stats_hist.get(sql_col, 0.0)
        recent_avg = stats_recent.get(sql_col, 0.0)
        weighted_avg = float(RECENT_WEIGHT) * recent_avg + float(HISTORICAL_WEIGHT) * hist_avg
        features[f"{role}_avg_{feat_name}"] = weighted_avg

    features[f"{role}_avg_xG"] = features.get(f"{role}_avg_Expected goals", 0)
    features[f"{role}_avg_shots"] = features.get(f"{role}_avg_Total shots", 0)
    features[f"{role}_avg_big_chances"] = features.get(f"{role}_avg_Big chances", 0)
    features[f"{role}_avg_goals_scored"] = features.get(f"{role}_avg_Goals", 0)

    form_features = _get_form_features_inline(team=team, role=role, league_id=league_id, n=n,
                                              recent_years=recent_years)
    features.update(form_features)

    return features


def get_team_features_cached(team: str, role: str, year: Optional[str],
                             league_id: str, n: int = 5) -> dict:
    """Wrapper con caché para get_team_features."""
    cache_key = f"{team}_{role}_{year}_{league_id}"
    cache_age = time.time() - _team_features_mtime.get(cache_key, 0)

    if cache_key in _team_features_cache and cache_age < _TEAM_FEATURES_TTL:
        return _team_features_cache[cache_key]

    features = get_team_features(team=team, role=role, year=year, league_id=league_id, n=n)
    _team_features_cache[cache_key] = features
    _team_features_mtime[cache_key] = time.time()

    return features


def _get_form_features_inline(team: str, role: str, league_id: str, n: int, recent_years: list) -> dict:
    """Helper interno para calcular win_rate, form_pts y goals_conceded.

    Para ligas internacionales consulta todos los torneos internacionales
    combinados, ya que una selección puede tener datos en varias competiciones
    (ej. Spain: liga 11 qualifying + liga 16 World Cup).
    """
    # Para internacionales, ampliar a todas las ligas de selecciones nacionales
    if league_id in INTERNATIONAL_LEAGUE_IDS:
        effective_leagues = list(INTERNATIONAL_LEAGUE_IDS)
    else:
        effective_leagues = league_id

    stats_hist_wr = get_team_win_rates(team, role, effective_leagues, recent_years, n=None)
    stats_recent_wr = get_team_win_rates(team, role, effective_leagues, recent_years, n=n)

    neutral_wr = NEUTRAL_TEAM_STATS['win_rate']
    neutral_draw = 0.27
    neutral_loss = 0.40

    if stats_hist_wr["total"] == 0:
        hist_win_rate = neutral_wr
        hist_draw_rate = neutral_draw
        hist_loss_rate = neutral_loss
    else:
        total_hist = float(stats_hist_wr["total"])
        if role == "home":
            hist_win_rate = float(stats_hist_wr["wins"]) / total_hist
            hist_draw_rate = float(stats_hist_wr["draws"]) / total_hist
            hist_loss_rate = float(stats_hist_wr["losses"]) / total_hist
        else:
            hist_win_rate = float(stats_hist_wr["losses"]) / total_hist
            hist_draw_rate = float(stats_hist_wr["draws"]) / total_hist
            hist_loss_rate = float(stats_hist_wr["wins"]) / total_hist

    if stats_recent_wr["total"] == 0:
        recent_win_rate = neutral_wr
        recent_draw_rate = neutral_draw
        recent_loss_rate = neutral_loss
        form_pts = NEUTRAL_TEAM_STATS['points_avg'] * 5
    else:
        total_recent = float(stats_recent_wr["total"])
        if role == "home":
            recent_win_rate = float(stats_recent_wr["wins"]) / total_recent
            recent_draw_rate = float(stats_recent_wr["draws"]) / total_recent
            recent_loss_rate = float(stats_recent_wr["losses"]) / total_recent
            form_pts = stats_recent_wr["wins"] * 3 + stats_recent_wr["draws"]
        else:
            recent_win_rate = float(stats_recent_wr["losses"]) / total_recent
            recent_draw_rate = float(stats_recent_wr["draws"]) / total_recent
            recent_loss_rate = float(stats_recent_wr["wins"]) / total_recent
            form_pts = stats_recent_wr["losses"] * 3 + stats_recent_wr["draws"]

    hist_conceded = get_team_goals_conceded_average(team, role, recent_years, n=None)
    recent_conceded = get_team_goals_conceded_average(team, role, recent_years, n=n)
    neutral_conceded = NEUTRAL_TEAM_STATS['goals_against_avg']

    return {
        f"{role}_win_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_win_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_win_rate,
        f"{role}_draw_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_draw_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_draw_rate,
        f"{role}_loss_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_loss_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_loss_rate,
        f"{role}_form_pts": form_pts,
        f"{role}_avg_goals_conceded_global": float(RECENT_WEIGHT) * (recent_conceded or neutral_conceded)
                                             + float(HISTORICAL_WEIGHT) * (hist_conceded or neutral_conceded),
    }


def get_team_season_count(team: str, year: Optional[str]) -> int:
    """Número de temporadas disponibles para este equipo."""
    recent_years = get_recent_years(year) if year else []
    if not recent_years:
        return 0
    return db_get_team_season_count(team=team, years=recent_years)


def get_neighbors(team: str, year: Optional[str], standings: list, n_neighbors: int = 3) -> list:
    """Devuelve equipos vecinos en tabla con histórico >= 2 temporadas (con caché)."""
    standings_tuple = tuple(t["team"] for t in standings)
    cache_key = (team, year, standings_tuple)

    if cache_key in _neighbors_cache:
        return _neighbors_cache[cache_key]

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
    result = [name for _, name in candidates[:n_neighbors]]

    _neighbors_cache[cache_key] = result
    return result


def get_features_from_neighbors(team: str, role: str, year: Optional[str],
                                league_id: str, standings: list) -> dict:
    """Estima features usando vecinos cuando el equipo no tiene histórico."""
    closest = get_neighbors(team, year, standings)
    if not closest:
        return {}

    all_features = [
        f for f in [
            get_team_features_cached(team=neighbor, role=role, year=year, league_id=league_id, n=5)
            for neighbor in closest
        ] if f
    ]

    if not all_features:
        return {}

    merged = {}
    for key in all_features[0]:
        vals = [f[key] for f in all_features if key in f]
        if vals:
            merged[key] = sum(vals) / len(vals)

    PROMOTED_TEAM_PENALTY = 0.75
    for key in [k for k in merged.keys() if 'elo' not in k.lower()]:
        merged[key] *= PROMOTED_TEAM_PENALTY

    return merged


# ─────────────────────────────────────────────
# FORM FEATURES CON PONDERACIÓN
# ─────────────────────────────────────────────

def get_win_rates_weighted(team: str, role: str, year: Optional[str],
                           league_id: str, n: int = 5) -> dict:
    """Calcula win rates con ponderación INVERTIDA: 35% reciente + 65% histórico."""
    recent_years = get_recent_years(year) if year else []

    stats_hist = get_team_win_rates(team, role, league_id, recent_years, n=None)
    stats_recent = get_team_win_rates(team, role, league_id, recent_years, n=n)

    if stats_hist["total"] == 0:
        hist_win_rate = hist_draw_rate = hist_loss_rate = 0
    else:
        total_hist = float(stats_hist["total"])
        if role == "home":
            hist_win_rate = float(stats_hist["wins"]) / total_hist
            hist_draw_rate = float(stats_hist["draws"]) / total_hist
            hist_loss_rate = float(stats_hist["losses"]) / total_hist
        else:
            hist_win_rate = float(stats_hist["losses"]) / total_hist
            hist_draw_rate = float(stats_hist["draws"]) / total_hist
            hist_loss_rate = float(stats_hist["wins"]) / total_hist

    if stats_recent["total"] == 0:
        recent_win_rate = recent_draw_rate = recent_loss_rate = 0
    else:
        total_recent = float(stats_recent["total"])
        if role == "home":
            recent_win_rate = float(stats_recent["wins"]) / total_recent
            recent_draw_rate = float(stats_recent["draws"]) / total_recent
            recent_loss_rate = float(stats_recent["losses"]) / total_recent
        else:
            recent_win_rate = float(stats_recent["losses"]) / total_recent
            recent_draw_rate = float(stats_recent["draws"]) / total_recent
            recent_loss_rate = float(stats_recent["wins"]) / total_recent

    return {
        f"{role}_win_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_win_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_win_rate,
        f"{role}_draw_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_draw_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_draw_rate,
        f"{role}_loss_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_loss_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_loss_rate,
    }


def get_form_points(team: str, role: str, year: Optional[str],
                    league_id: str, n: int = 5) -> int:
    """Calcula puntos de forma de últimos N partidos."""
    recent_years = get_recent_years(year) if year else []
    matches = get_team_recent_matches_goals(team, league_id, recent_years, role, n)

    if not matches:
        return 0

    points = 0
    for match in matches:
        hg = float(match["home_goals"] or 0)
        ag = float(match["away_goals"] or 0)

        if role == "home":
            if hg > ag:
                points += 3
            elif hg == ag:
                points += 1
        else:
            if ag > hg:
                points += 3
            elif ag == hg:
                points += 1

    return points


def get_goals_conceded_weighted(team: str, role: str, year: Optional[str], n: int = 5) -> float:
    """Calcula goles concedidos ponderados: 65% reciente + 35% histórico."""
    recent_years = get_recent_years(year) if year else []

    hist_avg = get_team_goals_conceded_average(team, role, recent_years, n=None)
    recent_avg = get_team_goals_conceded_average(team, role, recent_years, n=n)

    return float(RECENT_WEIGHT) * recent_avg + float(HISTORICAL_WEIGHT) * hist_avg


def get_form_features(home_team: str, away_team: str, league_id: str, year: Optional[str]) -> dict:
    """Calcula win rates y form points con ponderación."""
    features = {}

    features.update(get_win_rates_weighted(home_team, "home", year, league_id))
    features.update(get_win_rates_weighted(away_team, "away", year, league_id))

    features["home_form_pts"] = get_form_points(home_team, "home", year, league_id, n=5)
    features["away_form_pts"] = get_form_points(away_team, "away", year, league_id, n=5)

    features["home_avg_goals_conceded_global"] = get_goals_conceded_weighted(home_team, "home", year)
    features["away_avg_goals_conceded_global"] = get_goals_conceded_weighted(away_team, "away", year)

    features["home_win_rate_global"] = features.get("home_win_rate_home", 0)
    features["away_win_rate_global"] = features.get("away_win_rate_away", 0)

    return features


# ─────────────────────────────────────────────
# H2H Y OVER RATES CON PONDERACIÓN
# ─────────────────────────────────────────────

def get_h2h_features(home_team: str, away_team: str, league_id: str,
                     year: Optional[str], n: int = 5) -> dict:
    """Calcula features H2H con proxy de vecinos si < 2 enfrentamientos directos."""
    matches = get_h2h_matches(home_team, away_team, n)

    if len(matches) < 2:
        standings = get_league_standings(league_id, year) if year else []
        if standings:
            home_neighbors = get_neighbors(home_team, year, standings, n_neighbors=3)
            away_neighbors = get_neighbors(away_team, year, standings, n_neighbors=3)

            if home_neighbors and away_neighbors:
                matches = get_proxy_h2h_matches(home_neighbors, away_neighbors, n)

    if not matches:
        return {
            "h2h_home_wins": 0.33,
            "h2h_draws": 0.33,
            "h2h_away_wins": 0.33,
            "h2h_avg_goals": 0
        }

    total = len(matches)
    avg_goals = sum(float(m["home_goals"] or 0) + float(m["away_goals"] or 0) for m in matches) / total

    if total < 3:
        return {
            "h2h_home_wins": 0.33,
            "h2h_draws": 0.33,
            "h2h_away_wins": 0.33,
            "h2h_avg_goals": round(avg_goals, 2),
        }

    home_wins = sum(
        1 for m in matches if
        (m["homeTeam"] == home_team and float(m["home_goals"] or 0) > float(m["away_goals"] or 0)) or
        (m["awayTeam"] == home_team and float(m["away_goals"] or 0) > float(m["home_goals"] or 0))
    )
    draws = sum(1 for m in matches if float(m["home_goals"] or 0) == float(m["away_goals"] or 0))
    away_wins = sum(
        1 for m in matches if
        (m["homeTeam"] == away_team and float(m["home_goals"] or 0) > float(m["away_goals"] or 0)) or
        (m["awayTeam"] == away_team and float(m["away_goals"] or 0) > float(m["home_goals"] or 0))
    )

    return {
        "h2h_home_wins": home_wins / total,
        "h2h_draws": draws / total,
        "h2h_away_wins": away_wins / total,
        "h2h_avg_goals": round(avg_goals, 2),
    }


def get_h2h_features_cached(home_team: str, away_team: str,
                            league_id: str, year: Optional[str], n: int = 5) -> dict:
    """Wrapper con caché para H2H."""
    cache_key = f"{home_team}_vs_{away_team}_{year}_{league_id}_h2h"
    cache_age = time.time() - _team_features_mtime.get(cache_key, 0)

    if cache_key in _team_features_cache and cache_age < _TEAM_FEATURES_TTL:
        return _team_features_cache[cache_key]

    features = get_h2h_features(home_team, away_team, league_id, year, n)
    _team_features_cache[cache_key] = features
    _team_features_mtime[cache_key] = time.time()

    return features


def get_over_rates_weighted(team: str, year: Optional[str], n: int = 5) -> dict:
    """Calcula over rates con ponderación 65% reciente + 35% histórico."""
    recent_years = get_recent_years(year) if year else []

    matches_hist = get_team_matches_total_goals(team, recent_years, n=None)
    matches_recent = get_team_matches_total_goals(team, recent_years, n=n)

    features = {}

    for threshold in [0.5, 1.5, 2.5, 3.5]:
        col = f"over_{str(threshold).replace('.', '_')}_rate"

        hist_over_count = sum(1 for m in matches_hist if float(m["total_goals"] or 0) > threshold)
        hist_over_rate = float(hist_over_count) / float(len(matches_hist)) if matches_hist else 0.0

        recent_over_count = sum(1 for m in matches_recent if float(m["total_goals"] or 0) > threshold)
        recent_over_rate = float(recent_over_count) / float(len(matches_recent)) if matches_recent else 0.0

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


def get_over_rates_cached(home_team: str, away_team: str, year: Optional[str]) -> dict:
    """Wrapper con caché para over rates."""
    cache_key = f"{home_team}_{away_team}_{year}_over"
    cache_age = time.time() - _team_features_mtime.get(cache_key, 0)

    if cache_key in _team_features_cache and cache_age < _TEAM_FEATURES_TTL:
        return _team_features_cache[cache_key]

    features = get_over_rates(home_team, away_team, year)
    _team_features_cache[cache_key] = features
    _team_features_mtime[cache_key] = time.time()

    return features


# ─────────────────────────────────────────────
# BUILD FEATURES
# ─────────────────────────────────────────────

def build_features(home_team: str, away_team: str, year: Optional[str],
                   feature_cols: list, league_id: str = "8") -> pd.DataFrame:
    """Construye el vector de features con ponderación 65/35."""
    features = {}
    standings = get_league_standings(league_id, year) if year else []

    for team, role in [(home_team, "home"), (away_team, "away")]:
        season_count = get_team_season_count(team, year) if year else 2
        if season_count < 2 and standings:
            team_features = get_features_from_neighbors(team, role, year, league_id, standings)
        else:
            team_features = get_team_features_cached(team=team, role=role, year=year, league_id=league_id, n=5)
        features.update(team_features)

    features.update(get_form_features(home_team, away_team, league_id, year))

    features["home_avg_goals_conceded"] = features.get("home_avg_goals_conceded_global", 0)
    features["away_avg_goals_conceded"] = features.get("away_avg_goals_conceded_global", 0)

    features.update(get_h2h_features(home_team, away_team, league_id, year))
    features.update(get_over_rates(home_team, away_team, year))
    features.update(get_elo(home_team, away_team))

    h_scored = features.get("home_avg_goals_scored", 0)
    h_conceded = features.get("home_avg_goals_conceded", 0)
    a_scored = features.get("away_avg_goals_scored", 0)
    a_conceded = features.get("away_avg_goals_conceded", 0)
    features["xG_match"] = (h_scored + a_conceded + a_scored + h_conceded) / 2
    features["expected_xG_match"] = features.get("home_avg_xG", 0) + features.get("away_avg_xG", 0)

    row = {col: features.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row]).astype(float)


def build_features_1x2(home_team: str, away_team: str, year: Optional[str],
                       league_id: str = "8") -> np.ndarray:
    """
    Construye vector de features para modelos 1X2 (RF/XGBoost).
    - 40 features base para LaLiga/Serie A
    - 47 features (40 + 7 Premier) para Premier League
    """
    features = {}
    standings = get_league_standings(league_id, year) if year else []

    # ── 1. FEATURES ELO (3) ──
    elo_data = get_elo(home_team, away_team)
    features['elo_home'] = elo_data['elo_home']
    features['elo_away'] = elo_data['elo_away']
    features['elo_diff'] = elo_data['elo_diff']

    # ── 2. FEATURES DE EQUIPO (26 = 13 x 2) ──
    for team, role in [(home_team, "home"), (away_team, "away")]:
        season_count = get_team_season_count(team, year) if year else 2

        if season_count < 2 and standings:
            team_features = get_features_from_neighbors(team, role, year, league_id, standings)
        else:
            team_features = get_team_features_cached(team, role, year, league_id, n=5)

        features[f'{role}_win_rate'] = team_features.get(f'{role}_win_rate_{role}', 0)
        features[f'{role}_goals_for_avg'] = team_features.get(f'{role}_avg_goals_scored', 0)
        features[f'{role}_goals_against_avg'] = team_features.get(f'{role}_avg_goals_conceded_global', 0)
        features[f'{role}_points_avg'] = team_features.get(f'{role}_form_pts', 0) / 5.0

        features[f'{role}_shots_on_target_avg'] = team_features.get(f'{role}_avg_Shots on target', 0)
        features[f'{role}_possession_avg'] = team_features.get(f'{role}_avg_Ball possession', 0)
        features[f'{role}_total_shots_avg'] = team_features.get(f'{role}_avg_Total shots', 0)
        features[f'{role}_gk_saves_avg'] = team_features.get(f'{role}_avg_Goalkeeper saves', 0)
        features[f'{role}_big_chances_avg'] = team_features.get(f'{role}_avg_Big chances', 0)
        features[f'{role}_accurate_passes_avg'] = team_features.get(f'{role}_avg_Accurate passes', 0)
        features[f'{role}_tackles_won_avg'] = team_features.get(f'{role}_avg_Tackles won', 0)
        features[f'{role}_interceptions_avg'] = team_features.get(f'{role}_avg_Interceptions', 0)
        features[f'{role}_blocked_shots_avg'] = team_features.get(f'{role}_avg_Blocked shots', 0)

    # ── 3. FEATURES H2H (5) ──
    h2h = get_h2h_features_cached(home_team, away_team, league_id, year, n=5)
    features['h2h_home_win_rate'] = h2h.get('h2h_home_wins', 0.33)
    features['h2h_home_goals_avg'] = h2h.get('h2h_avg_goals', 0) / 2
    features['h2h_away_goals_avg'] = h2h.get('h2h_avg_goals', 0) / 2
    features['h2h_total_goals_avg'] = h2h.get('h2h_avg_goals', 0)
    features['h2h_used_proxy'] = 1 if len(h2h) < 2 else 0

    # ── 4. FEATURES DE BALANCE (6) ──
    features['elo_diff_abs'] = abs(features['elo_diff'])
    features['form_balance'] = abs(features['home_win_rate'] - features['away_win_rate'])
    features['goals_balance'] = abs(features['home_goals_for_avg'] - features['away_goals_for_avg'])
    features['possession_balance'] = abs(features['home_possession_avg'] - features['away_possession_avg'])
    features['shots_balance'] = abs(features['home_total_shots_avg'] - features['away_total_shots_avg'])
    features['shots_on_target_balance'] = abs(
        features['home_shots_on_target_avg'] - features['away_shots_on_target_avg'])

    # ── 5. ORDEN BASE DE FEATURES (40) ──
    feature_order = [
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

    # ── 6. SOLO PREMIER LEAGUE: 7 FEATURES ADICIONALES ──
    if league_id == '17':
        features['season_progress'] = 0.5
        features['is_early_season'] = 0
        features['is_late_season'] = 0
        features['home_form_momentum'] = features['home_goals_for_avg'] - features['home_win_rate']
        features['away_form_momentum'] = features['away_goals_for_avg'] - features['away_win_rate']

        elo_balanced = 1 if abs(features['elo_diff']) < 50 else 0
        form_balanced = 1 if features['form_balance'] < 0.1 else 0
        features['ultra_balanced'] = elo_balanced * form_balanced

        features['underdog_home_vs_strong_away'] = 1 if (
            features['elo_home'] < 1450 and features['elo_away'] > 1600) else 0

        feature_order.extend([
            'season_progress',
            'is_early_season',
            'is_late_season',
            'home_form_momentum',
            'away_form_momentum',
            'ultra_balanced',
            'underdog_home_vs_strong_away'
        ])

    # ── 7. CONVERTIR A NUMPY ARRAY ──
    feature_vector = np.array([[features.get(col, 0.0) for col in feature_order]], dtype=np.float64)

    return feature_vector


def _blend_tournament_stats(features_40: np.ndarray, home_tourn: dict, away_tourn: dict) -> np.ndarray:
    """Blend historical features with current tournament observed stats.

    Uses a credibility-weighted blend: w = n / (n + TOURN_CREDIBILITY_PSEUDO_COUNT).
    Only blends stats for teams that have current tournament data.
    Recomputes derived balance features after blending.
    """
    f = features_40.copy().flatten()  # work on a flat copy

    # Mapping: (role, stat_key) -> feature index
    stat_index_map = {
        ('home', 'win_rate'):            3,
        ('home', 'goals_for_avg'):       4,
        ('home', 'goals_against_avg'):   5,
        ('home', 'points_avg'):          6,
        ('home', 'shots_on_target_avg'): 7,
        ('home', 'possession_avg'):      8,
        ('home', 'total_shots_avg'):     9,
        ('home', 'gk_saves_avg'):        10,
        ('home', 'big_chances_avg'):     11,
        ('home', 'accurate_passes_avg'): 12,
        ('home', 'tackles_won_avg'):     13,
        ('home', 'interceptions_avg'):   14,
        ('home', 'blocked_shots_avg'):   15,
        ('away', 'win_rate'):            16,
        ('away', 'goals_for_avg'):       17,
        ('away', 'goals_against_avg'):   18,
        ('away', 'points_avg'):          19,
        ('away', 'shots_on_target_avg'): 20,
        ('away', 'possession_avg'):      21,
        ('away', 'total_shots_avg'):     22,
        ('away', 'gk_saves_avg'):        23,
        ('away', 'big_chances_avg'):     24,
        ('away', 'accurate_passes_avg'): 25,
        ('away', 'tackles_won_avg'):     26,
        ('away', 'interceptions_avg'):   27,
        ('away', 'blocked_shots_avg'):   28,
    }

    for (role, stat), idx in stat_index_map.items():
        tourn = home_tourn if role == 'home' else away_tourn
        n = tourn.get(f'n_{role}', 0)
        if n == 0:
            continue
        stat_val = tourn.get(stat)
        if stat_val is None:
            continue
        # Credibility: w = n / (n + pseudo_count)
        w = n / (n + TOURN_CREDIBILITY_PSEUDO_COUNT)
        f[idx] = w * stat_val + (1 - w) * f[idx]

    # Recompute derived balance features after blending
    f[35] = abs(f[3]  - f[16])   # form_balance
    f[36] = abs(f[4]  - f[17])   # goals_balance
    f[37] = abs(f[8]  - f[21])   # possession_balance
    f[38] = abs(f[9]  - f[22])   # shots_balance
    f[39] = abs(f[7]  - f[20])   # shots_on_target_balance

    return f.reshape(1, -1)


def build_features_qualy(home_team: str, away_team: str, year: Optional[str],
                         league_id: str, is_qualifier: int) -> np.ndarray:
    """
    Construye el vector de features (45 dimensiones) para modelos de clasificatorias.

    Las 40 features base son idénticas a build_features_1x2. Se pasa el año real
    en formato internacional ("2026") para que get_recent_years lo detecte y genere
    las temporadas correctas (ej: ["2026", "2022"]) al consultar la BD.

    Feature 41: is_qualifier (1 = partido de clasificatoria, 0 = torneo principal).
    Feature 42: home_xg_avg (Expected Goals promedio del equipo local, últimas 5 apariciones)
    Feature 43: away_xg_avg (Expected Goals promedio del equipo visitante, últimas 5 apariciones)
    Feature 44: home_tournament_points (puntos acumulados en el torneo actual)
    Feature 45: away_tournament_points (puntos acumulados en el torneo actual)

    Blends current tournament observed stats (credibility-weighted) into the feature
    vector so teams with real WC 2026 data are not treated identically to 4-year-old
    historical averages.
    """
    # Construir las 40 features base con el año real (formato internacional "2026")
    # → ELO correcto (caché global incluye partidos internacionales)
    # → Stats de equipo desde BD filtradas por años del ciclo internacional
    # → H2H basado en enfrentamientos históricos entre las selecciones
    features_40 = build_features_1x2(home_team, away_team, year=year, league_id=league_id)

    # Blend current tournament form for international leagues
    if year:
        home_tourn = get_current_tournament_averages(home_team, league_id, year)
        away_tourn = get_current_tournament_averages(away_team, league_id, year)
        if home_tourn or away_tourn:
            features_40 = _blend_tournament_stats(features_40, home_tourn, away_tourn)

    # xG features: average Expected Goals from last 5 appearances in recent cycles
    recent_years = get_recent_years(year) if year else []
    home_xg = get_team_xg_average(home_team, 'home', league_id, recent_years, n=5)
    away_xg = get_team_xg_average(away_team, 'away', league_id, recent_years, n=5)

    # Current tournament accumulated points (0 if no data)
    home_pts = get_team_current_tournament_points(home_team, league_id, year) if year else 0
    away_pts = get_team_current_tournament_points(away_team, league_id, year) if year else 0

    extra = np.array(
        [[float(is_qualifier), home_xg, away_xg, float(home_pts), float(away_pts)]],
        dtype=np.float64
    )
    return np.hstack([features_40, extra])

import pickle
import os
import threading
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from football_core.db import (run_query, TABLE, get_all_matches_chronological, get_league_matches,
                              get_team_season_count as db_get_team_season_count,
                              get_team_recent_matches_goals, get_h2h_matches, get_proxy_h2h_matches,
                              get_team_stat_average, get_team_multiple_stats_average,
                              get_team_win_rates, get_team_goals_conceded_average, get_team_matches_total_goals,
                              get_teams_by_league)
from football_core.feature_engineering import build_features_1x2, build_features_qualy, warm_cache
from tournament_simulation import run_simulation, build_probability_matrix

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
    MIN_PROB,
    FORM_WINDOW,
    H2H_LOOKBACK,
    MIN_H2H_MATCHES,
    LEAGUE_NAMES_NORMALIZED,
    LEAGUES,
    SAVES_THRESHOLDS,
    CORNERS_THRESHOLDS
)


# ══════════════════════════════════════════════════════════════════════════════
# DEFINICIÓN DE VARIABLES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
MODEL_1X2 = os.path.join(MODELS_DIR, "1x2", "production")
MODEL_1X2_LALIGA = os.path.join(MODEL_1X2, "laliga")
MODEL_1X2_PREMIER = os.path.join(MODEL_1X2, "premier_league")
MODEL_1X2_SERIEA = os.path.join(MODEL_1X2, "serie_a")

MODEL_OVER_UNDER_GOALS = os.path.join(MODELS_DIR, "over_under", "goals", "production")
MODEL_OVER_UNDER_GOALS_LALIGA = os.path.join(MODEL_OVER_UNDER_GOALS, "laliga")
MODEL_OVER_UNDER_GOALS_PREMIER = os.path.join(MODEL_OVER_UNDER_GOALS, "premier_league")
MODEL_OVER_UNDER_GOALS_SERIEA = os.path.join(MODEL_OVER_UNDER_GOALS, "serie_a")

# ── Clasificatorias/Torneos internacionales (modelo universal Mundial) ──
QUALY_LEAGUES = {'11', '16', '295', '140'}  # Europe, World Cup, Conmebol, Concacaf
QUALY_LEAGUE_SLUG = 'worldcup_all'
QUALY_FILE_SUFFIX = 'qualy'
QUALY_IS_QUALIFIER = {
    '11': 1,   # Qualy WC Europe — clasificatoria (terreno propio)
    '295': 1,  # Qualy WC Conmebol — clasificatoria (terreno propio)
    '140': 1,  # Qualy WC Concacaf — clasificatoria (terreno propio)
    '16': 0,   # World Cup — torneo en terreno neutral
}
MODEL_1X2_QUALY = os.path.join(MODEL_1X2, QUALY_LEAGUE_SLUG)
MODEL_OVER_UNDER_GOALS_QUALY = os.path.join(MODEL_OVER_UNDER_GOALS, QUALY_LEAGUE_SLUG)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL MANAGER PARA 1X2 (RF + ENSEMBLE)
# ══════════════════════════════════════════════════════════════════════════════

class ModelManager1X2:
    """Gestiona modelos 1X2: RF para LaLiga/Serie A, Ensemble para Premier"""

    def __init__(self):
        self.models = {}
        self.ensemble_config = None

    def load_all_models(self):
        """Carga todos los modelos 1X2 al iniciar"""
        print("🔄 Cargando modelos 1X2...")

        # LaLiga - Random Forest
        laliga_path = os.path.join(MODEL_1X2_LALIGA, "model_1x2_laliga.pkl")
        if os.path.exists(laliga_path):
            with open(laliga_path, 'rb') as f:
                self.models['8'] = {  # league_id 8 = LaLiga
                    'type': 'rf',
                    'model': pickle.load(f)
                }
            print("  ✅ LaLiga: Random Forest cargado")

        # Serie A - Random Forest
        seriea_path = os.path.join(MODEL_1X2_SERIEA, "model_1x2_serie_a.pkl")
        if os.path.exists(seriea_path):
            with open(seriea_path, 'rb') as f:
                self.models['23'] = {  # league_id 23 = Serie A
                    'type': 'rf',
                    'model': pickle.load(f)
                }
            print("  ✅ Serie A: Random Forest cargado")

        # Premier League - Ensemble (RF + XGBoost)
        premier_rf_path = os.path.join(MODEL_1X2_PREMIER, "model_1x2_premier_league.pkl")
        premier_xgb_path = os.path.join(MODEL_1X2_PREMIER, "model_xgboost_premier_league.pkl")
        ensemble_config_path = os.path.join(MODEL_1X2_PREMIER, "ensemble_config_premier_league.json")

        if os.path.exists(premier_rf_path) and os.path.exists(premier_xgb_path):
            with open(premier_rf_path, 'rb') as f:
                rf_model = pickle.load(f)

            with open(premier_xgb_path, 'rb') as f:
                xgb_model = pickle.load(f)

            # Cargar config ensemble (weights)
            import json
            if os.path.exists(ensemble_config_path):
                with open(ensemble_config_path, 'r') as f:
                    self.ensemble_config = json.load(f)
                    rf_weight = self.ensemble_config['rf_weight']
                    xgb_weight = self.ensemble_config['xgb_weight']
            else:
                # Default weights si no existe config
                rf_weight = 0.7
                xgb_weight = 0.3

            self.models['17'] = {  # league_id 17 = Premier
                'type': 'ensemble',
                'rf_model': rf_model,
                'xgb_model': xgb_model,
                'rf_weight': rf_weight,
                'xgb_weight': xgb_weight
            }
            print(f"  ✅ Premier: Ensemble cargado (RF={rf_weight}, XGB={xgb_weight})")

        # Qualy WC Europe (11) + World Cup (16) — modelo compartido
        qualy_path = os.path.join(MODEL_1X2_QUALY, "model_1x2_qualy.pkl")
        if os.path.exists(qualy_path):
            with open(qualy_path, 'rb') as f:
                qualy_model = pickle.load(f)
            for lid in QUALY_LEAGUES:
                self.models[lid] = {'type': 'rf', 'model': qualy_model}
            print(f"  ✅ Qualy WC (Europe + Conmebol + Concacaf): RF cargado ({QUALY_LEAGUE_SLUG})")

        print("✅ Modelos 1X2 cargados exitosamente")

    def predict_1x2(self, league_id: str, features: np.ndarray,
                    elo_diff: float) -> dict:  # ← Cambiar features_40 por features
        """
        Predice 1X2 usando Bayesian Blend dinámico: Elo + Modelo ML

        Args:
            league_id: '8', '17', o '23'
            features: Array numpy con features (40 para LaLiga/Serie A, 47 para Premier)  # ← Actualizar doc
            elo_diff: Diferencia elo_home - elo_away (para calcular blend weight)

        Returns:
            dict con probabilidades 1X2 blended
        """
        if league_id not in self.models:
            # Fallback: si no hay modelo específico, usar baseline neutro
            return {
                '1': 0.40,
                'X': 0.30,
                '2': 0.30
            }

        model_config = self.models[league_id]

        # ── PASO 1: PREDICCIÓN DEL MODELO ML ──
        if model_config['type'] == 'rf':
            # Random Forest simple (LaLiga, Serie A)
            model = model_config['model']
            probas_ml = model.predict_proba(features)[0]  # ← Cambiar features_40 por features

        elif model_config['type'] == 'ensemble':
            # Ensemble RF + XGBoost (Premier)
            rf_model = model_config['rf_model']
            xgb_model = model_config['xgb_model']
            rf_weight = model_config['rf_weight']
            xgb_weight = model_config['xgb_weight']

            # Predicciones de cada modelo
            rf_probas = rf_model.predict_proba(features)[0]  # ← Cambiar features_40 por features
            xgb_probas = xgb_model.predict_proba(features)[0]  # ← Cambiar features_40 por features

            # Weighted average
            probas_ml = (rf_weight * rf_probas) + (xgb_weight * xgb_probas)

        # ── PASO 2: PROBABILIDADES BASADAS EN ELO PURO ──
        # Torneos en terreno neutral (Mundial, Eurocopa): sin ventaja local
        _NEUTRAL_LEAGUES = {"16", "1"}
        _home_adv = 0 if league_id in _NEUTRAL_LEAGUES else 100
        probas_elo = elo_to_probabilities(elo_diff, home_advantage=_home_adv)  # Shape: (3,)

        # ── PASO 3: CALCULAR PESOS DINÁMICOS ──
        # Cuanto mayor |elo_diff|, más confiamos en Elo
        abs_elo_diff = abs(elo_diff)
        # print(f"[DEBUG BLEND] elo_diff={elo_diff:.1f}, abs={abs_elo_diff:.1f}")

        # Ajuste asimétrico: más peso a Elo cuando local es underdog
        if elo_diff < -150:  # Local es underdog claro (ej: Oviedo)
            # print(f"[DEBUG BLEND] → Caso: Local underdog claro")
            # Penalizar más fuerte: modelo ML tiende a sobreestimar locales débiles
            weight_elo = min(0.95, 0.50 + (abs_elo_diff / 250.0))
        elif elo_diff > 200:  # Local es favorito claro (ej: Barcelona, Madrid)
            # print(f"[DEBUG BLEND] → Caso: Local favorito claro")
            # Aumentar peso Elo: modelo ML subestima favoritos
            weight_elo = min(0.90, 0.30 + (abs_elo_diff / 300.0))
        else:  # Partido equilibrado (-150 < diff < 200)
            # SUB-CASO: partidos MUY equilibrados → más peso al ML (mejor para empates)
            if abs(elo_diff) < 80:  # Diferencia < 80 puntos Elo
                # print(f"[DEBUG BLEND] → Caso: Partido MUY equilibrado (<80)")
                weight_elo = 0.30  # Invertir: 30% Elo / 70% ML
            else:  # Equilibrado normal (80-150 puntos)
                # Balancear modelo ML y Elo
                # print(f"[DEBUG BLEND] → Caso: Partido equilibrado (80-150)")
                weight_elo = min(0.75, 0.35 + (abs_elo_diff / 400.0))

        # print(f"[DEBUG BLEND] → weight_elo={weight_elo:.2f}, weight_ml={1 - weight_elo:.2f}")
        weight_ml = 1.0 - weight_elo

        # ── PASO 4: BLEND BAYESIANO ──
        # print(f"[DEBUG BLEND] probas_ML:  1={probas_ml[0] * 100:.1f}% X={probas_ml[1] * 100:.1f}% 2={probas_ml[2] * 100:.1f}%")
        # print(f"[DEBUG BLEND] probas_Elo: 1={probas_elo[0] * 100:.1f}% X={probas_elo[1] * 100:.1f}% 2={probas_elo[2] * 100:.1f}%")

        probas_blended = (weight_ml * probas_ml) + (weight_elo * probas_elo)
        # print(f"[DEBUG BLEND] probas_BLEND: 1={probas_blended[0] * 100:.1f}% X={probas_blended[1] * 100:.1f}% 2={probas_blended[2] * 100:.1f}%")

        # ── PASO 5: APLICAR MIN_PROB FLOOR ──
        probas_blended = np.maximum(probas_blended, MIN_PROB)

        # ── PASO 6: RE-NORMALIZAR ──
        probas_blended = probas_blended / probas_blended.sum()

        # ── PASO 7: RETORNAR COMO DICT ──
        # Asumiendo orden: [0=1, 1=X, 2=2]
        return {
            '1': float(probas_blended[0]),
            'X': float(probas_blended[1]),
            '2': float(probas_blended[2]),
            '_debug': {  # Info de diagnóstico (opcional, quitar en prod)
                'weight_elo': round(weight_elo, 2),
                'weight_ml': round(weight_ml, 2),
                'elo_diff': round(elo_diff, 1),
                'probas_ml': probas_ml.tolist(),
                'probas_elo': probas_elo.tolist()
            }
        }


class ModelManagerOverUnderGoals:
    """Gestiona la carga de modelos Over/Under (0.5, 1.5, 2.5, 3.5)"""

    def __init__(self):
        self.models = {}
        self.thresholds = [0.5, 1.5, 2.5, 3.5]
        self.league_models = {}  # {league_id: {threshold: model}}

    def load_models(self, league_id: str = "8"):
        """Carga los 4 modelos Over/Under para una liga"""
        if league_id in self.league_models:
            return  # Ya cargados

        if league_id in QUALY_LEAGUES:
            league_dir = MODEL_OVER_UNDER_GOALS_QUALY
            file_suffix = QUALY_FILE_SUFFIX
        else:
            file_suffix = {
                "8": "laliga", "17": "premier_league", "23": "serie_a"
            }.get(league_id, "laliga")
            league_dir = {
                "8": MODEL_OVER_UNDER_GOALS_LALIGA,
                "17": MODEL_OVER_UNDER_GOALS_PREMIER,
                "23": MODEL_OVER_UNDER_GOALS_SERIEA
            }.get(league_id, MODEL_OVER_UNDER_GOALS_LALIGA)

        self.league_models[league_id] = {}

        for threshold in self.thresholds:
            threshold_str = str(threshold).replace(".", "_")
            model_path = os.path.join(league_dir, f"model_over_{threshold_str}_{file_suffix}.pkl")

            if not os.path.exists(model_path):
                print(f"⚠️  Modelo Over/Under {threshold} no encontrado para {file_suffix}")
                continue

            try:
                with open(model_path, 'rb') as f:
                    self.league_models[league_id][threshold] = pickle.load(f)
                print(f"  ✅ Over {threshold}: {file_suffix}")
            except Exception as e:
                print(f"  ❌ Error cargando Over {threshold}: {e}")

    def get_model(self, league_id: str, threshold: float):
        """Obtiene un modelo específico"""
        if league_id not in self.league_models:
            self.load_models(league_id)
        return self.league_models.get(league_id, {}).get(threshold)

    def load_all_models(self):
        """Precarga modelos de todas las ligas disponibles"""
        print("🔄 Cargando modelos Over/Under...")
        for league_id in ["8", "17", "23"]:
            print(f"  Liga {league_id}:")
            self.load_models(league_id)
        # Qualy — un único modelo compartido por 11 y 16
        print(f"  Qualy ({QUALY_LEAGUE_SLUG}):")
        self.load_models("11")
        # Reusar el mismo dict para el league_id 16
        if "11" in self.league_models:
            self.league_models["16"] = self.league_models["11"]
        print("✅ Modelos Over/Under cargados")

    def predict_over_under(self, home_team: str, away_team: str, year: str, league_id: str,
                           prebuilt_features: np.ndarray = None) -> dict:
        """
        Predice probabilidades Over/Under para umbrales 0.5, 1.5, 2.5, 3.5
        con restricción monotónica: P(>0.5) >= P(>1.5) >= P(>2.5) >= P(>3.5)

        prebuilt_features: si se pasa, se usa directamente (útil para qualy leagues
        donde el vector ya tiene la dimensión correcta incluyendo is_qualifier).
        """
        if prebuilt_features is not None:
            features = prebuilt_features  # Dimensión correcta para el modelo
        else:
            features = build_features_1x2(home_team, away_team, year, league_id)
            if features.shape[1] > 40:
                features = features[:, :40]

        if features.ndim == 3:
            features = features.reshape(1, -1)
        elif features.ndim == 1:
            features = features.reshape(1, -1)

        # Obtener probabilidades de cada modelo
        probs_over = []

        for threshold in self.thresholds:
            model = model_manager_ou_goals.get_model(league_id, threshold)

            if model is None:
                # Si no hay modelo, usar heurística simple
                probs_over.append(0.5)
                continue

            prob = model.predict_proba(features)[0, 1]  # Probabilidad de "Over"
            probs_over.append(prob)

        # Aplicar restricción monotónica: P(>0.5) >= P(>1.5) >= P(>2.5) >= P(>3.5)
        probs_over = np.array(probs_over)
        probs_over_monotonic = np.maximum.accumulate(probs_over[::-1])[::-1]

        # Construir respuesta
        result = {}
        for i, threshold in enumerate(self.thresholds):
            threshold_str = str(threshold).replace(".", "_")
            prob_over = float(probs_over_monotonic[i])
            prob_under = 1.0 - prob_over

            result[f"over_{threshold_str}"] = {
                "over": round(prob_over * 100, 1),
                "under": round(prob_under * 100, 1)
            }

        return result


class ModelManagerSaves:
    """
    Gestor de modelos Over/Under para SAVES (paradas de portero).
    Arquitectura: 9 modelos RF por liga con feature is_home.
    """

    def __init__(self, league_id: str = "8", dir_name: str = None, file_suffix: str = None):
        """
        Inicializa el gestor de modelos de saves para una liga.

        Args:
            league_id: ID de la liga ('8', '17', '23', '11', '16', ...)
            dir_name: Nombre del directorio bajo production/ (por defecto se infiere de league_id)
            file_suffix: Sufijo en el nombre del fichero pkl (por defecto igual a dir_name)
        """
        self.league_id = league_id
        self.league_name = dir_name or LEAGUE_NAMES_NORMALIZED.get(LEAGUES.get(league_id, ''), 'unknown')
        self.file_suffix = file_suffix or self.league_name

        # Thresholds disponibles
        self.thresholds = SAVES_THRESHOLDS

        # Directorio de modelos
        self.models_dir = os.path.join(
            MODELS_DIR,
            "over_under",
            "saves",
            "production",
            self.league_name
        )

        # Cache de modelos cargados
        self.models = {}

        print(f"📊 ModelManagerSaves inicializado para {LEAGUES.get(league_id, self.league_name)}")
        self._load_models()

    def _load_models(self):
        """Carga todos los modelos de saves de la liga."""
        for threshold in self.thresholds:
            threshold_str = str(threshold).replace('.', '_')
            model_filename = f"model_over_{threshold_str}_saves_{self.file_suffix}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)

            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[threshold] = pickle.load(f)
                    print(f"  ✅ Modelo Over {threshold} saves cargado")
                except Exception as e:
                    print(f"  ❌ Error cargando modelo Over {threshold} saves: {str(e)}")
            else:
                print(f"  ⚠️ Modelo Over {threshold} saves no encontrado: {model_path}")

    def predict(self, features: np.ndarray, is_home: int) -> dict:
        """
        Predice probabilidades Over/Under saves para un equipo.

        Args:
            features: Array de features (40 o 47 según liga)
            is_home: 1 si es portero local, 0 si visitante

        Returns:
            dict: {
                'over_0_5': 0.95,
                'over_1_5': 0.78,
                ...
                'over_8_5': 0.02
            }
        """
        if not self.models:
            print("❌ No hay modelos cargados para saves")
            return {}

        # Añadir feature is_home a las features base
        features_with_context = np.append(features.flatten(), [is_home]).reshape(1, -1)

        probabilities = {}

        for threshold in self.thresholds:
            if threshold not in self.models:
                continue

            try:
                model = self.models[threshold]
                # Probabilidad de over (clase 1)
                prob_over = model.predict_proba(features_with_context)[0, 1]

                # Aplicar MIN_PROB floor
                prob_over = max(prob_over, MIN_PROB)

                # Guardar
                threshold_key = f"over_{str(threshold).replace('.', '_')}"
                probabilities[threshold_key] = float(prob_over)

            except Exception as e:
                print(f"❌ Error prediciendo saves threshold {threshold}: {str(e)}")
                continue

        # Enforce monotonicity: P(>0.5) >= P(>1.5) >= ... >= P(>8.5)
        probabilities = self._enforce_monotonicity(probabilities)

        return probabilities

    def _enforce_monotonicity(self, probs: dict) -> dict:
        """
        Asegura que P(>0.5) >= P(>1.5) >= ... >= P(>8.5).

        Args:
            probs: Dict con probabilidades originales

        Returns:
            dict: Probabilidades corregidas
        """
        # Ordenar thresholds de menor a mayor
        sorted_thresholds = sorted(self.thresholds)

        # Extraer valores en orden
        values = []
        for t in sorted_thresholds:
            key = f"over_{str(t).replace('.', '_')}"
            if key in probs:
                values.append(probs[key])

        # Aplicar maximum accumulate (backwards)
        # P(>8.5) queda igual, P(>7.5) = max(P(>7.5), P(>8.5)), etc.
        corrected = np.maximum.accumulate(values[::-1])[::-1]

        # Reconstruir dict
        corrected_probs = {}
        for i, t in enumerate(sorted_thresholds):
            key = f"over_{str(t).replace('.', '_')}"
            if key in probs:
                corrected_probs[key] = float(corrected[i])

        return corrected_probs

    def predict_match(self, features: np.ndarray) -> dict:
        """
        Predice saves para ambos porteros (home y away).

        Args:
            features: Features base del partido (40 o 47)

        Returns:
            dict: {
                'home': {'over_0_5': 0.95, 'over_1_5': 0.78, ...},
                'away': {'over_0_5': 0.97, 'over_1_5': 0.82, ...}
            }
        """
        return {
            'home': self.predict(features, is_home=1),
            'away': self.predict(features, is_home=0)
        }


class ModelManagerCorners:
    """
    Gestor de modelos Over/Under para CORNERS (córners).
    Arquitectura: 10 modelos RF por liga con feature is_home.
    """

    def __init__(self, league_id: str = "8", dir_name: str = None, file_suffix: str = None):
        """
        Inicializa el gestor de modelos de corners para una liga.

        Args:
            league_id: ID de la liga ('8', '17', '23', '11', '16', ...)
            dir_name: Nombre del directorio bajo production/ (por defecto se infiere de league_id)
            file_suffix: Sufijo en el nombre del fichero pkl (por defecto igual a dir_name)
        """
        self.league_id = league_id
        self.league_name = dir_name or LEAGUE_NAMES_NORMALIZED.get(LEAGUES.get(league_id, ''), 'unknown')
        self.file_suffix = file_suffix or self.league_name

        # Thresholds disponibles
        self.thresholds = CORNERS_THRESHOLDS

        # Directorio de modelos
        self.models_dir = os.path.join(
            MODELS_DIR,
            "over_under",
            "corners",
            "production",
            self.league_name
        )

        # Cache de modelos cargados
        self.models = {}

        print(f"⚽ ModelManagerCorners inicializado para {LEAGUES.get(league_id, self.league_name)}")
        self._load_models()

    def _load_models(self):
        """Carga todos los modelos de corners de la liga."""
        for threshold in self.thresholds:
            threshold_str = str(threshold).replace('.', '_')
            model_filename = f"model_over_{threshold_str}_corners_{self.file_suffix}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)

            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[threshold] = pickle.load(f)
                    print(f"  ✅ Modelo Over {threshold} corners cargado")
                except Exception as e:
                    print(f"  ❌ Error cargando modelo Over {threshold} corners: {str(e)}")
            else:
                print(f"  ⚠️ Modelo Over {threshold} corners no encontrado: {model_path}")

    def predict(self, features: np.ndarray, is_home: int) -> dict:
        """
        Predice probabilidades Over/Under corners para un equipo.

        Args:
            features: Array de features (40 o 47 según liga)
            is_home: 1 si es equipo local, 0 si visitante

        Returns:
            dict: {
                'over_5_5': 0.42,
                'over_6_5': 0.31,
                ...
                'over_14_5': 0.01
            }
        """
        if not self.models:
            print("❌ No hay modelos cargados para corners")
            return {}

        # Añadir feature is_home a las features base
        features_with_context = np.append(features.flatten(), [is_home]).reshape(1, -1)

        probabilities = {}

        for threshold in self.thresholds:
            if threshold not in self.models:
                continue

            try:
                model = self.models[threshold]
                # Probabilidad de over (clase 1)
                prob_over = model.predict_proba(features_with_context)[0, 1]

                # Aplicar MIN_PROB floor
                prob_over = max(prob_over, MIN_PROB)

                # Guardar
                threshold_key = f"over_{str(threshold).replace('.', '_')}"
                probabilities[threshold_key] = float(prob_over)

            except Exception as e:
                print(f"❌ Error prediciendo corners threshold {threshold}: {str(e)}")
                continue

        # Enforce monotonicity: P(>5.5) >= P(>6.5) >= ... >= P(>14.5)
        probabilities = self._enforce_monotonicity(probabilities)

        return probabilities

    def _enforce_monotonicity(self, probs: dict) -> dict:
        """
        Asegura que P(>5.5) >= P(>6.5) >= ... >= P(>14.5).

        Args:
            probs: Dict con probabilidades originales

        Returns:
            dict: Probabilidades corregidas
        """
        # Ordenar thresholds de menor a mayor
        sorted_thresholds = sorted(self.thresholds)

        # Extraer valores en orden
        values = []
        for t in sorted_thresholds:
            key = f"over_{str(t).replace('.', '_')}"
            if key in probs:
                values.append(probs[key])

        # Aplicar maximum accumulate (backwards)
        corrected = np.maximum.accumulate(values[::-1])[::-1]

        # Reconstruir dict
        corrected_probs = {}
        for i, t in enumerate(sorted_thresholds):
            key = f"over_{str(t).replace('.', '_')}"
            if key in probs:
                corrected_probs[key] = float(corrected[i])

        return corrected_probs

    def predict_match(self, features: np.ndarray) -> dict:
        """
        Predice corners para ambos equipos (home y away).

        Args:
            features: Features base del partido (40 o 47)

        Returns:
            dict: {
                'home': {'over_5_5': 0.42, 'over_6_5': 0.31, ...},
                'away': {'over_5_5': 0.38, 'over_6_5': 0.27, ...}
            }
        """
        return {
            'home': self.predict(features, is_home=1),
            'away': self.predict(features, is_home=0)
        }


# Instancia global
model_manager_1x2 = ModelManager1X2()
model_manager_ou_goals = ModelManagerOverUnderGoals()
model_manager_saves = {}
model_manager_corners = {}

# ══════════════════════════════════════════════════════════════════════════════
# BAYESIAN BLEND: ELO → PROBABILIDADES 1X2
# ══════════════════════════════════════════════════════════════════════════════

def elo_to_probabilities(elo_diff: float, home_advantage: float = 100) -> np.ndarray:
    """
    Convierte diferencia Elo en probabilidades 1X2 usando logística + ajuste gaussiano.
    VERSIÓN CALIBRADA PARA FÚTBOL (LaLiga específicamente)

    Args:
        elo_diff: Diferencia elo_home - elo_away
        home_advantage: Ventaja de jugar en casa (default: 100)

    Returns:
        np.ndarray [prob_1, prob_X, prob_2] normalizado a suma=1
    """
    # ── AJUSTE 1: Home advantage adaptativo (MÁS AGRESIVO) ──
    # Reducir ventaja local cuando hay mucha diferencia de nivel
    abs_diff = abs(elo_diff)

    if elo_diff < -250:  # Local es underdog extremo (ej: Oviedo)
        effective_home_adv = home_advantage * 0.3
    elif elo_diff < -150:  # Local es underdog claro
        effective_home_adv = home_advantage * 0.5
    elif elo_diff < -50:  # Local es underdog moderado ← NUEVO
        effective_home_adv = home_advantage * 0.35 # 35 puntos
    elif elo_diff > 300:  # Local es favorito extremo (ej: Madrid)
        effective_home_adv = home_advantage * 0.6
    elif elo_diff > 150:  # Local es favorito claro (ej: Barça)
        effective_home_adv = home_advantage * 0.75
    elif elo_diff > 50:  # Local es favorito moderado ← NUEVO
        effective_home_adv = home_advantage * 0.85  # 85 puntos
    else:  # Partido equilibrado (-50 < diff < 50) ← AJUSTADO
        effective_home_adv = home_advantage  # 100 puntos

    adjusted_diff = elo_diff + effective_home_adv

    # ── PASO 2: Probabilidad home win (logística) ──
    exp_home = 1.0 / (1.0 + 10.0 ** (-adjusted_diff / ELO_SCALE))

    # ── PASO 3: Probabilidad empate (gaussiana + floor AUMENTADO) ──
    # Floor del 15% en LaLiga (liga con muchos empates)
    prob_draw_gaussian = 0.30 * np.exp(-(adjusted_diff ** 2) / (2 * 200 ** 2))
    prob_draw = max(0.15, prob_draw_gaussian)  # ← Floor aumentado a 15%

    # ── PASO 4: Probabilidad away win (residual) ──
    prob_away = 1.0 - exp_home - prob_draw

    # ── PASO 5: Floors mínimos (evitar 0% exacto) ──
    prob_away = max(0.04, prob_away)
    exp_home = max(0.04, exp_home)

    # ── PASO 6: Normalizar a suma=1 ──
    total = exp_home + prob_draw + prob_away

    return np.array([exp_home, prob_draw, prob_away]) / total

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precarga Elos y modelos al arrancar para que la primera request sea rápida."""
    print("🚀 Precargando caché de Elo...")
    try:
        warm_cache()
    except Exception as e:
        print(f"⚠️  Error precargando Elo: {e}")

    # ── NUEVOS MODELOS 1X2 ──
    print("🚀 Precargando modelos 1X2 (RF + Ensemble)...")
    try:
        model_manager_1x2.load_all_models()
    except Exception as e:
        print(f"⚠️  Error cargando modelos 1X2: {e}")

    # ── MODELOS OVER/UNDER ──
    print("🚀 Precargando modelos Over/Under...")
    try:
        model_manager_ou_goals.load_all_models()
    except Exception as e:
        print(f"⚠️  Error cargando modelos Over/Under: {e}")

    # ── AÑADIR ESTO: PRECARGAR SAVES Y CORNERS ──
    print("🚀 Precargando modelos Saves...")
    try:
        for league_id in ["8", "17", "23"]:
            model_manager_saves[league_id] = ModelManagerSaves(league_id=league_id)
        # Qualy — modelo compartido entre todos los leagues internacionales del Mundial
        qualy_saves = ModelManagerSaves(
            league_id="11", dir_name=QUALY_LEAGUE_SLUG, file_suffix=QUALY_FILE_SUFFIX
        )
        for lid in QUALY_LEAGUES:
            model_manager_saves[lid] = qualy_saves
        print("✅ Modelos Saves precargados para todas las ligas")
    except Exception as e:
        print(f"⚠️  Error cargando modelos Saves: {e}")

    print("🚀 Precargando modelos Corners...")
    try:
        for league_id in ["8", "17", "23"]:
            model_manager_corners[league_id] = ModelManagerCorners(league_id=league_id)
        # Qualy — modelo compartido entre todos los leagues internacionales del Mundial
        qualy_corners = ModelManagerCorners(
            league_id="11", dir_name=QUALY_LEAGUE_SLUG, file_suffix=QUALY_FILE_SUFFIX
        )
        for lid in QUALY_LEAGUES:
            model_manager_corners[lid] = qualy_corners
        print("✅ Modelos Corners precargados para todas las ligas")
    except Exception as e:
        print(f"⚠️  Error cargando modelos Corners: {e}")

    # ── PRE-CALENTAR MATRIZ DE PROBABILIDADES PARA SIMULACIÓN ──
    print("🚀 Pre-calentando matriz de probabilidades para simulación de torneo...")
    try:
        wc_model_entry = model_manager_1x2.models.get("16")
        if wc_model_entry:
            wc_teams: set = set()
            for lid in QUALY_LEAGUES:
                wc_teams.update(get_teams_by_league(league_id=lid))
            wc_teams_list = sorted(wc_teams)
            build_probability_matrix(wc_teams_list, wc_model_entry["model"], year=None)
            print(f"  ✅ Matriz precalculada para {len(wc_teams_list)} selecciones")
        else:
            print("  ⚠️  Modelo worldcup_all no disponible, saltando precalentamiento")
    except Exception as e:
        print(f"⚠️  Error precalentando matriz de simulación: {e}")

    print("✅ Precarga completada. API lista.")
    yield


app = FastAPI(title="Football Prediction API", version="1.0", lifespan=lifespan)

LEAGUE_MODEL_MAP = {
    "8": "laliga",
    "17": "premier",
    "23": "seriea"
}

# ─────────────────────────────────────────────
# CACHÉ GLOBAL DE MODELOS (en memoria)
# ─────────────────────────────────────────────

_models_cache: dict = {}
_models_mtime: dict = {}
_models_lock = threading.Lock()
_saves_managers: dict = {}
_corners_managers: dict = {}


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
# ODDS
# ─────────────────────────────────────────────

def calculate_odds(probabilities: dict, margin: float = 0.05) -> dict:
    """Convierte probabilidades (0-100) a cuotas decimales con margen de casa."""
    odds = {}
    for outcome, prob in probabilities.items():
        if prob > 0:
            fair_odd = 100.0 / prob
            odds[outcome] = round(fair_odd * (1 - margin), 2)
        else:
            odds[outcome] = 0.0
    return odds


# ─────────────────────────────────────────────
# CONFIDENCE
# ─────────────────────────────────────────────

def get_match_confidence(probabilities: dict, elo_diff: float, league_id: str) -> dict:
    """
    Calcula confianza basándose en el GAP entre probabilidades + ajuste por liga.
    Args:
        probabilities: dict con keys "1", "X", "2"
        elo_diff: Diferencia Elo (solo informativo)
        league_id: ID de la liga para ajustar umbrales
    """
    probs = [probabilities["1"], probabilities["X"], probabilities["2"]]
    sorted_probs = sorted(probs, reverse=True)
    max_prob = sorted_probs[0]
    second_prob = sorted_probs[1]
    gap = max_prob - second_prob

    # Umbrales de GAP por liga (LaLiga más empates, Premier más definido)
    gap_thresholds = {
        "8": {"very_high": 35, "high": 18, "medium": 7},   # LaLiga (más empates)
        "17": {"very_high": 31, "high": 15, "medium": 5},  # Premier (más definido)
        "23": {"very_high": 33, "high": 17, "medium": 6},  # Serie A (intermedio)
        "11": {"very_high": 30, "high": 15, "medium": 6},  # Qualy WC Europe
        "16": {"very_high": 28, "high": 14, "medium": 5},  # World Cup (más definido, fase final)
    }
    thresholds = gap_thresholds.get(league_id, gap_thresholds["8"])

    # Confidence score basado en Elo
    confidence_score = min(round((abs(elo_diff) / 200) * 100, 1), 100.0)

    # Clasificación basada en GAP de probabilidades
    if gap >= thresholds["very_high"]:
        level = "very_high"
        description = "Favorito muy claro"
    elif gap >= thresholds["high"]:
        level = "high"
        description = "Favorito claro"
    elif gap >= thresholds["medium"]:
        level = "medium"
        description = "Ligero favorito"
    else:
        level = "low"
        description = "Partido equilibrado"

    return {
        "level": level,
        "score": confidence_score,
        "elo_diff": round(elo_diff, 1),
        "gap": round(gap, 2),
        "description": description,
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
    try:
        if req.league_id in QUALY_LEAGUES:
            # Clasificatorias/torneos internacionales: 41 features (40 base + is_qualifier)
            is_qualifier = QUALY_IS_QUALIFIER.get(req.league_id, 0)
            X_1x2 = build_features_qualy(
                req.home_team, req.away_team, req.year, req.league_id, is_qualifier
            )
            # Saves/corners también usan las 41 features (modelo entrenado con is_qualifier)
            X_for_saves_corners = X_1x2
        else:
            # Ligas domésticas: 40 features (47 para Premier, truncado a 40 para saves/corners)
            X_1x2 = build_features_1x2(req.home_team, req.away_team, req.year, req.league_id)
            X_for_saves_corners = X_1x2[:, :40]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error construyendo features: {e}")

    # Alias histórico: X_base_40 sigue usándose en la sección de saves/corners
    X_base_40 = X_for_saves_corners

    # ══════════════════════════════════════════════════════════════════════════
    # PREDICCIÓN 1X2 CON NUEVOS MODELOS (RF o Ensemble según liga)
    # ══════════════════════════════════════════════════════════════════════════

    # Extraer elo_diff de X_1x2 ANTES de llamar predict_1x2
    elo_diff = float(X_1x2[0, 2])  # Posición 2 = elo_diff en feature_order

    # Llamar con blend dinámico
    result_proba_new = model_manager_1x2.predict_1x2(league_id=req.league_id, features=X_1x2, elo_diff=elo_diff)

    # Convertir a porcentajes
    result_proba = {
        k: round(v * 100, 4) for k, v in result_proba_new.items()
        if k != '_debug'  # ← FILTRAR _debug
    }

    # Predicción final (mayor probabilidad)
    predicted = max(result_proba, key=result_proba.get)

    # ══════════════════════════════════════════════════════════════════════════
    # ELO Y CONFIDENCE (tu código existente)
    # ══════════════════════════════════════════════════════════════════════════

    confidence = get_match_confidence(result_proba, elo_diff, req.league_id)
    odds = calculate_odds(result_proba, margin=0.07)

    # ══════════════════════════════════════════════════════════════════════════
    # OVER/UNDER GOALS
    # ══════════════════════════════════════════════════════════════════════════
    # Para ligas qualy pasamos X_1x2 ya construido (41 features) para evitar
    # rebuild y para que el modelo reciba la dimensión correcta.
    result_proba_ou = model_manager_ou_goals.predict_over_under(
        home_team=req.home_team, away_team=req.away_team,
        year=req.year, league_id=req.league_id,
        prebuilt_features=X_1x2 if req.league_id in QUALY_LEAGUES else None
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SAVES (PARADAS DE PORTERO)
    # ══════════════════════════════════════════════════════════════════════════
    try:
        saves_manager = model_manager_saves.get(req.league_id)
        if saves_manager:
            saves_predictions = saves_manager.predict_match(X_base_40)
        else:
            print(f"⚠️ No hay modelo de saves para liga {req.league_id}")
            saves_predictions = {"home": {}, "away": {}}
    except Exception as e:
        print(f"⚠️ Error prediciendo saves: {e}")
        saves_predictions = {"home": {}, "away": {}}

    # ══════════════════════════════════════════════════════════════════════════
    # CORNERS (CÓRNERS)
    # ══════════════════════════════════════════════════════════════════════════
    try:
        corners_manager = model_manager_corners.get(req.league_id)
        if corners_manager:
            corners_predictions = corners_manager.predict_match(X_base_40)
        else:
            print(f"⚠️ No hay modelo de corners para liga {req.league_id}")
            corners_predictions = {"home": {}, "away": {}}
    except Exception as e:
        print(f"⚠️ Error prediciendo corners: {e}")
        corners_predictions = {"home": {}, "away": {}}

    # ══════════════════════════════════════════════════════════════════════════
    # RESPONSE (añadir info de modelo usado)
    # ══════════════════════════════════════════════════════════════════════════

    model_type = model_manager_1x2.models.get(req.league_id, {}).get('type', 'unknown')

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "year": req.year,
        "league_id": req.league_id,
        "model_1x2": model_type,  # ← INFO: 'rf' o 'ensemble'
        "resultado": {
            "predicted": predicted,
            "probabilities": result_proba,
            "odds": odds,
            "confidence": confidence,
            "blend_info": result_proba_new.get('_debug', {}),
        },
        "over_under_goals": result_proba_ou,
        "saves": saves_predictions,
        "corners": corners_predictions
    }


# ─────────────────────────────────────────────
# SIMULATION ENDPOINT
# ─────────────────────────────────────────────

class SimulationRequest(BaseModel):
    teams: list[str]
    n_simulations: int = 10_000
    groups: Optional[dict] = None   # {group_name: [t1, t2, t3, t4]} — None = random draw


@app.post("/simulate")
def simulate(req: SimulationRequest):
    """
    Monte Carlo WC tournament simulation.
    Returns win probabilities for each team (top-10 sorted by probability DESC).
    Uses the worldcup_all qualy model (league_id='16', neutral ground).
    """
    model_entry = model_manager_1x2.models.get("16")
    if model_entry is None:
        raise HTTPException(status_code=503, detail="Modelo worldcup_all no disponible")

    raw_model = model_entry["model"]

    try:
        probs = run_simulation(
            teams=req.teams,
            model=raw_model,
            n_simulations=req.n_simulations,
            groups=req.groups,
            year=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en simulación: {e}")

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return {
        "n_simulations": req.n_simulations,
        "results": [{"team": t, "win_probability": round(p * 100, 2)} for t, p in sorted_probs],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
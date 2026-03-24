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
from football_agent.db import (run_query, TABLE, get_all_matches_chronological, get_league_matches,
                               get_team_season_count as db_get_team_season_count,
                               get_team_recent_matches_goals, get_h2h_matches, get_proxy_h2h_matches,
                               get_team_stat_average, get_team_multiple_stats_average,
                               get_team_win_rates, get_team_goals_conceded_average, get_team_matches_total_goals)

from constants import (
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
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_1X2 = os.path.join(MODELS_DIR, "1x2", "production")
MODEL_1X2_LALIGA = os.path.join(MODEL_1X2, "laliga")
MODEL_1X2_PREMIER = os.path.join(MODEL_1X2, "premier_league")
MODEL_1X2_SERIEA = os.path.join(MODEL_1X2, "serie_a")

MODEL_OVER_UNDER_GOALS = os.path.join(MODELS_DIR, "over_under", "goals", "production")
MODEL_OVER_UNDER_GOALS_LALIGA = os.path.join(MODEL_OVER_UNDER_GOALS, "laliga")
MODEL_OVER_UNDER_GOALS_PREMIER = os.path.join(MODEL_OVER_UNDER_GOALS, "premier_league")
MODEL_OVER_UNDER_GOALS_SERIEA = os.path.join(MODEL_OVER_UNDER_GOALS, "serie_a")

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
        probas_elo = elo_to_probabilities(elo_diff)  # Shape: (3,)

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

        league_name = {
            "8": "laliga",
            "17": "premier_league",
            "23": "serie_a"
        }.get(league_id, "laliga")

        league_dir_map = {
            "8": MODEL_OVER_UNDER_GOALS_LALIGA,
            "17": MODEL_OVER_UNDER_GOALS_PREMIER,
            "23": MODEL_OVER_UNDER_GOALS_SERIEA
        }

        league_dir = league_dir_map.get(league_id, MODEL_OVER_UNDER_GOALS_LALIGA)

        self.league_models[league_id] = {}

        for threshold in self.thresholds:
            threshold_str = str(threshold).replace(".", "_")
            model_path = os.path.join(league_dir, f"model_over_{threshold_str}_{league_name}.pkl")

            if not os.path.exists(model_path):
                print(f"⚠️  Modelo Over/Under {threshold} no encontrado para {league_name}")
                continue

            try:
                with open(model_path, 'rb') as f:
                    self.league_models[league_id][threshold] = pickle.load(f)
                print(f"  ✅ Over {threshold}: {league_name}")
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
        print("✅ Modelos Over/Under cargados")

    def predict_over_under(self, home_team: str, away_team: str, year: str, league_id: str) -> dict:
        """
        Predice probabilidades Over/Under para umbrales 0.5, 1.5, 2.5, 3.5
        con restricción monotónica: P(>0.5) >= P(>1.5) >= P(>2.5) >= P(>3.5)
        """
        # Reutilizar features de 1X2
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

    def __init__(self, league_id: str = "8"):
        """
        Inicializa el gestor de modelos de saves para una liga.

        Args:
            league_id: ID de la liga ('8', '17', '23')
        """
        self.league_id = league_id
        self.league_name = LEAGUE_NAMES_NORMALIZED.get(LEAGUES.get(league_id, ''), 'unknown')

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

        print(f"📊 ModelManagerSaves inicializado para {LEAGUES.get(league_id, 'unknown')}")
        self._load_models()

    def _load_models(self):
        """Carga todos los modelos de saves de la liga."""
        for threshold in self.thresholds:
            threshold_str = str(threshold).replace('.', '_')
            model_filename = f"model_over_{threshold_str}_saves_{self.league_name}.pkl"
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

    def __init__(self, league_id: str = "8"):
        """
        Inicializa el gestor de modelos de corners para una liga.

        Args:
            league_id: ID de la liga ('8', '17', '23')
        """
        self.league_id = league_id
        self.league_name = LEAGUE_NAMES_NORMALIZED.get(LEAGUES.get(league_id, ''), 'unknown')

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

        print(f"⚽ ModelManagerCorners inicializado para {LEAGUES.get(league_id, 'unknown')}")
        self._load_models()

    def _load_models(self):
        """Carga todos los modelos de corners de la liga."""
        for threshold in self.thresholds:
            threshold_str = str(threshold).replace('.', '_')
            model_filename = f"model_over_{threshold_str}_corners_{self.league_name}.pkl"
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
        global _elo_cache, _elo_cache_time
        _elo_cache = _build_elo_cache()
        _elo_cache_time = time.time()
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
        print("✅ Modelos Saves precargados para todas las ligas")
    except Exception as e:
        print(f"⚠️  Error cargando modelos Saves: {e}")

    print("🚀 Precargando modelos Corners...")
    try:
        for league_id in ["8", "17", "23"]:
            model_manager_corners[league_id] = ModelManagerCorners(league_id=league_id)
        print("✅ Modelos Corners precargados para todas las ligas")
    except Exception as e:
        print(f"⚠️  Error cargando modelos Corners: {e}")

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
# CACHÉ DE ELO
# ─────────────────────────────────────────────

_elo_cache: dict = {}
_elo_cache_time: float = 0.0
_ELO_TTL_SECONDS = 3600


def _build_elo_cache() -> dict:
    """
    Recalcula el Elo de todos los equipos desde el histórico completo.
    Parámetros: k=32, scale=600, home_advantage=100
    """
    print(f"🔄 Recalculando caché de Elo global: {datetime.now()}")

    # ✅ Solo llamada a db.py, sin SQL mezclado
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

# ─────────────────────────────────────────────
# CACHÉ DE FEATURES POR EQUIPO
# ─────────────────────────────────────────────

_team_features_cache: dict = {}
_team_features_mtime: dict = {}
_TEAM_FEATURES_TTL = 1800  # 30 minutos (actualizar más frecuente que Elo)
_neighbors_cache = {} # Cache: {(team, year, standings_tuple): [vecinos]}


def get_team_features_cached(team: str, role: str, year: Optional[str],
                             league_id: str, n: int = 5) -> dict:
    """
    Wrapper con caché para get_team_features.
    Cache key: team_role_year_league (ej: "Barcelona_home_25/26_8")
    """
    cache_key = f"{team}_{role}_{year}_{league_id}"
    cache_age = time.time() - _team_features_mtime.get(cache_key, 0)

    # Si está en caché y no expiró, retornar
    if cache_key in _team_features_cache and cache_age < _TEAM_FEATURES_TTL:
        return _team_features_cache[cache_key]

    # Si no, calcular y guardar
    features = get_team_features(team=team, role=role, year=year, league_id=league_id, n=n)
    _team_features_cache[cache_key] = features
    _team_features_mtime[cache_key] = time.time()

    return features


def get_h2h_features_cached(home_team: str, away_team: str,
                            league_id: str, year: Optional[str], n: int = 5) -> dict:
    """
    Wrapper con caché para H2H.
    Cache key: home_vs_away_year_league
    """
    cache_key = f"{home_team}_vs_{away_team}_{year}_{league_id}_h2h"
    cache_age = time.time() - _team_features_mtime.get(cache_key, 0)

    if cache_key in _team_features_cache and cache_age < _TEAM_FEATURES_TTL:
        return _team_features_cache[cache_key]

    features = get_h2h_features(home_team, away_team, league_id, year, n)
    _team_features_cache[cache_key] = features
    _team_features_mtime[cache_key] = time.time()

    return features


def get_over_rates_cached(home_team: str, away_team: str, year: Optional[str]) -> dict:
    """
    Wrapper con caché para over rates.
    Cache key: home_away_year_over
    """
    cache_key = f"{home_team}_{away_team}_{year}_over"
    cache_age = time.time() - _team_features_mtime.get(cache_key, 0)

    if cache_key in _team_features_cache and cache_age < _TEAM_FEATURES_TTL:
        return _team_features_cache[cache_key]

    features = get_over_rates(home_team, away_team, year)
    _team_features_cache[cache_key] = features
    _team_features_mtime[cache_key] = time.time()

    return features


def get_league_standings(league_id: str, year: Optional[str]) -> list:
    """
    Calcula la clasificación de una liga/año con caché.
    """
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
    recent_years = get_recent_years(year) if year else []

    # ✅ Histórico completo (sin límite de partidos)
    historical_avg = get_team_stat_average(team, role, stat_name, recent_years, n=None)

    # ✅ Forma reciente (últimos n partidos)
    recent_avg = get_team_stat_average(team, role, stat_name, recent_years, n=n)

    # ── PONDERACIÓN 65% reciente + 35% histórico ──
    weighted_avg = float(RECENT_WEIGHT) * recent_avg + float(HISTORICAL_WEIGHT) * historical_avg

    return weighted_avg


def get_team_features(team: str, role: str, year: Optional[str], league_id: str, n: int = 5) -> dict:
    """
    Versión OPTIMIZADA: 2 queries en lugar de 18
    Calcula features de equipo con ponderación 65% reciente + 35% histórico.
    """
    recent_years = get_recent_years(year) if year else []

    if not recent_years:
        # Sin año, retornar features vacías
        return {f"{role}_avg_{feat}": 0.0 for feat in FEATURES}

    # ✅ QUERY 1: HISTÓRICO COMPLETO (todas las stats en una sola query)
    stats_hist = get_team_multiple_stats_average(team, role, recent_years, n=None)

    # ✅ QUERY 2: FORMA RECIENTE (últimos n partidos)
    stats_recent = get_team_multiple_stats_average(team, role, recent_years, n=n)

    # ══════════════════════════════════════════════════════════════════════
    # PONDERACIÓN 65% reciente + 35% histórico
    # ══════════════════════════════════════════════════════════════════════
    features = {}

    # Mapeo entre nombres SQL (snake_case) y nombres de features
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

    # Aliases especiales (mantener compatibilidad con código existente)
    features[f"{role}_avg_xG"] = features.get(f"{role}_avg_Expected goals", 0)
    features[f"{role}_avg_shots"] = features.get(f"{role}_avg_Total shots", 0)
    features[f"{role}_avg_big_chances"] = features.get(f"{role}_avg_Big chances", 0)
    features[f"{role}_avg_goals_scored"] = features.get(f"{role}_avg_Goals", 0)

    # ══════════════════════════════════════════════════════════════════════
    # AÑADIR FORM FEATURES INLINE (win_rate, form_pts, goals_conceded)
    # ══════════════════════════════════════════════════════════════════════
    form_features = _get_form_features_inline(team=team, role=role, league_id=league_id, n=n,
                                              recent_years=recent_years)
    features.update(form_features)

    return features


def _get_form_features_inline(team: str, role: str, league_id: str, n: int, recent_years: list) -> dict:
    """
    Helper interno para calcular win_rate, form_pts y goals_conceded.
    Ahora solo 4 queries (2 win_rates + 2 goals_conceded).
    """

    # ✅ WIN RATES HISTÓRICO
    stats_hist_wr = get_team_win_rates(team, role, league_id, recent_years, n=None)

    # ✅ WIN RATES RECIENTE
    stats_recent_wr = get_team_win_rates(team, role, league_id, recent_years, n=n)

    # Calcular win rates ponderados
    if stats_hist_wr["total"] == 0:
        hist_win_rate = hist_draw_rate = hist_loss_rate = 0
    else:
        total_hist = float(stats_hist_wr["total"])
        if role == "home":
            hist_win_rate = float(stats_hist_wr["wins"]) / total_hist
            hist_draw_rate = float(stats_hist_wr["draws"]) / total_hist
            hist_loss_rate = float(stats_hist_wr["losses"]) / total_hist
        else:  # away: invertir wins/losses
            hist_win_rate = float(stats_hist_wr["losses"]) / total_hist
            hist_draw_rate = float(stats_hist_wr["draws"]) / total_hist
            hist_loss_rate = float(stats_hist_wr["wins"]) / total_hist

    if stats_recent_wr["total"] == 0:
        recent_win_rate = recent_draw_rate = recent_loss_rate = 0
        form_pts = 0
    else:
        total_recent = float(stats_recent_wr["total"])
        if role == "home":
            recent_win_rate = float(stats_recent_wr["wins"]) / total_recent
            recent_draw_rate = float(stats_recent_wr["draws"]) / total_recent
            recent_loss_rate = float(stats_recent_wr["losses"]) / total_recent
            form_pts = stats_recent_wr["wins"] * 3 + stats_recent_wr["draws"]
        else:  # away: invertir wins/losses
            recent_win_rate = float(stats_recent_wr["losses"]) / total_recent
            recent_draw_rate = float(stats_recent_wr["draws"]) / total_recent
            recent_loss_rate = float(stats_recent_wr["wins"]) / total_recent
            form_pts = stats_recent_wr["losses"] * 3 + stats_recent_wr["draws"]

    # ✅ GOLES CONCEDIDOS HISTÓRICO
    hist_conceded = get_team_goals_conceded_average(team, role, recent_years, n=None)

    # ✅ GOLES CONCEDIDOS RECIENTE
    recent_conceded = get_team_goals_conceded_average(team, role, recent_years, n=n)

    # Retornar features consolidadas
    return {
        f"{role}_win_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_win_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_win_rate,
        f"{role}_draw_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_draw_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_draw_rate,
        f"{role}_loss_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_loss_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_loss_rate,
        f"{role}_form_pts": form_pts,
        f"{role}_avg_goals_conceded_global": float(RECENT_WEIGHT) * recent_conceded + float(
            HISTORICAL_WEIGHT) * hist_conceded,
    }


def get_team_season_count(team: str, year: Optional[str]) -> int:
    """Número de temporadas disponibles para este equipo."""
    recent_years = get_recent_years(year) if year else []
    if not recent_years:
        return 0
    return db_get_team_season_count(team=team, years=recent_years)


def get_neighbors(team: str, year: Optional[str], standings: list, n_neighbors: int = 3) -> list:
    """Devuelve equipos vecinos en tabla con histórico >= 2 temporadas (con caché)."""
    # Cache key: depende del equipo, año y clasificación
    standings_tuple = tuple(t["team"] for t in standings)  # ← CORREGIDO
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

    # Guardar en caché
    _neighbors_cache[cache_key] = result
    return result


def get_features_from_neighbors(team: str, role: str, year: Optional[str],
                                league_id: str, standings: list) -> dict:
    """Estima features usando vecinos cuando el equipo no tiene histórico."""
    t_start = time.time()  # ← AÑADIR

    closest = get_neighbors(team, year, standings)
    if not closest:
        return {}

    t_neighbors = time.time()
    # print(f"⏱️  get_neighbors: {(t_neighbors - t_start) * 1000:.0f}ms")

    all_features = [f for f in [get_team_features_cached(team=neighbor, role=role, year=year, league_id=league_id, n=5) for neighbor in closest] if f]

    t_features = time.time()
    # print(f"⏱️  get_team_features x{len(closest)}: {(t_features - t_neighbors) * 1000:.0f}ms")

    if not all_features:
        return {}
    merged = {}
    for key in all_features[0]:
        vals = [f[key] for f in all_features if key in f]
        if vals:
            merged[key] = sum(vals) / len(vals)

    # ── PENALTY PARA EQUIPOS RECIÉN ASCENDIDOS ──
    # Reducir stats en ~25% porque equipos promovidos suelen rendir peor
    PROMOTED_TEAM_PENALTY = 0.75

    # Aplicar penalty a todas las features EXCEPTO Elo (que tiene su propia lógica)
    keys_to_penalize = [k for k in merged.keys() if 'elo' not in k.lower()]
    for key in keys_to_penalize:
        merged[key] *= PROMOTED_TEAM_PENALTY

    # print(f"⚠️  {team} sin histórico — proxy de vecinos con penalty 25%: {closest}")
    t_end = time.time()
    # print(f"⏱️  TOTAL get_features_from_neighbors: {(t_end - t_start) * 1000:.0f}ms")
    return merged

# ─────────────────────────────────────────────
# FORM FEATURES CON PONDERACIÓN
# ─────────────────────────────────────────────
def get_win_rates_weighted(team: str, role: str, year: Optional[str],
                           league_id: str, n: int = 5) -> dict:
    """
    Calcula win rates con ponderación INVERTIDA: 35% reciente + 65% histórico.
    Usa la misma función de db.py que _get_form_features_inline.
    """
    recent_years = get_recent_years(year) if year else []

    # ✅ WIN RATES HISTÓRICO (reutilizamos función ya creada en db.py)
    stats_hist = get_team_win_rates(team, role, league_id, recent_years, n=None)

    # ✅ WIN RATES RECIENTE
    stats_recent = get_team_win_rates(team, role, league_id, recent_years, n=n)

    # Calcular win rates desde perspectiva del equipo
    if stats_hist["total"] == 0:
        hist_win_rate = hist_draw_rate = hist_loss_rate = 0
    else:
        total_hist = float(stats_hist["total"])
        if role == "home":
            hist_win_rate = float(stats_hist["wins"]) / total_hist
            hist_draw_rate = float(stats_hist["draws"]) / total_hist
            hist_loss_rate = float(stats_hist["losses"]) / total_hist
        else:  # away: invertir wins/losses
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
        else:  # away: invertir wins/losses
            recent_win_rate = float(stats_recent["losses"]) / total_recent
            recent_draw_rate = float(stats_recent["draws"]) / total_recent
            recent_loss_rate = float(stats_recent["wins"]) / total_recent

    # ── PONDERACIÓN INVERTIDA: 35% reciente + 65% histórico ──
    return {
        f"{role}_win_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_win_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_win_rate,
        f"{role}_draw_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_draw_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_draw_rate,
        f"{role}_loss_rate_{role}": WIN_RATE_RECENT_WEIGHT * recent_loss_rate + WIN_RATE_HISTORICAL_WEIGHT * hist_loss_rate,
    }


def get_form_points(team: str, role: str, year: Optional[str],
                    league_id: str, n: int = 5) -> int:
    """
    Calcula puntos de forma de últimos N partidos.
    """
    recent_years = get_recent_years(year) if year else []

    # ✅ Solo llamada a db.py
    matches = get_team_recent_matches_goals(team, league_id, recent_years, role, n)

    if not matches:
        return 0

    # Calcular puntos según role
    points = 0
    for match in matches:
        hg = float(match["home_goals"] or 0)
        ag = float(match["away_goals"] or 0)

        if role == "home":
            if hg > ag:
                points += 3
            elif hg == ag:
                points += 1
        else:  # away
            if ag > hg:
                points += 3
            elif ag == hg:
                points += 1

    return points


def get_goals_conceded_weighted(team: str, role: str, year: Optional[str], n: int = 5) -> float:
    """
    Calcula goles concedidos ponderados: 65% reciente + 35% histórico.
    Reutiliza get_team_goals_conceded_average de db.py (ya creada en función #8).
    """
    recent_years = get_recent_years(year) if year else []

    # ✅ Histórico completo (reutilizamos función de db.py)
    hist_avg = get_team_goals_conceded_average(team, role, recent_years, n=None)

    # ✅ Forma reciente
    recent_avg = get_team_goals_conceded_average(team, role, recent_years, n=n)

    # Ponderación 65% reciente + 35% histórico
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
    """
    Calcula features H2H con proxy de vecinos si < 2 enfrentamientos directos.
    """
    matches = get_h2h_matches(home_team, away_team, n)

    # Proxy con vecinos si < 2 enfrentamientos
    if len(matches) < 2:
        standings = get_league_standings(league_id, year) if year else []
        if standings:
            home_neighbors = get_neighbors(home_team, year, standings, n_neighbors=3)
            away_neighbors = get_neighbors(away_team, year, standings, n_neighbors=3)

            if home_neighbors and away_neighbors:
                matches = get_proxy_h2h_matches(home_neighbors, away_neighbors, n)
                # if matches:
                #     print(f"⚠️  H2H {home_team} vs {away_team}: usando proxy de vecinos")

    # Si no hay datos, devolver distribución neutra
    if not matches:
        return {
            "h2h_home_wins": 0.33,
            "h2h_draws": 0.33,
            "h2h_away_wins": 0.33,
            "h2h_avg_goals": 0
        }

    # Calcular estadísticas H2H
    total = len(matches)
    avg_goals = sum(float(m["home_goals"] or 0) + float(m["away_goals"] or 0) for m in matches) / total

    # Si muy pocos datos, devolver distribución neutra pero con avg_goals real
    if total < 3:
        return {
            "h2h_home_wins": 0.33,
            "h2h_draws": 0.33,
            "h2h_away_wins": 0.33,
            "h2h_avg_goals": round(avg_goals, 2),
        }

    # Contar victorias/empates desde perspectiva del home_team
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


def get_over_rates_weighted(team: str, year: Optional[str], n: int = 5) -> dict:
    """
    Calcula over rates con ponderación 65% reciente + 35% histórico.
    Evalúa thresholds: 0.5, 1.5, 2.5, 3.5 goles totales.
    """
    recent_years = get_recent_years(year) if year else []

    # ✅ HISTÓRICO COMPLETO
    matches_hist = get_team_matches_total_goals(team, recent_years, n=None)

    # ✅ FORMA RECIENTE (últimos n partidos)
    matches_recent = get_team_matches_total_goals(team, recent_years, n=n)

    features = {}

    # Calcular over rates para cada threshold
    for threshold in [0.5, 1.5, 2.5, 3.5]:
        col = f"over_{str(threshold).replace('.', '_')}_rate"

        # Histórico
        hist_over_count = sum(1 for m in matches_hist if float(m["total_goals"] or 0) > threshold)
        hist_over_rate = float(hist_over_count) / float(len(matches_hist)) if matches_hist else 0.0

        # Reciente
        recent_over_count = sum(1 for m in matches_recent if float(m["total_goals"] or 0) > threshold)
        recent_over_rate = float(recent_over_count) / float(len(matches_recent)) if matches_recent else 0.0

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
            team_features = get_team_features_cached(team=team, role=role, year=year, league_id=league_id, n=5)
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


def build_features_1x2(home_team: str, away_team: str, year: Optional[str],
                       league_id: str = "8") -> np.ndarray:
    """
    Construye vector de features para modelos 1X2 (RF/XGBoost)
    - 40 features base para LaLiga/Serie A
    - 47 features (40 + 7 Premier) para Premier League
    """
    features = {}
    standings = get_league_standings(league_id, year) if year else []

    # ── 1. FEATURES ELO (3 features) ──
    elo_data = get_elo(home_team, away_team)
    features['elo_home'] = elo_data['elo_home']
    features['elo_away'] = elo_data['elo_away']
    features['elo_diff'] = elo_data['elo_diff']

    # ── 2. FEATURES DE EQUIPO (26 features = 13 x 2) ──
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

    # ── 3. FEATURES H2H (5 features) ──
    h2h = get_h2h_features_cached(home_team, away_team, league_id, year, n=5)
    features['h2h_home_win_rate'] = h2h.get('h2h_home_wins', 0.33)
    features['h2h_home_goals_avg'] = h2h.get('h2h_avg_goals', 0) / 2
    features['h2h_away_goals_avg'] = h2h.get('h2h_avg_goals', 0) / 2
    features['h2h_total_goals_avg'] = h2h.get('h2h_avg_goals', 0)
    features['h2h_used_proxy'] = 1 if len(h2h) < 2 else 0

    # ── 4. FEATURES DE BALANCE (6 features) ──
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

    # ── 6. ✅ SOLO PARA PREMIER LEAGUE: AÑADIR 7 FEATURES ADICIONALES ──
    if league_id == '17':  # ← CRÍTICO: Solo Premier League
        # 1. Season progress
        features['season_progress'] = 0.5

        # 2-3. Early/Late season
        features['is_early_season'] = 0
        features['is_late_season'] = 0

        # 4. Home form momentum
        features['home_form_momentum'] = features['home_goals_for_avg'] - features['home_win_rate']

        # 5. Away form momentum
        features['away_form_momentum'] = features['away_goals_for_avg'] - features['away_win_rate']

        # 6. Ultra balanced
        elo_balanced = 1 if abs(features['elo_diff']) < 50 else 0
        form_balanced = 1 if features['form_balance'] < 0.1 else 0
        features['ultra_balanced'] = elo_balanced * form_balanced

        # 7. Underdog home vs strong away
        features['underdog_home_vs_strong_away'] = 1 if (
                    features['elo_home'] < 1450 and features['elo_away'] > 1600) else 0

        # ✅ Extender feature_order SOLO para Premier
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
    # LaLiga/Serie A: 40 features
    # Premier League: 47 features
    feature_vector = np.array([[features.get(col, 0.0) for col in feature_order]], dtype=np.float64)

    return feature_vector

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
        "8": {"very_high": 35, "high": 18, "medium": 7},  # LaLiga (más empates)
        "17": {"very_high": 31, "high": 15, "medium": 5},  # Premier (más definido)
        "23": {"very_high": 33, "high": 17, "medium": 6}  # Serie A (intermedio)
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
    # model_result, calibrated_model, label_encoder, feature_cols, elo_threshold, over_models = load_models(req.league_id)
    try:
        # ── FEATURES PARA 1X2 (40 o 47 dimensiones según liga) ──
        X_1x2 = build_features_1x2(req.home_team, req.away_team, req.year, req.league_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error construyendo features: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # ✅ EXTRAER SOLO LAS 40 FEATURES BASE (para saves/corners)
    # ══════════════════════════════════════════════════════════════════════════
    # Premier tiene 47 features, pero saves/corners solo necesitan las primeras 40
    X_base_40 = X_1x2[:, :40]  # Tomar solo las primeras 40 columnas

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
    result_proba_ou = model_manager_ou_goals.predict_over_under(home_team=req.home_team, away_team=req.away_team,
                                                                year=req.year, league_id=req.league_id)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
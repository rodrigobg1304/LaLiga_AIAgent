"""
Script de entrenamiento GENÉRICO para modelos Over/Under SAVES
Arquitectura: 9 Random Forest independientes con calibración isotónica
Feature is_home para distinguir portero local vs visitante
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import pickle
import json
from datetime import datetime
from tqdm import tqdm

# Añadir paths para imports
from football_core.db import get_matches_with_saves

from football_core.feature_engineering import build_features_1x2
from football_core.constants import (
    LEAGUE_CONFIG,
    TRAIN_SEASONS,
    TEST_SEASONS,
    SAVES_THRESHOLDS,
    MODEL_BASE_DIR
)


# ═══════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Entrenar modelos Over/Under para Saves (paradas de portero)'
    )
    parser.add_argument(
        '--leagues',
        nargs='+',
        choices=['laliga', 'premier', 'serie_a', 'all'],
        default=['all'],
        help='Ligas a entrenar (default: all)'
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════

def get_all_matches(league_id: str, years: list) -> pd.DataFrame:
    """Obtiene todos los partidos con saves."""
    rows = get_matches_with_saves(league_id, years)
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No se encontraron partidos para liga {league_id}")

    print(f"✅ Cargados {len(df)} partidos")
    print(f"   Distribución de saves:")
    print(f"   Home saves: mean={df['home_saves'].mean():.1f}, max={df['home_saves'].max()}")
    print(f"   Away saves: mean={df['away_saves'].mean():.1f}, max={df['away_saves'].max()}")

    return df


def prepare_dataset_saves(df: pd.DataFrame, league_id: str) -> tuple:
    """
    Prepara dataset para saves con arquitectura is_home.

    Cada partido genera 2 rows:
    - Row 1: features + is_home=1 → target: home_saves
    - Row 2: features + is_home=0 → target: away_saves
    """
    print(f"\n🔄 Construyendo features para {len(df)} partidos...")

    X_list = []
    y_saves_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Procesando"):
        try:
            # Features base (40 o 47 para Premier)
            base_features = build_features_1x2(
                home_team=row['homeTeam'],
                away_team=row['awayTeam'],
                year=row['Year'],
                league_id=row['LeagueId']
            )

            if base_features is None:
                continue

            # Flatten si es necesario
            if base_features.ndim == 2:
                base_features = base_features.flatten()

            # Row 1: Home saves (is_home=1)
            features_home = np.append(base_features, [1])
            X_list.append(features_home)
            y_saves_list.append(row['home_saves'])

            # Row 2: Away saves (is_home=0)
            features_away = np.append(base_features, [0])
            X_list.append(features_away)
            y_saves_list.append(row['away_saves'])

        except Exception as e:
            continue

    X = np.vstack(X_list)
    y_saves = np.array(y_saves_list)

    print(f"✅ Features: {X.shape} | Total examples: {len(y_saves)}")
    print(f"   (cada partido genera 2 rows: home + away)")

    # Crear targets binarios para cada threshold
    y_dict = {}
    for threshold in SAVES_THRESHOLDS:
        col_name = f'over_{str(threshold).replace(".", "_")}'
        y_dict[col_name] = (y_saves > threshold).astype(int)

    return X, y_dict, y_saves


def train_model_for_threshold(X_train, y_train, X_test, y_test, threshold: float):
    """Entrena modelo RF para un threshold de saves."""
    print(f"\n🎯 Over {threshold} saves")

    pos_rate = y_train.mean()

    # Class weights
    if pos_rate < 0.3 or pos_rate > 0.7:
        class_weight = {0: pos_rate, 1: 1 - pos_rate}
    else:
        class_weight = None

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)

    # Calibración isotónica
    calibrated = CalibratedClassifierCV(
        rf,
        method='isotonic',
        cv=5,
        n_jobs=-1
    )
    calibrated.fit(X_train, y_train)

    # Evaluación
    y_pred_test = calibrated.predict(X_test)
    y_proba_test = calibrated.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_proba_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='binary', zero_division=0
    )

    print(f"   Test Acc: {test_acc:.3f} | AUC: {test_auc:.3f} | F1: {f1:.3f}")
    print(f"   Over: {pos_rate * 100:.1f}% train, {y_test.mean() * 100:.1f}% test")

    metrics = {
        'threshold': threshold,
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'test_f1': float(f1),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'positive_rate_train': float(pos_rate),
        'positive_rate_test': float(y_test.mean()),
        'trained_at': datetime.now().isoformat()
    }

    return calibrated, metrics


# ═══════════════════════════════════════════════════════════
# ENTRENAMIENTO POR LIGA
# ═══════════════════════════════════════════════════════════

def train_league(league_key: str):
    """Entrena modelos Over/Under Saves para una liga."""

    league_config = LEAGUE_CONFIG[league_key]
    league_id = league_config['id']
    league_slug = league_config['slug']
    league_name = league_config['name']

    print("\n" + "=" * 60)
    print(f"🚀 ENTRENAMIENTO SAVES - {league_name}")
    print("=" * 60)
    print(f"Liga ID: {league_id}")
    print(f"Train: {TRAIN_SEASONS}")
    print(f"Test:  {TEST_SEASONS}")
    print(f"Thresholds: {SAVES_THRESHOLDS}")

    # Directorio de salida
    output_dir = Path(MODEL_BASE_DIR) / "models" / "over_under" / "saves" / "production" / league_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    print("\n📥 Cargando datos de entrenamiento...")
    df_train = get_all_matches(league_id, TRAIN_SEASONS)

    print("\n📥 Cargando datos de test...")
    df_test = get_all_matches(league_id, TEST_SEASONS)

    # 2. Construir features con is_home
    X_train, y_train_dict, saves_train = prepare_dataset_saves(df_train, league_id)
    X_test, y_test_dict, saves_test = prepare_dataset_saves(df_test, league_id)

    # 3. Entrenar modelo por threshold
    all_metrics = {}

    for threshold in SAVES_THRESHOLDS:
        col_name = f'over_{str(threshold).replace(".", "_")}'

        y_train = y_train_dict[col_name]
        y_test = y_test_dict[col_name]

        # Entrenar
        model, metrics = train_model_for_threshold(
            X_train, y_train,
            X_test, y_test,
            threshold
        )

        # Guardar modelo
        model_filename = f"model_{col_name}_saves_{league_slug}.pkl"
        model_path = output_dir / model_filename

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"   💾 Modelo guardado: {model_path}")

        all_metrics[col_name] = metrics

    # 4. Guardar métricas
    metrics_path = output_dir / f"metrics_saves_{league_slug}.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n💾 Métricas guardadas: {metrics_path}")

    # 5. Resumen
    print("\n" + "=" * 60)
    print(f"✅ {league_name} SAVES - ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print("\nResumen de modelos:")
    for threshold in SAVES_THRESHOLDS:
        col_name = f'over_{str(threshold).replace(".", "_")}'
        m = all_metrics[col_name]
        print(f"  Over {threshold}: Test Acc={m['test_accuracy']:.3f} | AUC={m['test_auc']:.3f}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Determinar ligas
    if 'all' in args.leagues:
        leagues_to_train = list(LEAGUE_CONFIG.keys())
    else:
        leagues_to_train = args.leagues

    print("\n" + "=" * 60)
    print("🎯 ENTRENAMIENTO SAVES - SCRIPT GENÉRICO")
    print("=" * 60)
    print(f"Ligas seleccionadas: {', '.join([LEAGUE_CONFIG[k]['name'] for k in leagues_to_train])}")

    # Entrenar cada liga
    for league_key in leagues_to_train:
        try:
            train_league(league_key)
        except Exception as e:
            print(f"\n❌ ERROR en {LEAGUE_CONFIG[league_key]['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("✅ TODOS LOS ENTRENAMIENTOS SAVES COMPLETADOS")
    print("=" * 60)


if __name__ == "__main__":
    main()
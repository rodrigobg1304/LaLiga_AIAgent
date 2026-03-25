"""
Script de entrenamiento GENÉRICO para modelos Over/Under (0.5, 1.5, 2.5, 3.5)
Arquitectura: 4 Random Forest independientes con calibración isotónica
Soporta múltiples ligas mediante argumentos CLI
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
from football_core.db import get_matches_with_goals

from football_core.feature_engineering import build_features_1x2
from football_core.constants import (
    LEAGUE_CONFIG,
    TRAIN_SEASONS,
    TEST_SEASONS,
    OVER_UNDER_THRESHOLDS,
    MODEL_BASE_DIR
)


# ═══════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ═══════════════════════════════════════════════════════════

def parse_args():
    """Parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelos Over/Under para una o más ligas'
    )
    parser.add_argument(
        '--leagues',
        nargs='+',
        choices=['laliga', 'premier', 'serie_a', 'all'],
        default=['all'],
        help='Ligas a entrenar (default: all). Ejemplos: --leagues laliga premier'
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════

def get_all_matches(league_id: str, years: list) -> pd.DataFrame:
    """Obtiene todos los partidos con goles totales."""
    rows = get_matches_with_goals(league_id, years)
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No se encontraron partidos para liga {league_id}")

    # Calcular goles totales
    df['total_goals'] = df['home_goals'] + df['away_goals']

    # Crear targets binarios para cada umbral
    for threshold in OVER_UNDER_THRESHOLDS:
        col_name = f'over_{str(threshold).replace(".", "_")}'
        df[col_name] = (df['total_goals'] > threshold).astype(int)

    print(f"✅ Cargados {len(df)} partidos")
    print(f"   Distribución de goles totales:")
    print(df['total_goals'].describe())

    return df


def build_features_for_match(row: pd.Series) -> np.ndarray:
    """Construye features para un partido usando build_features_1x2."""
    try:
        features = build_features_1x2(
            home_team=row['homeTeam'],
            away_team=row['awayTeam'],
            year=row['Year'],
            league_id=row['LeagueId']
        )
        return features
    except Exception:
        return None


def prepare_dataset(df: pd.DataFrame) -> tuple:
    """Prepara X (features) e y (targets) para entrenamiento."""
    print(f"\n🔄 Construyendo features para {len(df)} partidos...")

    X_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Procesando"):
        features = build_features_for_match(row)
        if features is not None:
            X_list.append(features)
            valid_indices.append(idx)

    X = np.vstack(X_list)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    df_valid = df.loc[valid_indices].reset_index(drop=True)

    print(f"✅ Features: {X.shape} | Descartados: {len(df) - len(valid_indices)}")

    return X, df_valid


def train_model_for_threshold(X_train, y_train, X_test, y_test, threshold: float):
    """Entrena un modelo Random Forest para un umbral específico."""
    print(f"\n🎯 Over {threshold}")

    pos_rate = y_train.mean()

    # Class weights
    if pos_rate < 0.3 or pos_rate > 0.7:
        class_weight = {0: pos_rate, 1: 1 - pos_rate}
    else:
        class_weight = None

    # Entrenar Random Forest
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
    """Entrena modelos Over/Under para una liga específica."""

    league_config = LEAGUE_CONFIG[league_key]
    league_id = league_config['id']
    league_slug = league_config['slug']
    league_name = league_config['name']

    print("\n" + "=" * 60)
    print(f"🚀 ENTRENAMIENTO OVER/UNDER - {league_name}")
    print("=" * 60)
    print(f"Liga ID: {league_id}")
    print(f"Train: {TRAIN_SEASONS}")
    print(f"Test:  {TEST_SEASONS}")
    print(f"Umbrales: {OVER_UNDER_THRESHOLDS}")

    # Directorio de salida
    output_dir = Path(MODEL_BASE_DIR) / "models" / "over_under" / "goals" / "production" / league_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    print("\n📥 Cargando datos de entrenamiento...")
    df_train = get_all_matches(league_id, TRAIN_SEASONS)

    print("\n📥 Cargando datos de test...")
    df_test = get_all_matches(league_id, TEST_SEASONS)

    # 2. Construir features
    X_train, df_train_valid = prepare_dataset(df_train)
    X_test, df_test_valid = prepare_dataset(df_test)

    # 3. Entrenar un modelo por cada umbral
    all_metrics = {}

    for threshold in OVER_UNDER_THRESHOLDS:
        col_name = f'over_{str(threshold).replace(".", "_")}'

        y_train = df_train_valid[col_name].values
        y_test = df_test_valid[col_name].values

        # Entrenar
        model, metrics = train_model_for_threshold(
            X_train, y_train,
            X_test, y_test,
            threshold
        )

        # Guardar modelo
        model_filename = f"model_{col_name}_{league_slug}.pkl"
        model_path = output_dir / model_filename

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"   💾 Modelo guardado: {model_path}")

        all_metrics[col_name] = metrics

    # 4. Guardar métricas
    metrics_path = output_dir / f"metrics_over_under_{league_slug}.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n💾 Métricas guardadas: {metrics_path}")

    # 5. Resumen
    print("\n" + "=" * 60)
    print(f"✅ {league_name} - ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print("\nResumen de modelos:")
    for threshold in OVER_UNDER_THRESHOLDS:
        col_name = f'over_{str(threshold).replace(".", "_")}'
        m = all_metrics[col_name]
        print(f"  Over {threshold}: Test Acc={m['test_accuracy']:.3f} | AUC={m['test_auc']:.3f}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """Función principal con manejo de múltiples ligas."""
    args = parse_args()

    # Determinar qué ligas entrenar
    if 'all' in args.leagues:
        leagues_to_train = list(LEAGUE_CONFIG.keys())
    else:
        leagues_to_train = args.leagues

    print("\n" + "=" * 60)
    print("🎯 ENTRENAMIENTO OVER/UNDER - SCRIPT GENÉRICO")
    print("=" * 60)
    print(f"Ligas seleccionadas: {', '.join([LEAGUE_CONFIG[k]['name'] for k in leagues_to_train])}")

    # Entrenar cada liga
    for league_key in leagues_to_train:
        try:
            train_league(league_key)
        except Exception as e:
            print(f"\n❌ ERROR en {LEAGUE_CONFIG[league_key]['name']}: {str(e)}")
            continue

    # Resumen final
    print("\n" + "=" * 60)
    print("✅ TODOS LOS ENTRENAMIENTOS COMPLETADOS")
    print("=" * 60)


if __name__ == "__main__":
    main()
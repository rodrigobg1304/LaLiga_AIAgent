"""
Script de entrenamiento para modelos Over/Under (0.5, 1.5, 2.5, 3.5)
Arquitectura: 4 Random Forest independientes con calibración isotónica
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle
import json
from datetime import datetime

# Añadir path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
# sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from football_agent.db import run_query

# Importar función de features desde predict.py
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from predict import build_features_1x2

# ═══════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════

LEAGUE_ID = "8"  # LaLiga
YEARS_TRAIN = ['19/20', '20/21', '21/22', '22/23', '23/24']  # Últimas 5 temporadas
YEARS_TEST = ['24/25']  # Temporada actual para validación

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLDS = [0.5, 1.5, 2.5, 3.5]


# ═══════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════

def get_all_matches(league_id: str, years: list) -> pd.DataFrame:
    """Obtiene todos los partidos con goles totales."""
    years_str = ",".join([f"'{y}'" for y in years])

    sql = f"""
    SELECT 
        matchId,
        homeTeam, 
        awayTeam,
        LeagueId,
        Year,
        SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') 
            THEN CAST(homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
        SUM(CASE WHEN name='Goals' AND period IN ('1ST','2ND') 
            THEN CAST(awayValue AS DECIMAL) ELSE 0 END) AS away_goals
    FROM Leagues
    WHERE LeagueId = %s 
      AND Year IN ({years_str})
      AND name = 'Goals'
    GROUP BY matchId
    ORDER BY Year ASC, CAST(Round AS SIGNED) ASC
    """

    rows = run_query(sql, (league_id,))
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No se encontraron partidos para liga {league_id}")

    # Calcular goles totales
    df['total_goals'] = df['home_goals'] + df['away_goals']

    # Crear targets binarios para cada umbral
    for threshold in THRESHOLDS:
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
        # Silenciar errores individuales
        return None


def prepare_dataset(df: pd.DataFrame) -> tuple:
    """Prepara X (features) e y (targets) para entrenamiento."""
    print(f"\n🔄 Construyendo features para {len(df)} partidos...")

    from tqdm import tqdm  # Barra de progreso

    X_list = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Procesando"):
        features = build_features_for_match(row)
        if features is not None:
            X_list.append(features)
            valid_indices.append(idx)

    X = np.vstack(X_list)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)  # (n_samples, 1, 40) -> (n_samples, 40)

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

    # Entrenar Random Forest (sin prints)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        verbose=0  # ← Silenciar
    )
    rf.fit(X_train, y_train)

    # Calibración (sin prints)
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

    # Solo métricas clave
    from sklearn.metrics import precision_recall_fscore_support
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
# ENTRENAMIENTO PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🚀 ENTRENAMIENTO MODELOS OVER/UNDER")
    print("=" * 60)
    print(f"Liga: {LEAGUE_ID}")
    print(f"Train: {YEARS_TRAIN}")
    print(f"Test:  {YEARS_TEST}")
    print(f"Umbrales: {THRESHOLDS}")

    # 1. Cargar datos
    print("\n📥 Cargando datos de entrenamiento...")
    df_train = get_all_matches(LEAGUE_ID, YEARS_TRAIN)

    print("\n📥 Cargando datos de test...")
    df_test = get_all_matches(LEAGUE_ID, YEARS_TEST)

    # 2. Construir features
    X_train, df_train_valid = prepare_dataset(df_train)
    X_test, df_test_valid = prepare_dataset(df_test)

    # 3. Entrenar un modelo por cada umbral
    all_metrics = {}

    for threshold in THRESHOLDS:
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
        model_filename = f"model_{col_name}_laliga.pkl"
        model_path = os.path.join(OUTPUT_DIR, model_filename)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"   💾 Modelo guardado: {model_path}")

        all_metrics[col_name] = metrics

    # 4. Guardar métricas
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_over_under_laliga.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n💾 Métricas guardadas: {metrics_path}")

    # 5. Resumen final
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print("\nResumen de modelos:")
    for threshold in THRESHOLDS:
        col_name = f'over_{str(threshold).replace(".", "_")}'
        m = all_metrics[col_name]
        print(f"  Over {threshold}: Test Acc={m['test_accuracy']:.3f} | AUC={m['test_auc']:.3f}")


if __name__ == "__main__":
    main()
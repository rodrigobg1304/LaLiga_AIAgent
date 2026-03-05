# train_cascade.py
import os, sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
import pickle
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from football_agent.db import TABLE, run_query

# Importar funciones del train.py principal
from train import (
    logger, Config, load_data, calculate_elo,
    calculate_form_features, calculate_h2h_features,
    prepare_features, save_model
)


def train_cascade_premier():
    """
    Entrena sistema de cascada para Premier League:
    - Modelo 1: Detector de empates (Binary: X vs No-X)
    - Modelo 2: Predictor 1 vs 2 (Binary: 1 vs 2, solo para no-empates)
    """
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO EN CASCADA - PREMIER LEAGUE")
    logger.info("=" * 60)

    # 1. Cargar datos
    league_id = '17'
    df = load_data(league_id)
    logger.success(f"Datos cargados: {len(df)} partidos")

    # 2. Features
    df = calculate_elo(df)
    df = calculate_form_features(df)
    df = calculate_h2h_features(df)

    # 3. Split temporal
    logger.info(f"Split temporal: test season = {Config.TEST_SEASON}")
    train_df = df[df['season'] != Config.TEST_SEASON].copy()
    test_df = df[df['season'] == Config.TEST_SEASON].copy()

    logger.info(f"Train: {len(train_df)} partidos")
    logger.info(f"Test: {len(test_df)} partidos")

    # 4. Preparar features (CON features Premier)
    X_train, y_train = prepare_features(train_df, league_name='premier_league')
    X_test, y_test = prepare_features(test_df, league_name='premier_league')

    logger.info(f"Features shape: {X_train.shape}")

    # ═══════════════════════════════════════════════════════════
    # MODELO 1: DETECTOR DE EMPATES (X vs No-X)
    # ═══════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 60)
    logger.info("MODELO 1: DETECTOR DE EMPATES")
    logger.info("=" * 60)

    # Convertir a binario: 1=Empate, 0=No-Empate
    y_train_draw = (y_train == 1).astype(int)
    y_test_draw = (y_test == 1).astype(int)

    logger.info(f"Distribución train: Empates={np.sum(y_train_draw)}, No-Empates={np.sum(~y_train_draw.astype(bool))}")
    logger.info(f"Distribución test: Empates={np.sum(y_test_draw)}, No-Empates={np.sum(~y_test_draw.astype(bool))}")

    # Entrenar detector de empates
    model_draw = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,  # Menos profundo para binario
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight={0: 1.0, 1: 3.5},  # Peso FUERTE a empates
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )

    logger.info("  - Entrenando detector de empates...")
    model_draw.fit(X_train, y_train_draw)

    # Calibrar
    logger.info("  - Calibrando...")
    model_draw_cal = CalibratedClassifierCV(
        estimator=model_draw,
        method='isotonic',
        cv=5,
        n_jobs=-1
    )
    model_draw_cal.fit(X_train, y_train_draw)

    # Evaluar
    train_acc_draw = model_draw_cal.score(X_train, y_train_draw)
    test_acc_draw = model_draw_cal.score(X_test, y_test_draw)

    logger.info(f"  - Train accuracy: {train_acc_draw:.3f}")
    logger.info(f"  - Test accuracy: {test_acc_draw:.3f}")

    # Predicciones con threshold
    probs_draw = model_draw_cal.predict_proba(X_test)[:, 1]

    # Probar diferentes thresholds
    logger.info("\n  - Testing thresholds:")
    for threshold in [0.25, 0.28, 0.30, 0.33, 0.35, 0.40]:
        preds_draw = (probs_draw >= threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_test_draw, preds_draw, zero_division=0)
        recall = recall_score(y_test_draw, preds_draw)
        logger.info(f"    Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}")

    # ═══════════════════════════════════════════════════════════
    # MODELO 2: PREDICTOR 1 vs 2 (solo para no-empates)
    # ═══════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 60)
    logger.info("MODELO 2: PREDICTOR 1 vs 2")
    logger.info("=" * 60)

    # Filtrar solo partidos que NO son empates en training
    no_draw_mask_train = (y_train != 1)
    X_train_1v2 = X_train[no_draw_mask_train]
    y_train_1v2 = y_train[no_draw_mask_train]

    no_draw_mask_test = (y_test != 1)
    X_test_1v2 = X_test[no_draw_mask_test]
    y_test_1v2 = y_test[no_draw_mask_test]

    logger.info(f"Train 1v2: {len(X_train_1v2)} partidos (sin empates)")
    logger.info(f"Test 1v2: {len(X_test_1v2)} partidos (sin empates)")
    logger.info(f"Distribución train 1v2: 1={np.sum(y_train_1v2 == 0)}, 2={np.sum(y_train_1v2 == 2)}")

    # Entrenar predictor 1 vs 2
    model_1v2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )

    logger.info("  - Entrenando predictor 1 vs 2...")
    model_1v2.fit(X_train_1v2, y_train_1v2)

    # Calibrar
    logger.info("  - Calibrando...")
    model_1v2_cal = CalibratedClassifierCV(
        estimator=model_1v2,
        method='isotonic',
        cv=5,
        n_jobs=-1
    )
    model_1v2_cal.fit(X_train_1v2, y_train_1v2)

    # Evaluar en subset sin empates
    train_acc_1v2 = model_1v2_cal.score(X_train_1v2, y_train_1v2)
    test_acc_1v2 = model_1v2_cal.score(X_test_1v2, y_test_1v2)

    logger.info(f"  - Train accuracy (1v2): {train_acc_1v2:.3f}")
    logger.info(f"  - Test accuracy (1v2): {test_acc_1v2:.3f}")

    # ═══════════════════════════════════════════════════════════
    # EVALUACIÓN EN CASCADA COMPLETA
    # ═══════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 60)
    logger.info("EVALUACIÓN SISTEMA COMPLETO EN CASCADA")
    logger.info("=" * 60)

    # Threshold óptimo (ajustar basado en resultados anteriores)
    OPTIMAL_THRESHOLD = 0.37

    predictions_cascade = predict_cascade(
        X_test,
        model_draw_cal,
        model_1v2_cal,
        threshold=OPTIMAL_THRESHOLD
    )

    # Métricas
    accuracy_cascade = accuracy_score(y_test, predictions_cascade)

    logger.success(f"  - Accuracy cascada (threshold={OPTIMAL_THRESHOLD}): {accuracy_cascade:.3f}")

    # Distribución de predicciones
    pred_dist = {
        '1': np.sum(predictions_cascade == 0),
        'X': np.sum(predictions_cascade == 1),
        '2': np.sum(predictions_cascade == 2)
    }

    real_dist = {
        '1': np.sum(y_test == 0),
        'X': np.sum(y_test == 1),
        '2': np.sum(y_test == 2)
    }

    logger.info(f"  - Distribución real: 1={real_dist['1']}, X={real_dist['X']}, 2={real_dist['2']}")
    logger.info(f"  - Distribución predicha: 1={pred_dist['1']}, X={pred_dist['X']}, 2={pred_dist['2']}")

    # Classification report
    logger.info("\n  - Métricas por clase:")
    target_names = ['1 (Home)', 'X (Draw)', '2 (Away)']
    report = classification_report(y_test, predictions_cascade, target_names=target_names)
    print(report)

    # ═══════════════════════════════════════════════════════════
    # GUARDAR MODELOS
    # ═══════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 60)
    logger.info("GUARDANDO MODELOS")
    logger.info("=" * 60)

    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # Guardar modelo detector empates
    draw_model_path = f"{Config.MODELS_DIR}/cascade_draw_premier.pkl"
    with open(draw_model_path, 'wb') as f:
        pickle.dump(model_draw_cal, f)
    logger.success(f"  - Modelo detector empates guardado: {draw_model_path}")

    # Guardar modelo 1v2
    model_1v2_path = f"{Config.MODELS_DIR}/cascade_1v2_premier.pkl"
    with open(model_1v2_path, 'wb') as f:
        pickle.dump(model_1v2_cal, f)
    logger.success(f"  - Modelo 1v2 guardado: {model_1v2_path}")

    # Guardar threshold óptimo
    config_path = f"{Config.MODELS_DIR}/cascade_config_premier.json"
    import json
    cascade_config = {
        'optimal_threshold': OPTIMAL_THRESHOLD,
        'features': X_train.shape[1],
        'accuracy_draw_detector': float(test_acc_draw),
        'accuracy_1v2': float(test_acc_1v2),
        'accuracy_cascade': float(accuracy_cascade)
    }
    with open(config_path, 'w') as f:
        json.dump(cascade_config, f, indent=2)
    logger.success(f"  - Config guardada: {config_path}")

    logger.info("=" * 60)
    logger.success("ENTRENAMIENTO CASCADA COMPLETADO")
    logger.info("=" * 60)


def predict_cascade(X, model_draw, model_1v2, threshold=0.37):
    """
    Predicción en cascada

    Args:
        X: Features
        model_draw: Modelo detector empates
        model_1v2: Modelo predictor 1 vs 2
        threshold: Threshold para clasificar como empate

    Returns:
        np.array: Predicciones (0=1, 1=X, 2=2)
    """
    # PASO 1: Detectar empates
    probs_draw = model_draw.predict_proba(X)[:, 1]  # P(Empate)

    predictions = []

    for i in range(len(X)):
        if probs_draw[i] >= threshold:
            # Predecir empate
            predictions.append(1)
        else:
            # Predecir 1 vs 2
            pred_1v2 = model_1v2.predict(X[i:i + 1])[0]
            predictions.append(pred_1v2)

    return np.array(predictions)


if __name__ == "__main__":
    train_cascade_premier()
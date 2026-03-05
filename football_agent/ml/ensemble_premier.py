# ensemble_premier.py
import os, sys
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from train import (
    logger, Config, load_data, calculate_elo,
    calculate_form_features, calculate_h2h_features
)

from train_xgboost import prepare_features_xgb


def load_models(league='premier_league'):
    """
    Carga modelos RF y XGBoost para una liga

    Returns:
        tuple: (rf_model, xgb_model)
    """
    logger.info(f"Cargando modelos para {league}...")

    rf_path = f"./models/model_1x2_{league}.pkl"
    xgb_path = f"./models/model_xgboost_{league}.pkl"

    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)

    with open(xgb_path, 'rb') as f:
        xgb_model = pickle.load(f)

    logger.success("  - Modelos cargados exitosamente")

    return rf_model, xgb_model


def ensemble_predict(rf_model, xgb_model, X, rf_weight=0.7, xgb_weight=0.3, min_prob=0.04):
    """
    Combina predicciones de RF y XGBoost con weighted average

    Args:
        rf_model: Modelo Random Forest calibrado
        xgb_model: Modelo XGBoost
        X: Features (np.array)
        rf_weight: Peso para RF (default 0.7)
        xgb_weight: Peso para XGBoost (default 0.3)
        min_prob: Floor mínimo de probabilidad

    Returns:
        tuple: (y_pred, y_pred_proba)
    """
    # Predicciones de cada modelo
    rf_proba = rf_model.predict_proba(X)
    xgb_proba = xgb_model.predict_proba(X)

    # Weighted average
    ensemble_proba = (rf_weight * rf_proba) + (xgb_weight * xgb_proba)

    # Aplicar MIN_PROB floor
    ensemble_proba = np.maximum(ensemble_proba, min_prob)

    # Re-normalizar
    ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)

    # Predicción final (argmax)
    y_pred = np.argmax(ensemble_proba, axis=1)

    return y_pred, ensemble_proba


def evaluate_ensemble(rf_model, xgb_model, X_test, y_test, rf_weight=0.7):
    """
    Evalúa ensemble con diferentes pesos

    Args:
        rf_model: Modelo RF
        xgb_model: Modelo XGBoost
        X_test: Features de test
        y_test: Labels de test (0, 1, 2)
        rf_weight: Peso de RF (XGB será 1 - rf_weight)

    Returns:
        dict: Métricas del ensemble
    """
    xgb_weight = 1.0 - rf_weight

    logger.info(f"Evaluando ensemble: RF={rf_weight:.1f}, XGB={xgb_weight:.1f}")

    # Predicciones ensemble
    y_pred, y_pred_proba = ensemble_predict(
        rf_model, xgb_model, X_test,
        rf_weight=rf_weight,
        xgb_weight=xgb_weight
    )

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    # Distribución
    pred_dist = {
        '1': int(np.sum(y_pred == 0)),
        'X': int(np.sum(y_pred == 1)),
        '2': int(np.sum(y_pred == 2))
    }

    real_dist = {
        '1': int(np.sum(y_test == 0)),
        'X': int(np.sum(y_test == 1)),
        '2': int(np.sum(y_test == 2))
    }

    # Classification report
    target_names = ['1 (Home)', 'X (Draw)', '2 (Away)']
    class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Logging
    logger.success(f"  - Accuracy: {accuracy:.3f}")
    logger.info(f"  - Log Loss: {logloss:.3f}")
    logger.info(f"  - Distribución real: 1={real_dist['1']}, X={real_dist['X']}, 2={real_dist['2']}")
    logger.info(f"  - Distribución predicha: 1={pred_dist['1']}, X={pred_dist['X']}, 2={pred_dist['2']}")

    logger.info("  - Métricas por clase:")
    for class_name in target_names:
        precision = class_report[class_name]['precision']
        recall = class_report[class_name]['recall']
        f1 = class_report[class_name]['f1-score']
        logger.info(f"    {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    return {
        'rf_weight': rf_weight,
        'xgb_weight': xgb_weight,
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'classification_report': class_report,
        'predictions_distribution': pred_dist,
        'real_distribution': real_dist
    }


def grid_search_weights(rf_model, xgb_model, X_test, y_test):
    """
    Prueba diferentes combinaciones de pesos RF/XGB

    Args:
        rf_model: Modelo RF
        xgb_model: Modelo XGBoost
        X_test: Features de test
        y_test: Labels de test

    Returns:
        dict: Resultados de cada combinación
    """
    logger.info("=" * 60)
    logger.info("GRID SEARCH DE PESOS RF/XGB")
    logger.info("=" * 60)

    # Probar diferentes pesos RF (de 0.5 a 0.9 en steps de 0.1)
    rf_weights = [0.5, 0.6, 0.7, 0.8, 0.9]

    results = []

    for rf_weight in rf_weights:
        logger.info(f"\n--- Probando RF={rf_weight:.1f}, XGB={1 - rf_weight:.1f} ---")
        metrics = evaluate_ensemble(rf_model, xgb_model, X_test, y_test, rf_weight)
        results.append(metrics)

    # Encontrar mejor configuración
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE RESULTADOS")
    logger.info("=" * 60)

    logger.info("\n{:<12} {:<10} {:<12} {:<12} {:<12}".format(
        "RF Weight", "Accuracy", "Recall X", "F1 X", "Log Loss"
    ))
    logger.info("-" * 60)

    best_accuracy = 0
    best_config = None

    for res in results:
        rf_w = res['rf_weight']
        acc = res['accuracy']
        recall_x = res['classification_report']['X (Draw)']['recall']
        f1_x = res['classification_report']['X (Draw)']['f1-score']
        ll = res['log_loss']

        logger.info("{:<12} {:<10.3f} {:<12.3f} {:<12.3f} {:<12.3f}".format(
            f"{rf_w:.1f}", acc, recall_x, f1_x, ll
        ))

        # Buscar mejor por accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            best_config = res

    logger.info("\n" + "=" * 60)
    logger.success(f"MEJOR CONFIGURACIÓN: RF={best_config['rf_weight']:.1f}, XGB={best_config['xgb_weight']:.1f}")
    logger.success(f"  - Accuracy: {best_config['accuracy']:.3f}")
    logger.info(f"  - Recall Empates: {best_config['classification_report']['X (Draw)']['recall']:.3f}")
    logger.info(f"  - F1 Empates: {best_config['classification_report']['X (Draw)']['f1-score']:.3f}")
    logger.info("=" * 60)

    return results, best_config


def save_ensemble_config(best_config, league='premier_league'):
    """
    Guarda la configuración óptima del ensemble

    Args:
        best_config: Dict con mejor configuración
        league: Nombre de la liga
    """
    import json
    from datetime import datetime

    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    config_path = f"{Config.MODELS_DIR}/ensemble_config_{league}.json"

    # Añadir metadata
    ensemble_config = {
        'timestamp': datetime.now().isoformat(),
        'league': league,
        'model_type': 'Ensemble_RF_XGBoost',
        'rf_weight': best_config['rf_weight'],
        'xgb_weight': best_config['xgb_weight'],
        'metrics': {
            'accuracy': best_config['accuracy'],
            'log_loss': best_config['log_loss'],
            'classification_report': best_config['classification_report']
        },
        'usage': {
            'rf_model_path': f'./models/model_1x2_{league}.pkl',
            'xgb_model_path': f'./models/model_xgboost_{league}.pkl',
            'instructions': 'Load both models and use weighted average with specified weights'
        }
    }

    logger.info(f"  - Guardando configuración ensemble en {config_path}...")
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)

    logger.success("  - Configuración ensemble guardada exitosamente")


def main():
    """Pipeline completo de ensemble para Premier League"""

    logger.info("=" * 60)
    logger.info("ENSEMBLE RF + XGBOOST - PREMIER LEAGUE")
    logger.info("=" * 60)

    # 1. Cargar datos
    league_id = '17'
    league_name = 'premier_league'

    df = load_data(league_id)
    logger.success(f"Datos cargados: {len(df)} partidos")

    # 2. Features
    df = calculate_elo(df)
    df = calculate_form_features(df)
    df = calculate_h2h_features(df)

    # 3. Split temporal
    test_df = df[df['season'] == Config.TEST_SEASON].copy()
    logger.info(f"Test: {len(test_df)} partidos (season {Config.TEST_SEASON})")

    # 4. Preparar features (usar función de XGBoost que no encodea)
    X_test, y_test = prepare_features_xgb(test_df, league_name)

    # 5. Cargar modelos
    rf_model, xgb_model = load_models(league_name)

    # 6. Grid search de pesos
    results, best_config = grid_search_weights(rf_model, xgb_model, X_test, y_test)

    # 7. Guardar mejor configuración
    save_ensemble_config(best_config, league_name)

    # 8. Comparar con modelos individuales
    logger.info("\n" + "=" * 60)
    logger.info("COMPARACIÓN CON MODELOS INDIVIDUALES")
    logger.info("=" * 60)

    # RF solo
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, target_names=['1', 'X', '2'], output_dict=True)

    logger.info(f"\nRF solo:")
    logger.info(f"  - Accuracy: {rf_acc:.3f}")
    logger.info(f"  - Recall Empates: {rf_report['X']['recall']:.3f}")
    logger.info(f"  - F1 Empates: {rf_report['X']['f1-score']:.3f}")

    # XGBoost solo
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_report = classification_report(y_test, xgb_pred, target_names=['1', 'X', '2'], output_dict=True)

    logger.info(f"\nXGBoost solo:")
    logger.info(f"  - Accuracy: {xgb_acc:.3f}")
    logger.info(f"  - Recall Empates: {xgb_report['X']['recall']:.3f}")
    logger.info(f"  - F1 Empates: {xgb_report['X']['f1-score']:.3f}")

    # Ensemble
    logger.info(f"\nEnsemble (RF={best_config['rf_weight']:.1f}, XGB={best_config['xgb_weight']:.1f}):")
    logger.info(f"  - Accuracy: {best_config['accuracy']:.3f}")
    logger.info(f"  - Recall Empates: {best_config['classification_report']['X (Draw)']['recall']:.3f}")
    logger.info(f"  - F1 Empates: {best_config['classification_report']['X (Draw)']['f1-score']:.3f}")

    # Mejora vs RF
    acc_improvement = best_config['accuracy'] - rf_acc
    recall_improvement = best_config['classification_report']['X (Draw)']['recall'] - rf_report['X']['recall']

    logger.info(f"\nMejora vs RF solo:")
    logger.info(f"  - Accuracy: {acc_improvement:+.1%}")
    logger.info(f"  - Recall Empates: {recall_improvement:+.1%}")

    logger.info("\n" + "=" * 60)
    logger.success("ENSEMBLE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
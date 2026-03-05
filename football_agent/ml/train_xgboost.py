# train_xgboost.py
import os, sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, classification_report
import pickle
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Importar funciones del train.py principal
from train import (
    logger, Config, load_data, calculate_elo,
    calculate_form_features, calculate_h2h_features
)


def prepare_features_xgb(df, league_name='laliga'):
    """
    Igual que prepare_features() pero sin encodear target
    XGBoost necesita labels 0, 1, 2 directamente

    Returns:
        X: np.array de features
        y: np.array con valores 0, 1, 2
    """
    logger.info("Preparando features para XGBoost...")

    # Calcular features de balance
    df = df.copy()
    df['elo_diff_abs'] = np.abs(df['elo_diff'])
    df['form_balance'] = np.abs(df['home_win_rate'] - df['away_win_rate'])
    df['goals_balance'] = np.abs(df['home_goals_for_avg'] - df['away_goals_for_avg'])
    df['possession_balance'] = np.abs(df['home_possession_avg'] - df['away_possession_avg'])
    df['shots_balance'] = np.abs(df['home_total_shots_avg'] - df['away_total_shots_avg'])
    df['shots_on_target_balance'] = np.abs(df['home_shots_on_target_avg'] - df['away_shots_on_target_avg'])

    # Lista de features (40 base)
    feature_columns = [
        # Elo (3)
        'elo_home', 'elo_away', 'elo_diff',

        # Form - Home BÁSICAS (4)
        'home_win_rate', 'home_goals_for_avg', 'home_goals_against_avg', 'home_points_avg',

        # Form - Home AVANZADAS (9)
        'home_shots_on_target_avg', 'home_possession_avg', 'home_total_shots_avg',
        'home_gk_saves_avg', 'home_big_chances_avg', 'home_accurate_passes_avg',
        'home_tackles_won_avg', 'home_interceptions_avg', 'home_blocked_shots_avg',

        # Form - Away BÁSICAS (4)
        'away_win_rate', 'away_goals_for_avg', 'away_goals_against_avg', 'away_points_avg',

        # Form - Away AVANZADAS (9)
        'away_shots_on_target_avg', 'away_possession_avg', 'away_total_shots_avg',
        'away_gk_saves_avg', 'away_big_chances_avg', 'away_accurate_passes_avg',
        'away_tackles_won_avg', 'away_interceptions_avg', 'away_blocked_shots_avg',

        # H2H (5)
        'h2h_home_win_rate', 'h2h_home_goals_avg', 'h2h_away_goals_avg',
        'h2h_total_goals_avg', 'h2h_used_proxy',

        # FEATURES DE BALANCE (6)
        'elo_diff_abs', 'form_balance', 'goals_balance',
        'possession_balance', 'shots_balance', 'shots_on_target_balance'
    ]

    # Validar columnas
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes: {missing_cols}")

    # Convertir a float64
    df_features = df[feature_columns].copy()
    for col in feature_columns:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

    # Imputar NaN si existen
    if np.isnan(df_features.values).any():
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(df_features.values)
    else:
        X = df_features.values

    # Mapear target: '1'->0, 'X'->1, '2'->2
    result_mapping = {'1': 0, 'X': 1, '2': 2}
    y = df['result'].map(result_mapping).values

    logger.success(f"  - Features preparadas: {X.shape}")
    logger.info(f"  - Total features: {len(feature_columns)}")
    logger.info(f"  - Distribución target: 0={np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%), "
                f"1={np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%), "
                f"2={np.sum(y == 2)} ({np.sum(y == 2) / len(y) * 100:.1f}%)")

    return X, y


def calculate_sample_weights(y_train, league_name):
    """
    Calcula sample weights para cada instancia
    XGBoost los usa para dar más importancia a ciertas clases

    Args:
        y_train: Labels (0, 1, 2)
        league_name: Nombre de la liga

    Returns:
        np.array: Weight por cada muestra
    """
    # Contar cada clase
    unique, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)

    # Calcular weight base (inverso de frecuencia)
    class_weights = {}
    for cls, count in zip(unique, counts):
        class_weights[cls] = n_samples / (len(unique) * count)

    # Ajustar según liga
    if league_name == 'premier_league':
        # Premier: boost FUERTE a empates
        class_weights[1] *= 3.0  # Clase 1 = empates
        logger.info(f"  - Premier League: Aumentando peso de empates x3.0")
    elif league_name == 'laliga':
        class_weights[1] *= 1.3
        logger.info(f"  - LaLiga: Aumentando peso de empates x1.3")
    elif league_name == 'serie_a':
        class_weights[1] *= 1.1
        logger.info(f"  - Serie A: Aumentando peso de empates x1.1")

    # Crear array de weights
    sample_weights = np.array([class_weights[label] for label in y_train])

    logger.info(f"  - Class weights: {class_weights}")
    logger.info(f"  - Sample weights shape: {sample_weights.shape}")

    return sample_weights


def train_xgboost_model(X_train, y_train, X_val, y_val, league_name):
    """
    Entrena modelo XGBoost con early stopping

    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento (0, 1, 2)
        X_val: Features de validación
        y_val: Labels de validación
        league_name: Nombre de la liga

    Returns:
        xgb.XGBClassifier: Modelo entrenado
    """
    logger.info("Entrenando modelo XGBoost...")

    # Calcular sample weights
    sample_weights = calculate_sample_weights(y_train, league_name)

    # Configurar modelo
    model = xgb.XGBClassifier(
        objective='multi:softprob',  # Multiclase con softmax
        num_class=3,  # 3 clases

        # Árboles
        n_estimators=300,  # Número de árboles (con early stopping)
        max_depth=6,  # Profundidad moderada
        learning_rate=0.05,  # Learning rate bajo (más conservador)

        # Regularización
        min_child_weight=10,  # Mínimo peso en hoja (evita overfitting)
        gamma=0.1,  # Penalización por split adicional
        subsample=0.8,  # 80% de datos por árbol (bagging)
        colsample_bytree=0.8,  # 80% de features por árbol

        # Evaluación
        eval_metric='mlogloss',  # Multi-class log loss
        early_stopping_rounds=20,  # Parar si no mejora en 20 rounds

        # Sistema
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=1  # Mostrar progreso
    )

    logger.info(f"  - Configuración XGBoost:")
    logger.info(f"    max_depth={model.max_depth}, learning_rate={model.learning_rate}")
    logger.info(f"    subsample={model.subsample}, colsample={model.colsample_bytree}")

    # Crear eval_set para early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]

    # Entrenar con early stopping
    logger.info("  - Entrenando con early stopping...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=10  # Imprimir cada 10 rounds
    )

    # Resultados de early stopping
    best_iteration = model.best_iteration
    best_score = model.best_score

    logger.info(f"  - Best iteration: {best_iteration}")
    logger.info(f"  - Best validation mlogloss: {best_score:.4f}")

    # Accuracy en train
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    logger.info(f"  - Train accuracy: {train_acc:.3f}")

    # Accuracy en validación
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    logger.info(f"  - Validation accuracy: {val_acc:.3f}")

    logger.success("  - Modelo XGBoost entrenado exitosamente")

    return model


def evaluate_xgboost(model, X_test, y_test):
    """
    Evalúa modelo XGBoost y genera métricas

    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Labels de test (0, 1, 2)

    Returns:
        dict: Métricas de evaluación
    """
    logger.info("Evaluando modelo XGBoost...")

    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Aplicar MIN_PROB floor y re-normalizar
    y_pred_proba = np.maximum(y_pred_proba, Config.MIN_PROB)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    # Métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    # Classification report
    target_names = ['1 (Home)', 'X (Draw)', '2 (Away)']
    class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Distribución de predicciones
    pred_dist = {
        '1': int(np.sum(y_pred == 0)),
        'X': int(np.sum(y_pred == 1)),
        '2': int(np.sum(y_pred == 2))
    }

    # Distribución real
    real_dist = {
        '1': int(np.sum(y_test == 0)),
        'X': int(np.sum(y_test == 1)),
        '2': int(np.sum(y_test == 2))
    }

    # Logging
    logger.success(f"  - Accuracy: {accuracy:.3f}")
    logger.info(f"  - Log Loss: {logloss:.3f}")
    logger.info(f"  - Distribución real: 1={real_dist['1']}, X={real_dist['X']}, 2={real_dist['2']}")
    logger.info(f"  - Distribución predicha: 1={pred_dist['1']}, X={pred_dist['X']}, 2={pred_dist['2']}")

    # Precision/Recall por clase
    logger.info("  - Métricas por clase:")
    for class_name in target_names:
        precision = class_report[class_name]['precision']
        recall = class_report[class_name]['recall']
        f1 = class_report[class_name]['f1-score']
        logger.info(f"    {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # Feature importance
    feature_names = [
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

    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1][:15]

    logger.info("  - Top 15 features más importantes:")
    for i, idx in enumerate(indices, 1):
        logger.info(f"    {i}. {feature_names[idx]}: {feature_importances[idx]:.4f}")

    # Preparar dict de métricas
    metrics = {
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'classification_report': class_report,
        'predictions_distribution': pred_dist,
        'real_distribution': real_dist,
        'test_size': len(y_test),
        'feature_importances': {
            feature_names[i]: float(feature_importances[i])
            for i in range(len(feature_importances))
        },
        'best_iteration': int(model.best_iteration),
        'best_score': float(model.best_score)
    }

    return metrics


def save_xgboost_model(model, metrics, league):
    """
    Guarda modelo XGBoost y métricas

    Args:
        model: Modelo entrenado
        metrics: Dict con métricas
        league: Nombre de la liga
    """
    import json

    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    model_path = f"{Config.MODELS_DIR}/model_xgboost_{league}.pkl"
    metrics_path = f"{Config.MODELS_DIR}/training_metrics_xgboost_{league}.json"

    # Guardar modelo
    logger.info(f"  - Guardando modelo en {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Añadir metadata
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['league'] = league
    metrics['model_type'] = 'XGBoost'
    metrics['config'] = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': model.max_depth,
        'learning_rate': model.learning_rate,
        'n_estimators': model.n_estimators,
        'subsample': model.subsample,
        'colsample_bytree': model.colsample_bytree,
        'min_child_weight': model.min_child_weight,
        'gamma': model.gamma,
        'min_prob': Config.MIN_PROB,
        'test_season': Config.TEST_SEASON
    }

    # Guardar métricas
    logger.info(f"  - Guardando métricas en {metrics_path}...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.success("  - Modelo y métricas guardados exitosamente")


def train_league_xgboost(league_id, league_name):
    """
    Pipeline de entrenamiento XGBoost para una liga

    Args:
        league_id: ID de la liga
        league_name: Nombre de la liga
    """
    logger.info("=" * 60)
    logger.info(f"ENTRENANDO XGBOOST PARA: {league_name} (ID: {league_id})")
    logger.info("=" * 60)

    # 1. Cargar datos
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

    # Split validation del train (últimos 20% de train)
    val_split_idx = int(len(train_df) * 0.8)
    train_df_final = train_df.iloc[:val_split_idx]
    val_df = train_df.iloc[val_split_idx:]

    logger.info(f"Train: {len(train_df_final)} partidos")
    logger.info(f"Validation: {len(val_df)} partidos")
    logger.info(f"Test: {len(test_df)} partidos")

    # 4. Preparar features
    league_name_normalized = league_name.lower().replace(" ", "_")
    X_train, y_train = prepare_features_xgb(train_df_final, league_name_normalized)
    X_val, y_val = prepare_features_xgb(val_df, league_name_normalized)
    X_test, y_test = prepare_features_xgb(test_df, league_name_normalized)

    # 5. Entrenar
    model = train_xgboost_model(X_train, y_train, X_val, y_val, league_name_normalized)

    # 6. Evaluar en test
    metrics = evaluate_xgboost(model, X_test, y_test)

    # 7. Guardar
    save_xgboost_model(model, metrics, league_name_normalized)
    logger.success(f"Modelo XGBoost {league_name} guardado exitosamente")

    return metrics


def main():
    """Pipeline completo - entrena XGBoost por liga"""

    logger.info("=" * 60)
    logger.info("INICIO ENTRENAMIENTO XGBOOST 1X2")
    logger.info("=" * 60)

    all_metrics = {}

    # Entrenar cada liga
    for league_id, league_name in Config.LEAGUES.items():
        try:
            metrics = train_league_xgboost(league_id, league_name)
            all_metrics[league_name] = metrics
        except Exception as e:
            logger.error(f"Error entrenando {league_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("=" * 60)
    logger.success("ENTRENAMIENTO XGBOOST COMPLETADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
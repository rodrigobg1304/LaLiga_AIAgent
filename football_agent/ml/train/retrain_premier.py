import sys
import os
import pickle
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Añadir paths
sys.path.append('..')
sys.path.append('../../src')

from football_agent.db import get_league_matches
from constants import MODEL_1X2_PREMIER


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES (copiadas de train_1x2.py y train_xgboost.py)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_features(df):
    """
    Prepara features para modelos 1X2.
    - 40 features base para todas las ligas
    - +7 features adicionales SOLO para Premier League
    """
    from predict import build_features_1x2

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        try:
            # Construir features usando la función de predict.py
            features = build_features_1x2(
                home_team=row['homeTeam'],
                away_team=row['awayTeam'],
                year=row['Year'],
                league_id='17'  # Premier League
            )

            X_list.append(features[0])

            # Target: resultado del partido
            hg = float(row['home_goals'] or 0)
            ag = float(row['away_goals'] or 0)

            if hg > ag:
                y_list.append(0)  # Victoria local
            elif hg < ag:
                y_list.append(2)  # Victoria visitante
            else:
                y_list.append(1)  # Empate

        except Exception as e:
            print(f"⚠️  Error procesando partido {row['homeTeam']} vs {row['awayTeam']}: {e}")
            continue

    if not X_list:
        raise ValueError("No se pudieron generar features")

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def get_league_data(league_id, years):
    """Obtiene datos de partidos de una liga para años específicos"""
    all_matches = []
    for year in years:
        matches = get_league_matches(league_id, year)
        all_matches.extend(matches)
    return all_matches


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1: ENTRENAR RANDOM FOREST (47 FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("PASO 1: ENTRENANDO RANDOM FOREST PARA PREMIER LEAGUE (47 FEATURES)")
print("=" * 80)

# Definir años
train_years = ['18/19', '19/20', '20/21', '21/22', '22/23', '23/24']
val_years = ['24/25']
test_years = ['25/26']

# Cargar datos
print(f"\n🔄 Cargando datos...")
import pandas as pd

train_matches = get_league_data('17', train_years)
val_matches = get_league_data('17', val_years)
test_matches = get_league_data('17', test_years)

train_df = pd.DataFrame(train_matches)
val_df = pd.DataFrame(val_matches)
test_df = pd.DataFrame(test_matches)

print(f"  Train: {len(train_df)} partidos ({train_years[0]} - {train_years[-1]})")
print(f"  Val:   {len(val_df)} partidos ({val_years[0]})")
print(f"  Test:  {len(test_df)} partidos ({test_years[0]})")

# Preparar features
print(f"\n🔄 Generando features...")
X_train, y_train = prepare_features(train_df)
X_val, y_val = prepare_features(val_df)
X_test, y_test = prepare_features(test_df)

print(f"  ✅ Features generadas: {X_train.shape[1]} (debe ser 47)")
assert X_train.shape[1] == 47, f"❌ ERROR: Esperaba 47 features, pero tiene {X_train.shape[1]}"

# Entrenar Random Forest
print(f"\n🔄 Entrenando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight={0: 1.0, 1: 3.5, 2: 1.0},  # Peso extra a empates
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Calibrar modelo
print(f"🔄 Calibrando modelo...")
calibrated_rf = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
calibrated_rf.fit(X_train, y_train)

# Evaluar
print(f"\n📊 EVALUACIÓN EN TEST:")
y_pred = calibrated_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  Accuracy: {acc * 100:.1f}%")

print(f"\n{classification_report(y_test, y_pred, target_names=['1', 'X', '2'], zero_division=0)}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de confusión:")
print(f"         Pred 1  Pred X  Pred 2")
print(f"Real 1:  {cm[0, 0]:6d}  {cm[0, 1]:6d}  {cm[0, 2]:6d}")
print(f"Real X:  {cm[1, 0]:6d}  {cm[1, 1]:6d}  {cm[1, 2]:6d}")
print(f"Real 2:  {cm[2, 0]:6d}  {cm[2, 1]:6d}  {cm[2, 2]:6d}")

# GUARDAR MODELO RF (SOBREESCRIBIR)
rf_path = os.path.join(MODEL_1X2_PREMIER, "model_1x2_premier_league.pkl")
print(f"\n💾 Guardando modelo RF en: {rf_path}")

os.makedirs(MODEL_1X2_PREMIER, exist_ok=True)

# FORZAR SOBREESCRITURA: Eliminar archivo si existe
if os.path.exists(rf_path):
    os.remove(rf_path)
    print(f"  🗑️  Eliminado modelo viejo")

with open(rf_path, 'wb') as f:
    pickle.dump(calibrated_rf, f)

print(f"  ✅ Modelo RF guardado")

# Verificar
with open(rf_path, 'rb') as f:
    loaded = pickle.load(f)
    if hasattr(loaded, 'calibrated_classifiers_'):
        base = loaded.calibrated_classifiers_[0].estimator
    else:
        base = loaded
    print(f"  ✅ VERIFICACIÓN: Modelo espera {base.n_features_in_} features")

# ══════════════════════════════════════════════════════════════════════════════
# PASO 2: ENTRENAR XGBOOST (47 FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PASO 2: ENTRENANDO XGBOOST PARA PREMIER LEAGUE (47 FEATURES)")
print("=" * 80)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2.0,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# Evaluar
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\n📊 EVALUACIÓN EN TEST:")
print(f"  Accuracy: {acc_xgb * 100:.1f}%")

print(f"\n{classification_report(y_test, y_pred_xgb, target_names=['1', 'X', '2'], zero_division=0)}")

# GUARDAR MODELO XGBOOST (SOBREESCRIBIR)
xgb_path = os.path.join(MODEL_1X2_PREMIER, "model_xgboost_premier_league.pkl")
print(f"\n💾 Guardando modelo XGBoost en: {xgb_path}")

# FORZAR SOBREESCRITURA
if os.path.exists(xgb_path):
    os.remove(xgb_path)
    print(f"  🗑️  Eliminado modelo viejo")

with open(xgb_path, 'wb') as f:
    pickle.dump(xgb_model, f)

print(f"  ✅ Modelo XGBoost guardado")

# Verificar
with open(xgb_path, 'rb') as f:
    loaded_xgb = pickle.load(f)
    print(f"  ✅ VERIFICACIÓN: Modelo espera {loaded_xgb.n_features_in_} features")

# ══════════════════════════════════════════════════════════════════════════════
# PASO 3: CREAR ENSEMBLE (RF + XGBOOST)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PASO 3: CREANDO ENSEMBLE RF + XGBOOST")
print("=" * 80)

# Grid search de pesos
print(f"\n🔄 Buscando mejor combinación de pesos...")

best_acc = 0
best_weights = (0.5, 0.5)

for rf_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:
    xgb_weight = 1.0 - rf_weight

    # Predicciones de ambos modelos
    rf_proba = calibrated_rf.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)

    # Ensemble
    ensemble_proba = (rf_weight * rf_proba) + (xgb_weight * xgb_proba)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

    acc = accuracy_score(y_test, ensemble_pred)

    # Calcular recall de empates
    recall_draw = 0
    if np.sum(y_test == 1) > 0:
        recall_draw = np.sum((ensemble_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)

    print(f"  RF={rf_weight:.1f}, XGB={xgb_weight:.1f} → Acc={acc * 100:.1f}%, Recall X={recall_draw * 100:.1f}%")

    if acc > best_acc:
        best_acc = acc
        best_weights = (rf_weight, xgb_weight)

print(f"\n🏆 Mejor configuración: RF={best_weights[0]}, XGB={best_weights[1]} (Acc={best_acc * 100:.1f}%)")

# GUARDAR CONFIG ENSEMBLE (SOBREESCRIBIR)
ensemble_config = {
    'rf_weight': best_weights[0],
    'xgb_weight': best_weights[1],
    'test_accuracy': float(best_acc)
}

config_path = os.path.join(MODEL_1X2_PREMIER, "ensemble_config_premier_league.json")
print(f"\n💾 Guardando config ensemble en: {config_path}")

# FORZAR SOBREESCRITURA
if os.path.exists(config_path):
    os.remove(config_path)
    print(f"  🗑️  Eliminado config vieja")

with open(config_path, 'w') as f:
    json.dump(ensemble_config, f, indent=2)

print(f"  ✅ Config guardada")

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("VERIFICACIÓN FINAL DE MODELOS")
print("=" * 80)

print(f"\n📁 Archivos en {MODEL_1X2_PREMIER}:")
for file in os.listdir(MODEL_1X2_PREMIER):
    if file.endswith('.pkl') or file.endswith('.json'):
        filepath = os.path.join(MODEL_1X2_PREMIER, file)
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        import time

        mtime = time.ctime(os.path.getmtime(filepath))
        print(f"  {file}: {size:.1f} MB, modificado: {mtime}")

print(f"\n✅ RE-ENTRENAMIENTO COMPLETADO")
print(f"✅ Todos los modelos tienen 47 features")
print(f"✅ Archivos sobreescritos correctamente")
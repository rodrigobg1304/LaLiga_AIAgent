import pickle

# Cargar el modelo RF de Premier
model_path = "/Users/rodrigobenitogarcia/PycharmProjects/LaLiga_AIAgent/football_agent/ml/models/1x2/production/premier_league/model_1x2_premier_league.pkl"

with open(model_path, 'rb') as f:
    rf_model = pickle.load(f)

# El RF está calibrado, así que accedemos al estimador base
base_estimator = rf_model.calibrated_classifiers_[0].estimator

# Número de features que espera
n_features = base_estimator.n_features_in_

print(f"✅ Modelo RF de Premier League espera: {n_features} features")

if n_features == 47:
    print("✅ CORRECTO: Tiene las 47 features (40 base + 7 Premier)")
elif n_features == 40:
    print("❌ INCORRECTO: Solo tiene 40 features base")
    print("   → Necesitas re-entrenar con train_1x2.py")
# training

Scripts de entrenamiento de los modelos ML y scheduler de reentrenamiento automatico. Los modelos generados se guardan en `models/` (raíz del proyecto) y son consumidos por `services/prediction/`.

---

## Requisitos

| Elemento | Version |
|---|---|
| Python | `3.12` (imagen Docker) / `>=3.10` local |
| scikit-learn | `>=1.3` |
| xgboost | `>=2.0` |
| numpy | `>=1.24` |
| pandas | `>=2.0` |
| tqdm | `>=4.0` |
| football-core | instalado desde `../football-core` |

---

## Estructura

```
training/
├── Dockerfile
├── requirements.txt
├── retrain_scheduler.py       # Scheduler automatico de reentrenamiento
└── train/
    ├── train_1x2.py           # Modelo de resultado (1/X/2) — Random Forest
    ├── train_xgboost.py       # Modelo de resultado — XGBoost
    ├── ensemble_premier.py    # Ensemble RF+XGBoost para Premier League
    ├── retrain_premier.py     # Reentrenamiento especifico Premier League
    ├── train_over_under_goals.py    # Over/Under goles
    ├── train_over_under_saves.py    # Over/Under paradas de portero
    └── train_over_under_corners.py  # Over/Under corners
```

---

## Modelos entrenados

### `train_1x2.py` — Resultado del partido (LaLiga / Serie A)

Entrena un `RandomForestClassifier` con calibracion de probabilidades (`CalibratedClassifierCV`) para predecir el resultado final: **1** (local), **X** (empate), **2** (visitante).

**Features utilizadas** (definidas en `football_core.constants.FEATURES`):
- Goals, Ball possession, Total shots, Shots on target
- Goalkeeper saves, Big chances, Accurate passes
- Tackles won, Interceptions, Blocked shots
- Elo rating (home/away), forma reciente, H2H, win rates ponderados

**Ligas:** LaLiga (`8`), Serie A (`23`)
**Output:** modelos `.pkl` en `models/1x2/production/{liga}/`

---

### `train_xgboost.py` — Resultado del partido (XGBoost)

Alternativa XGBoost al modelo RF. Mismas features. Se usa como componente del ensemble de Premier League.

---

### `ensemble_premier.py` — Ensemble para Premier League

Combina las predicciones del RF y del XGBoost mediante un meta-clasificador logístico. Genera el modelo de produccion para Premier League.

**Output:** `models/1x2/production/premier_league/`

---

### `retrain_premier.py` — Reentrenamiento de Premier League

Script independiente para reentrenar únicamente el modelo de Premier League sin tocar las otras ligas.

---

### `train_over_under_goals.py` — Over/Under goles

Entrena modelos binarios (uno por umbral) para predecir si el total de goles del partido supera cada umbral: `0.5`, `1.5`, `2.5`, `3.5`.

**Output:** `models/over_under/goals/production/{liga}/`

---

### `train_over_under_saves.py` — Over/Under paradas

Predice si el total de paradas de portero supera cada umbral: `0.5` a `6.5`.

**Output:** `models/over_under/saves/production/{liga}/`

---

### `train_over_under_corners.py` — Over/Under corners

Predice si el total de corners supera cada umbral: `2.5` a `9.5`.

**Output:** `models/over_under/corners/production/{liga}/`

---

## Scheduler de reentrenamiento automatico

`retrain_scheduler.py` se ejecuta en segundo plano y monitoriza la base de datos. Si detecta nuevos partidos (cambio en el hash del estado de la BD), lanza el reentrenamiento automaticamente.

**Intervalo de comprobacion:** cada 7 dias (configurable con `CHECK_INTERVAL`)

**Como funciona:**
1. Consulta el numero total de partidos, el ultimo `matchId` y la ultima temporada.
2. Genera un hash MD5 del estado.
3. Si el hash difiere del guardado en `models/last_state.pkl`, lanza el script de entrenamiento en un subproceso.
4. Actualiza el estado guardado si el reentrenamiento fue exitoso.

**Log:** `retrain.log` en el directorio de trabajo.

---

## Ejecucion manual

### Entrenar todos los modelos desde cero

```bash
cd training

# Resultado 1X2 - LaLiga y Serie A
python train/train_1x2.py

# Resultado 1X2 - XGBoost
python train/train_xgboost.py

# Ensemble Premier League
python train/ensemble_premier.py

# Over/Under goles
python train/train_over_under_goals.py

# Over/Under paradas
python train/train_over_under_saves.py

# Over/Under corners
python train/train_over_under_corners.py
```

### Lanzar el scheduler

```bash
python training/retrain_scheduler.py
```

---

## Ejecucion con Docker

El Dockerfile instala `football-core` y copia los scripts. Por defecto ejecuta el scheduler:

```bash
docker build -t football-training -f training/Dockerfile .
docker run --env-file .env football-training

# Para ejecutar un script concreto en lugar del scheduler:
docker run --env-file .env football-training python train/train_1x2.py
```

---

## Variables de entorno necesarias

Las mismas que `football-core` para la conexion a la BD, mas:

| Variable | Descripcion | Por defecto |
|---|---|---|
| `MODELS_DIR` | Directorio donde se guardan los modelos | `./models` |
| `DB_HOST` | Host MySQL | — |
| `DB_PORT` | Puerto MySQL | `3306` |
| `DB_DATABASE` | Base de datos | — |
| `DB_USER` | Usuario | — |
| `DB_PASSWORD` | Contraseña | — |
| `DB_TABLE` | Tabla de estadísticas | — |

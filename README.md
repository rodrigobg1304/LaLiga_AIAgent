# Football AI Agent

Sistema de predicción y análisis de fútbol basado en Machine Learning e Inteligencia Artificial, con arquitectura de microservicios desplegable mediante Docker.

---

## Estructura del proyecto

```
LaLiga_AIAgent/
│
├── football-core/                        # Paquete compartido instalable (pip install -e)
│   ├── pyproject.toml
│   └── src/football_core/
│       ├── db.py                         # Conexión y queries a MySQL
│       ├── constants.py                  # Constantes globales (thresholds, rutas de modelos)
│       ├── config.py                     # Opciones de liga para la UI
│       └── feature_engineering.py        # Construcción de features ML compartida
│
├── services/                             # Microservicios Docker
│   ├── prediction/                       # Servicio de predicción ML  →  :8001
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── predict.py                    # API FastAPI con modelos RF + XGBoost
│   │
│   ├── agent/                            # Servicio de agente conversacional  →  :8002
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── agent_api.py                  # API FastAPI que orquesta CrewAI
│   │   ├── src/football_agent/           # Paquete CrewAI (crew, tools, config)
│   │   └── knowledge/                    # Ficheros de conocimiento para el agente
│   │
│   └── streamlit/                        # Interfaz web  →  :8501
│       ├── Dockerfile
│       ├── requirements.txt
│       └── streamlit_app.py              # Dashboard interactivo
│
├── training/                             # Scripts de entrenamiento (batch/CronJob)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── retrain_scheduler.py              # Scheduler de reentrenamiento automático
│   └── train/
│       ├── train_1x2.py                  # Modelo resultado (1X2) — Random Forest
│       ├── train_xgboost.py              # Modelo resultado — XGBoost (Premier)
│       ├── ensemble_premier.py           # Ensemble RF + XGBoost para Premier League
│       ├── retrain_premier.py            # Reentrenamiento incremental Premier
│       ├── train_over_under_goals.py     # Modelo Over/Under goles
│       ├── train_over_under_saves.py     # Modelo Over/Under paradas de portero
│       └── train_over_under_corners.py   # Modelo Over/Under córners
│
├── models/                               # Artefactos ML entrenados (.pkl) — montado como volumen Docker
│   ├── 1x2/production/                   # Modelos de resultado por liga
│   └── over_under/                       # Modelos Over/Under (goals, saves, corners)
│
├── football_agent/                       # Entorno de desarrollo local del agente CrewAI
│   ├── pyproject.toml
│   └── src/football_agent/              # Paquete CrewAI (crew, tools, config)
│
├── scripts/                              # Utilidades y análisis exploratorio
│   ├── eda_analysis.py
│   ├── explore_stats.py
│   └── create_indexes.py
│
├── docker-compose.yml                    # Orquestación local de todos los servicios
└── football_agent/.env                   # Variables de entorno (no subir a git)
```

---

## Arquitectura de servicios

```
                    ┌─────────────────┐
                    │  Usuario / Web  │
                    └────────┬────────┘
                             │ :8501
                    ┌────────▼────────┐
                    │  streamlit-ui   │  Dashboard interactivo
                    │   (Streamlit)   │  Visualización, selectores de liga/equipo
                    └────────┬────────┘
                             │ HTTP
            ┌────────────────┴────────────────┐
            │ :8001                            │ :8002
   ┌────────▼────────┐               ┌────────▼────────┐
   │prediction-service│              │  agent-service   │
   │   (FastAPI ML)   │              │ (FastAPI + CrewAI)│
   │                  │              │                  │
   │ · Modelos 1X2    │              │ · Agente LLM     │
   │ · Over/Under     │              │ · Herramientas   │
   │   Goals/Saves    │              │   de consulta BD │
   │   Corners        │              │ · Claude AI      │
   └────────┬─────────┘              └────────┬─────────┘
            │                                 │
            └────────────┬────────────────────┘
                         │
                ┌────────▼────────┐
                │   MySQL (host)  │  Base de datos histórica
                │  RegularLeagues │  LaLiga · Premier · Serie A
                └─────────────────┘
```

### Descripción de cada servicio

| Servicio | Puerto | Tecnología | Función |
|---|---|---|---|
| `prediction-service` | 8001 | FastAPI + scikit-learn + XGBoost | Recibe un partido (local, visitante, temporada, liga) y devuelve probabilidades 1X2, Over/Under goles, paradas y córners |
| `agent-service` | 8002 | FastAPI + CrewAI + Claude | Agente conversacional que responde preguntas sobre estadísticas, clasificaciones y resultados consultando la BD |
| `streamlit-ui` | 8501 | Streamlit | Dashboard web con selector de liga, predicción de partidos, visualización de probabilidades y cuotas |
| `training` | — | scikit-learn + XGBoost | Scripts de entrenamiento y reentrenamiento de modelos (se ejecutan como batch jobs o CronJobs en K8s) |

---

## Paquete compartido `football-core`

Todos los servicios comparten el mismo paquete instalable `football-core`, lo que elimina duplicación de código y dependencias circulares.

| Módulo | Contenido |
|---|---|
| `db.py` | Conexión MySQL, todas las queries (partidos, clasificaciones, estadísticas) |
| `constants.py` | Thresholds de modelos, configuración de ligas, rutas de modelos |
| `config.py` | Opciones de liga para selectboxes de la UI |
| `feature_engineering.py` | Construcción de features ML: ELO, forma, H2H, Over rates |

---

## Modelos ML

### Resultado del partido (1X2)
- **LaLiga / Serie A**: Random Forest calibrado con isotonic regression
- **Premier League**: Ensemble RF + XGBoost con blend dinámico basado en diferencia ELO

### Over/Under
- **Goles**: 4 modelos independientes (Over 0.5, 1.5, 2.5, 3.5)
- **Paradas de portero**: 7 modelos (Over 0.5 → 6.5), con feature `is_home`
- **Córners**: 8 modelos (Over 2.5 → 9.5), con feature `is_home`

### Features principales (40 base + 7 exclusivas Premier)
Sistema ELO histórico · Forma reciente · Head-to-head · Estadísticas de temporada · Clasificación · Vecinos más cercanos en tabla

---

## Puesta en marcha con Docker

### Requisitos
- Docker Desktop con al menos 4 CPUs y 6 GB RAM asignados
- Base de datos MySQL levantada en el host
- Fichero `football_agent/.env` configurado

### Variables de entorno (`football_agent/.env`)
```env
DB_HOST=host.docker.internal
DB_PORT=3306
DB_USER=tu_usuario
DB_PASSWORD=tu_password
DB_DATABASE=RegularLeagues
DB_TABLE=Leagues
ANTHROPIC_API_KEY=tu_api_key
MODEL=claude-haiku-4-5-20251001
```

### Comandos

```bash
# Clonar y entrar al repositorio
git clone <url-del-repo>
cd LaLiga_AIAgent

# Primera vez (construye las imágenes)
docker-compose up --build

# Siguientes veces
docker-compose up

# Parar los servicios
docker-compose down
```

Una vez levantado, accede a la web en: **http://localhost:8501**

---

## Desarrollo local (sin Docker)

```bash
# Instalar el paquete compartido en modo editable
pip install -e football-core/

# Lanzar el servidor de predicción
cd services/prediction
MODELS_DIR=../../models uvicorn predict:app --port 8001 --reload

# Lanzar el agente
cd services/agent
uvicorn agent_api:app --port 8002 --reload

# Lanzar la UI
cd services/streamlit
streamlit run streamlit_app.py
```

### Entrenamiento de modelos

```bash
cd training/train

# Entrenar modelos 1X2 para todas las ligas
python train_1x2.py --leagues all

# Entrenar Over/Under goles
python train_over_under_goals.py --leagues all

# Entrenar Over/Under paradas
python train_over_under_saves.py --leagues all

# Entrenar Over/Under córners
python train_over_under_corners.py --leagues all
```

---

## Ligas soportadas

| Liga | ID | Modelos disponibles |
|---|---|---|
| LaLiga | 8 | 1X2 · Goals · Saves · Corners |
| Premier League | 17 | 1X2 (Ensemble) · Goals · Saves · Corners |
| Serie A | 23 | 1X2 · Goals · Saves · Corners |

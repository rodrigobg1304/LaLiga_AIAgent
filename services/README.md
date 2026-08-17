# services

Microservicios que componen el sistema en produccion. Se despliegan con Docker Compose desde la raiz del proyecto. Todos dependen de `football-core` como libreria compartida.

---

## Arquitectura

```
                    ┌──────────────────┐
                    │  streamlit-ui    │  :8501
                    │  (Streamlit)     │
                    └────────┬─────────┘
                             │ HTTP
              ┌──────────────┴─────────────┐
              │                            │
   ┌──────────▼──────────┐   ┌─────────────▼─────────┐
   │  prediction-service │   │    agent-service      │
   │  (FastAPI)  :8001   │   │    (FastAPI)  :8002   │
   └─────────────────────┘   └───────────────────────┘
              │                            │
              └──────────────┬─────────────┘
                             │
                    ┌────────▼─────────┐
                    │   MySQL (BD)     │
                    └──────────────────┘
```

---

## Despliegue

```bash
# Desde la raiz del proyecto
docker compose up --build
```

Variables de entorno necesarias en `.env` (raíz del proyecto):

```env
ANTHROPIC_API_KEY=sk-ant-...
MODEL=claude-haiku-4-5-20251001
DB_HOST=...
DB_PORT=3306
DB_DATABASE=...
DB_USER=...
DB_PASSWORD=...
DB_TABLE=...
```

---

## Servicios

### `prediction/` — Servicio de prediccion ML

**Puerto:** `8001`
**Imagen base:** `python:3.12-slim`

API REST que carga los modelos ML entrenados y devuelve predicciones para un partido dado.

**Dependencias:**

| Paquete | Version |
|---|---|
| fastapi | `==0.111.0` |
| uvicorn | `==0.29.0` |
| scikit-learn | `>=1.3` |
| xgboost | `>=2.0` |
| numpy | `>=1.24` |
| pandas | `>=2.0` |

**Endpoint principal:**

```
POST /predict
{
  "home_team": "real-madrid",
  "away_team": "barcelona",
  "league_id": "8",
  "season": "25/26"
}
```

Responde con probabilidades para:
- **1/X/2**: resultado del partido (RF para LaLiga/Serie A, Ensemble para Premier League)
- **Over/Under goles**: umbrales 0.5, 1.5, 2.5, 3.5
- **Over/Under paradas (saves)**: umbrales 0.5 a 6.5
- **Over/Under corners**: umbrales 2.5 a 9.5

Los modelos se montan desde `models/` en la raíz del proyecto (volumen de solo lectura). La ruta es configurable con `MODELS_DIR`.

---

### `agent/` — Servicio del agente conversacional

**Puerto:** `8002`
**Imagen base:** `python:3.12-slim` (hereda la cadena de instalacion de `football-core`)

Expone el agente CrewAI como API REST. El agente recibe una pregunta en lenguaje natural, llama a las herramientas necesarias (queries a BD o prediccion ML) y devuelve una respuesta estructurada en español.

**Dependencias:**

| Paquete | Version |
|---|---|
| fastapi | `>=0.111.0` |
| uvicorn | `>=0.31.1` |
| crewai[tools] | `==1.9.3` |

**Endpoint:**

```
POST /agent
{
  "query": "¿Cuántos goles ha marcado el Real Madrid esta temporada?"
}
```

El agente (`football_analyst`) dispone de las herramientas:
- `team_results_tool` — resultados de partidos
- `goals_tool` — goles marcados/encajados
- `stat_tool` — estadística concreta
- `standings_tool` — clasificacion
- `prediction_tool` — prediccion de partido (llama al `prediction-service`)

Requiere `OPENAI_API_KEY` y `PREDICTION_URL=http://prediction-service:8001` en el entorno.

---

### `streamlit/` — Interfaz web

**Puerto:** `8501`
**Imagen base:** `python:3.12-slim`

Dashboard interactivo construido con Streamlit. Permite:
- Ver la clasificacion de LaLiga, Premier League y Serie A
- Consultar resultados y estadísticas por equipo y temporada
- Lanzar predicciones de partidos con probabilidades y odds
- Chatear con el agente conversacional desde la misma interfaz

**Dependencias:**

| Paquete | Version |
|---|---|
| streamlit | `>=1.30` |
| pandas | `>=2.0` |
| plotly | `>=5.0` |
| requests | `>=2.31` |

Requiere `PREDICTION_URL` y `AGENT_URL` para conectar con los otros servicios.

---

## Variables de entorno por servicio

| Variable | `prediction` | `agent` | `streamlit` |
|---|---|---|---|
| `DB_HOST` / `DB_USER` / ... | Si | Si | Si |
| `MODELS_DIR` | Si | — | — |
| `OPENAI_API_KEY` | — | Si | — |
| `PREDICTION_URL` | — | Si | Si |
| `AGENT_URL` | — | — | Si |
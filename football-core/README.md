# football-core

Librería Python compartida entre todos los servicios y scripts del proyecto. Centraliza el acceso a la base de datos, las constantes globales, la configuración de ligas y la ingeniería de features para los modelos ML.

---

## Requisitos

| Elemento | Versión |
|---|---|
| Python | `>=3.10, <3.14` |
| mysql-connector-python | `>=8.0` |

---

## Instalación

Se instala como paquete local con `pip`. Todos los Dockerfiles del proyecto lo instalan así:

```bash
pip install ./football-core
```

Para desarrollo local:

```bash
pip install -e ./football-core
```

---

## Estructura

```
football-core/
├── pyproject.toml
└── src/football_core/
    ├── __init__.py
    ├── db.py                  # Capa de acceso a MySQL
    ├── constants.py           # Constantes globales del sistema
    ├── config.py              # Opciones de liga para la UI
    └── feature_engineering.py # Construccion de features para ML
```

---

## Modulos

### `db.py` — Acceso a base de datos

Gestiona la conexión a MySQL y expone todas las queries del sistema como funciones tipadas.

**Configuracion de conexion** (prioridad: variables de entorno > `mainconfig_secret.ini`):

| Variable de entorno | Descripcion |
|---|---|
| `DB_HOST` | Host del servidor MySQL |
| `DB_PORT` | Puerto (por defecto `3306`) |
| `DB_DATABASE` / `DB_NAME` | Nombre de la base de datos |
| `DB_USER` | Usuario |
| `DB_PASSWORD` / `DB_PASS` | Contraseña |
| `DB_TABLE` | Tabla principal de estadísticas |

**Funciones principales:**

| Funcion | Descripcion |
|---|---|
| `get_teams_by_league(league_id, season)` | Equipos de una liga/temporada |
| `get_team_results(team, year, top_n)` | Resultados de un equipo |
| `get_standings(leagueId, year)` | Clasificacion de una liga |
| `get_team_stats(team, stat_name, year)` | Estadística concreta por partido |
| `get_all_matches_chronological(league_id)` | Todos los partidos ordenados |
| `get_league_all_stats(league_id)` | Stats completas por partido para entrenamiento |
| `get_matches_with_saves(league_id, years)` | Partidos con datos de paradas |
| `get_matches_with_corners(league_id, years)` | Partidos con datos de corners |
| `get_h2h_matches(team1, team2, n)` | Historial de enfrentamientos directos |
| `get_team_multiple_stats_average(team, role, years, n)` | Media de 10 stats en bloque |
| `get_team_win_rates(team, role, league_id, years, n)` | Ratios de victorias/empates/derrotas |

---

### `constants.py` — Constantes globales

Centraliza todos los parámetros del sistema para evitar duplicacion entre scripts de entrenamiento y prediccion.

**Ligas soportadas:**

| ID | Liga | País |
|---|---|---|
| `8` | LaLiga | España |
| `17` | Premier League | Inglaterra |
| `23` | Serie A | Italia |

**Temporadas:**
- Entrenamiento: `19/20` → `24/25`
- Test / Produccion: `25/26`

**Parametros de modelo:**

| Constante | Valor | Descripcion |
|---|---|---|
| `RECENT_WEIGHT` | `0.65` | Peso de la forma reciente en stats generales |
| `HISTORICAL_WEIGHT` | `0.35` | Peso del histórico completo |
| `WIN_RATE_RECENT_WEIGHT` | `0.35` | Peso reciente para win rates (mas volatiles) |
| `ELO_K` | `40` | Factor K del sistema Elo |
| `ELO_INITIAL` | `1500` | Rating Elo inicial para equipos nuevos |
| `ELO_SEASON_REGRESSION` | `0.75` | Regresion a la media entre temporadas |
| `FORM_WINDOW` | `10` | Últimos N partidos para forma reciente |
| `H2H_LOOKBACK` | `5` | Últimos N enfrentamientos H2H |
| `RANDOM_STATE` | `42` | Seed para reproducibilidad |

**Umbrales Over/Under:**

| Mercado | Umbrales |
|---|---|
| Goles | `0.5, 1.5, 2.5, 3.5` |
| Paradas (saves) | `0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5` |
| Corners | `2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5` |

**Ruta de modelos:**
Configurable mediante la variable de entorno `MODELS_DIR`. Si no se define, apunta a la raíz del proyecto (donde vive `models/`).

---

### `config.py` — Opciones de liga

Expone `get_league_options()`: devuelve un diccionario `{nombre: id}` con las ligas presentes en la base de datos. Usado por la UI de Streamlit para poblar selectboxes.

---

### `feature_engineering.py` — Features ML

Contiene `build_features_1x2()` y `warm_cache()`. Construye el vector de features para prediccion de resultado (1/X/2) a partir de los datos historicos: Elo, forma reciente, H2H, stats promediadas y win rates ponderados.

---

## Variables de entorno relevantes

| Variable | Descripcion | Por defecto |
|---|---|---|
| `DB_HOST` | Host MySQL | — |
| `DB_PORT` | Puerto MySQL | `3306` |
| `DB_DATABASE` | Nombre BD | — |
| `DB_USER` | Usuario BD | — |
| `DB_PASSWORD` | Contraseña BD | — |
| `DB_TABLE` | Tabla de estadísticas | — |
| `MODELS_DIR` | Directorio de modelos | raíz del proyecto (`models/`) |
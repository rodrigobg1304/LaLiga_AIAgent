# football_agent

Agente conversacional de analisis de futbol construido con [CrewAI](https://crewai.com). Acepta preguntas en lenguaje natural en español y responde con datos reales extraidos de la base de datos MySQL y predicciones de los modelos ML. No usa conocimiento interno del LLM: todas las respuestas pasan por herramientas que consultan datos reales.

En produccion este paquete vive dentro de `services/agent/` y se expone como API REST. Este directorio es el entorno de **desarrollo y pruebas local**.

---

## Requisitos

| Elemento | Version |
|---|---|
| Python | `>=3.10, <3.14` |
| crewai\[tools\] | `==1.9.3` |
| uv | ultima version estable |
| MySQL | base de datos accesible con los datos de partidos |
| prediction-service | `services/prediction/` corriendo en `:8001` (para predicciones) |

---

## Instalacion

Este proyecto usa [uv](https://docs.astral.sh/uv/) para la gestion del entorno y dependencias.

```bash
# Instalar uv si no esta disponible
pip install uv

# Entrar al directorio y resolver dependencias
cd football_agent
crewai install
```

El entorno virtual se crea automaticamente en `.venv/` con Python 3.12. El fichero `uv.lock` garantiza reproducibilidad.

---

## Configuracion

Antes de ejecutar, crear el fichero `.env` dentro de `football_agent/`:

```env
# LLM
OPENAI_API_KEY=sk-...

# Base de datos MySQL
DB_HOST=localhost
DB_PORT=3306
DB_DATABASE=nombre_bd
DB_USER=usuario
DB_PASSWORD=contraseña
DB_TABLE=nombre_tabla

# Servicio de prediccion (para la herramienta prediction_tool)
PREDICTION_URL=http://localhost:8001
```

Alternativamente, las credenciales de BD se pueden poner en `src/football_agent/mainconfig_secret.ini` (las variables de entorno tienen prioridad):

```ini
[MySQL]
host     = localhost
port     = 3306
database = nombre_bd
user     = usuario
password = contraseña
table    = nombre_tabla
```

> **Importante:** `mainconfig_secret.ini` esta en `.gitignore`. No committear credenciales.

---

## Ejecucion

### Modo interactivo (CLI)

```bash
cd football_agent
crewai run
```

Lanza un bucle interactivo donde se puede escribir cualquier pregunta en español:

```
⚽ Pregunta: ¿Cuántos goles ha marcado el Real Madrid esta temporada?
⚽ Pregunta: Dame la clasificacion de LaLiga en la temporada 24/25
⚽ Pregunta: ¿Quién tiene más posesion media, Barcelona o Atletico?
⚽ Pregunta: Predice el partido Real Madrid vs Barcelona
⚽ Pregunta: salir
```

Escribir `salir`, `exit` o `q` para cerrar.

### Comandos disponibles

Los scripts del `pyproject.toml` se pueden lanzar con `uv run` o `crewai`:

| Comando | Descripcion |
|---|---|
| `crewai run` | Modo interactivo (bucle de preguntas) |
| `crewai train` | Entrena el crew con ejemplos |
| `crewai replay` | Repite la ultima ejecucion guardada |
| `crewai test` | Ejecuta tests del crew |

---

## Estructura

```
football_agent/
├── pyproject.toml                     # Dependencias y scripts del paquete
├── uv.lock                            # Lock file de dependencias
├── .env                               # Variables de entorno (no commitear)
├── knowledge/
│   └── user_preference.txt            # Contexto del usuario para el agente
└── src/football_agent/
    ├── crew.py                        # Definicion del Crew, agentes y tareas
    ├── main.py                        # Punto de entrada del CLI (bucle interactivo)
    ├── config.py                      # Configuracion auxiliar de liga
    ├── mainconfig_secret.ini          # Credenciales BD alternativas (no commitear)
    ├── config/
    │   ├── agents.yaml                # Rol, objetivo y backstory del agente
    │   └── tasks.yaml                 # Definicion de la tarea principal
    └── tools/
        ├── __init__.py
        └── custom_tool.py             # Plantilla para añadir herramientas nuevas
```

---

## Como funciona CrewAI

El flujo sigue el patron **Crew → Agent → Task → Tools**:

```
Usuario escribe pregunta
        │
        ▼
  Crew.kickoff({"query": "..."})
        │
        ▼
  Task: football_query_task
  (instruccion: usar tools, no responder de memoria)
        │
        ▼
  Agent: football_analyst
  (LLM decide qué tool llamar y con qué parametros)
        │
        ├── team_results_tool(team, year, limit)
        ├── goals_tool(team, year)
        ├── stat_tool(team, stat, year)
        ├── standings_tool(league, year)
        └── prediction_tool(home_team, away_team, year)
                │
                ▼ (esta tool llama al prediction-service via HTTP)
        ◄── Datos de BD / prediccion ML
        │
        ▼
  Respuesta estructurada en español (markdown)
```

---

## El agente: `football_analyst`

Definido en `config/agents.yaml`.

- **Rol:** Football Data Analyst
- **Objetivo:** Responder preguntas sobre rendimiento de equipos consultando la base de datos
- **Comportamiento:** Responde siempre en español con datos precisos. Nunca inventa estadísticas. Si no puede obtener datos de las tools, lo indica explicitamente.
- **Delegacion:** Desactivada (`allow_delegation=False`). Un unico agente maneja todo.

---

## Herramientas disponibles

Definidas en `services/agent/src/football_agent/tools/tools.py` (la version de produccion es identica en comportamiento).

### `team_results_tool` — Resultados de partidos

Devuelve los partidos de un equipo con resultado, puntos obtenidos y jornada.

| Parametro | Tipo | Descripcion |
|---|---|---|
| `team` | `str` | Nombre del equipo en slug (ver formato abajo) |
| `year` | `str` | Temporada: `'25/26'`, `'24/25'`, ... (opcional) |
| `limit` | `int` | Maximo de partidos a devolver (opcional) |

Ejemplo de respuesta: lista de dicts con `Round`, `homeTeam`, `awayTeam`, `home_goals`, `away_goals`, `result`, `points`.

---

### `goals_tool` — Goles por temporada

Devuelve el resumen de goles marcados y encajados de un equipo, desglosado por temporada y por condicion (local/visitante).

| Parametro | Tipo | Descripcion |
|---|---|---|
| `team` | `str` | Nombre del equipo en slug |
| `year` | `str` | Temporada (opcional) |

---

### `stat_tool` — Estadística concreta

Devuelve el valor de una estadística especifica por partido.

| Parametro | Tipo | Descripcion |
|---|---|---|
| `team` | `str` | Nombre del equipo en slug |
| `stat` | `str` | Nombre de la estadística (ver lista abajo) |
| `year` | `str` | Temporada (opcional) |

**Estadísticas disponibles** (nombres en inglés tal como estan en la BD):

| Estadística | Descripcion |
|---|---|
| `Goals` | Goles |
| `Ball possession` | Posesion de balon (%) |
| `Total shots` | Tiros totales |
| `Shots on target` | Tiros a puerta |
| `Goalkeeper saves` | Paradas del portero |
| `Big chances` | Ocasiones claras |
| `Accurate passes` | Pases precisos |
| `Tackles won` | Entradas ganadas |
| `Interceptions` | Intercepciones |
| `Blocked shots` | Tiros bloqueados |
| `Corner kicks` | Saques de esquina |

---

### `standings_tool` — Clasificacion

Devuelve la tabla de clasificacion de una liga en una temporada determinada.

| Parametro | Tipo | Descripcion |
|---|---|---|
| `league` | `str` | ID de liga: `'8'` (LaLiga), `'17'` (Premier), `'23'` (Serie A) |
| `year` | `str` | Temporada: `'25/26'`, `'24/25'`, ... |

---

### `prediction_tool` — Prediccion de partido

Llama al `prediction-service` (`localhost:8001`) y devuelve probabilidades de resultado y mercados over/under para un partido concreto.

| Parametro | Tipo | Descripcion |
|---|---|---|
| `home_team` | `str` | Equipo local en slug |
| `away_team` | `str` | Equipo visitante en slug |
| `year` | `str` | Temporada (opcional, usa la actual por defecto) |

Respuesta incluye:
- Probabilidades **1/X/2** (local gana / empate / visitante gana)
- Over/Under **goles** (umbrales 0.5, 1.5, 2.5, 3.5)
- Over/Under **paradas** (umbrales 0.5 a 6.5)
- Over/Under **corners** (umbrales 2.5 a 9.5)

> Si el `prediction-service` no esta corriendo, la herramienta devuelve un mensaje de error sin romper el agente.

---

## Formato de nombres de equipo (slug)

Los equipos se pasan siempre en formato slug (minusculas, guiones en lugar de espacios):

| Nombre real | Slug |
|---|---|
| Real Madrid | `real-madrid` |
| FC Barcelona | `barcelona` |
| Atletico de Madrid | `atletico-madrid` |
| Real Betis | `real-betis` |
| Manchester City | `manchester-city` |
| Inter Milan | `inter` |

El LLM realiza esta conversion automaticamente a partir del nombre natural que escribe el usuario.

---

## Formato de temporadas

Las temporadas siguen el formato `'AA/AA'`:

| Temporada | Formato |
|---|---|
| 2024-2025 | `'24/25'` |
| 2025-2026 | `'25/26'` |
| 2023-2024 | `'23/24'` |

Si no se especifica temporada, el agente selecciona la mas reciente disponible.

---

## Anadir una herramienta nueva

`tools/custom_tool.py` contiene una plantilla lista para extender:

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class MiHerramientaInput(BaseModel):
    equipo: str = Field(..., description="Nombre del equipo en slug")

class MiHerramienta(BaseTool):
    name: str = "Nombre de la herramienta"
    description: str = "Descripcion clara para que el agente sepa cuándo usarla"
    args_schema = MiHerramientaInput

    def _run(self, equipo: str) -> str:
        # Logica aqui
        return resultado
```

Despues registrarla en `crew.py`:

```python
from football_agent.tools.custom_tool import MiHerramienta

@agent
def football_analyst(self) -> Agent:
    return Agent(
        config=self.agents_config["football_analyst"],
        tools=[..., MiHerramienta()],
        ...
    )
```

---

## Relacion con el resto del proyecto

```
football_agent/          ← desarrollo y pruebas local (este directorio)
     │
     │  mismo codigo adaptado
     ▼
services/agent/          ← version de produccion, expuesta como API REST en :8002
     │
     │  llama a
     ▼
services/prediction/     ← predicciones ML en :8001

services/streamlit/      ← UI que consume ambos servicios y permite chatear con el agente
```

Para desplegar todo junto con Docker ver `services/README.md` y el `docker-compose.yml` de la raiz.

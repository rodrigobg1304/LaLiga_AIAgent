# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Football match prediction system for LaLiga, Premier League, and Serie A. Microservices architecture with ML inference, a CrewAI conversational agent, and a Streamlit dashboard.

## Running the Project

### Docker (recommended)
```bash
docker-compose up --build   # first time
docker-compose up           # subsequent runs
docker-compose down         # stop
```

Services:
- Streamlit UI: http://localhost:8501
- Prediction API: http://localhost:8001
- Agent API: http://localhost:8002

### Local Development (no Docker)
```bash
# Install shared library first (required by all services)
pip install -e football-core/

# Run each service in a separate terminal:
cd services/prediction && MODELS_DIR=../../models uvicorn predict:app --port 8001 --reload
cd services/agent && uvicorn agent_api:app --port 8002 --reload
cd services/streamlit && streamlit run streamlit_app.py
```

### Weekly Data Collection (Sofascore → MySQL)
```bash
cd scripts
pip install -r requirements.txt

cd leagues
# Find the season ID for a league
python collect_leagues.py --league 8 --list-seasons

# Collect a specific round
python collect_leagues.py --league 8 --season 77559 --round 27

# Collect multiple leagues at once (same round)
python collect_leagues.py --league 8 17 23 --season 77559 76986 76457 --round 27

# Backfill a full season
python collect_leagues.py --league 8 --season 77559 --round-start 1 --round-end 38
```

Sofascore imposes rate limits — the client adds a 10-second delay between match requests automatically.
Known issue (2025-11): daily blocking has been observed; if requests fail, verify the API is accessible.
Uses `INSERT IGNORE` so re-running is always safe (duplicates skipped via PRIMARY KEY).

### Training Models
```bash
cd training
pip install -r requirements.txt
python train/train_1x2.py          # Random Forest 1X2
python train/train_xgboost.py      # XGBoost 1X2
python train/train_over_under_goals.py
python train/train_over_under_saves.py
python train/train_over_under_corners.py
python retrain_scheduler.py        # Auto-retrains every 7 days
```

## Required Environment Variables

```
DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_DATABASE, DB_TABLE
ANTHROPIC_API_KEY
MODEL=claude-haiku-4-5-20251001
MODELS_DIR          # defaults to ./models
PREDICTION_URL      # URL of prediction service (for agent)
AGENT_URL           # URL of agent service (for streamlit)
```

## Architecture

```
streamlit-ui (:8501)
    ├── → prediction-service (:8001)  [ML inference via FastAPI]
    └── → agent-service (:8002)       [CrewAI + Claude via FastAPI]
                └── → prediction-service (:8001)
                └── → MySQL (host DB)

All services share: football-core (pip package)
```

### Key Components

**`football-core/`** — Shared pip-installable library used by all services:
- `db.py` — All MySQL queries (15+ functions for matches, stats, standings)
- `feature_engineering.py` — Builds ML feature vectors from DB data
- `constants.py` — Thresholds, ELO config, league IDs (LaLiga=8, Premier=17, SerieA=23)
- `config.py` — League selector options for UI

**`services/prediction/predict.py`** — FastAPI app that loads `.pkl` models from `MODELS_DIR` and returns match outcome probabilities. Models: 1X2 (RF + Ensemble) and Over/Under (goals, saves, corners) per league.

**`services/agent/agent_api.py`** — FastAPI + CrewAI agent (`FootballAnalyst`). Has tools for querying match history, stats, standings, and calling the prediction service. Responds in Spanish. **Pending improvement:** agent responses need to be more concise (current behavior is too verbose).

**`services/streamlit/streamlit_app.py`** — Dashboard with standings tables, match results, prediction interface (with betting odds), and chat with the agent. Custom warm color scheme (cream/orange/dark blue).

**`training/`** — Scripts to train models per league. Output `.pkl` files go to `models/{1x2,over_under}/{production,experiments}/`. Validation metrics: accuracy, F1-score, AUC (standard classification metrics). Only models that pass validation go to `/production`. `/experiments` holds older lower-precision models kept for historical reference only — they are not used in production and have no roadmap for reactivation.

### Supported Leagues and Models

| League | ID | 1X2 Model | Over/Under |
|--------|-----|-----------|------------|
| LaLiga | 8 | Random Forest | Goals, Saves, Corners |
| Premier League | 17 | Ensemble (RF+XGBoost) | Goals, Saves, Corners |
| Serie A | 23 | Random Forest | Goals, Saves, Corners |
| Qualy WC Europe | 11 | Random Forest (qualy) | Goals, Saves, Corners |
| World Cup | 16 | Random Forest (qualy, shared) | Goals, Saves, Corners |

**Why Ensemble only for Premier League:** The RF alone left draws (X) severely underrepresented in Premier League predictions. The RF+XGBoost ensemble corrects this class imbalance. LaLiga and Serie A RF models already produce well-balanced class predictions on their own.

**Qualy/International model details:**
- Models live in `models/*/production/worldcup_europe/` with `_qualy` suffix.
- Trained with `train/train_qualy.py` on 856 matches (Qualy WC Europe + World Cup, 3 campaigns: 2018, 2022, 2026).
- 41 features: 40 standard base features + `is_qualifier` (1=qualifier round, 0=World Cup match).
- Seasons use plain year format ("2026") instead of the domestic "YY/YY" format.
- At prediction time, `get_recent_years` detects the plain year format ("2026") and generates international cycle years (e.g. ["2026", "2022"]), so team stats are queried correctly from the DB. ELO is computed from the global cache (includes all leagues).
- Trained with `--qualifier 11` so is_qualifier=1 for Qualy (11) and 0 for World Cup (16).

**Corners thresholds:** Range from 2.5 to 9.5 in steps of 1 (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5). Upper bound is 9.5 because it is extremely rare for a team (home or away) to take 10+ corners in a single match.

**Training data range:** Seasons 19/20 through 24/25. Current active season is 25/26 (test set). Convention: when a season ends, it moves to training and the next season becomes the new test set (e.g. next cycle: train 19/20→25/26, test 26/27).

**Planned expansions:** Additional domestic leagues, plus international tournaments (Champions League, World Cup, Euros, Copa América).

### Data Flow

1. Features are built in `football-core/feature_engineering.py` using historical match data from MySQL
2. Prediction service loads pre-trained `.pkl` models and applies feature vectors
3. Agent service uses CrewAI tools to query DB or call prediction service, then generates natural language responses via Claude
4. Streamlit calls prediction/agent APIs via HTTP and renders results

## Daily Automation (WC 2026 / active tournaments)

Two launchd agents run daily (replacing cron — launchd catches up if Mac was asleep):

| Time | Agent plist | Script | Log |
|------|-------------|--------|-----|
| 09:00 | `com.laliga.collect-retrain` | `scripts/collect_and_retrain.py` | `scripts/logs/nightly.log` |
| 10:00 | `com.laliga.daily-pipeline` | `scripts/daily_pipeline.py` | `scripts/logs/daily_pipeline.log` |

**`collect_and_retrain.py` (9:00):** Collects Sofascore stats for completed WC 2026 rounds that have scores in Matches but no stats in Leagues. Retrains qualy models only if new rows were added. Safe to re-run (INSERT IGNORE). WC season id = 58210, league id = 16.

**`daily_pipeline.py` (10:00):**
1. Refreshes fixtures with placeholder team names (e.g. `w73`, `2a`) from Sofascore.
2. Fetches last 36h results from DB, calls prediction API for each, shows ✅/❌ vs actual outcome.
3. Fetches next 24h matches, calls prediction API, builds Telegram message and sends it.

**Telegram:** `scripts/telegram_notifier.py`. Reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from env or `scripts/.env`. HTML parse mode. Sends one message per run.

**Scoreline prediction:** Poisson + Dixon-Coles (ρ=−0.10). λ estimated from O/U survival sum; split home/away by 1X2 weights. Best scoreline must be consistent with predicted 1X2 outcome and O/U goals minimum.

**Draw calibration in pipeline:** raw argmax rarely selects X. Override: if `px > 20%` AND `px > 0.65 × max(p1, p2)` → predict draw.

**Manage launchd agents:**
```bash
launchctl start com.laliga.collect-retrain   # force run now
launchctl start com.laliga.daily-pipeline    # force run now
launchctl list | grep laliga                 # check status
```
Plist files: `~/Library/LaunchAgents/com.laliga.collect-retrain.plist` and `com.laliga.daily-pipeline.plist`.

## Code Conventions

- **Function names and variable names in English**, even if comments or docstrings are in Spanish (PEP8 style).
- DB updates for WC 2026 are automated daily via `collect_and_retrain.py`. Domestic leagues still updated manually.

## Critical Rules

- **Never modify models in `/production` without explicit user approval.** Always ask before touching any `.pkl` file under `models/{1x2,over_under}/production/`.

## No Tests or Linting Configured

The project has no automated test suite and no linting configuration (no pytest, flake8, black, ruff, mypy, or pre-commit hooks).

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

## Daily Automation

Four launchd agents (replacing cron — launchd catches up if Mac was asleep, but NOT if the Mac was fully shut down or in deep sleep past `standbydelaylow`; if runs are missed for several days in a row, check `pmset -g | grep powernap` — Power Nap is off by default, so scheduled jobs won't fire while the lid is closed):

| Time | Agent plist | Script | Log | Status |
|------|-------------|--------|-----|--------|
| always on | `com.laliga.prediction-service` | `services/prediction/predict.py` (uvicorn, port 8001) | `scripts/logs/prediction_service.log` | active |
| always on | `com.laliga.crew` | `crew/main.py` (internal APScheduler, see below) | `crew/logs/agent_runs.log` | active |
| 09:00 | `com.laliga.collect-retrain` | `scripts/collect_and_retrain.py` | `scripts/logs/nightly.log` | active |
| 10:00 | `com.laliga.daily-pipeline` | `scripts/daily_pipeline.py` | `scripts/logs/daily_pipeline.log` | **disabled** (see below) |

**`com.laliga.prediction-service` (always on):** Runs the FastAPI prediction service locally (not via Docker) with `RunAtLoad` + `KeepAlive`, so it auto-starts at login and restarts if it crashes. `daily_pipeline.py` calls it at `localhost:8001` (`PREDICTION_URL` in `scripts/.env`) — if this agent isn't loaded, predictions silently fail with "Sin predicción disponible" / HTTP errors in `daily_pipeline.log`, even though the rest of the pipeline runs fine. Uses the `football-agent` conda env (has fastapi/uvicorn/sklearn/xgboost + `football-core` installed editable). Env vars (`DB_HOST=localhost`, `DB_TABLE=Leagues`, etc.) are baked into the plist directly, not read from a `.env` file.

**`com.laliga.crew` (always on):** Runs `crew/main.py`, a separate APScheduler-based process (not launchd `StartCalendarInterval`) with two independent internal jobs: domestic leagues (LaLiga/Premier/Serie A) every Tue/Fri at 08:00, active only within the 1 Aug – 1 Jun season window; World Cup 2026 daily at 09:00, active only 11 Jun – 20 Jul 2026 (now outside that window, so it's a no-op until the next tournament). `python crew/main.py --run-now` / `--run-worldcup-now` trigger a run manually.

**`collect_and_retrain.py` (9:00):** Collects Sofascore stats for completed rounds that have scores in `Matches` but no stats in `Leagues`, for **LaLiga (8), Premier League (17), Serie A (23) — season 26/27** (season ids `97268`/`96668`/`95836`) plus the World Cup (league 16, season id 58210, kept for the next tournament cycle). Domestic collection uses `scripts/leagues/collect_leagues.py`; WC uses `scripts/tournaments/collect_tournaments.py`. Safe to re-run (INSERT IGNORE).
- **WC:** auto-retrains the qualy models if new stat rows were added, then syncs to `worldcup_all`.
- **Domestic auto-retrain (2026-09-11):** `discover_next_round_fixtures()` (called for all three domestic leagues, not just LaLiga — a fix on the same date: it used to only run as a side effect of the LaLiga-only Telegram step, so Premier/Serie A never advanced past round 1) brings each new round's fixtures into `Matches` as soon as Sofascore publishes them, so `process_domestic_league()` picks up and stat-collects a finished round without waiting on anything else. Once a round becomes fully stat-collected in *any* domestic league, `maybe_trigger_retrain()` launches `scripts/retrain_and_promote.py` in the background (`subprocess.Popen`, detached) — it retrains 1X2 (RF + XGBoost + Premier ensemble) and Over/Under (goals, saves, corners) for **all three domestic leagues at once** (`train_1x2.py`/`train_xgboost.py` have no per-league filter) and copies the results straight into `models/{1x2,over_under}/production/`, then restarts `com.laliga.prediction-service` and sends a Telegram summary. Takes roughly 40-60 minutes (the O/U feature loop dominates); a lock in `scripts/state/retrain_state.json` (`_lock.pid`, cleared automatically once stale) stops a second run from overlapping. Per-league progress is tracked in the same file (`last_retrained_round`) so it only re-triggers when a league has genuinely moved past what it was last retrained on.
- Domestic season ids are hardcoded in `DOMESTIC_LEAGUES` in the script — bump them manually each August when a new season starts (`python scripts/leagues/collect_leagues.py --league <id> --list-seasons`).
- **Round notifications (LaLiga only):** `process_round_notifications()` drives `scripts/round_pipeline.py` to send the **entire round in one Telegram message** as soon as its earliest kickoff is within 24h (project decision, 2026-09-11 — previously announced per-match as each fixture's own 24h window arrived, fragmenting a round across several days), full predictions: winner %, goals O/U, saves, corners, scoreline — same format as the WC daily messages. Also sends a "summary" message (✅/❌ vs actual) once every match in the round has a final score. State (which rounds were already announced/summarized, per round not per match) is tracked in `scripts/state/round_notify_state.json` (gitignored) so re-runs don't duplicate messages. Premier League and Serie A fixtures/stats are still collected above but intentionally don't get Telegram messages — project decision to keep notifications LaLiga-only for now.

**`daily_pipeline.py` (10:00, currently disabled):**
1. Refreshes fixtures with placeholder team names (e.g. `w73`, `2a`) from Sofascore.
2. Fetches last 36h results from DB, calls prediction API for each, shows ✅/❌ vs actual outcome.
3. Fetches next 24h matches, calls prediction API, builds Telegram message and sends it.

**LaLiga (8), Premier League (17) and Serie A (23) are all excluded from steps 2 and 3** (`LeagueId NOT IN ('8','17','23')`) — project decision (2026-09-14): Telegram predictions must only ever come from LaLiga, and LaLiga already gets its own per-round announce/summary messages from `round_pipeline.py` (see above), so duplicating it here was never wanted either. That leaves this script covering only Qualy/World Cup leagues (11, 16, 1, 27) if it's ever re-enabled — Premier/Serie A data is still collected/trained on as usual, just never surfaced by this script. It was disabled on 2026-08-11 (World Cup ended, plist moved to `~/Library/LaunchAgents/disabled/`) to stop sending the Telegram message while the 26/27 domestic season data is still being set up. Re-enable once `collect_and_retrain.py` has populated `Matches`/`Leagues` for the new season:
```bash
mv ~/Library/LaunchAgents/disabled/com.laliga.daily-pipeline.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.laliga.daily-pipeline.plist
```

**Telegram:** `scripts/telegram_notifier.py`. Reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from env or `scripts/.env`. HTML parse mode. Sends one message per run.

**Scoreline prediction:** Poisson + Dixon-Coles (ρ=−0.10). λ estimated from O/U survival sum; split home/away by 1X2 weights. Best scoreline is the joint distribution's top cell consistent with the predicted 1X2 outcome (no separate goals-minimum floor — an earlier version forced total goals ≥ the highest O/U threshold crossing 50%, but that discarded genuinely more likely low-scoring picks like 1-0 in favor of a merely-consistent 2-0, which is why 0-0/1-0/0-1 and real blowouts never appeared; removed 2026-09-11).

**Draw calibration in pipeline:** raw argmax rarely selects X. Override: if `px > 20%` AND `px > 0.90 × max(p1, p2)` → predict draw. Applies across all leagues in the message (domestic + international).

**Manage launchd agents:**
```bash
launchctl start com.laliga.collect-retrain   # force run now
launchctl start com.laliga.daily-pipeline    # force run now (only works once re-enabled, see above)
launchctl list | grep laliga                 # check status (PID column: "-" = not running now, that's normal for the two scheduled jobs between runs)
curl -s localhost:8001/docs -o /dev/null -w "%{http_code}\n"   # verify prediction-service is up
```
Plist files: `~/Library/LaunchAgents/com.laliga.{collect-retrain,daily-pipeline,prediction-service,crew}.plist`. Disabled plists live in `~/Library/LaunchAgents/disabled/`.

## Code Conventions

- **Function names and variable names in English**, even if comments or docstrings are in Spanish (PEP8 style).
- Data collection (stats + scores) for LaLiga, Premier League, Serie A and the World Cup is automated daily via `collect_and_retrain.py` (see [Daily Automation](#daily-automation)). Domestic and WC qualy `/production` models both auto-retrain once a round/round-of-the-tournament finishes — see the auto-retrain bullet above.

## Critical Rules

- **Never modify models in `/production` without explicit user approval** — except the standing, user-authorized automation described under [Daily Automation](#daily-automation) (`retrain_and_promote.py`, triggered by `collect_and_retrain.py` when a round finishes). That one pipeline is pre-approved to retrain-and-promote on its own; nothing else is. Any other change to a `.pkl` under `models/{1x2,over_under}/production/` — ad hoc retraining, a manual edit, anything outside that specific script — still requires asking first.

## No Tests or Linting Configured

The project has no automated test suite and no linting configuration (no pytest, flake8, black, ruff, mypy, or pre-commit hooks).

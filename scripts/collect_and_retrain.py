"""
Nightly data collection + conditional retrain.

Runs at ~23:30 each day:
  1. Domestic leagues (LaLiga, Premier League, Serie A — season 26/27):
       Refreshes scores, collects stats for completed rounds. Does NOT
       retrain — domestic /production models are retrained manually
       (see training/train_*.py) per project convention.
  2. World Cup 2026 (league 16): same steps, plus auto-retrain of the
       qualy models if new stat rows were collected (kept from the WC
       cycle; will simply find nothing to do once the tournament data
       is no longer changing).
  3. LaLiga round notifications (Telegram): discovers the next round's
       fixtures, sends a one-off "announce" message (full predictions)
       the first time a round appears, and a "summary" message
       (✅/❌ vs actual) once every match in a round has a final score.
       LaLiga only, per project decision — Premier/Serie A stats are
       still collected above but don't get Telegram messages.
  4. Updates DailyPredictions results (winner_correct, ou_correct, etc.)
     across all leagues.

Schedule (add to crontab):
    30 23 * * * cd /path/to/scripts && python collect_and_retrain.py >> logs/nightly.log 2>&1

Required env vars: DB_*, PREDICTION_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
(same as daily_pipeline.py / round_pipeline.py)
"""
import os
import sys
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_connection, LEAGUES_TABLE, MATCHES_TABLE, update_prediction_results, insert_match_metadata

# ── Config ──────────────────────────────────────────────────────
TRAINING_DIR = Path(__file__).parent.parent / "training"
SCRIPTS_DIR  = Path(__file__).parent
NOTIFY_STATE_FILE = SCRIPTS_DIR / "state" / "round_notify_state.json"

# Domestic leagues: (league_id, season_id, Year label as stored in Leagues table, name)
DOMESTIC_LEAGUES = [
    (8,  97268, "26/27", "LaLiga"),
    (17, 96668, "26/27", "Premier League"),
    (23, 95836, "26/27", "Serie A"),
]

# Round notifications via Telegram: LaLiga only (see project decision above)
NOTIFY_LEAGUE_ID  = 8
NOTIFY_SEASON_ID  = 97268
NOTIFY_YEAR       = "26/27"

# World Cup (kept for the next tournament cycle; auto-retrains qualy models)
WC_LEAGUE_ID = 16
WC_SEASON_ID = 58210
WC_YEAR      = "2026"


def log(msg: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}")


# ── Step 0: Refresh scores for finished-but-unscored matches ────

def refresh_finished_match_scores(league_id: int, season_id: int) -> list[str]:
    """
    Finds rounds where matches should have finished (started >3h ago) but
    still have homeScore=NULL in the DB. Fetches current metadata from
    Sofascore and updates scores via insert_match_metadata (COALESCE guard
    prevents overwriting valid scores with NULL).
    Returns list of round IDs that got at least one score updated.
    """
    cutoff = datetime.now() - timedelta(hours=3)
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(f"""
                SELECT DISTINCT Round FROM {MATCHES_TABLE}
                WHERE LeagueId=%s AND SeasonId=%s
                  AND homeScore IS NULL
                  AND MatchDateLocal < %s
            """, (league_id, season_id, cutoff))
            rounds = [r["Round"] for r in cur.fetchall()]
    finally:
        conn.close()

    if not rounds:
        return []

    from sofascore_client import collect_round_fixtures

    refreshed = []
    for rnd in rounds:
        log(f"  Refreshing scores for round {rnd}...")
        try:
            fixtures = collect_round_fixtures(league_id, season_id, rnd)
            scored = [f for f in fixtures if f.get("homeScore") is not None]
            if scored:
                insert_match_metadata(fixtures)
                log(f"    → {len(scored)}/{len(fixtures)} matches with scores updated")
                refreshed.append(rnd)
            else:
                log(f"    → No completed matches found yet")
        except Exception as e:
            log(f"    → Error: {e}")

    return refreshed


# ── Step 1: Collect stats for completed rounds ───────────────────

def get_completed_rounds_in_matches(league_id: int, season_id: int) -> list[str]:
    """Returns rounds where at least one match is completed in Matches table."""
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(f"""
                SELECT DISTINCT Round FROM {MATCHES_TABLE}
                WHERE LeagueId=%s AND SeasonId=%s AND homeScore IS NOT NULL
            """, (league_id, season_id))
            return [r["Round"] for r in cur.fetchall()]
    finally:
        conn.close()


def count_stats_for_round(league_id: int, year: str, round_id: str) -> int:
    """Count unique matches with stats in Leagues table for this round."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(DISTINCT matchId) FROM {LEAGUES_TABLE}
                WHERE leagueId=%s AND Year=%s
                  AND CAST(Round AS SIGNED) = %s
            """, (league_id, year, round_id.split("/")[0]))
            return cur.fetchone()[0]
    finally:
        conn.close()


def count_completed_in_round(league_id: int, season_id: int, round_id: str) -> int:
    """Count completed matches in Matches table for this round."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) FROM {MATCHES_TABLE}
                WHERE LeagueId=%s AND SeasonId=%s AND Round=%s
                  AND homeScore IS NOT NULL
            """, (league_id, season_id, round_id))
            return cur.fetchone()[0]
    finally:
        conn.close()


def collect_round_qualifier(round_id: str) -> int:
    """Collect WC stats for a round via tournaments/collect_tournaments.py."""
    log(f"  Collecting WC round {round_id}...")
    result = subprocess.run(
        [sys.executable, "tournaments/collect_tournaments.py",
         "--type", "qualifier",
         "--league", str(WC_LEAGUE_ID),
         "--season", str(WC_SEASON_ID),
         "--round", round_id],
        cwd=str(SCRIPTS_DIR),
        capture_output=True, text=True, timeout=600
    )
    return _parse_collect_output(result)


def collect_round_domestic(league_id: int, season_id: int, round_id: str) -> int:
    """Collect domestic-league stats for a round via leagues/collect_leagues.py."""
    log(f"  Collecting round {round_id}...")
    result = subprocess.run(
        [sys.executable, "leagues/collect_leagues.py",
         "--league", str(league_id),
         "--season", str(season_id),
         "--round", str(round_id)],
        cwd=str(SCRIPTS_DIR),
        capture_output=True, text=True, timeout=600
    )
    return _parse_collect_output(result)


def _parse_collect_output(result: subprocess.CompletedProcess) -> int:
    for line in result.stdout.splitlines():
        if "new inserted" in line or "rows collected" in line:
            log(f"    {line.strip()}")
    if result.returncode != 0:
        log(f"    Error: {result.stderr[:200]}")
        return 0
    for line in result.stdout.splitlines():
        if "stat rows collected," in line:
            try:
                return int(line.strip().split(",")[1].split()[0])
            except Exception:
                pass
    return 0


# ── Step 2: Retrain (WC / qualy only) ─────────────────────────────

def retrain_qualy():
    log("  Retraining qualy models...")
    env = {**os.environ, "PYTHONPATH": str(TRAINING_DIR)}
    result = subprocess.run(
        [sys.executable, "train/train_qualy.py",
         "--league", "11", "16", "--qualifier", "11"],
        cwd=str(TRAINING_DIR),
        env=env, capture_output=True, text=True, timeout=600
    )
    for line in result.stdout.splitlines():
        if "accuracy" in line.lower() or "completado" in line.lower() or "ERROR" in line:
            log(f"    {line.strip()}")
    if result.returncode != 0:
        log(f"    Retrain error: {result.stderr[:300]}")
        return False

    # Sync to worldcup_all
    models_dir = TRAINING_DIR.parent / "models"
    src = models_dir
    for cat in ["1x2", "over_under/goals", "over_under/saves", "over_under/corners"]:
        s = src / cat / "production" / "qualy_worldcup_europe"
        d = src / cat / "production" / "worldcup_all"
        if s.exists() and d.exists():
            for f in s.glob("*.pkl"):
                (d / f.name).write_bytes(f.read_bytes())
            for f in s.glob("*.json"):
                (d / f.name).write_bytes(f.read_bytes())
    log("  Models synced to worldcup_all")
    return True


# ── Per-league processing ─────────────────────────────────────────

def process_domestic_league(league_id: int, season_id: int, year: str, name: str) -> int:
    log(f"-- {name} (season {year}) --")

    refreshed = refresh_finished_match_scores(league_id, season_id)
    if refreshed:
        log(f"  Scores refreshed for rounds: {refreshed}")

    completed_rounds = get_completed_rounds_in_matches(league_id, season_id)
    log(f"  Completed rounds in Matches table: {completed_rounds}")

    total_new = 0
    for rnd in completed_rounds:
        n_stats     = count_stats_for_round(league_id, year, rnd)
        n_completed = count_completed_in_round(league_id, season_id, rnd)
        if n_stats < n_completed:
            log(f"  Round {rnd}: {n_stats} stats vs {n_completed} completed → collecting")
            total_new += collect_round_domestic(league_id, season_id, rnd)
        else:
            log(f"  Round {rnd}: {n_stats} stats ok, skipping")

    log(f"  {name}: {total_new} new stat rows")
    if total_new > 0:
        log(f"  {name}: new data collected — retrain manually with training/train_*.py when ready")
    return total_new


def process_world_cup() -> int:
    log("-- World Cup 2026 --")

    refreshed = refresh_finished_match_scores(WC_LEAGUE_ID, WC_SEASON_ID)
    if refreshed:
        log(f"  Scores refreshed for rounds: {refreshed}")

    completed_rounds = get_completed_rounds_in_matches(WC_LEAGUE_ID, WC_SEASON_ID)
    log(f"  Completed rounds in Matches table: {completed_rounds}")

    total_new = 0
    for rnd in completed_rounds:
        n_stats     = count_stats_for_round(WC_LEAGUE_ID, WC_YEAR, rnd)
        n_completed = count_completed_in_round(WC_LEAGUE_ID, WC_SEASON_ID, rnd)
        if n_stats < n_completed:
            log(f"  Round {rnd}: {n_stats} stats vs {n_completed} completed → collecting")
            total_new += collect_round_qualifier(rnd)
        else:
            log(f"  Round {rnd}: {n_stats} stats ok, skipping")

    log(f"  World Cup: {total_new} new stat rows")
    if total_new > 0:
        log("  World Cup: new data found → retraining qualy models")
        ok = retrain_qualy()
        log("  Retrain completed successfully" if ok else "  Retrain failed — models unchanged")
    return total_new


# ── Step 3: Round notifications (Telegram, LaLiga only) ──────────

def _load_notify_state() -> dict:
    if NOTIFY_STATE_FILE.exists():
        return json.loads(NOTIFY_STATE_FILE.read_text())
    return {}


def _save_notify_state(state: dict):
    NOTIFY_STATE_FILE.parent.mkdir(exist_ok=True)
    NOTIFY_STATE_FILE.write_text(json.dumps(state, indent=2))


def process_round_notifications(league_id: int, season_id: int, year: str) -> None:
    """
    Discovers the next round's fixtures and sends the two Telegram
    lifecycle messages per round (announce once fixtures appear,
    summary once every match has a final score). Idempotent — tracked
    in scripts/state/round_notify_state.json so re-runs don't re-send.
    """
    from sofascore_client import collect_round_fixtures
    from round_pipeline import get_round_matches, build_and_send_announcement, build_and_send_summary

    lid = str(league_id)
    log(f"-- Round notifications: league {lid} --")

    state = _load_notify_state()
    league_state = state.setdefault(lid, {"announced": [], "summarized": []})

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT MAX(CAST(Round AS SIGNED)) FROM {MATCHES_TABLE}
                WHERE LeagueId=%s AND SeasonId=%s
            """, (league_id, season_id))
            row = cur.fetchone()
            max_round = row[0] if row and row[0] else 0
    finally:
        conn.close()

    # Try to pull in the next round's fixtures (no-op if Sofascore hasn't
    # published them yet — collect_round_fixtures returns an empty list).
    next_round = max_round + 1
    try:
        fixtures = collect_round_fixtures(league_id, season_id, next_round)
        if fixtures:
            insert_match_metadata(fixtures)
            log(f"  Round {next_round}: fixtures collected ({len(fixtures)} matches)")
    except Exception as e:
        log(f"  Round {next_round}: could not fetch fixtures yet ({e})")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT Round FROM {MATCHES_TABLE}
                WHERE LeagueId=%s AND SeasonId=%s
            """, (league_id, season_id))
            rounds = sorted((r[0] for r in cur.fetchall()), key=lambda x: int(x))
    finally:
        conn.close()

    for rnd in rounds:
        matches = get_round_matches(lid, season_id, rnd)
        if not matches:
            continue

        if rnd not in league_state["announced"]:
            log(f"  Round {rnd}: sending announcement")
            if build_and_send_announcement(lid, season_id, rnd, year):
                league_state["announced"].append(rnd)
                _save_notify_state(state)

        all_played = all(m["homeScore"] is not None for m in matches)
        if all_played and rnd not in league_state["summarized"]:
            log(f"  Round {rnd}: all matches played — sending summary")
            if build_and_send_summary(lid, season_id, rnd):
                league_state["summarized"].append(rnd)
                _save_notify_state(state)


# ── Main ─────────────────────────────────────────────────────────

def run():
    log("=== Nightly collect & retrain started ===")

    for league_id, season_id, year, name in DOMESTIC_LEAGUES:
        process_domestic_league(league_id, season_id, year, name)

    process_world_cup()

    try:
        process_round_notifications(NOTIFY_LEAGUE_ID, NOTIFY_SEASON_ID, NOTIFY_YEAR)
    except Exception as e:
        log(f"  Round notifications error: {e}")

    updated = update_prediction_results()
    log(f"  Prediction results updated: {updated} matches")

    log("=== Nightly run complete ===\n")


if __name__ == "__main__":
    run()

"""
Nightly data collection + conditional retrain.

Runs at ~23:30 each day:
  1. Collects stats for any WC 2026 round with completed-but-uncollected matches
  2. Updates DailyPredictions results (winner_correct, ou_correct, etc.)
  3. Retrains qualy models if new WC 2026 matches were added to Leagues table

Schedule (add to crontab):
    30 23 * * * cd /path/to/scripts && python collect_and_retrain.py >> logs/nightly.log 2>&1

Required env vars: DB_*, PREDICTION_URL (same as daily_pipeline.py)
"""
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_connection, LEAGUES_TABLE, MATCHES_TABLE, update_prediction_results, insert_match_metadata

# ── Config ──────────────────────────────────────────────────────
WC_LEAGUE_ID  = 16
WC_SEASON_ID  = 58210
TRAINING_DIR  = Path(__file__).parent.parent / "training"
SCRIPTS_DIR   = Path(__file__).parent


def log(msg: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}")


# ── Step 0: Refresh scores for finished-but-unscored matches ────

def refresh_finished_match_scores() -> list[str]:
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
            """, (WC_LEAGUE_ID, WC_SEASON_ID, cutoff))
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
            fixtures = collect_round_fixtures(WC_LEAGUE_ID, WC_SEASON_ID, rnd)
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

def get_completed_rounds_in_matches() -> list[str]:
    """Returns rounds where at least one match is completed in Matches table."""
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(f"""
                SELECT DISTINCT Round FROM {MATCHES_TABLE}
                WHERE LeagueId=%s AND SeasonId=%s AND homeScore IS NOT NULL
            """, (WC_LEAGUE_ID, WC_SEASON_ID))
            return [r["Round"] for r in cur.fetchall()]
    finally:
        conn.close()


def count_stats_for_round(round_id: str) -> int:
    """Count unique matches with stats in Leagues table for this round."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(DISTINCT matchId) FROM {LEAGUES_TABLE}
                WHERE leagueId=%s AND Year='2026'
                  AND CAST(Round AS SIGNED) = %s
            """, (WC_LEAGUE_ID, round_id.split("/")[0]))
            return cur.fetchone()[0]
    finally:
        conn.close()


def count_completed_in_round(round_id: str) -> int:
    """Count completed matches in Matches table for this round."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) FROM {MATCHES_TABLE}
                WHERE LeagueId=%s AND SeasonId=%s AND Round=%s
                  AND homeScore IS NOT NULL
            """, (WC_LEAGUE_ID, WC_SEASON_ID, round_id))
            return cur.fetchone()[0]
    finally:
        conn.close()


def collect_round(round_id: str) -> int:
    """Collect stats for a round via collect_tournaments.py. Returns new rows inserted."""
    log(f"  Collecting round {round_id}...")
    result = subprocess.run(
        [sys.executable, "tournaments/collect_tournaments.py",
         "--type", "qualifier",
         "--league", str(WC_LEAGUE_ID),
         "--season", str(WC_SEASON_ID),
         "--round", round_id],
        cwd=str(SCRIPTS_DIR),
        capture_output=True, text=True, timeout=600
    )
    for line in result.stdout.splitlines():
        if "new inserted" in line or "rows collected" in line:
            log(f"    {line.strip()}")
    if result.returncode != 0:
        log(f"    Error: {result.stderr[:200]}")
        return 0
    # Count new rows from output
    for line in result.stdout.splitlines():
        if "stat rows collected," in line:
            try:
                return int(line.strip().split(",")[1].split()[0])
            except Exception:
                pass
    return 0


# ── Step 2: Retrain if new data ──────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────

def run():
    log("=== Nightly collect & retrain started ===")

    # Step 0: Update scores for matches that have finished since last run
    refreshed = refresh_finished_match_scores()
    if refreshed:
        log(f"  Scores refreshed for rounds: {refreshed}")

    # Update prediction results in DB
    updated = update_prediction_results()
    log(f"  Prediction results updated: {updated} matches")

    # Check which completed rounds need stats collected
    completed_rounds = get_completed_rounds_in_matches()
    log(f"  Completed rounds in Matches table: {completed_rounds}")

    total_new = 0
    for rnd in completed_rounds:
        n_stats     = count_stats_for_round(rnd)
        n_completed = count_completed_in_round(rnd)
        if n_stats < n_completed:
            log(f"  Round {rnd}: {n_stats} stats vs {n_completed} completed → collecting")
            new = collect_round(rnd)
            total_new += new
        else:
            log(f"  Round {rnd}: {n_stats} stats ok, skipping")

    log(f"  Total new stat rows: {total_new}")

    if total_new > 0:
        log("  New data found → retraining")
        ok = retrain_qualy()
        if ok:
            log("  Retrain completed successfully")
        else:
            log("  Retrain failed — models unchanged")
    else:
        log("  No new data → skipping retrain")

    log("=== Nightly run complete ===\n")


if __name__ == "__main__":
    run()

"""
Nightly data collection + conditional retrain.

Runs at 09:00 each day (com.laliga.collect-retrain, launchd StartCalendarInterval —
NOT the "~23:30 crontab" this docstring used to claim, stale since the move off cron):
  1. Domestic leagues (LaLiga, Premier League, Serie A — season 26/27):
       Discovers the next round's fixtures (so a new round enters the
       Matches table before any of its games have finished — otherwise
       stats collection never sees it), refreshes scores, and collects
       stats for completed rounds. Does NOT retrain — domestic
       /production models are retrained manually (see training/train_*.py)
       per project convention.
  2. World Cup 2026 (league 16): same steps, plus auto-retrain of the
       qualy models if new stat rows were collected (kept from the WC
       cycle; will simply find nothing to do once the tournament data
       is no longer changing).
  3. LaLiga round notifications (Telegram): sends a one-off "announce"
       message (full predictions) the first time a round's fixtures are
       within 24h of kickoff, and a "summary" message (✅/❌ vs actual)
       once every match in a round has a final score. LaLiga only, per
       project decision — Premier/Serie A fixtures/stats are still
       collected in step 1 above but don't get Telegram messages.
  4. Updates DailyPredictions results (winner_correct, ou_correct, etc.)
     across all leagues.

Schedule: ~/Library/LaunchAgents/com.laliga.collect-retrain.plist (09:00 daily).
Force a run now: launchctl start com.laliga.collect-retrain

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


# ── Step 0a: Discover next round's fixtures ──────────────────────

def discover_next_round_fixtures(league_id: int, season_id: int) -> int:
    """
    Finds the next round (max known round + 1) and inserts its fixtures
    into Matches if Sofascore has published them yet (no-op — empty list —
    otherwise). Without this, a new round never enters the Matches table
    until something else happens to pull it in (kickoffs land as
    homeScore=NULL rows only once refresh_finished_match_scores/collection
    already know about the round), so completed-round detection and stats
    collection silently stall on it. Safe to re-run (insert_match_metadata
    uses INSERT IGNORE / COALESCE).
    Returns the number of fixtures found (0 if not published yet).
    """
    from sofascore_client import collect_round_fixtures

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

    next_round = max_round + 1
    try:
        fixtures = collect_round_fixtures(league_id, season_id, next_round)
    except Exception as e:
        log(f"  Round {next_round}: could not fetch fixtures yet ({e})")
        return 0

    if fixtures:
        insert_match_metadata(fixtures)
        log(f"  Round {next_round}: fixtures collected ({len(fixtures)} matches)")
    return len(fixtures)


# ── Step 0b: Refresh scores for finished-but-unscored matches ────

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

def process_domestic_league(league_id: int, season_id: int, year: str, name: str) -> tuple[int, str | None]:
    """Returns (new_stat_rows, latest_fully_stated_round_or_None)."""
    log(f"-- {name} (season {year}) --")

    discover_next_round_fixtures(league_id, season_id)

    refreshed = refresh_finished_match_scores(league_id, season_id)
    if refreshed:
        log(f"  Scores refreshed for rounds: {refreshed}")

    completed_rounds = get_completed_rounds_in_matches(league_id, season_id)
    log(f"  Completed rounds in Matches table: {completed_rounds}")

    total_new = 0
    fully_stated_rounds = []
    for rnd in completed_rounds:
        n_stats     = count_stats_for_round(league_id, year, rnd)
        n_completed = count_completed_in_round(league_id, season_id, rnd)
        if n_stats < n_completed:
            log(f"  Round {rnd}: {n_stats} stats vs {n_completed} completed → collecting")
            total_new += collect_round_domestic(league_id, season_id, rnd)
            n_stats = count_stats_for_round(league_id, year, rnd)
        else:
            log(f"  Round {rnd}: {n_stats} stats ok, skipping")
        if n_stats >= n_completed and n_completed > 0:
            fully_stated_rounds.append(rnd)

    log(f"  {name}: {total_new} new stat rows")
    latest = max(fully_stated_rounds, key=int) if fully_stated_rounds else None
    return total_new, latest


# ── Step 2b: Auto-retrain trigger (all domestic leagues, all-or-nothing) ──

RETRAIN_STATE_FILE = SCRIPTS_DIR / "state" / "retrain_state.json"


def _load_retrain_state() -> dict:
    if RETRAIN_STATE_FILE.exists():
        return json.loads(RETRAIN_STATE_FILE.read_text())
    return {}


def _save_retrain_state(state: dict):
    RETRAIN_STATE_FILE.parent.mkdir(exist_ok=True)
    RETRAIN_STATE_FILE.write_text(json.dumps(state, indent=2))


def maybe_trigger_retrain(latest_rounds: dict[str, str]):
    """
    Launches retrain_and_promote.py in the background once any domestic
    league's completed round has advanced past what it was last retrained
    on. Per project decision (2026-09-11) — see that script's docstring for
    the standing /production-approval exception this represents.

    `latest_rounds` is {league_id_str: latest_fully_stated_round_or_None}
    for all three domestic leagues (collected after processing each one),
    since training always covers all three at once (train_1x2.py /
    train_xgboost.py have no per-league filter) — so a single retrain run
    updates every league's "last retrained" marker together, rather than
    each league re-triggering the whole cycle separately for data the
    others already picked up.
    """
    state = _load_retrain_state()

    advanced = {
        lid: rnd for lid, rnd in latest_rounds.items()
        if rnd is not None and state.get(lid, {}).get("last_retrained_round") != rnd
    }
    if not advanced:
        return

    lock = state.get("_lock")
    if lock and lock.get("pid"):
        try:
            os.kill(lock["pid"], 0)
            running = True
        except (ProcessLookupError, PermissionError):
            running = False
        if running:
            log(f"  Auto-retrain: {advanced} advanced, but a run (pid={lock['pid']}, "
                f"triggered by {lock.get('trigger')}) is already in progress — deferring")
            return
        log(f"  Stale retrain lock (pid={lock['pid']} no longer running) — clearing")
        state.pop("_lock", None)
        _save_retrain_state(state)

    trigger_lid, trigger_round = next(iter(advanced.items()))
    trigger_name = dict((str(l), n) for l, _, _, n in DOMESTIC_LEAGUES).get(trigger_lid, trigger_lid)
    rounds_arg = ",".join(f"{lid}:{rnd}" for lid, rnd in latest_rounds.items() if rnd is not None)

    log(f"  Auto-retrain: {trigger_name} round {trigger_round} fully collected — "
        f"launching background retrain for all domestic leagues ({rounds_arg})")
    log_path = SCRIPTS_DIR / "logs" / "retrain_promote.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a") as logf:
        subprocess.Popen(
            [sys.executable, "retrain_and_promote.py",
             "--round", f"{trigger_lid}:{trigger_round}",
             "--rounds", rounds_arg],
            cwd=str(SCRIPTS_DIR), stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )


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
    Sends the two Telegram lifecycle messages per round (announce once a
    round's fixtures are within 24h of kickoff, summary once every match
    has a final score). Idempotent — tracked in
    scripts/state/round_notify_state.json so re-runs don't re-send.

    Announce state is tracked per ROUND, not per match: the whole round is
    sent in a single message as soon as its earliest kickoff is within 24h —
    project decision (2026-09-11) so a reader gets the full round's
    predictions at once instead of it trickling in over several days as
    each match's own 24h window arrives.
    """
    from round_pipeline import get_round_matches, build_and_send_announcement, build_and_send_summary

    lid = str(league_id)
    log(f"-- Round notifications: league {lid} --")

    state = _load_notify_state()
    league_state = state.setdefault(lid, {"announced": [], "summarized": []})

    # Fixtures for the next round are already discovered by
    # process_domestic_league() earlier in run(); calling it again here is
    # cheap (idempotent, INSERT IGNORE) and keeps this function safe to call
    # standalone.
    discover_next_round_fixtures(league_id, season_id)

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

    now = datetime.now()
    for rnd in rounds:
        if rnd in league_state["announced"]:
            continue

        matches = get_round_matches(lid, season_id, rnd)
        if not matches:
            continue

        kickoffs = [m["MatchDateLocal"] for m in matches if m["MatchDateLocal"] is not None]
        if not kickoffs:
            continue

        earliest = min(kickoffs)
        if earliest < now:
            # First kickoff already passed without ever being announced (e.g. a round
            # that predates this notification logic) — sending a "preview" for a match
            # that already happened makes no sense, so just mark it done silently.
            log(f"  Round {rnd}: first kickoff already passed without being announced — "
                f"skipping (marking done)")
            league_state["announced"].append(rnd)
            _save_notify_state(state)
        elif earliest - now <= timedelta(hours=24):
            log(f"  Round {rnd}: first kickoff within 24h — sending full round announcement "
                f"({len(matches)} partidos)")
            sent_ids = build_and_send_announcement(lid, season_id, rnd, year, matches=matches)
            if sent_ids:
                league_state["announced"].append(rnd)
                _save_notify_state(state)
        # else: >24h to the first kickoff — leave pending for a future run

        all_played = all(m["homeScore"] is not None for m in matches)
        if all_played and rnd not in league_state["summarized"]:
            log(f"  Round {rnd}: all matches played — sending summary")
            if build_and_send_summary(lid, season_id, rnd):
                league_state["summarized"].append(rnd)
                _save_notify_state(state)


# ── Main ─────────────────────────────────────────────────────────

def run():
    log("=== Nightly collect & retrain started ===")

    latest_rounds = {}
    for league_id, season_id, year, name in DOMESTIC_LEAGUES:
        _, latest = process_domestic_league(league_id, season_id, year, name)
        latest_rounds[str(league_id)] = latest
    maybe_trigger_retrain(latest_rounds)

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

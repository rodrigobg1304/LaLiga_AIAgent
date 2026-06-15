"""
Scheduled football data-collection pipeline.

Two independent scheduled jobs:
  • Domestic leagues (LaLiga, Premier League, Serie A):
      Every Tuesday and Friday at 08:00 (Europe/Madrid).
      Season window: 1 Aug → 1 Jun (skips Jun 2 – Jul 31).
  • World Cup 2026 (tournament ID 16, season 58210):
      Every day at 09:00 (Europe/Madrid).
      Active window: 11 Jun 2026 → 20 Jul 2026 only.

All output is logged to crew/logs/agent_runs.log with a timestamped header per run.

Usage:
    # Start the scheduler (blocks until Ctrl+C)
    python main.py

    # Trigger a domestic-leagues run immediately (useful for manual testing)
    python main.py --run-now

    # Trigger a World Cup run immediately
    python main.py --run-worldcup-now
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────

_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE, override=False)

# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "agent_runs.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Pre-flight check ──────────────────────────────────────────────────────────

def _sofascore_accessible(tournament_id: int = 8) -> bool:
    """
    Quick HTTP probe against Sofascore before spending LLM credits on the crew.
    Returns True only when the API responds with HTTP 200.

    tournament_id — Sofascore unique-tournament ID to probe:
        8  → LaLiga  (domestic leagues pre-flight)
        16 → World Cup (tournament pre-flight)
    """
    try:
        import tls_client
        session = tls_client.Session(client_identifier="chrome_120", random_tls_extension_order=True)
        resp = session.get(
            f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons"
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.warning(f"Sofascore pre-flight request failed (tournament {tournament_id}): {exc}")
        return False


# ── Season windows ────────────────────────────────────────────────────────────

def _in_domestic_season() -> bool:
    """True between 1 Aug and 1 Jun (inclusive) — the domestic football season."""
    from datetime import date
    today = date.today()
    m, d = today.month, today.day
    # Aug–Dec and Jan–May are always in-season; Jun 1 is the last allowed day.
    return m in range(8, 13) or m in range(1, 6) or (m == 6 and d == 1)


# ── Main job ──────────────────────────────────────────────────────────────────

def run_collection_crew() -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 60

    logger.info(separator)
    logger.info(f"RUN START [DOMESTIC]  {run_ts}")
    logger.info(separator)

    if not _in_domestic_season():
        logger.info("Outside domestic season window (Aug 1 – Jun 1). Skipping.")
        logger.info(f"RUN END    {run_ts}  STATUS=SKIPPED (off-season)")
        return

    # Pre-flight: avoid spinning up agents if Sofascore is blocked
    if not _sofascore_accessible():
        logger.warning(
            "Sofascore API is not accessible (possible daily blocking). "
            "Skipping this run — will retry at next scheduled time."
        )
        logger.info(f"RUN END    {run_ts}  STATUS=SKIPPED (Sofascore blocked)")
        return

    logger.info("Sofascore pre-flight check passed. Launching domestic collection crew.")

    try:
        # Lazy import so env vars are already loaded before crew.py is parsed
        from crew import build_crew

        collection_crew = build_crew()
        result = collection_crew.kickoff()

        logger.info("Domestic crew completed successfully.")
        logger.info(f"Crew output:\n{result}")
        logger.info(f"RUN END    {run_ts}  STATUS=COMPLETED")

    except Exception as exc:
        logger.error(f"Domestic crew run failed: {exc}", exc_info=True)
        logger.info(f"RUN END    {run_ts}  STATUS=FAILED")


def run_worldcup_crew() -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 60

    logger.info(separator)
    logger.info(f"RUN START [WORLD CUP]  {run_ts}")
    logger.info(separator)

    if not _sofascore_accessible(tournament_id=16):
        logger.warning(
            "Sofascore API is not accessible (tournament 16 / World Cup). "
            "Skipping World Cup run — will retry tomorrow at 09:00."
        )
        logger.info(f"RUN END    {run_ts}  STATUS=SKIPPED (Sofascore blocked)")
        return

    logger.info("Sofascore pre-flight check passed. Launching World Cup collection crew.")

    try:
        from worldcup_crew import build_worldcup_crew

        wc_crew = build_worldcup_crew()
        result = wc_crew.kickoff()

        logger.info("World Cup crew completed successfully.")
        logger.info(f"Crew output:\n{result}")
        logger.info(f"RUN END    {run_ts}  STATUS=COMPLETED")

    except Exception as exc:
        logger.error(f"World Cup crew run failed: {exc}", exc_info=True)
        logger.info(f"RUN END    {run_ts}  STATUS=FAILED")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Execute a single domestic-leagues run immediately, then exit.",
    )
    parser.add_argument(
        "--run-worldcup-now",
        action="store_true",
        help="Execute a single World Cup collection run immediately, then exit.",
    )
    args = parser.parse_args()

    if args.run_now:
        logger.info("Manual domestic run triggered via --run-now.")
        run_collection_crew()
        return

    if args.run_worldcup_now:
        logger.info("Manual World Cup run triggered via --run-worldcup-now.")
        run_worldcup_crew()
        return

    scheduler = BlockingScheduler(timezone="Europe/Madrid")

    scheduler.add_job(
        run_collection_crew,
        trigger=CronTrigger(day_of_week="tue,fri", hour=8, minute=0),
        id="football_data_collection",
        name="Football Data Collection (LaLiga + Premier League + Serie A)",
        misfire_grace_time=3600,
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        run_worldcup_crew,
        trigger=CronTrigger(
            hour=9, minute=0,
            start_date="2026-06-11 09:00:00",
            end_date="2026-07-20 23:59:59",
        ),
        id="worldcup_data_collection",
        name="World Cup 2026 Daily Data Collection",
        misfire_grace_time=3600,
        max_instances=1,
        coalesce=True,
    )

    logger.info("Football Data Collection Scheduler started.")
    logger.info("Schedule (domestic)  : Tue + Fri at 08:00 Europe/Madrid  [Aug 1 – Jun 1]")
    logger.info("Schedule (World Cup) : daily at 09:00 Europe/Madrid  [Jun 11 – Jul 20 2026]")
    logger.info("Log file : %s", _LOG_DIR / "agent_runs.log")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()

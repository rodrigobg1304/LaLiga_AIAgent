"""
Scheduled football data-collection pipeline.

Runs the CrewAI collection crew every Monday and Friday at 08:00 (Europe/Madrid).
All output is logged to crew/logs/agent_runs.log with a timestamped header per run.

Usage:
    # Start the scheduler (blocks until Ctrl+C)
    python main.py

    # Trigger a single run immediately (useful for manual testing)
    python main.py --run-now
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

def _sofascore_accessible() -> bool:
    """
    Quick HTTP probe against Sofascore before spending LLM credits on the crew.
    Returns True only when the API responds with HTTP 200.
    """
    try:
        from curl_cffi import requests as cffi_requests

        resp = cffi_requests.get(
            "https://api.sofascore.com/api/v1/unique-tournament/8/seasons",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=15,
            impersonate="chrome120",
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.warning(f"Sofascore pre-flight request failed: {exc}")
        return False


# ── Main job ──────────────────────────────────────────────────────────────────

def run_collection_crew() -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 60

    logger.info(separator)
    logger.info(f"RUN START  {run_ts}")
    logger.info(separator)

    # Pre-flight: avoid spinning up agents if Sofascore is blocked
    if not _sofascore_accessible():
        logger.warning(
            "Sofascore API is not accessible (possible daily blocking). "
            "Skipping this run — will retry at next scheduled time."
        )
        logger.info(f"RUN END    {run_ts}  STATUS=SKIPPED (Sofascore blocked)")
        return

    logger.info("Sofascore pre-flight check passed. Launching collection crew.")

    try:
        # Lazy import so env vars are already loaded before crew.py is parsed
        from crew import build_crew

        collection_crew = build_crew()
        result = collection_crew.kickoff()

        logger.info("Crew completed successfully.")
        logger.info(f"Crew output:\n{result}")
        logger.info(f"RUN END    {run_ts}  STATUS=COMPLETED")

    except Exception as exc:
        logger.error(f"Crew run failed: {exc}", exc_info=True)
        logger.info(f"RUN END    {run_ts}  STATUS=FAILED")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Execute a single collection run immediately, then exit (skips the scheduler).",
    )
    args = parser.parse_args()

    if args.run_now:
        logger.info("Manual run triggered via --run-now.")
        run_collection_crew()
        return

    scheduler = BlockingScheduler(timezone="Europe/Madrid")

    scheduler.add_job(
        run_collection_crew,
        trigger=CronTrigger(day_of_week="mon,fri", hour=8, minute=0),
        id="football_data_collection",
        name="Football Data Collection (LaLiga + Premier League + Serie A)",
        misfire_grace_time=3600,  # tolerate up to 1 h of scheduler downtime
        max_instances=1,          # never run two collection jobs concurrently
        coalesce=True,            # collapse missed runs into a single catch-up run
    )

    logger.info("Football Data Collection Scheduler started.")
    logger.info("Schedule : every Monday and Friday at 08:00 Europe/Madrid")
    logger.info("Log file : %s", _LOG_DIR / "agent_runs.log")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()

"""
Full domestic retrain + auto-promote, triggered when a round finishes.

Project decision (2026-09-11): once collect_and_retrain.py detects that a
round just became fully stat-collected in any of the three domestic leagues
(LaLiga/Premier/Serie A), it launches this script in the background. It
retrains 1X2 (RF + XGBoost + Premier ensemble) and Over/Under (goals, saves,
corners) for all three leagues against the standard training split, then
copies the results straight into models/{1x2,over_under}/production/ and
restarts the prediction service — no per-run human approval.

This is a deliberate, standing exception to the general
"never touch /production without asking" rule (see CLAUDE.md Critical
Rules): the user explicitly asked for this specific automated pipeline to
retrain-and-promote on its own so predictions stay current with the latest
round's data. It does NOT authorize ad-hoc /production changes outside this
script — those still require asking first.

Training takes roughly 40-60 minutes per full run (mostly the Over/Under
goals/saves/corners feature loop). A lock (state/retrain_state.json,
in_progress_pid) prevents a second run from starting while one is still in
flight; if training fails partway, nothing is promoted and a Telegram alert
is sent instead.

Usage:
    python retrain_and_promote.py --round "8:5"
    (--round is "<league_id>:<round_id>" of whichever league triggered this,
    used only for logging/the state file and the Telegram message — the
    training itself always covers all three domestic leagues.)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
TRAINING_DIR = SCRIPTS_DIR.parent / "training"
MODELS_DIR = SCRIPTS_DIR.parent / "models"
CANDIDATE_DIR = SCRIPTS_DIR.parent / "models_candidate_tmp"
RETRAIN_STATE_FILE = SCRIPTS_DIR / "state" / "retrain_state.json"
LOG_FILE = SCRIPTS_DIR / "logs" / "retrain_promote.log"

LEAGUE_NAMES = {8: "LaLiga", 17: "Premier League", 23: "Serie A"}
LEAGUE_SLUGS = ["laliga", "premier_league", "serie_a"]

sys.path.insert(0, str(SCRIPTS_DIR))
from telegram_notifier import send_message


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def _load_state() -> dict:
    if RETRAIN_STATE_FILE.exists():
        return json.loads(RETRAIN_STATE_FILE.read_text())
    return {}


def _save_state(state: dict):
    RETRAIN_STATE_FILE.parent.mkdir(exist_ok=True)
    RETRAIN_STATE_FILE.write_text(json.dumps(state, indent=2))


def run_training_script(script: str, extra_args: list[str], timeout: int) -> bool:
    env = {**os.environ, "MODELS_DIR": str(CANDIDATE_DIR), "PYTHONPATH": str(TRAINING_DIR)}
    log(f"  Running {script} {' '.join(extra_args)} (timeout={timeout}s)...")
    try:
        result = subprocess.run(
            [sys.executable, script] + extra_args,
            cwd=str(TRAINING_DIR), env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log(f"  {script}: TIMED OUT after {timeout}s")
        return False
    if result.returncode != 0:
        log(f"  {script}: FAILED (exit {result.returncode}) — {result.stderr[-500:]}")
        return False
    return True


def promote_all() -> int:
    """Copy every candidate production file (1X2 + O/U, all 3 leagues) over the real ones."""
    copied = 0
    family_dirs = [Path("1x2") / "production" / slug for slug in LEAGUE_SLUGS]
    for family in ["goals", "saves", "corners"]:
        family_dirs += [Path("over_under") / family / "production" / slug for slug in LEAGUE_SLUGS]

    for rel in family_dirs:
        src = CANDIDATE_DIR / "models" / rel
        if not src.exists():
            continue
        dst = MODELS_DIR / rel
        dst.mkdir(parents=True, exist_ok=True)
        for f in list(src.glob("*.pkl")) + list(src.glob("*.json")):
            shutil.copy2(f, dst / f.name)
            copied += 1
    return copied


def restart_prediction_service():
    subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.laliga.prediction-service"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True,
                        help='"<league_id>:<round_id>" of the round that triggered this run (for logging)')
    parser.add_argument("--rounds", required=True,
                        help='"<league_id>:<round_id>,..." latest fully-collected round per domestic '
                             'league — all of them get marked as retrained-through on success, since '
                             'training always covers all three leagues together')
    args = parser.parse_args()
    trigger_league_id, trigger_round = args.round.split(":", 1)
    trigger_name = LEAGUE_NAMES.get(int(trigger_league_id), trigger_league_id)
    rounds_by_league = dict(pair.split(":", 1) for pair in args.rounds.split(",") if pair)

    state = _load_state()
    lock = state.setdefault("_lock", {})
    if lock.get("pid"):
        log(f"Another retrain is already in progress (pid={lock['pid']}) — aborting")
        sys.exit(1)

    lock["pid"] = os.getpid()
    lock["started_at"] = datetime.now().isoformat(timespec="seconds")
    lock["trigger"] = args.round
    _save_state(state)

    log(f"=== Auto-retrain started (triggered by {trigger_name} round {trigger_round}) ===")

    if CANDIDATE_DIR.exists():
        shutil.rmtree(CANDIDATE_DIR)
    CANDIDATE_DIR.mkdir(parents=True)

    steps = [
        ("train/train_1x2.py", [], 900),
        ("train/train_xgboost.py", [], 900),
        ("train/ensemble_premier.py", [], 300),
        ("train/train_over_under_goals.py", ["--leagues", "all"], 1800),
        ("train/train_over_under_saves.py", ["--leagues", "all"], 1800),
        ("train/train_over_under_corners.py", ["--leagues", "all"], 1800),
    ]
    ok = True
    for script, extra_args, timeout in steps:
        if not run_training_script(script, extra_args, timeout):
            ok = False
            break

    if not ok:
        log("Training failed — NOT promoting, production models unchanged")
        send_message(
            f"⚠️ Reentrenamiento automático falló (disparado por {trigger_name} jornada "
            f"{trigger_round}). Modelos de producción sin cambios — revisar "
            f"scripts/logs/retrain_promote.log"
        )
        shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
        state = _load_state()
        state.pop("_lock", None)
        _save_state(state)
        sys.exit(1)

    copied = promote_all()
    restart_prediction_service()
    shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
    log(f"{copied} archivos promocionados a producción, prediction-service reiniciado")

    state = _load_state()
    state.pop("_lock", None)
    now_iso = datetime.now().isoformat(timespec="seconds")
    for lid, rnd in rounds_by_league.items():
        league_state = state.setdefault(lid, {})
        league_state["last_retrained_round"] = rnd
        league_state["last_retrained_at"] = now_iso
    _save_state(state)

    send_message(
        f"🔄 Modelos reentrenados y promocionados a producción automáticamente "
        f"(disparado por {trigger_name} jornada {trigger_round})."
    )
    log("=== Auto-retrain complete ===")


if __name__ == "__main__":
    main()

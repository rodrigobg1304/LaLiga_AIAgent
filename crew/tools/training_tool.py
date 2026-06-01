"""
Training tool for the data-collection crew.

Wraps each script under training/train/ as a CrewAI tool.
Subprocesses inherit the crew environment; DB_* and PYTHONPATH are set
explicitly so football_core imports and DB connections work correctly.
"""
import os
import sys
import subprocess
from pathlib import Path
from crewai.tools import tool

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_TRAINING_DIR  = _PROJECT_ROOT / "training"
_FOOTBALL_CORE = _PROJECT_ROOT / "football-core" / "src"

# Ordered list of scripts that must run sequentially to produce a complete
# set of production models.  The training scripts iterate all leagues
# internally and handle per-league failures with try/except.
TRAINING_PIPELINE = [
    ("train_1x2",                "train/train_1x2.py",               "Random Forest 1X2"),
    ("train_xgboost",            "train/train_xgboost.py",            "XGBoost 1X2 (Premier League ensemble)"),
    ("train_over_under_goals",   "train/train_over_under_goals.py",   "Over/Under Goals"),
    ("train_over_under_saves",   "train/train_over_under_saves.py",   "Over/Under Saves"),
    ("train_over_under_corners", "train/train_over_under_corners.py", "Over/Under Corners"),
]

_VALID_SCRIPT_NAMES = {name for name, _, _ in TRAINING_PIPELINE}


def _training_env() -> dict:
    """
    Build the subprocess environment for a training script.
    - Sets PYTHONPATH so football_core is importable without pip install.
    - Maps MYSQL_* → DB_* so football_core.db.get_connection() finds credentials.
    - Adds DB_TABLE=Leagues (required by training queries).
    """
    env = os.environ.copy()

    # football_core path — matches how retrain_scheduler.py sets it
    existing_pythonpath = env.get("PYTHONPATH", "")
    fc_path = str(_FOOTBALL_CORE)
    env["PYTHONPATH"] = f"{fc_path}:{existing_pythonpath}" if existing_pythonpath else fc_path

    # Credential bridge: MYSQL_* → DB_* (crew uses MYSQL_*, football_core uses DB_*)
    bridge = {
        "DB_HOST":     os.environ.get("MYSQL_HOST"),
        "DB_PORT":     os.environ.get("MYSQL_PORT"),
        "DB_USER":     os.environ.get("MYSQL_USER"),
        "DB_PASSWORD": os.environ.get("MYSQL_PASSWORD"),
        "DB_DATABASE": os.environ.get("MYSQL_DATABASE"),
    }
    for db_key, mysql_val in bridge.items():
        if mysql_val and db_key not in env:
            env[db_key] = mysql_val

    # DB_TABLE is required by training queries; default matches the project convention
    env.setdefault("DB_TABLE", "Leagues")

    return env


@tool("run_training_script")
def run_training_script(script_name: str) -> dict:
    """
    Run one training script from training/train/ and return its result.

    Valid values for script_name:
      - train_1x2               → Random Forest 1X2 for all leagues
      - train_xgboost           → XGBoost 1X2 + Premier League ensemble
      - train_over_under_goals  → Over/Under Goals for all leagues
      - train_over_under_saves  → Over/Under Saves for all leagues
      - train_over_under_corners → Over/Under Corners for all leagues

    Returns a dict:
      script_name  — as provided
      description  — human-readable label
      success      — True if exit code 0
      returncode   — process exit code
      stdout       — last 3000 chars of stdout (training logs)
      stderr       — last 1000 chars of stderr (errors if any)
      error        — exception message if the subprocess could not be started
    """
    # Look up description
    description = next(
        (desc for name, _, desc in TRAINING_PIPELINE if name == script_name),
        script_name,
    )

    if script_name not in _VALID_SCRIPT_NAMES:
        return {
            "script_name": script_name, "description": description,
            "success": False, "returncode": -1,
            "stdout": "", "stderr": "",
            "error": (
                f"Unknown script '{script_name}'. "
                f"Valid names: {sorted(_VALID_SCRIPT_NAMES)}"
            ),
        }

    script_path = next(path for name, path, _ in TRAINING_PIPELINE if name == script_name)
    full_path = _TRAINING_DIR / script_path

    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            capture_output=True,
            text=True,
            cwd=str(_TRAINING_DIR),
            env=_training_env(),
            timeout=1800,  # 30-minute ceiling per script
        )
        return {
            "script_name": script_name,
            "description": description,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "script_name": script_name, "description": description,
            "success": False, "returncode": -1,
            "stdout": "", "stderr": "",
            "error": "Script timed out after 1800 seconds.",
        }
    except Exception as exc:
        return {
            "script_name": script_name, "description": description,
            "success": False, "returncode": -1,
            "stdout": "", "stderr": "",
            "error": str(exc),
        }


@tool("get_training_pipeline")
def get_training_pipeline() -> list:
    """
    Returns the ordered list of training scripts that must be executed
    to produce a complete set of production models.

    Use this tool first to know which scripts to run and in what order
    before calling run_training_script for each one.

    Returns a list of dicts: [{step, script_name, description}]
    """
    return [
        {"step": i + 1, "script_name": name, "description": desc}
        for i, (name, _, desc) in enumerate(TRAINING_PIPELINE)
    ]

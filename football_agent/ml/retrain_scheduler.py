"""
Reentrenamiento automático cuando hay nuevos partidos en la BD.
Ejecutar en background: python ml/retrain_scheduler.py
"""

import os
import sys
import time
import pickle
import hashlib
import logging
import subprocess
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from football_agent.db import run_query, TABLE

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

CHECK_INTERVAL  = 604800          # Comprobar cada semana (segundos)
STATE_FILE      = os.path.join(os.path.dirname(__file__), "models/last_state.pkl")
LOG_FILE        = os.path.join(os.path.dirname(__file__), "retrain.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ESTADO DE LA BD
# ─────────────────────────────────────────────

def get_db_state() -> dict:
    """Obtiene el estado actual de la BD: número de partidos y último matchId."""
    rows = run_query(f"""
        SELECT 
            COUNT(DISTINCT matchId) AS total_matches,
            MAX(matchId)            AS last_match_id,
            MAX(Year)               AS last_year
        FROM {TABLE}
        WHERE name = 'Goals'
          AND period IN ('1ST', '2ND')
    """, ())
    return rows[0] if rows else {}


def get_state_hash(state: dict) -> str:
    """Genera un hash del estado para detectar cambios."""
    state_str = f"{state.get('total_matches')}_{state.get('last_match_id')}_{state.get('last_year')}"
    return hashlib.md5(state_str.encode()).hexdigest()


def load_last_state() -> str:
    """Carga el hash del último estado conocido."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return None


def save_state(state_hash: str):
    """Guarda el hash del estado actual."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state_hash, f)


# ─────────────────────────────────────────────
# REENTRENAMIENTO
# ─────────────────────────────────────────────

def retrain():
    """Lanza el script de entrenamiento en un subproceso."""
    log.info("🔄 Iniciando reentrenamiento...")
    start = datetime.now()

    train_script = os.path.join(os.path.dirname(__file__), "train.py")
    src_path     = os.path.join(os.path.dirname(__file__), "../src")

    result = subprocess.run(
        [sys.executable, train_script],
        env={**os.environ, "PYTHONPATH": src_path},
        capture_output=True,
        text=True
    )

    duration = (datetime.now() - start).seconds

    if result.returncode == 0:
        log.info(f"✅ Reentrenamiento completado en {duration}s")
        log.info(result.stdout[-1000:])  # últimas líneas del output
        return True
    else:
        log.error(f"❌ Error en reentrenamiento: {result.stderr[-500:]}")
        return False


# ─────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────

def run():
    log.info("🚀 Scheduler iniciado — comprobando cada %ds", CHECK_INTERVAL)
    log.info("📋 Monitorizando tabla: %s", TABLE)

    while True:
        try:
            current_state      = get_db_state()
            current_hash       = get_state_hash(current_state)
            last_hash          = load_last_state()

            log.info(
                "🔍 Estado BD — Partidos: %s | Último ID: %s | Temporada: %s",
                current_state.get("total_matches"),
                current_state.get("last_match_id"),
                current_state.get("last_year")
            )

            if last_hash is None:
                log.info("📌 Primera ejecución — guardando estado inicial")
                save_state(current_hash)

            elif current_hash != last_hash:
                log.info("🆕 Nuevos datos detectados — lanzando reentrenamiento")
                success = retrain()
                if success:
                    save_state(current_hash)
                    log.info("💾 Estado actualizado")
                else:
                    log.warning("⚠️  Reentrenamiento fallido — se reintentará en el próximo ciclo")

            else:
                log.info("✓ Sin cambios en la BD")

        except Exception as e:
            log.error("❌ Error en el scheduler: %s", e)

        log.info("⏳ Próxima comprobación en %dm", CHECK_INTERVAL // 60)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run()
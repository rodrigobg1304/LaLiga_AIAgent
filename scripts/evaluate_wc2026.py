"""
Evaluación del modelo sobre los partidos completados del Mundial 2026.

Métricas calculadas:
  · 1X2: accuracy, F1-macro, F1 por clase, matriz de confusión
  · O/U goles 1.5 y 2.5: accuracy, precision, recall, F1
  · Brier score (calibración probabilística)
  · Log-loss

Uso:
    cd scripts
    python evaluate_wc2026.py
"""
import sys
import time
import requests
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_connection, LEAGUES_TABLE

PREDICTION_URL = "http://localhost:8001"
LEAGUE_ID      = "16"
YEAR           = "2026"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_completed_matches() -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(f"""
                SELECT matchId, homeTeam, awayTeam,
                    MAX(CASE WHEN name='Goals' THEN CAST(homeValue AS SIGNED) END) AS hg,
                    MAX(CASE WHEN name='Goals' THEN CAST(awayValue AS SIGNED) END) AS ag
                FROM {LEAGUES_TABLE}
                WHERE leagueId=%s AND Year=%s
                GROUP BY matchId, homeTeam, awayTeam
                HAVING hg IS NOT NULL AND ag IS NOT NULL
                ORDER BY matchId
            """, (LEAGUE_ID, YEAR))
            return cur.fetchall()
    finally:
        conn.close()


def get_prediction(home: str, away: str) -> dict | None:
    try:
        r = requests.post(f"{PREDICTION_URL}/predict",
                          json={"home_team": home, "away_team": away,
                                "league_id": LEAGUE_ID, "year": YEAR},
                          timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def actual_outcome(hg: int, ag: int) -> str:
    return "1" if hg > ag else ("2" if hg < ag else "X")


def ou_actual(hg: int, ag: int, threshold: float) -> int:
    return 1 if (hg + ag) > threshold else 0


# ─────────────────────────────────────────────
# Metrics (no sklearn dependency)
# ─────────────────────────────────────────────

def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def f1_per_class(y_true, y_pred, classes):
    results = {}
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[c] = {"precision": prec, "recall": rec, "f1": f1,
                      "support": sum(t == c for t in y_true)}
    macro_f1 = np.mean([v["f1"] for v in results.values()])
    return results, macro_f1


def brier_score(y_true_bin, y_prob):
    return np.mean([(p - t) ** 2 for t, p in zip(y_true_bin, y_prob)])


def log_loss_single(y_true_bin, y_prob, eps=1e-7):
    return -np.mean([t * np.log(max(p, eps)) + (1-t) * np.log(max(1-p, eps))
                     for t, p in zip(y_true_bin, y_prob)])


def confusion_matrix(y_true, y_pred, classes):
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────

def run():
    matches = get_completed_matches()
    print(f"\n{'═'*60}")
    print(f"  EVALUACIÓN MUNDIAL 2026 — {len(matches)} partidos")
    print(f"{'═'*60}\n")

    y_true_1x2,  y_pred_1x2  = [], []
    p1_list, px_list, p2_list = [], [], []

    y_true_ou15, y_prob_ou15 = [], []
    y_true_ou25, y_prob_ou25 = [], []

    skipped = 0
    for i, m in enumerate(matches, 1):
        home, away = m["homeTeam"], m["awayTeam"]
        hg, ag     = int(m["hg"]), int(m["ag"])
        actual     = actual_outcome(hg, ag)

        pred = get_prediction(home, away)
        if pred is None:
            skipped += 1
            continue

        resultado = pred.get("resultado", {})
        probs     = resultado.get("probabilities", {})
        raw_pred = resultado.get("predicted", "?")
        _px = probs.get("X", 0)
        _p1 = probs.get("1", 0)
        _p2 = probs.get("2", 0)
        if raw_pred != "X" and _px > 20 and _px > 0.65 * max(_p1, _p2):
            predicted = "X"
        else:
            predicted = raw_pred
        ou_goals  = pred.get("over_under_goals", {})

        p1 = probs.get("1", 0) / 100
        px = probs.get("X", 0) / 100
        p2 = probs.get("2", 0) / 100

        y_true_1x2.append(actual)
        y_pred_1x2.append(predicted)
        p1_list.append(p1); px_list.append(px); p2_list.append(p2)

        # O/U 1.5
        ou15 = ou_goals.get("over_1_5", {})
        p_ou15 = (ou15.get("over", 0) / 100) if isinstance(ou15, dict) else 0.0
        y_true_ou15.append(ou_actual(hg, ag, 1.5))
        y_prob_ou15.append(p_ou15)

        # O/U 2.5
        ou25 = ou_goals.get("over_2_5", {})
        p_ou25 = (ou25.get("over", 0) / 100) if isinstance(ou25, dict) else 0.0
        y_true_ou25.append(ou_actual(hg, ag, 2.5))
        y_prob_ou25.append(p_ou25)

        status = "✅" if predicted == actual else "❌"
        print(f"  {status} {home:<28} {hg}-{ag} {away:<25} pred={predicted} actual={actual}")

        if i % 10 == 0:
            time.sleep(0.3)   # brief pause every 10 requests

    n = len(y_true_1x2)
    print(f"\n  ({skipped} partidos sin predicción)")

    # ── 1X2 ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  1X2 — CLASIFICACIÓN")
    print(f"{'─'*60}")

    acc = accuracy(y_true_1x2, y_pred_1x2)
    per_class, macro_f1 = f1_per_class(y_true_1x2, y_pred_1x2, ["1", "X", "2"])

    print(f"\n  Accuracy : {acc:.3f}  ({int(acc*n)}/{n})")
    print(f"  F1 macro : {macro_f1:.3f}\n")
    print(f"  {'Clase':<8} {'Prec':>7} {'Recall':>8} {'F1':>7} {'Soporte':>9}")
    labels = {"1": "Local (1)", "X": "Empate (X)", "2": "Visit. (2)"}
    for c in ["1", "X", "2"]:
        m2 = per_class[c]
        print(f"  {labels[c]:<8}   {m2['precision']:.3f}    {m2['recall']:.3f}   {m2['f1']:.3f}     {m2['support']:>4}")

    # Brier score por clase
    brier_1 = brier_score([1 if t == "1" else 0 for t in y_true_1x2], p1_list)
    brier_x = brier_score([1 if t == "X" else 0 for t in y_true_1x2], px_list)
    brier_2 = brier_score([1 if t == "2" else 0 for t in y_true_1x2], p2_list)
    print(f"\n  Brier score → Local: {brier_1:.3f} | Empate: {brier_x:.3f} | Visit.: {brier_2:.3f}")
    print(f"  (0 = perfecto, 0.25 = aleatorio)")

    # Confusion matrix
    cm = confusion_matrix(y_true_1x2, y_pred_1x2, ["1", "X", "2"])
    print(f"\n  Matriz de confusión (filas=real, cols=pred):")
    print(f"  {'':>12} {'Pred 1':>8} {'Pred X':>8} {'Pred 2':>8}")
    for real in ["1", "X", "2"]:
        row = " ".join(f"{cm[real][p]:>8}" for p in ["1", "X", "2"])
        print(f"  Real {labels[real]:<7}  {row}")

    # Distribución real vs predicha
    real_dist  = Counter(y_true_1x2)
    pred_dist  = Counter(y_pred_1x2)
    print(f"\n  Distribución real:     1={real_dist['1']} ({real_dist['1']/n:.0%})  "
          f"X={real_dist['X']} ({real_dist['X']/n:.0%})  "
          f"2={real_dist['2']} ({real_dist['2']/n:.0%})")
    print(f"  Distribución predicha: 1={pred_dist['1']} ({pred_dist['1']/n:.0%})  "
          f"X={pred_dist.get('X',0)} ({pred_dist.get('X',0)/n:.0%})  "
          f"2={pred_dist['2']} ({pred_dist['2']/n:.0%})")

    # ── O/U Goals ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  OVER/UNDER GOLES")
    print(f"{'─'*60}")

    for label, y_true_ou, y_prob_ou, thr in [
        ("+1.5 goles", y_true_ou15, y_prob_ou15, 1.5),
        ("+2.5 goles", y_true_ou25, y_prob_ou25, 2.5),
    ]:
        y_pred_ou = [1 if p >= 0.5 else 0 for p in y_prob_ou]
        acc_ou    = accuracy(y_true_ou, y_pred_ou)
        per_ou, f1_ou = f1_per_class(y_true_ou, y_pred_ou, [0, 1])
        bs_ou     = brier_score(y_true_ou, y_prob_ou)
        real_over = sum(y_true_ou)
        pred_over = sum(y_pred_ou)
        avg_prob  = np.mean(y_prob_ou) * 100

        print(f"\n  Over {label}:")
        print(f"    Accuracy  : {acc_ou:.3f}  ({int(acc_ou*n)}/{n})")
        print(f"    F1 macro  : {f1_ou:.3f}")
        print(f"    F1 over   : {per_ou[1]['f1']:.3f}  (prec={per_ou[1]['precision']:.3f} rec={per_ou[1]['recall']:.3f})")
        print(f"    F1 under  : {per_ou[0]['f1']:.3f}  (prec={per_ou[0]['precision']:.3f} rec={per_ou[0]['recall']:.3f})")
        print(f"    Brier     : {bs_ou:.3f}")
        print(f"    Real over : {real_over}/{n} ({real_over/n:.0%})  |  "
              f"Pred over: {pred_over}/{n} ({pred_over/n:.0%})  |  "
              f"Prob media: {avg_prob:.1f}%")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    run()

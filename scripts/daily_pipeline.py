"""
Daily notification pipeline.

At 10:00 each day:
  1. Refresh fixtures for rounds with placeholder team names (e.g. '2a', 'w73')
  2. Show yesterday's results vs predictions
  3. Show upcoming matches (next 24h) with predictions

Schedule:
    0 10 * * * cd /scripts && DB_... python daily_pipeline.py >> logs/daily_pipeline.log 2>&1

Required env vars (in scripts/.env or shell):
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_DATABASE
    PREDICTION_URL      — e.g. http://localhost:8001
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""
import os
import re
import sys
import math
import requests
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_connection, MATCHES_TABLE, LEAGUES_TABLE
from telegram_notifier import send_message

PREDICTION_URL = os.environ.get("PREDICTION_URL", "http://localhost:8001")

LEAGUE_EMOJI = {
    "8":   "🇪🇸",
    "17":  "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "23":  "🇮🇹",
    "11":  "🌍",
    "16":  "🏆",
    "1":   "🌍",
    "27":  "🌍",
}

LEAGUE_NAMES = {
    "8":   "LaLiga",
    "17":  "Premier League",
    "23":  "Serie A",
    "11":  "Qualy WC Europa",
    "16":  "Mundial 2026",
    "1":   "Eurocopa",
    "27":  "Qualy Euro",
}

OUTCOME_LABELS = {"1": "Local", "X": "Empate", "2": "Visitante"}
OUTCOME_EMOJI  = {"1": "🏠",    "X": "🤝",      "2": "✈️"}


# ─────────────────────────────────────────────
# Fixture refresh (Point 2)
# ─────────────────────────────────────────────

def _is_placeholder(name: str) -> bool:
    """Detects Sofascore placeholder names: '2a', 'w73', 'g1', '3a3b3c3d3f', etc."""
    return bool(re.match(r'^(\d|[wg]\d)', name or ""))


def refresh_upcoming_fixtures():
    """
    Re-fetches fixture metadata from Sofascore for rounds that still have
    placeholder team names (e.g. 'w73', '2a', 'g1'). Safe to call daily —
    uses ON DUPLICATE KEY UPDATE so no data is lost.
    """
    sql = f"""
        SELECT DISTINCT LeagueId, SeasonId, Round
        FROM {MATCHES_TABLE}
        WHERE homeScore IS NULL
          AND (homeTeam REGEXP '^[0-9]|^[wg][0-9]'
               OR awayTeam  REGEXP '^[0-9]|^[wg][0-9]')
        ORDER BY LeagueId, SeasonId, Round
    """
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql)
            rounds = cur.fetchall()
    finally:
        conn.close()

    if not rounds:
        return

    try:
        from sofascore_client import collect_round_fixtures
        from db_utils import insert_match_metadata
    except ImportError:
        print("  [fixtures] sofascore_client not available — skipping refresh")
        return

    for r in rounds:
        lid, sid, rnd = r["LeagueId"], r["SeasonId"], r["Round"]
        print(f"  [fixtures] Refreshing league={lid} season={sid} round={rnd}")
        try:
            fixtures = collect_round_fixtures(lid, sid, rnd)
            if fixtures:
                inserted = insert_match_metadata(fixtures)
                known = sum(1 for f in fixtures if not _is_placeholder(f.get("homeTeam", "")))
                print(f"    → {len(fixtures)} fixtures, {known} with real names, {inserted} rows updated")
        except Exception as e:
            print(f"    → Error: {e}")


# ─────────────────────────────────────────────
# Yesterday's results (Point 3)
# ─────────────────────────────────────────────

def get_yesterday_results() -> list[dict]:
    """Returns matches that finished in the last 36 hours."""
    cutoff_end   = datetime.now()
    cutoff_start = cutoff_end - timedelta(hours=36)
    sql = f"""
        SELECT MatchId, LeagueId, Round, homeTeam, awayTeam,
               MatchDateLocal, homeScore, awayScore, SeasonId
        FROM {MATCHES_TABLE}
        WHERE homeScore IS NOT NULL
          AND MatchDateLocal BETWEEN %s AND %s
        ORDER BY MatchDateLocal ASC
    """
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (cutoff_start, cutoff_end))
            return cur.fetchall()
    finally:
        conn.close()


def _actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "1"
    if home_score < away_score:
        return "2"
    return "X"


def _goals_ou_check(ou_goals: dict, actual_total: int) -> tuple[str, bool]:
    """
    Returns (label, was_correct) for the most likely O/U goals threshold.
    Compares predicted threshold vs actual total goals.
    """
    if not ou_goals:
        return "—", False
    sorted_keys = sorted(ou_goals.keys(), key=_key_to_float)

    def over_pct(k):
        v = ou_goals[k]
        return v.get("over", 0) if isinstance(v, dict) else v * 100

    best_key = None
    for k in sorted_keys:
        if over_pct(k) >= 50:
            best_key = k

    if best_key:
        t = _key_to_float(best_key)
        pct = over_pct(best_key)
        correct = actual_total > t
        return f"+{t:.1f} ({pct:.0f}%)", correct
    else:
        k = sorted_keys[0]
        t = _key_to_float(k)
        pct = over_pct(k)
        correct = actual_total <= t
        return f"-{t:.1f} ({100-pct:.0f}%)", correct


def build_yesterday_block(results: list[dict], year_cache: dict) -> str:
    if not results:
        return ""

    lines = ["📊 <b>Resultados de ayer</b>", "━━━━━━━━━━━━━━━━━━━━━━"]

    winner_correct = 0
    goals_correct  = 0

    for m in results:
        lid  = str(m["LeagueId"])
        year = year_cache.get(lid)
        home = _fmt_team(m["homeTeam"])
        away = _fmt_team(m["awayTeam"])
        hs, as_ = int(m["homeScore"]), int(m["awayScore"])
        actual_outcome = _actual_outcome(hs, as_)
        actual_total   = hs + as_

        pred = get_prediction(m["homeTeam"], m["awayTeam"], lid, year)

        # ── Header ──────────────────────────────────────────────
        lines.append(f"⚽ <b>{home} {hs}-{as_} {away}</b>")

        if pred:
            resultado = pred.get("resultado", {})
            probs     = resultado.get("probabilities", {})
            predicted = resultado.get("predicted", "?")
            best_prob = probs.get(predicted, 0)
            pred_label = OUTCOME_LABELS.get(predicted, predicted)
            actual_label = OUTCOME_LABELS.get(actual_outcome, actual_outcome)

            # ── Winner line ──────────────────────────────────────
            winner_hit = predicted == actual_outcome
            winner_icon = "✅" if winner_hit else "❌"
            if winner_hit:
                winner_correct += 1
                lines.append(
                    f"  {winner_icon} Ganador: {OUTCOME_EMOJI.get(predicted,'')} "
                    f"<b>{pred_label}</b> ({best_prob:.0f}%)"
                )
            else:
                lines.append(
                    f"  {winner_icon} Ganador: pred {OUTCOME_EMOJI.get(predicted,'')} "
                    f"{pred_label} ({best_prob:.0f}%) → fue {OUTCOME_EMOJI.get(actual_outcome,'')} "
                    f"<b>{actual_label}</b>"
                )

            # ── Goals line ───────────────────────────────────────
            ou = pred.get("over_under_goals", {})
            p1, px, p2 = probs.get("1", 0), probs.get("X", 0), probs.get("2", 0)
            eg_h, eg_a = _expected_goals_split(ou, p1, px, p2)
            pred_total = eg_h + eg_a
            ou_label, goals_hit = _goals_ou_check(ou, actual_total)
            goals_icon = "✅" if goals_hit else "❌"
            if goals_hit:
                goals_correct += 1
            lines.append(
                f"  {goals_icon} Goles: pred ~{pred_total:.1f} (O/U {ou_label}) · "
                f"real <b>{actual_total}</b>"
            )
        else:
            actual_label = OUTCOME_LABELS.get(actual_outcome, actual_outcome)
            lines.append(f"  ⬜ Sin predicción · {OUTCOME_EMOJI.get(actual_outcome,'')} {actual_label}")

        lines.append("")

    n = len(results)
    lines.append(
        f"<i>Ganador: {winner_correct}/{n} ✅  ·  Goles O/U: {goals_correct}/{n} ✅</i>"
    )
    return "\n".join(lines).rstrip()


# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────

def get_matches_next_24h() -> list[dict]:
    now = datetime.now()
    cutoff = now + timedelta(hours=24)
    sql = f"""
        SELECT MatchId, LeagueId, Round, homeTeam, awayTeam,
               MatchDate, MatchDateLocal
        FROM {MATCHES_TABLE}
        WHERE homeScore IS NULL
          AND MatchDateLocal BETWEEN %s AND %s
          AND homeTeam NOT REGEXP '^[0-9]|^[wg][0-9]'
          AND awayTeam  NOT REGEXP '^[0-9]|^[wg][0-9]'
        ORDER BY MatchDateLocal ASC
    """
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (now, cutoff))
            return cur.fetchall()
    finally:
        conn.close()


def get_current_year_for_league(league_id: str) -> str | None:
    sql = f"""
        SELECT Year FROM {LEAGUES_TABLE}
        WHERE leagueId = %s
        GROUP BY Year
        ORDER BY MAX(matchId) DESC
        LIMIT 1
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (league_id,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_team_tournament_stats(team: str, league_id: str, year: str) -> dict:
    sql = f"""
        SELECT
            matchId, homeTeam, awayTeam,
            MAX(CASE WHEN name = 'Goalkeeper saves' THEN
                CASE WHEN homeTeam = %s THEN CAST(homeValue AS DECIMAL(6,2))
                     ELSE CAST(awayValue AS DECIMAL(6,2)) END
            END) AS gk_saves,
            MAX(CASE WHEN name = 'Corner kicks' THEN
                CASE WHEN homeTeam = %s THEN CAST(homeValue AS DECIMAL(6,2))
                     ELSE CAST(awayValue AS DECIMAL(6,2)) END
            END) AS corners
        FROM {LEAGUES_TABLE}
        WHERE leagueId = %s AND Year = %s
          AND (homeTeam = %s OR awayTeam = %s)
        GROUP BY matchId, homeTeam, awayTeam
    """
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (team, team, league_id, year, team, team))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {}
    saves = [r["gk_saves"] for r in rows if r["gk_saves"] is not None]
    crnrs = [r["corners"]  for r in rows if r["corners"]  is not None]
    n = len(rows)
    return {
        "n_matches":   n,
        "saves_avg":   round(sum(saves) / len(saves), 1) if saves else None,
        "corners_avg": round(sum(crnrs) / len(crnrs), 1) if crnrs else None,
    }


# ─────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────

def get_prediction(home_team: str, away_team: str, league_id: str, year: str | None) -> dict | None:
    try:
        resp = requests.post(
            f"{PREDICTION_URL}/predict",
            json={"home_team": home_team, "away_team": away_team,
                  "league_id": league_id, "year": year},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [predict] Error for {home_team} vs {away_team}: {e}")
        return None


# ─────────────────────────────────────────────
# Scoreline prediction (Poisson + Dixon-Coles)
# ─────────────────────────────────────────────

def _poisson(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dixon_coles_tau(h: int, a: int, lam_h: float, lam_a: float, rho: float = -0.10) -> float:
    """
    Correction factor for low-scoring scorelines.
    ρ < 0 boosts 0-0 and 1-1 (more frequent than independent Poisson predicts)
    and slightly reduces 1-0 and 0-1.
    """
    if h == 0 and a == 0:
        return 1 - lam_h * lam_a * rho
    if h == 1 and a == 0:
        return 1 + lam_a * rho
    if h == 0 and a == 1:
        return 1 + lam_h * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def scoreline_probs(ou_goals: dict, p1: float, px: float, p2: float,
                    max_goals: int = 5, rho: float = -0.10) -> list[tuple[int, int, float]]:
    """
    Returns top scorelines sorted by probability (descending).

    λ_total is estimated from the sum of O/U survival probabilities:
        E[total] = P(>0.5) + P(>1.5) + P(>2.5) + P(>3.5)
    Then split into home/away using 1X2 weights.
    Dixon-Coles correction applied for h,a ∈ {0,1}.
    """
    if not ou_goals:
        return []

    e_total = sum(
        (v.get("over", 0) / 100 if isinstance(v, dict) else float(v))
        for v in ou_goals.values()
    )
    if e_total <= 0:
        return []

    denom = p1 + px + p2 or 100.0
    home_share = (p1 + 0.45 * px) / denom
    lam_h = max(e_total * home_share,       0.05)
    lam_a = max(e_total * (1 - home_share), 0.05)

    raw: dict[tuple[int, int], float] = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = _poisson(h, lam_h) * _poisson(a, lam_a)
            p *= _dixon_coles_tau(h, a, lam_h, lam_a, rho)
            raw[(h, a)] = p

    total = sum(raw.values())
    normalized = {k: v / total for k, v in raw.items()}
    return sorted(normalized.items(), key=lambda x: x[1], reverse=True)


def _min_goals_from_ou(ou_goals: dict) -> int:
    """
    Returns the minimum total goals the scoreline must have, derived from the
    highest O/U threshold where over-probability >= 50%.

    E.g. if P(>1.5) = 68% and P(>2.5) = 42%  → min_goals = 2
         if P(>2.5) = 55% and P(>3.5) = 28%  → min_goals = 3
         if P(>0.5) = 91% and P(>1.5) = 45%  → min_goals = 1
    """
    best_threshold = -1.0
    for key, val in ou_goals.items():
        over_pct = val.get("over", 0) if isinstance(val, dict) else val * 100
        if over_pct >= 50:
            t = _key_to_float(key)
            if t > best_threshold:
                best_threshold = t
    return int(best_threshold) + 1 if best_threshold >= 0 else 0


def _best_scoreline(ou_goals: dict, p1: float, px: float, p2: float,
                    predicted_outcome: str = None) -> str:
    """
    Returns the most likely scoreline that is consistent with BOTH:
    1. The predicted 1X2 outcome (1=home win, X=draw, 2=away win)
    2. The O/U goals model — scoreline total ≥ min_goals derived from P(>T) ≥ 50%

    Fallback: relax the goals constraint if no scoreline satisfies both filters.
    """
    scores = scoreline_probs(ou_goals, p1, px, p2)
    if not scores:
        return "—"

    min_g = _min_goals_from_ou(ou_goals)

    def outcome_ok(h, a):
        if predicted_outcome == "1":
            return h > a
        if predicted_outcome == "2":
            return h < a
        if predicted_outcome == "X":
            return h == a
        return True

    # Primary filter: match outcome AND goals minimum
    filtered = [(s, p) for s, p in scores if outcome_ok(s[0], s[1]) and s[0] + s[1] >= min_g]

    # Fallback 1: only outcome filter (drop goals constraint)
    if not filtered:
        filtered = [(s, p) for s, p in scores if outcome_ok(s[0], s[1])]

    (h, a), _ = (filtered[0] if filtered else scores[0])
    return f"{h}-{a}"


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────

def _fmt_team(slug: str) -> str:
    replacements = {
        "dr-congo":               "DR Congo",
        "bosnia-and-herzegovina": "Bosnia y Herz.",
        "south-korea":            "Corea del Sur",
        "south-africa":           "Sudáfrica",
        "saudi-arabia":           "Arabia Saudí",
        "cote-divoire":           "Costa de Marfil",
        "cape-verde":             "Cabo Verde",
        "new-zealand":            "Nueva Zelanda",
        "czech-republic":         "Rep. Checa",
        "usa":                    "EE.UU.",
        "curacao":                "Curaçao",
    }
    if slug in replacements:
        return replacements[slug]
    return " ".join(w.capitalize() for w in slug.replace("-", " ").split())


def _key_to_float(key: str) -> float:
    return float(key.replace("over_", "").replace("_", "."))


def _top2_ou(ou_data: dict, multiplier: float = 1.0) -> str:
    if not ou_data:
        return "—"
    sorted_keys = sorted(ou_data.keys(), key=_key_to_float)

    def get_over_pct(k) -> float:
        v = ou_data[k]
        return v.get("over", 0) if isinstance(v, dict) else v * multiplier

    above = [k for k in sorted_keys if get_over_pct(k) >= 50]
    if above:
        parts = [f"+{_key_to_float(k):.1f} ({get_over_pct(k):.0f}%)" for k in above[-2:]]
        return " · ".join(parts)

    k = sorted_keys[0]
    return f"-{_key_to_float(k):.1f} ({100 - get_over_pct(k):.0f}%)"


def _expected_goals_split(ou_goals: dict, p1: float, px: float, p2: float) -> tuple[float, float]:
    total = sum(
        (v.get("over", 0) / 100 if isinstance(v, dict) else v)
        for v in ou_goals.values()
    )
    denom = p1 + px + p2 or 1
    home_share = (p1 + 0.45 * px) / denom
    return round(total * home_share, 1), round(total * (1 - home_share), 1)


# ─────────────────────────────────────────────
# Match block
# ─────────────────────────────────────────────

def format_match_block(match: dict, pred: dict | None,
                       home_stats: dict = None, away_stats: dict = None) -> str:
    I = "       "   # indent — aligns stats under the emoji label

    kick_off = match["MatchDateLocal"]
    time_str = kick_off.strftime("%H:%M") if isinstance(kick_off, datetime) else str(kick_off)
    home = _fmt_team(match["homeTeam"])
    away = _fmt_team(match["awayTeam"])
    header = f"🕐 <b>{time_str}</b>  {home} 🆚 {away}"

    if pred is None:
        return header + "\n⚠️ Sin predicción disponible"

    resultado = pred.get("resultado", {})
    probs = resultado.get("probabilities", {})
    odds  = resultado.get("odds", {})
    conf  = resultado.get("confidence", {})
    conf_label = conf.get("label") or conf.get("description", "")

    p1 = probs.get("1", 0)
    px = probs.get("X", 0)
    p2 = probs.get("2", 0)

    best = max(probs, key=lambda k: probs.get(k, 0)) if probs else "?"
    best_label = OUTCOME_LABELS.get(best, best)
    best_emoji = OUTCOME_EMOJI.get(best, "")

    # ── 1X2 ──────────────────────────────────────────────────────
    block_1x2 = (
        f"🏆 <b>1X2:</b> {best_emoji} {best_label} — {conf_label}\n"
        f"{I}🏠 {p1:.0f}% ({odds.get('1',0)}) · "
        f"🤝 {px:.0f}% ({odds.get('X',0)}) · "
        f"✈️ {p2:.0f}% ({odds.get('2',0)})"
    )

    # ── Scoreline — consistent with predicted 1X2 outcome ────────
    ou_goals = pred.get("over_under_goals", {})
    scoreline = _best_scoreline(ou_goals, p1, px, p2, predicted_outcome=best)
    block_score = f"🎲 <b>Resultado:</b> {scoreline}"

    # ── Goals ─────────────────────────────────────────────────────
    eg_h, eg_a = _expected_goals_split(ou_goals, p1, px, p2)

    def _ou_pct(key: str) -> str:
        v = ou_goals.get(key)
        if v is None:
            return "—"
        pct = v.get("over", 0) if isinstance(v, dict) else v * 100
        return f"{pct:.0f}%"

    block_goals = (
        f"⚽ <b>Goles:</b> +1.5 ({_ou_pct('over_1_5')}) · +2.5 ({_ou_pct('over_2_5')})\n"
        f"{I}🏠~{eg_h} · ✈️~{eg_a}"
    )

    # ── Saves & Corners ───────────────────────────────────────────
    def _fmt(val, label):
        return f"{val:.1f} {label}/p" if val is not None else "—"

    h_stats = home_stats or {}
    a_stats = away_stats or {}
    h_saves   = h_stats.get("saves_avg")
    a_saves   = a_stats.get("saves_avg")
    h_corners = h_stats.get("corners_avg")
    a_corners = a_stats.get("corners_avg")
    n = max(h_stats.get("n_matches", 0), a_stats.get("n_matches", 0))

    if h_saves is not None or a_saves is not None:
        tag = f"({n}p)"
        block_saves   = f"🧤 <b>Paradas</b> {tag}:\n{I}🏠 {_fmt(h_saves,'par')}  ·  ✈️ {_fmt(a_saves,'par')}"
        block_corners = f"📐 <b>Córners</b> {tag}:\n{I}🏠 {_fmt(h_corners,'cor')}  ·  ✈️ {_fmt(a_corners,'cor')}"
    else:
        saves   = pred.get("saves", {})
        corners = pred.get("corners", {})
        sh = _top2_ou(saves.get("home",   {}), 100)
        sa = _top2_ou(saves.get("away",   {}), 100)
        ch = _top2_ou(corners.get("home", {}), 100)
        ca = _top2_ou(corners.get("away", {}), 100)
        block_saves   = f"🧤 <b>Paradas:</b>\n{I}🏠 {sh}  ·  ✈️ {sa}"
        block_corners = f"📐 <b>Córners:</b>\n{I}🏠 {ch}  ·  ✈️ {ca}"

    return "\n\n".join([header, block_1x2, block_score, block_goals, block_saves, block_corners])


# ─────────────────────────────────────────────
# Full message
# ─────────────────────────────────────────────

def build_telegram_message(yesterday_block: str, matches: list[dict],
                           predictions: dict, team_stats: dict) -> str:
    now_str = datetime.now().strftime("%d/%m/%Y — %H:%M")
    lines = [f"📅 <b>{now_str}</b>", ""]

    if yesterday_block:
        lines.append(yesterday_block)
        lines.append("")

    by_league: dict[str, list[dict]] = {}
    for m in matches:
        by_league.setdefault(str(m["LeagueId"]), []).append(m)

    if not matches:
        lines.append("😴 No hay partidos en las próximas 24 horas.")
        return "\n".join(lines)

    for lid, league_matches in by_league.items():
        emoji = LEAGUE_EMOJI.get(lid, "⚽")
        name  = LEAGUE_NAMES.get(lid, f"Liga {lid}")
        lines += [f"{emoji} <b>{name}</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
        for m in league_matches:
            lines.append(format_match_block(
                m,
                predictions.get(m["MatchId"]),
                team_stats.get(m["homeTeam"], {}),
                team_stats.get(m["awayTeam"], {}),
            ))
            lines.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")
        lines.append("")

    return "\n".join(lines).rstrip()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run():
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Daily pipeline started")

    # Point 2: refresh fixtures with real team names
    print("  Refreshing upcoming fixtures...")
    refresh_upcoming_fixtures()

    # Point 3: yesterday's results
    yesterday = get_yesterday_results()
    print(f"  Yesterday's results: {len(yesterday)} matches")

    year_cache: dict[str, str | None] = {}

    def _get_year(lid: str) -> str | None:
        if lid not in year_cache:
            year_cache[lid] = get_current_year_for_league(lid)
        return year_cache[lid]

    # Pre-populate year cache from yesterday's matches
    for m in yesterday:
        _get_year(str(m["LeagueId"]))

    yesterday_block = build_yesterday_block(yesterday, year_cache)

    # Upcoming matches
    matches = get_matches_next_24h()
    print(f"  Upcoming matches (next 24h): {len(matches)}")

    predictions: dict = {}
    team_stats:  dict = {}

    for m in matches:
        lid  = str(m["LeagueId"])
        year = _get_year(lid)
        home, away = m["homeTeam"], m["awayTeam"]
        print(f"  Predicting: {home} vs {away} (league={lid}, year={year})")
        predictions[m["MatchId"]] = get_prediction(home, away, lid, year)

        for team in (home, away):
            if team not in team_stats and year:
                team_stats[team] = get_team_tournament_stats(team, lid, year)

    message = build_telegram_message(yesterday_block, matches, predictions, team_stats)
    print("\n--- Telegram message ---")
    print(message)
    print("------------------------\n")

    ok = send_message(message)
    print("Telegram notification sent." if ok else "Telegram notification failed.")


if __name__ == "__main__":
    run()

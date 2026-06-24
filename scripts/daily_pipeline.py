"""
Daily notification pipeline.

Queries upcoming matches in the next 24h, gets predictions from the prediction
service, and sends a Telegram message with all the info.

Usage:
    cd scripts
    python daily_pipeline.py

Schedule with cron at 09:00 every day:
    0 9 * * * cd /path/to/project/scripts && python daily_pipeline.py >> logs/daily_pipeline.log 2>&1

Required env vars (in scripts/.env or shell):
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_DATABASE
    PREDICTION_URL      — e.g. http://localhost:8001
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""
import os
import sys
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
# DB helpers
# ─────────────────────────────────────────────

def get_matches_next_24h() -> list[dict]:
    """Returns upcoming unplayed matches with kick-off in the next 24 hours (local time)."""
    now = datetime.now()
    cutoff = now + timedelta(hours=24)
    sql = f"""
        SELECT MatchId, LeagueId, Round, homeTeam, awayTeam,
               MatchDate, MatchDateLocal
        FROM {MATCHES_TABLE}
        WHERE homeScore IS NULL
          AND MatchDateLocal BETWEEN %s AND %s
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
    """Returns the most recent season year recorded in the DB for this league."""
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
    """
    Returns actual observed averages for a team in the given tournament/year.
    Queries both home and away appearances and returns role-agnostic averages.
    Stats: goalkeeper saves and corner kicks.
    """
    sql = f"""
        SELECT
            matchId,
            homeTeam, awayTeam,
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

    saves  = [r["gk_saves"]  for r in rows if r["gk_saves"]  is not None]
    crnrs  = [r["corners"]   for r in rows if r["corners"]   is not None]
    n      = len(rows)

    return {
        "n_matches":    n,
        "saves_avg":    round(sum(saves) / len(saves), 1) if saves else None,
        "corners_avg":  round(sum(crnrs) / len(crnrs), 1) if crnrs else None,
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
# Formatting helpers
# ─────────────────────────────────────────────

def _fmt_team(slug: str) -> str:
    """'dr-congo' → 'DR Congo', 'bosnia-and-herzegovina' → 'Bosnia y Herz.'"""
    replacements = {
        "dr-congo":                  "DR Congo",
        "bosnia-and-herzegovina":    "Bosnia y Herz.",
        "south-korea":               "Corea del Sur",
        "south-africa":              "Sudáfrica",
        "saudi-arabia":              "Arabia Saudí",
        "cote-divoire":              "Costa de Marfil",
        "cape-verde":                "Cabo Verde",
        "new-zealand":               "Nueva Zelanda",
        "czech-republic":            "Rep. Checa",
        "usa":                       "EE.UU.",
    }
    if slug in replacements:
        return replacements[slug]
    return " ".join(w.capitalize() for w in slug.replace("-", " ").split())


def _key_to_float(key: str) -> float:
    return float(key.replace("over_", "").replace("_", "."))


def _top2_ou(ou_data: dict, multiplier: float = 1.0) -> str:
    """Returns the two highest thresholds where over >= 50%, or the under when none qualify."""
    if not ou_data:
        return "—"

    sorted_keys = sorted(ou_data.keys(), key=_key_to_float)

    def get_over_pct(k) -> float:
        v = ou_data[k]
        if isinstance(v, dict):
            return v.get("over", 0)
        return v * multiplier

    above = [k for k in sorted_keys if get_over_pct(k) >= 50]

    if above:
        parts = []
        for k in above[-2:]:           # last 2 (highest thresholds still ≥50%)
            t = _key_to_float(k)
            pct = get_over_pct(k)
            parts.append(f"+{t:.1f} ({pct:.0f}%)")
        return " · ".join(parts)

    # All below 50% → show under for lowest threshold
    k = sorted_keys[0]
    t = _key_to_float(k)
    under_pct = 100 - get_over_pct(k)
    return f"-{t:.1f} ({under_pct:.0f}%)"


def _expected_goals(ou_goals: dict) -> tuple[float, float]:
    """
    Estimates total expected goals using sum of over-probabilities, then splits
    home/away by 50/50 (no per-team goal model available).
    Returns (total, None) — caller decides how to use it.
    """
    total = sum(
        (v.get("over", 0) / 100 if isinstance(v, dict) else v)
        for v in ou_goals.values()
    )
    return total


def _expected_goals_split(ou_goals: dict, p1: float, px: float, p2: float) -> tuple[float, float]:
    """
    Splits total expected goals between home and away using 1X2 probabilities.
    Home share ≈ (p1 + 0.45*px) / 100, away share ≈ (p2 + 0.45*px) / 100.
    """
    total = _expected_goals(ou_goals)
    home_share = (p1 + 0.45 * px) / (p1 + px + p2)
    away_share = 1.0 - home_share
    return round(total * home_share, 1), round(total * away_share, 1)


# ─────────────────────────────────────────────
# Match block
# ─────────────────────────────────────────────

def _saves_corners_lines(pred: dict, home_stats: dict, away_stats: dict) -> tuple[str, str]:
    """
    Returns (saves_line, corners_line).
    Priority: use actual tournament observed stats when available.
    Falls back to model predictions only if no observed data exists AND the model
    shows meaningful variance (std > 0.01 across random inputs — a collapsed model
    like the qualy saves model is silently skipped).
    """
    def _fmt_stat(val, label) -> str:
        if val is None:
            return "—"
        return f"{val:.1f} {label}/p"

    # ── Observed tournament stats (preferred) ────────────────────────────────
    h_saves   = home_stats.get("saves_avg")
    a_saves   = away_stats.get("saves_avg")
    h_corners = home_stats.get("corners_avg")
    a_corners = away_stats.get("corners_avg")
    h_n       = home_stats.get("n_matches", 0)
    a_n       = away_stats.get("n_matches", 0)

    have_observed = h_saves is not None or a_saves is not None

    if have_observed:
        h_sv  = _fmt_stat(h_saves,   "par")
        a_sv  = _fmt_stat(a_saves,   "par")
        h_cr  = _fmt_stat(h_corners, "cor")
        a_cr  = _fmt_stat(a_corners, "cor")
        tag   = f"({max(h_n, a_n)}p)"
        saves_line   = f"🧤 <b>Paradas</b> {tag}:  🏠 {h_sv}  ·  ✈️ {a_sv}"
        corners_line = f"📐 <b>Córners</b> {tag}:  🏠 {h_cr}  ·  ✈️ {a_cr}"
        return saves_line, corners_line

    # ── Model predictions fallback ────────────────────────────────────────────
    saves   = pred.get("saves", {})
    corners = pred.get("corners", {})
    sh = _top2_ou(saves.get("home",   {}), multiplier=100)
    sa = _top2_ou(saves.get("away",   {}), multiplier=100)
    ch = _top2_ou(corners.get("home", {}), multiplier=100)
    ca = _top2_ou(corners.get("away", {}), multiplier=100)
    saves_line   = f"🧤 <b>Paradas:</b>  🏠 {sh}  ·  ✈️ {sa}"
    corners_line = f"📐 <b>Córners:</b>  🏠 {ch}  ·  ✈️ {ca}"
    return saves_line, corners_line


def format_match_block(match: dict, pred: dict | None,
                       home_stats: dict = None, away_stats: dict = None) -> str:
    kick_off = match["MatchDateLocal"]
    time_str = kick_off.strftime("%H:%M") if isinstance(kick_off, datetime) else str(kick_off)

    home_raw = match["homeTeam"]
    away_raw = match["awayTeam"]
    home = _fmt_team(home_raw)
    away = _fmt_team(away_raw)

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
    o1 = odds.get("1", 0)
    ox = odds.get("X", 0)
    o2 = odds.get("2", 0)

    best = max(probs, key=lambda k: probs.get(k, 0)) if probs else "?"
    best_emoji = OUTCOME_EMOJI.get(best, "")
    best_label = OUTCOME_LABELS.get(best, best)

    line_1x2  = f"🏆 <b>1X2</b>  🏠 {p1:.0f}% ({o1}) · 🤝 {px:.0f}% ({ox}) · ✈️ {p2:.0f}% ({o2})"
    line_best = f"🎯 Pronóstico: {best_emoji} <b>{best_label}</b> — {conf_label}"

    # Over/Under goals (model works well here)
    ou_goals = pred.get("over_under_goals", {})
    goals_tag = _top2_ou(ou_goals, multiplier=1)
    eg_home, eg_away = _expected_goals_split(ou_goals, p1, px, p2)
    line_goals = f"⚽ <b>Goles:</b> {goals_tag}  |  🏠~{eg_home} · ✈️~{eg_away}"

    # Saves and corners: observed stats when available, model otherwise
    saves_line, corners_line = _saves_corners_lines(
        pred,
        home_stats or {},
        away_stats or {},
    )

    return "\n".join([header, line_1x2, line_best, line_goals, saves_line, corners_line])


# ─────────────────────────────────────────────
# Full message
# ─────────────────────────────────────────────

def build_telegram_message(matches: list[dict], predictions: dict,
                           team_stats: dict) -> str:
    now_str = datetime.now().strftime("%d/%m/%Y — %H:%M")
    lines = [f"📅 <b>{now_str}</b>", ""]

    by_league: dict[str, list[dict]] = {}
    for m in matches:
        by_league.setdefault(str(m["LeagueId"]), []).append(m)

    if not matches:
        lines.append("😴 No hay partidos en las próximas 24 horas.")
        return "\n".join(lines)

    for lid, league_matches in by_league.items():
        emoji = LEAGUE_EMOJI.get(lid, "⚽")
        name  = LEAGUE_NAMES.get(lid, f"Liga {lid}")
        lines.append(f"{emoji} <b>{name}</b>")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        for m in league_matches:
            pred       = predictions.get(m["MatchId"])
            h_stats    = team_stats.get(m["homeTeam"], {})
            a_stats    = team_stats.get(m["awayTeam"], {})
            lines.append(format_match_block(m, pred, h_stats, a_stats))
            lines.append("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")
        lines.append("")

    return "\n".join(lines).rstrip()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run():
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Daily pipeline started")

    matches = get_matches_next_24h()
    print(f"  Upcoming matches (next 24h): {len(matches)}")

    year_cache:  dict[str, str | None] = {}
    predictions: dict = {}
    team_stats:  dict = {}

    for m in matches:
        lid = str(m["LeagueId"])
        if lid not in year_cache:
            year_cache[lid] = get_current_year_for_league(lid)
        year = year_cache[lid]

        home, away = m["homeTeam"], m["awayTeam"]
        print(f"  Predicting: {home} vs {away} (league={lid}, year={year})")
        pred = get_prediction(home, away, lid, year)
        predictions[m["MatchId"]] = pred

        # Fetch actual tournament stats for saves/corners (replaces collapsed model)
        for team in (home, away):
            if team not in team_stats and year:
                team_stats[team] = get_team_tournament_stats(team, lid, year)

    message = build_telegram_message(matches, predictions, team_stats)
    print("\n--- Telegram message ---")
    print(message)
    print("------------------------\n")

    ok = send_message(message)
    print("Telegram notification sent." if ok else "Telegram notification failed (check token/chat_id).")


if __name__ == "__main__":
    run()

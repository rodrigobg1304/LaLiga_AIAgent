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


def build_yesterday_block(results: list[dict], year_cache: dict) -> str:
    if not results:
        return ""

    lines = ["📊 <b>Resultados de ayer</b>", "━━━━━━━━━━━━━━━━━━━━━━"]

    correct = 0
    total_pred_goals = 0.0
    total_real_goals = 0

    for m in results:
        lid  = str(m["LeagueId"])
        year = year_cache.get(lid)
        home = _fmt_team(m["homeTeam"])
        away = _fmt_team(m["awayTeam"])
        hs, as_ = int(m["homeScore"]), int(m["awayScore"])
        actual = _actual_outcome(hs, as_)
        total_real_goals += hs + as_

        pred = get_prediction(m["homeTeam"], m["awayTeam"], lid, year)

        if pred:
            resultado = pred.get("resultado", {})
            probs = resultado.get("probabilities", {})
            predicted = resultado.get("predicted", "?")
            best_prob = probs.get(predicted, 0)
            hit = "✅" if predicted == actual else "❌"
            if predicted == actual:
                correct += 1

            pred_emoji = OUTCOME_EMOJI.get(predicted, "")
            ou = pred.get("over_under_goals", {})
            eg_h, eg_a = _expected_goals_split(ou, probs.get("1", 0), probs.get("X", 0), probs.get("2", 0))
            total_pred_goals += eg_h + eg_a

            lines.append(
                f"{hit} {home} <b>{hs}-{as_}</b> {away}  "
                f"({pred_emoji}{OUTCOME_LABELS.get(predicted, predicted)} {best_prob:.0f}%  "
                f"⚽~{eg_h+eg_a:.1f})"
            )
        else:
            actual_emoji = OUTCOME_EMOJI.get(actual, "")
            lines.append(f"⬜ {home} <b>{hs}-{as_}</b> {away}  ({actual_emoji})")

    n_pred = sum(1 for m in results if get_prediction is not None)
    pct = f"{correct}/{len(results)}" if results else "0/0"
    avg_goals = f"{total_pred_goals/len(results):.1f}" if results else "—"
    lines.append(f"<i>Aciertos: {pct} · Goles pred ~{avg_goals} vs real {total_real_goals/len(results):.1f}</i>")

    return "\n".join(lines)


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


def _saves_corners_lines(pred: dict, home_stats: dict, away_stats: dict) -> tuple[str, str]:
    def _fmt(val, label):
        return f"{val:.1f} {label}/p" if val is not None else "—"

    h_saves   = home_stats.get("saves_avg")
    a_saves   = away_stats.get("saves_avg")
    h_corners = home_stats.get("corners_avg")
    a_corners = away_stats.get("corners_avg")
    n = max(home_stats.get("n_matches", 0), away_stats.get("n_matches", 0))

    if h_saves is not None or a_saves is not None:
        tag = f"({n}p)"
        return (
            f"🧤 <b>Paradas</b> {tag}:  🏠 {_fmt(h_saves,'par')}  ·  ✈️ {_fmt(a_saves,'par')}",
            f"📐 <b>Córners</b> {tag}:  🏠 {_fmt(h_corners,'cor')}  ·  ✈️ {_fmt(a_corners,'cor')}",
        )

    saves   = pred.get("saves", {})
    corners = pred.get("corners", {})
    return (
        f"🧤 <b>Paradas:</b>  🏠 {_top2_ou(saves.get('home',{}), 100)}  ·  ✈️ {_top2_ou(saves.get('away',{}), 100)}",
        f"📐 <b>Córners:</b>  🏠 {_top2_ou(corners.get('home',{}), 100)}  ·  ✈️ {_top2_ou(corners.get('away',{}), 100)}",
    )


# ─────────────────────────────────────────────
# Match block
# ─────────────────────────────────────────────

def format_match_block(match: dict, pred: dict | None,
                       home_stats: dict = None, away_stats: dict = None) -> str:
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
    line_1x2 = (
        f"🏆 <b>1X2</b>  🏠 {p1:.0f}% ({odds.get('1',0)}) · "
        f"🤝 {px:.0f}% ({odds.get('X',0)}) · "
        f"✈️ {p2:.0f}% ({odds.get('2',0)})"
    )
    line_best = (
        f"🎯 Pronóstico: {OUTCOME_EMOJI.get(best,'')} "
        f"<b>{OUTCOME_LABELS.get(best, best)}</b> — {conf_label}"
    )

    ou_goals = pred.get("over_under_goals", {})
    eg_h, eg_a = _expected_goals_split(ou_goals, p1, px, p2)
    line_goals = f"⚽ <b>Goles:</b> {_top2_ou(ou_goals, 1)}  |  🏠~{eg_h} · ✈️~{eg_a}"

    saves_line, corners_line = _saves_corners_lines(pred, home_stats or {}, away_stats or {})

    return "\n".join([header, line_1x2, line_best, line_goals, saves_line, corners_line])


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

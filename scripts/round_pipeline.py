"""
Round-based Telegram notifications (one message per league per matchday),
as opposed to daily_pipeline.py's rolling 24h/36h window.

Two modes:
  --mode announce   Send one message per league with the full round's
                     predictions (winner, goals O/U, saves, corners,
                     scoreline) — reuses the exact match-block format
                     used for the World Cup 2026 daily messages.
                     Also stores each prediction in DailyPredictions so
                     the summary can compare against it later.

  --mode summary     Once every match in the round has a final score,
                     send one message per league with ✅/❌ for winner
                     and goals O/U across the whole round, plus a
                     round accuracy tally. Refuses to send if the round
                     isn't fully played yet (use --force to override).

Usage:
    cd scripts
    python round_pipeline.py --league 8 --season 97268 --round 1 --mode announce
    python round_pipeline.py --league 8 --season 97268 --round 1 --mode summary

Required env vars: DB_*, PREDICTION_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
(same as daily_pipeline.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from db_utils import get_connection, MATCHES_TABLE, save_prediction, update_prediction_results
from telegram_notifier import send_message
from daily_pipeline import (
    LEAGUE_EMOJI, LEAGUE_NAMES, OUTCOME_LABELS, OUTCOME_EMOJI,
    format_match_block, get_prediction, get_team_tournament_stats,
    _fmt_team, _actual_outcome, _goals_ou_check, _expected_goals_split,
    _best_scoreline, get_stored_prediction_outcome,
)

TELEGRAM_MAX_LEN = 4096
CHUNK_SAFETY_MARGIN = 3500  # leave headroom for header/footer added per chunk


# ─────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────

def get_round_matches(league_id: str, season_id: int, round_id: str) -> list[dict]:
    sql = f"""
        SELECT MatchId, LeagueId, Round, homeTeam, awayTeam,
               MatchDateLocal, homeScore, awayScore
        FROM {MATCHES_TABLE}
        WHERE LeagueId = %s AND SeasonId = %s AND Round = %s
        ORDER BY MatchDateLocal ASC
    """
    conn = get_connection()
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, (league_id, season_id, str(round_id)))
            return cur.fetchall()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Message chunking (Telegram 4096-char limit)
# ─────────────────────────────────────────────

def _chunk_blocks(blocks: list[str], header: str) -> list[str]:
    """Groups match blocks into <=CHUNK_SAFETY_MARGIN-char messages, each
    prefixed with the header (with a page indicator if split)."""
    chunks: list[list[str]] = [[]]
    current_len = len(header)
    for block in blocks:
        block_len = len(block) + len("\n─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n")
        if current_len + block_len > CHUNK_SAFETY_MARGIN and chunks[-1]:
            chunks.append([])
            current_len = len(header)
        chunks[-1].append(block)
        current_len += block_len

    n = len(chunks)
    messages = []
    for i, chunk in enumerate(chunks, 1):
        page = f" ({i}/{n})" if n > 1 else ""
        text = header.replace("{page}", page) + "\n\n" + "\n\n─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─\n\n".join(chunk)
        messages.append(text)
    return messages


def _send_chunked(messages: list[str]) -> bool:
    ok = True
    for msg in messages:
        ok = send_message(msg) and ok
    return ok


# ─────────────────────────────────────────────
# Announce
# ─────────────────────────────────────────────

def build_and_send_announcement(league_id: str, season_id: int, round_id: str, year: str) -> bool:
    matches = get_round_matches(league_id, season_id, round_id)
    if not matches:
        print(f"  No matches found for league={league_id} season={season_id} round={round_id}")
        return False

    emoji = LEAGUE_EMOJI.get(league_id, "⚽")
    name = LEAGUE_NAMES.get(league_id, f"Liga {league_id}")
    header = f"{emoji} <b>{name} — Jornada {round_id}{{page}}</b>\n━━━━━━━━━━━━━━━━━━━━━━"

    blocks = []
    for m in matches:
        home, away = m["homeTeam"], m["awayTeam"]
        print(f"  Predicting: {home} vs {away}")
        pred = get_prediction(home, away, league_id, year)
        home_stats = get_team_tournament_stats(home, league_id, year)
        away_stats = get_team_tournament_stats(away, league_id, year)
        blocks.append(format_match_block(m, pred, home_stats, away_stats))

        if pred:
            try:
                resultado = pred.get("resultado", {})
                probs = resultado.get("probabilities", {})
                p1 = probs.get("1", 0); px = probs.get("X", 0); p2 = probs.get("2", 0)
                raw_best = resultado.get("predicted", "?")
                if raw_best != "X" and px > 20 and px > 0.90 * max(p1, p2):
                    pred_outcome = "X"
                else:
                    pred_outcome = raw_best
                ou = pred.get("over_under_goals", {})
                ou15 = ou.get("over_1_5", {}); ou25 = ou.get("over_2_5", {})
                ou15_p = ou15.get("over", 0) if isinstance(ou15, dict) else 0
                ou25_p = ou25.get("over", 0) if isinstance(ou25, dict) else 0
                eg_h, eg_a = _expected_goals_split(ou, p1, px, p2)
                score = _best_scoreline(ou, p1, px, p2, predicted_outcome=pred_outcome)
                is_value = max(p1, px, p2) >= 60
                save_prediction(m, pred, pred_outcome, score, eg_h, eg_a, ou15_p, ou25_p, is_value)
            except Exception as e:
                print(f"  [db] Error saving prediction: {e}")

    messages = _chunk_blocks(blocks, header)
    print(f"  Sending {len(messages)} message(s) for {name} — Jornada {round_id}")
    return _send_chunked(messages)


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

def build_and_send_summary(league_id: str, season_id: int, round_id: str, force: bool = False) -> bool:
    matches = get_round_matches(league_id, season_id, round_id)
    if not matches:
        print(f"  No matches found for league={league_id} season={season_id} round={round_id}")
        return False

    unplayed = [m for m in matches if m["homeScore"] is None]
    if unplayed and not force:
        print(f"  Round {round_id} not finished yet ({len(unplayed)}/{len(matches)} pending). "
              f"Use --force to send a partial summary.")
        return False

    update_prediction_results()

    emoji = LEAGUE_EMOJI.get(league_id, "⚽")
    name = LEAGUE_NAMES.get(league_id, f"Liga {league_id}")
    header = f"{emoji} <b>{name} — Resumen Jornada {round_id}{{page}}</b>\n━━━━━━━━━━━━━━━━━━━━━━"

    blocks = []
    winner_correct = goals_correct = played = 0
    for m in matches:
        home, away = _fmt_team(m["homeTeam"]), _fmt_team(m["awayTeam"])
        if m["homeScore"] is None:
            blocks.append(f"⚽ <b>{home} vs {away}</b>\n  ⬜ Pendiente")
            continue

        played += 1
        hs, as_ = int(m["homeScore"]), int(m["awayScore"])
        actual_outcome = _actual_outcome(hs, as_)
        actual_total = hs + as_
        pred = get_prediction(m["homeTeam"], m["awayTeam"], league_id, str(round_id))

        lines = [f"⚽ <b>{home} {hs}-{as_} {away}</b>"]
        if pred:
            resultado = pred.get("resultado", {})
            probs = resultado.get("probabilities", {})
            stored = get_stored_prediction_outcome(m["MatchId"])
            predicted = stored if stored is not None else resultado.get("predicted", "?")
            best_prob = probs.get(predicted, 0)
            pred_label = OUTCOME_LABELS.get(predicted, predicted)
            actual_label = OUTCOME_LABELS.get(actual_outcome, actual_outcome)

            winner_hit = predicted == actual_outcome
            icon = "✅" if winner_hit else "❌"
            if winner_hit:
                winner_correct += 1
                lines.append(f"  {icon} Ganador: {OUTCOME_EMOJI.get(predicted,'')} <b>{pred_label}</b> ({best_prob:.0f}%)")
            else:
                lines.append(
                    f"  {icon} Ganador: pred {OUTCOME_EMOJI.get(predicted,'')} {pred_label} ({best_prob:.0f}%) "
                    f"→ fue {OUTCOME_EMOJI.get(actual_outcome,'')} <b>{actual_label}</b>"
                )

            ou = pred.get("over_under_goals", {})
            p1, px, p2 = probs.get("1", 0), probs.get("X", 0), probs.get("2", 0)
            eg_h, eg_a = _expected_goals_split(ou, p1, px, p2)
            ou_label, goals_hit = _goals_ou_check(ou, actual_total)
            g_icon = "✅" if goals_hit else "❌"
            if goals_hit:
                goals_correct += 1
            lines.append(f"  {g_icon} Goles: pred ~{eg_h+eg_a:.1f} (O/U {ou_label}) · real <b>{actual_total}</b>")
        else:
            actual_label = OUTCOME_LABELS.get(actual_outcome, actual_outcome)
            lines.append(f"  ⬜ Sin predicción · {OUTCOME_EMOJI.get(actual_outcome,'')} {actual_label}")

        blocks.append("\n".join(lines))

    footer = (
        f"<i>Ganador: {winner_correct}/{played} ✅  ·  Goles O/U: {goals_correct}/{played} ✅</i>"
        if played else "<i>Sin partidos jugados todavía.</i>"
    )
    blocks.append(footer)

    messages = _chunk_blocks(blocks, header)
    print(f"  Sending {len(messages)} message(s) for {name} — Resumen Jornada {round_id}")
    return _send_chunked(messages)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, help="League id, e.g. 8")
    ap.add_argument("--season", required=True, type=int, help="Season id, e.g. 97268")
    ap.add_argument("--round", required=True, help="Round/matchday number")
    ap.add_argument("--mode", required=True, choices=["announce", "summary"])
    ap.add_argument("--year", default=None, help="Season label for feature lookup, e.g. 26/27 "
                                                   "(defaults to same value as --round-season label if omitted)")
    ap.add_argument("--force", action="store_true", help="For --mode summary: send even if round isn't fully played")
    args = ap.parse_args()

    league_id = str(args.league)
    year = args.year

    if args.mode == "announce":
        if not year:
            print("  --year is required for --mode announce (e.g. --year 26/27)")
            sys.exit(1)
        ok = build_and_send_announcement(league_id, args.season, args.round, year)
    else:
        ok = build_and_send_summary(league_id, args.season, args.round, force=args.force)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

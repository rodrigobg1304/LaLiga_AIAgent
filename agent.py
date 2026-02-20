import re
import os
from typing import Optional
from db import (
    get_team_results, get_goals_scored, get_team_stats,
    get_standings, get_top_stats
)

def is_api() -> bool:
    """
    Function to know if it is an API or not.
    :return: True or False
    """
    return os.getenv("RUNTIME") == "api"

# ─────────────────────────────────────────────
# CASOS PREDEFINIDOS (menú seleccionable)
# ─────────────────────────────────────────────

MENU_CASES = {
    "1": {
        "label": "📋 Resultados de un equipo",
        "params": ["team", "league?", "year?"],
        "handler": "results"
    },
    "2": {
        "label": "⚽ Goles marcados y encajados",
        "params": ["team", "league?", "year?"],
        "handler": "goals"
    },
    "3": {
        "label": "📊 Estadística específica de un equipo",
        "params": ["team", "stat", "league?", "year?"],
        "handler": "stat"
    },
    "4": {
        "label": "🏆 Clasificación de una liga",
        "params": ["league", "year"],
        "handler": "standings"
    },
    "5": {
        "label": "🔝 Top equipos por estadística",
        "params": ["stat", "league", "year", "top_n?"],
        "handler": "top_stats"
    },
    "6": {
        "label": "🗂️  Informe completo de equipo (multi-hoja Excel)",
        "params": ["team", "league?", "year?"],
        "handler": "full_report"
    },
}

AVAILABLE_STATS = [
    "goals", "possession", "shots", "shots_on_target",
    "xg", "corners", "fouls", "yellow_cards", "red_cards",
    "passes", "passes_accuracy"
]


# ─────────────────────────────────────────────
# DETECCIÓN DE INTENCIÓN POR REGEX
# ─────────────────────────────────────────────

INTENT_PATTERNS = [
    (r"(resultado(s)|partido(s)|jugado(s)|ganado(s)|perdido(s)|empate(s))", "results"),
    (r"(gol(es)|marcado(s)|encajado(s)|anotado(s)|score(s)",                "goals"),
    (r"(posesión|possession|tiro|disparo|falta|córner)",                    "stat"),
    (r"(clasificación|tabla|standing|puntos totales)",                      "standings"),
    (r"(top|ranking|mejor|líder)",                                          "top_stats"),
    (r"(informe|reporte|completo|todo)",                                    "full_report"),
]


def detect_intent(text: str) -> Optional[str]:
    text_lower = text.lower()
    for pattern, intent in INTENT_PATTERNS:
        if re.search(pattern, text_lower):
            return intent
    return None


def extract_year(text: str) -> Optional[int]:
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else None


def extract_stat(text: str) -> Optional[str]:
    text_lower = text.lower()
    for stat in AVAILABLE_STATS:
        if stat in text_lower or stat.replace("_", " ") in text_lower:
            return stat
    return None


# ─────────────────────────────────────────────
# FORMATEO DE RESULTADOS EN TEXTO
# ─────────────────────────────────────────────

def format_results(data: list[dict], team: str) -> str | list[dict]:
    if not data:
        return f"No se encontraron partidos para {team}."

    array_t = {}
    wins = home_wins = away_wins = draws = home_draws = away_draws = losses = home_losses = away_losses = 0
    lines = [f"\n{'─' * 55}", f"  RESULTADOS — {team.upper()}", f"{'─' * 55}"]

    jornada_str = "JORNADA"
    partido_str = "PARTIDO"
    resultado_str = "RESULTADO"
    lines.append(f" +{'-' * 9}+{'-' * 52}+{'-' * 12}+")
    lines.append(f" | {jornada_str.center(7)} | {partido_str.center(50)} | {resultado_str.center(10)} |")
    lines.append(f" +{'-' * 9}+{'-' * 52}+{'-' * 12}+")
    for row in data:
        hg = int(row.get("home_goals") or 0)
        ag = int(row.get("away_goals") or 0)
        home = row["homeTeam"]
        away = row["awayTeam"]
        rnd = row.get("Round", "?")
        season = row["Year"]

        marker = ""
        if home == team:
            result = "W" if hg > ag else ("D" if hg == ag else "L")
            score = f"{hg} - {ag}"
        else:
            result = "W" if ag > hg else ("D" if ag == hg else "L")
            score = f"{ag} - {hg}"

        if result == "W":
            wins += 1;   marker = "✅"
        elif result == "D":
            draws += 1;  marker = "➖"
        else:
            losses += 1; marker = "❌"

        if result == "W" and home == team:
            home_wins += 1
        elif result == "W" and away == team:
            away_wins += 1
        elif result == "D" and home == team:
            home_draws += 1
        elif result == "D" and away == team:
            away_draws += 1
        elif result == "L" and home == team:
            home_losses += 1
        elif result == "L" and away == team:
            away_losses += 1

        match_str = f"{home} vs {away}"
        if is_api():
            array_match = {
                "round": rnd,
                "homeTeam": home,
                "awayTeam": away,
                "match": match_str,
                "home_goals": hg,
                "away_goals": ag,
                "result": result
            }
            if season not in array_t:
                array_t[season] = []
            array_t[season].append(array_match)
        else:
            lines.append(f" | {str(rnd).center(7)} | {match_str.center(50)} | [{score}] {marker} |")

    if is_api():
        season_resume = []
        for s, m in array_t.items():
            wins = sum(1 for x in m if x["result"] == "W")
            home_wins = sum(1 for x in m if x["result"] == "W" and x["homeTeam"] == team)
            away_wins = sum(1 for x in m if x["result"] == "W" and x["awayTeam"] == team)
            draws = sum(1 for x in m if x["result"] == "D")
            home_draws = sum(1 for x in m if x["result"] == "D" and x["homeTeam"] == team)
            away_draws = sum(1 for x in m if x["result"] == "D" and x["awayTeam"] == team)
            losses = sum(1 for x in m if x["result"] == "L")
            home_losses = sum(1 for x in m if x["result"] == "L" and x["homeTeam"] == team)
            away_losses = sum(1 for x in m if x["result"] == "L" and x["awayTeam"] == team)
            season_resume.append({
                "season": s,
                "summary": {
                    "played": len(m),
                    "wins": wins,
                    "home_wins": home_wins,
                    "away_wins": away_wins,
                    "draws": draws,
                    "home_draws": home_draws,
                    "away_draws": away_draws,
                    "losses": losses,
                    "home_losses": home_losses,
                    "away_losses": away_losses,
                    "points": wins * 3 + draws
                },
                "matches": m
            })
        return season_resume
    else:
        total = wins + draws + losses
        pts = wins * 3 + draws

        lines += [
            f"{'─' * 80}",
            f"  PJ:{total}  W:{wins} (H:{home_wins}, A:{away_wins})  D:{draws} (H:{home_draws}, A:{away_draws})  "
            f"L:{losses} (H:{home_losses}, A:{away_losses}) Pts:{pts}",
            f"{'─' * 80}\n"
        ]
        return "\n".join(lines)


def format_goals(data: list[dict], team: str) -> str | list[dict]:
    if not data or not data[0]:
        return [{"msg": "No se encontraron datos de goles."}]

    array_t = []
    separator = f" +{'-' * 17}+{'-' * 8}+{'-' * 20}+{'-' * 22}+{'-' * 12}+"
    lines = [
        f"\n{'─' * 55}",
        f"  GOLES — {team.upper()}",
        f"{'─' * 55}",
        separator,
        f" | {'TEAM'.center(15)} | {'SEASON'.center(6)} | {'Total Goals Scored'.center(18)} | {'Total Goals Conceded'.center(20)} | {'Total Diff'.center(10)} |",
        separator,
    ]

    for d in data:
        row = d
        season = row['year']
        scored    = int(row.get("total_goals_scored")   or 0)
        home_scored = int(row.get("home_goals_scored") or 0)
        away_scored = int(row.get("away_goals_scored") or 0)
        conceded  = int(row.get("total_goals_conceded") or 0)
        home_conceded = int(row.get("home_goals_conceded") or 0)
        away_conceded = int(row.get("away_goals_conceded") or 0)
        diff      = scored - conceded
        home_diff = home_scored - home_conceded
        away_diff = away_scored - away_conceded
        sign      = "+" if diff >= 0 else ""
        home_sign = "+" if home_diff >= 0 else ""
        away_sign = "+" if away_diff >= 0 else ""

        diff_str = f"{sign}{diff}"

        lines.append(
            f" | {team.center(15)} | {str(season).center(6)} | {str(scored).center(18)} | {str(conceded).center(20)} | {diff_str.center(10)} |")
        lines.append(separator)

        array_season = {
            "team": team,
            "year": season,
            "total_goals_scored": scored,
            "total_goals_conceded": conceded,
            "total_diff": f"{sign}{diff}",
            "home_goals_scored": home_scored,
            "home_goals_conceded": home_conceded,
            "home_diff": f"{home_sign}{home_diff}",
            "away_goals_scored": away_scored,
            "away_goals_conceded": away_conceded,
            "away_diff": f"{away_sign}{away_diff}"
            }
        array_t.append(array_season)

    if is_api():
        return array_t
    else:
        return "\n".join(lines)


def format_stat(data: list[dict], team: str, stat: str) -> str | list[dict]:
    if not data:
        return f"No se encontraron datos de '{stat}' para {team}."

    lines = [f"\n📊 {stat.upper()} — {team.upper()}", f"{'─'*50}"]
    values = []
    array_t = {}
    for row in data:
        season = row.get("Year")
        home = row["homeTeam"]
        val  = float(row["homeValue"] if home == team else row["awayValue"])
        opp  = row["awayTeam"] if home == team else row["homeTeam"]
        lines.append(f"  vs {opp:<25} {val:>8.2f}")
        values.append(val)

        if season not in array_t:
            array_t[season] = {"values": [], "matches": []}

        array_t[season]["values"].append(val)
        array_t[season]["matches"].append({
            "home": home,
            "away": opp,
            f"{stat}": val
        })

    if values:
        avg = sum(values) / len(values)
        lines += [f"{'─'*50}", f"  Promedio: {avg:.2f}\n"]

    if is_api():
        season_resume = []
        for s, m in array_t.items():
            avg = round(sum(m["values"]) / len(m["values"]), 2) if m["values"] else 0
            season_resume.append({"season": s, "stat": stat, f"avg_{stat}": f"{avg:.2f}", "matches": m["matches"]})

        return season_resume
    else:
        return "\n".join(lines)


def format_standings(data: list[dict], year: str) -> str | list[dict]:
    if not data:
        return "No se encontraron datos de clasificación."
    array_t = []
    lines = [
        f"\n🏆 CLASIFICACIÓN — {year}",
        f"{'─'*65}",
        f"  {'#':>2}  {'Equipo':<28} PJ   V   E   D   GF  GA  DIF  PTS",
        f"{'─'*65}"
    ]
    for i, row in enumerate(data, 1):
        lines.append(
            f"  {i:>2}  {row['team']:<28} "
            f"{int(row['played']):>2}  {int(row['wins']):>2}  {int(row['draws']):>2}  "
            f"{int(row['losses']):>2}  {int(row['goals_for']):>3} {int(row['goals_against']):>3}  "
            f"{int(row['goal_diff']):>+4}  {int(row['points']):>3}"
        )
        array_s = {
            "classification": i,
            "team": row["team"],
            "played": row["played"],
            "wins": row["wins"],
            "draws": row["draws"],
            "losses": row["losses"],
            "goals_for": row["goals_for"],
            "goals_against": row["goals_against"],
            "goal_diff": row["goal_diff"],
            "points": row["points"]
        }
        array_t.append(array_s)
    lines.append(f"{'─'*65}\n")

    if is_api():
        return array_t
    else:
        return "\n".join(lines)


def format_top_stats(data: list[dict], stat: str) -> str | list[dict]:
    if not data:
        return "No se encontraron datos."
    col_w = 9  # ancho de cada columna de stats
    team_w = 30
    array_t = []
    separator = f" +{'─' * 4}+{'─' * (team_w + 2)}+{'─' * (col_w * 5 + 2)}+{'─' * (col_w * 5 + 2)}+"
    lines = [
        f"\n🔝 RANKING — {stat.upper()}",
        separator,
        f" | {'#':^2} | {'TEAM':^{team_w}} | {'── HOME ──':^{col_w * 5 }} | {'── AWAY ──':^{col_w * 5 }} |",
        f" | {'':^2} | {'':^{team_w}} | {'SUM FOR':^{col_w}} | {'SUM AG':^{col_w}} | {'AVG FOR':^{col_w}} | {'AVG AG':^{col_w}} | {'SUM FOR':^{col_w}} | {'SUM AG':^{col_w}} | {'AVG FOR':^{col_w}} | {'AVG AG':^{col_w}} |",
        separator,
    ]
    for i, row in enumerate(data, 1):
        lines.append(
            f" | {i:>2} | {row['team']:<{team_w}} "
            f"| {float(row['sum_stat_home_for']):^{col_w}.2f} "
            f"| {float(row['sum_stat_home_against']):^{col_w}.2f} "
            f"| {float(row['avg_stat_home_for']):^{col_w}.2f} "
            f"| {float(row['avg_stat_home_against']):^{col_w}.2f} "
            f"| {float(row['sum_stat_away_for']):^{col_w}.2f} "
            f"| {float(row['sum_stat_away_against']):^{col_w}.2f} "
            f"| {float(row['avg_stat_away_for']):^{col_w}.2f} "
            f"| {float(row['avg_stat_away_against']):^{col_w}.2f} |"
        )
        lines.append(separator)
        array_s = {
            "ranking": i,
            "team": row["team"],
            "sum_stat_home_for": row["sum_stat_home_for"],
            "sum_stat_home_against": row["sum_stat_home_against"],
            "avg_stat_home_for": row["avg_stat_home_for"],
            "avg_stat_home_against": row["avg_stat_home_against"],
            "sum_stat_away_for": row["sum_stat_away_for"],
            "sum_stat_away_against": row["sum_stat_away_against"],
            "avg_stat_away_for": row["avg_stat_away_for"],
            "avg_stat_away_against": row["avg_stat_away_against"]
        }
        array_t.append(array_s)

    if is_api():
        return array_t
    else:
        return "\n".join(lines)


# ─────────────────────────────────────────────
# DISPATCHER PRINCIPAL
# ─────────────────────────────────────────────

def run_case(handler: str, team: str = None, league: str = None,
             year: str = None, stat: str = None, top_n: int = 10) -> dict:
    """
    Devuelve {"text": str, "data": list[dict], "handler": str}
    """
    if handler == "results":
        data = get_team_results(team=team, year=year, top_n=None)
        return {"text": format_results(data, team), "data": data, "handler": handler}

    elif handler == "goals":
        data = get_goals_scored(team=team, year=year)
        return {"text": format_goals(data, team), "data": data, "handler": handler}

    elif handler == "stat":
        # stat = stat or "possession"
        data = get_team_stats(team=team, stat_name=stat, year=year)
        return {"text": format_stat(data, team, stat), "data": data, "handler": handler}

    elif handler == "standings":
        data = get_standings(year=year)
        return {"text": format_standings(data, year), "data": data, "handler": handler}

    elif handler == "top_stats":
        stat = stat or "Goals"
        data = get_top_stats(stat, year, top_n)
        return {"text": format_top_stats(data, stat), "data": data, "handler": handler}

    elif handler == "full_report":
        results_data  = get_team_results(team, league, year)
        goals_data    = get_goals_scored(team, league, year)
        poss_data     = get_team_stats(team, "possession", league, year)
        shots_data    = get_team_stats(team, "shots", league, year)
        text = (
            format_results(results_data, team) +
            format_goals(goals_data, team) +
            format_stat(poss_data, team, "possession") +
            format_stat(shots_data, team, "shots")
        )
        return {
            "text": text,
            "data": {
                "Resultados":  results_data,
                "Goles":       goals_data,
                "Posesion":    poss_data,
                "Tiros":       shots_data,
            },
            "handler": handler
        }

    return {"text": "Caso no reconocido.", "data": [], "handler": "unknown"}

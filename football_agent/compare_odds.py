"""
Comparador de cuotas: Modelo vs Bet365
Jornada 27/02 - 02/03/2026

Uso:
    python compare_odds.py
    python compare_odds.py --url http://localhost:8001   # URL custom del servidor
    python compare_odds.py --league laliga               # filtrar liga
"""

import requests
import argparse
import json
from typing import Optional

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
API_URL = "http://localhost:8001/predict"

# ─────────────────────────────────────────────
# PARTIDOS CON CUOTAS BET365 (fuente: bet365.es / bet365.it)
# ─────────────────────────────────────────────
FIXTURES = [
    # ── LALIGA J26 (27/02 - 02/03/2026) ──────────────────────────────
    {
        "league": "LaLiga", "date": "27/02",
        "home": "levante",           "away": "deportivo-alaves",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 2.62, "X": 3.10, "2": 2.87},
    },
    {
        "league": "LaLiga", "date": "28/02",
        "home": "rayo-vallecano",    "away": "athletic-club",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 2.40, "X": 3.10, "2": 3.20},
    },
    {
        "league": "LaLiga", "date": "28/02",
        "home": "barcelona",         "away": "villarreal",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 1.30, "X": 6.50, "2": 7.00},
    },
    {
        "league": "LaLiga", "date": "28/02",
        "home": "mallorca",          "away": "real-sociedad",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 3.00, "X": 3.30, "2": 2.37},
    },
    {
        "league": "LaLiga", "date": "28/02",
        "home": "real-oviedo",       "away": "atletico-de-madrid",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 5.75, "X": 3.60, "2": 1.66},
    },
    {
        "league": "LaLiga", "date": "01/03",
        "home": "elche",             "away": "espanyol",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 2.37, "X": 3.30, "2": 3.00},
    },
    {
        "league": "LaLiga", "date": "01/03",
        "home": "valencia",          "away": "osasuna",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 2.35, "X": 3.20, "2": 3.25},
    },
    {
        "league": "LaLiga", "date": "01/03",
        "home": "real-betis",        "away": "sevilla",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 1.75, "X": 3.80, "2": 4.50},
    },
    {
        "league": "LaLiga", "date": "01/03",
        "home": "girona",            "away": "celta-de-vigo",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 2.45, "X": 3.30, "2": 2.90},
    },
    {
        "league": "LaLiga", "date": "02/03",
        "home": "real-madrid",       "away": "getafe",
        "year": "25/26",             "league_id": "8",
        "bet365": {"1": 1.30, "X": 5.50, "2": 10.00},
    },

    # ── PREMIER LEAGUE GW28 (27/02 - 01/03/2026) ─────────────────────
    {
        "league": "Premier", "date": "27/02",
        "home": "wolverhampton-wanderers", "away": "aston-villa",
        "year": "25/26",                   "league_id": "17",
        "bet365": {"1": 3.90, "X": 3.60, "2": 1.90},
    },
    {
        "league": "Premier", "date": "28/02",
        "home": "bournemouth",       "away": "sunderland",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 1.75, "X": 3.80, "2": 4.50},
    },
    {
        "league": "Premier", "date": "28/02",
        "home": "burnley",           "away": "brentford",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 4.50, "X": 3.90, "2": 1.72},
    },
    {
        "league": "Premier", "date": "28/02",
        "home": "liverpool",         "away": "west-ham-united",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 1.40, "X": 5.00, "2": 7.00},
    },
    {
        "league": "Premier", "date": "28/02",
        "home": "newcastle-united",  "away": "everton",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 1.70, "X": 4.00, "2": 4.75},
    },
    {
        "league": "Premier", "date": "28/02",
        "home": "leeds-united",      "away": "manchester-city",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 5.00, "X": 4.33, "2": 1.60},
    },
    {
        "league": "Premier", "date": "01/03",
        "home": "brighton",          "away": "nottingham-forest",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 2.05, "X": 3.60, "2": 3.40},
    },
    {
        "league": "Premier", "date": "01/03",
        "home": "fulham",            "away": "tottenham-hotspur",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 2.10, "X": 3.60, "2": 3.30},
    },
    {
        "league": "Premier", "date": "01/03",
        "home": "manchester-united", "away": "crystal-palace",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 1.53, "X": 4.50, "2": 5.50},
    },
    {
        "league": "Premier", "date": "01/03",
        "home": "arsenal",           "away": "chelsea",
        "year": "25/26",             "league_id": "17",
        "bet365": {"1": 1.60, "X": 3.80, "2": 5.50},
    },

    # ── SERIE A G27 (27/02 - 02/03/2026) ─────────────────────────────
    {
        "league": "SerieA", "date": "27/02",
        "home": "parma",             "away": "cagliari",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 2.40, "X": 2.90, "2": 3.30},
    },
    {
        "league": "SerieA", "date": "28/02",
        "home": "como",              "away": "lecce",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 1.42, "X": 4.33, "2": 8.50},
    },
    {
        "league": "SerieA", "date": "28/02",
        "home": "hellas-verona",     "away": "napoli",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 7.00, "X": 3.80, "2": 1.53},
    },
    {
        "league": "SerieA", "date": "28/02",
        "home": "inter",             "away": "genoa",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 1.30, "X": 5.00, "2": 11.00},
    },
    {
        "league": "SerieA", "date": "01/03",
        "home": "cremonese",         "away": "ac-milan",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 6.25, "X": 4.20, "2": 1.50},
    },
    {
        "league": "SerieA", "date": "01/03",
        "home": "sassuolo",          "away": "atalanta",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 4.00, "X": 3.60, "2": 1.90},
    },
    {
        "league": "SerieA", "date": "01/03",
        "home": "torino",            "away": "lazio",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 2.70, "X": 3.00, "2": 2.80},
    },
    {
        "league": "SerieA", "date": "01/03",
        "home": "as-roma",           "away": "juventus",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 2.62, "X": 3.10, "2": 2.90},
    },
    {
        "league": "SerieA", "date": "02/03",
        "home": "pisa",              "away": "bologna",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 4.10, "X": 3.20, "2": 2.00},
    },
    {
        "league": "SerieA", "date": "02/03",
        "home": "udinese",           "away": "fiorentina",
        "year": "25/26",             "league_id": "23",
        "bet365": {"1": 3.25, "X": 3.20, "2": 2.35},
    },
]


# ─────────────────────────────────────────────
# LLAMADA AL MODELO
# ─────────────────────────────────────────────
def get_model_odds(fixture: dict, api_url: str) -> Optional[dict]:
    try:
        resp = requests.post(api_url, json={
            "home_team": fixture["home"],
            "away_team": fixture["away"],
            "year":      fixture["year"],
            "league_id": fixture["league_id"],
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["resultado"]["odds"]
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# COMPARACIÓN
# ─────────────────────────────────────────────
def compare(fixture: dict, model_odds: dict) -> None:
    b = fixture["bet365"]
    m = model_odds

    home = fixture["home"].replace("-", " ").title()
    away = fixture["away"].replace("-", " ").title()
    league = fixture["league"]
    date = fixture["date"]

    print(f"\n{'─'*62}")
    print(f"  {league} {date}  |  {home} vs {away}")
    print(f"{'─'*62}")

    if "error" in m:
        print(f"  ⚠️  ERROR modelo: {m['error']}")
        return

    header = f"  {'':8} {'1':>7} {'X':>7} {'2':>7}"
    print(header)
    print(f"  {'Bet365':8} {b.get('1','-'):>7} {b.get('X','-'):>7} {b.get('2','-'):>7}")
    print(f"  {'Modelo':8} {m.get('1','-'):>7} {m.get('X','-'):>7} {m.get('2','-'):>7}")

    # Diferencias
    diffs = {}
    for k in ["1", "X", "2"]:
        bv = b.get(k)
        mv = m.get(k)
        if bv and mv:
            diffs[k] = round(mv - bv, 2)

    signs = {k: ("+" if v > 0 else "") + str(v) for k, v in diffs.items()}
    print(f"  {'Diff':8} {signs.get('1',''):>7} {signs.get('X',''):>7} {signs.get('2',''):>7}")

    # Value bets: modelo da cuota > Bet365 (el modelo cree que hay valor)
    value = [k for k, v in diffs.items() if v > 0.2]
    if value:
        print(f"  💡 Posible valor en: {', '.join(value)}")


def main():
    parser = argparse.ArgumentParser(description="Comparador cuotas modelo vs Bet365")
    parser.add_argument("--url",    default=API_URL, help="URL del endpoint del modelo")
    parser.add_argument("--league", default=None,    help="Filtrar por liga: laliga, premier, seriea")
    parser.add_argument("--json",   action="store_true", help="Output en JSON en lugar de texto")
    args = parser.parse_args()

    fixtures = FIXTURES
    if args.league:
        league_map = {"laliga": "LaLiga", "premier": "Premier", "seriea": "SerieA"}
        target = league_map.get(args.league.lower())
        if target:
            fixtures = [f for f in fixtures if f["league"] == target]

    results = []
    total = len(fixtures)

    print(f"\n🔄 Consultando modelo para {total} partidos...")
    print(f"   API: {args.url}\n")

    for i, fix in enumerate(fixtures, 1):
        home = fix["home"].replace("-", " ").title()
        away = fix["away"].replace("-", " ").title()
        print(f"  [{i:02d}/{total}] {fix['league']} | {home} vs {away}...", end=" ", flush=True)
        model_odds = get_model_odds(fix, args.url)
        print("✓" if "error" not in model_odds else "✗")
        results.append({"fixture": fix, "model_odds": model_odds})

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # ── Imprimir comparativa ──
    current_league = None
    for r in results:
        fix = r["fixture"]
        if fix["league"] != current_league:
            current_league = fix["league"]
            league_labels = {"LaLiga": "⚽ LA LIGA — Jornada 26", "Premier": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 PREMIER LEAGUE — GW28", "SerieA": "🇮🇹 SERIE A — Giornata 27"}
            print(f"\n\n{'═'*62}")
            print(f"  {league_labels.get(current_league, current_league)}")
            print(f"{'═'*62}")
        compare(fix, r["model_odds"])

    print(f"\n\n{'═'*62}")
    print("  ✅ Comparativa completada")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
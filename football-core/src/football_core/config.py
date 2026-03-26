from football_core.db import run_query, TABLE

# Mapeo LeagueId → Nombre legible
# Actualizar cuando se añadan nuevas ligas a la BD
LEAGUES = {
    "8": "LaLiga",
    "17": "Premier League",
    "23": "Serie A",
    "11": "Qualy WC Europe",
    "16": "World Cup",
}


def get_league_options() -> dict:
    """Devuelve {nombre: id} para usar en selectboxes."""
    rows = run_query(f"SELECT DISTINCT LeagueId FROM {TABLE}", ())
    existing_ids = {str(row["LeagueId"]) for row in rows}
    return {
        name: lid
        for lid, name in LEAGUES.items()
        if lid in existing_ids
    }
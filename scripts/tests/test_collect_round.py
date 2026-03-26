"""
Test manual de recopilación de datos para LaLiga, temporada 77559, jornada 29.

Permite inspeccionar el DataFrame antes de insertarlo en MySQL.

Ejecución:
    cd scripts
    python tests/test_collect_round.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sofascore_client import get_seasons_dict_result, collect_data_statistics_from
from db_utils import insert_statistics

# ─── Parámetros ───────────────────────────────────────────────────────────────
LEAGUE_ID  = 8
SEASON_ID  = 77559
ROUND      = 29

if __name__ == "__main__":
    print("1. Cargando temporadas de LaLiga...")
    get_seasons_dict_result(LEAGUE_ID)

    print(f"\n2. Recopilando estadísticas — Jornada {ROUND}...")
    df = collect_data_statistics_from(my_league=LEAGUE_ID, my_season=SEASON_ID, my_round=ROUND)

    print(f"\n3. Datos recopilados: {len(df)} filas, {df['MatchId'].nunique()} partidos")
    print(df.to_string())

    # print("\n4. Insertando en MySQL...")
    # inserted = insert_statistics(df)
    # print(f"   Filas nuevas insertadas: {inserted} (de {len(df)} recopiladas)")

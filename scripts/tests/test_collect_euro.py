"""
Recopilación completa de la última Eurocopa (Euro 2024).

Itera automáticamente por todas las rondas disponibles (fase de grupos + eliminatorias)
usando get_rounds_list() para no hardcodear los slugs.

Ejecución:
    cd scripts
    python tests/test_collect_euro.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sofascore_client import get_seasons_dict_result, get_rounds_list, collect_data_statistics_from, collect_round_metadata
from db_utils import insert_statistics, insert_match_metadata

# ─── Parámetros ───────────────────────────────────────────────────────────────
LEAGUE_ID  = 1
SEASON_ID  = 56953   # Euro 2024

if __name__ == "__main__":
    print("1. Cargando temporadas de la Eurocopa...")
    get_seasons_dict_result(LEAGUE_ID)

    print("\n2. Consultando rondas disponibles...")
    rounds = get_rounds_list(LEAGUE_ID, SEASON_ID)
    print(f"   Rondas encontradas: {rounds}")

    total_rows = 0
    total_inserted = 0

    print("\n3. Recopilando estadísticas por ronda...")
    for round_id in rounds:
        print(f"\n   Ronda: {round_id}")
        df = collect_data_statistics_from(my_league=LEAGUE_ID, my_season=SEASON_ID, my_round=round_id)
        metadata = collect_round_metadata(my_league=LEAGUE_ID, my_season=SEASON_ID, my_round=round_id)
        if df.empty:
            print("   Sin partidos completados.")
            continue
        print(f"   {df['MatchId'].nunique()} partidos — {len(df)} filas recopiladas")
        total_rows += len(df)

        inserted = insert_statistics(df)
        insert_match_metadata(metadata)
        total_inserted += inserted
        print(f"   {inserted} filas nuevas insertadas.")

    print(f"\n{'─' * 50}")
    print(f"Total recopilado : {total_rows} filas")
    print(f"Total insertado  : {total_inserted} filas nuevas")

"""
Borra todos los registros de una liga/torneo de la tabla Leagues.
Útil para limpiar datos antes de una recolección completa.

Ejecución:
    cd scripts
    python tests/delete_league_data.py --league 1              # Eurocopa
    python tests/delete_league_data.py --league 1 --dry-run    # Ver cuántos registros se borrarían
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from db_utils import get_connection, LEAGUES_TABLE


def count_rows(league_id: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {LEAGUES_TABLE} WHERE LeagueId = %s", (league_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def delete_rows(league_id: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"DELETE FROM {LEAGUES_TABLE} WHERE LeagueId = %s", (league_id,))
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Borrar datos de una liga/torneo de la DB.")
    parser.add_argument("--league", type=int, required=True, help="LeagueId a borrar (e.g. 1=Eurocopa)")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra cuántos registros se borrarían, sin borrar")
    args = parser.parse_args()

    count = count_rows(args.league)
    print(f"Registros encontrados para LeagueId={args.league}: {count}")

    if args.dry_run:
        print("Dry-run activado — no se ha borrado nada.")
        return

    if count == 0:
        print("No hay datos que borrar.")
        return

    confirm = input(f"¿Confirmas el borrado de {count} registros? (s/n): ")
    if confirm.lower() != 's':
        print("Operación cancelada.")
        return

    deleted = delete_rows(args.league)
    print(f"Borrados {deleted} registros de LeagueId={args.league}.")


if __name__ == "__main__":
    main()

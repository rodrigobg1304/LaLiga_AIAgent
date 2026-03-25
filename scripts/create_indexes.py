import sys
import os

from football_core.db import run_query, TABLE

print(f"📊 Tabla detectada: {TABLE}")
print("🔨 Creando índices...")

indices = [
    f"CREATE INDEX idx_home_team_year_name ON {TABLE} (homeTeam, Year, name, period)",
    f"CREATE INDEX idx_away_team_year_name ON {TABLE} (awayTeam, Year, name, period)",
    f"CREATE INDEX idx_h2h_teams ON {TABLE} (homeTeam, awayTeam, Year, name)",
    f"CREATE INDEX idx_year_round ON {TABLE} (Year, Round)",
    f"CREATE INDEX idx_elo_calc ON {TABLE} (Year, Round, homeTeam, awayTeam, name, period)"
]

for idx, sql in enumerate(indices, 1):
    try:
        run_query(sql, ())
        print(f"  ✅ Índice {idx}/5 creado")
    except Exception as e:
        if "1061" in str(e) or "Duplicate" in str(e):
            print(f"  ℹ️  Índice {idx}/5 ya existe (OK)")
        else:
            print(f"  ❌ Índice {idx}/5 - Error: {str(e)}")

print("✅ Proceso completado")
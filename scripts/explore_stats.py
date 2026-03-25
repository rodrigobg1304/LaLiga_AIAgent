import sys
import os
from football_core.db import run_query, TABLE
import pandas as pd

# 1. Ver todas las stats disponibles
print("="*60)
print("STATS DISPONIBLES EN LA BASE DE DATOS")
print("="*60)

sql_stats = f"SELECT DISTINCT name FROM {TABLE} ORDER BY name"
stats = run_query(sql_stats)
stats_list = [row['name'] for row in stats]

print(f"\nTotal stats diferentes: {len(stats_list)}")
print("\nLista completa:")
for i, stat in enumerate(stats_list, 1):
    print(f"{i:2}. {stat}")

# 2. Ver cobertura de cada stat (% de partidos que la tienen)
print("\n" + "="*60)
print("COBERTURA DE CADA STAT (% partidos con dato)")
print("="*60)

# Total de partidos únicos
total_matches = run_query(f"SELECT COUNT(DISTINCT matchId) as total FROM {TABLE}")
total = total_matches[0]['total']

coverage = []
for stat in stats_list:
    sql = f"""
        SELECT COUNT(DISTINCT matchId) as count 
        FROM {TABLE} 
        WHERE name = %s
    """
    result = run_query(sql, (stat,))
    count = result[0]['count']
    pct = (count / total) * 100
    coverage.append({'stat': stat, 'matches': count, 'coverage_pct': pct})

coverage_df = pd.DataFrame(coverage).sort_values('coverage_pct', ascending=False)
print(coverage_df.to_string(index=False))

# 3. Ver ejemplo de datos para las stats más importantes
print("\n" + "="*60)
print("EJEMPLO DE DATOS - TOP 5 STATS CON MEJOR COBERTURA")
print("="*60)

top_stats = coverage_df.head(5)['stat'].tolist()
for stat in top_stats:
    print(f"\n--- {stat} ---")
    sql = f"""
        SELECT homeTeam, awayTeam, Year, 
               homeValue, awayValue
        FROM {TABLE}
        WHERE name = %s
        LIMIT 5
    """
    examples = run_query(sql, (stat,))
    print(pd.DataFrame(examples))

# 4. Verificar stats por liga (puede haber diferencias)
print("\n" + "="*60)
print("STATS POR LIGA")
print("="*60)

sql_by_league = f"""
    SELECT 
        LeagueId,
        COUNT(DISTINCT name) as num_stats,
        COUNT(DISTINCT matchId) as num_matches
    FROM {TABLE}
    GROUP BY LeagueId
    ORDER BY LeagueId
"""
league_stats = run_query(sql_by_league)
print(pd.DataFrame(league_stats))


"""
TOP 10 Features adicionales (por orden de importancia):
Shots on target
Ball possession
Total shots
Goalkeeper saves
Big chances
Accurate passes
Tackles won
Interceptions
Blocked shots
Shots inside box
"""
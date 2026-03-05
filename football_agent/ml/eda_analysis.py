import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from football_agent.db import run_query, TABLE
import pandas as pd
import numpy as np

print("=" * 60)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 60)

# Cargar datos de una liga (ej: LaLiga)
LEAGUE_ID = '8'

# 1. ESTRUCTURA BÁSICA
print("\n1. ESTRUCTURA BÁSICA")
print("-" * 60)

sql_basic = f"""
    SELECT 
        matchId,
        Year,
        CAST(Round AS SIGNED) as round,
        homeTeam,
        awayTeam,
        LeagueId,
        name,
        homeValue,
        awayValue
    FROM {TABLE}
    WHERE LeagueId = %s
    LIMIT 1000
"""
df_sample = pd.DataFrame(run_query(sql_basic, (LEAGUE_ID,)))

print(f"Columnas: {df_sample.columns.tolist()}")
print(f"\nTipos de datos:")
print(df_sample.dtypes)
print(f"\nShape: {df_sample.shape}")

# 2. VALORES NULOS
print("\n2. VALORES NULOS")
print("-" * 60)
print(df_sample.isnull().sum())

# 3. DISTRIBUCIÓN DE STATS
print("\n3. DISTRIBUCIÓN DE VALORES POR STAT")
print("-" * 60)

# Seleccionar stats importantes (ajustar según Paso 1)
important_stats = [
    'Goals',
    'Shots on target',
    'Ball possession',
    'Total shots',
    'Goalkeeper saves',
    'Big chances',
    'Accurate passes',
    'Tackles won',
    'Interceptions',
    'Blocked shots'
]

for stat in important_stats:
    print(f"\n--- {stat} ---")
    sql_dist = f"""
        SELECT 
            homeValue,
            COUNT(*) as frequency
        FROM {TABLE}
        WHERE name = %s AND LeagueId = %s
        GROUP BY homeValue
        ORDER BY homeValue
    """
    dist = run_query(sql_dist, (stat, LEAGUE_ID))
    dist_df = pd.DataFrame(dist)

    if len(dist_df) > 0:
        values = [float(row['homeValue']) for row in dist]
        print(f"Min: {min(values)}, Max: {max(values)}")
        print(f"Mean: {np.mean(values):.2f}, Median: {np.median(values):.2f}")
        print(f"Valores únicos: {len(values)}")

# 4. CONSISTENCIA TEMPORAL
print("\n4. CONSISTENCIA POR TEMPORADA")
print("-" * 60)

sql_temporal = f"""
    SELECT 
        Year,
        COUNT(DISTINCT matchId) as matches,
        COUNT(DISTINCT name) as stats_available
    FROM {TABLE}
    WHERE LeagueId = %s
    GROUP BY Year
    ORDER BY Year
"""
temporal = pd.DataFrame(run_query(sql_temporal, (LEAGUE_ID,)))
print(temporal)

# 5. PARTIDOS CON DATOS INCOMPLETOS
print("\n5. DETECCIÓN DE PARTIDOS CON DATOS INCOMPLETOS")
print("-" * 60)

sql_incomplete = f"""
    SELECT 
        matchId,
        COUNT(DISTINCT name) as num_stats
    FROM {TABLE}
    WHERE LeagueId = %s
    GROUP BY matchId
    HAVING COUNT(DISTINCT name) < 5  -- Ajustar threshold
    LIMIT 10
"""
incomplete = run_query(sql_incomplete, (LEAGUE_ID,))
print(f"Partidos con <5 stats: {len(incomplete)}")
if incomplete:
    print(pd.DataFrame(incomplete))

# 6. RANGOS EXTRAÑOS (outliers potenciales)
print("\n6. DETECCIÓN DE OUTLIERS EN STATS CLAVE")
print("-" * 60)

for stat in ['Goals', 'Ball possession']:  # Ajustar
    sql_outliers = f"""
        SELECT 
            matchId,
            homeTeam,
            awayTeam,
            CAST(homeValue AS SIGNED) as value
        FROM {TABLE}
        WHERE name = %s AND LeagueId = %s
        ORDER BY CAST(homeValue AS SIGNED) DESC
        LIMIT 5
    """
    outliers = run_query(sql_outliers, (stat, LEAGUE_ID))
    print(f"\n{stat} - Top 5 valores más altos:")
    print(pd.DataFrame(outliers))

print("\n" + "=" * 60)
print("EDA COMPLETADO")
print("=" * 60)
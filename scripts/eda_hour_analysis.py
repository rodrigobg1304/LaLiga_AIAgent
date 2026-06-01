"""
Análisis exploratorio: rendimiento de equipos por franja horaria.

Objetivo: identificar si existe un patrón estadístico en el rendimiento
(win rate, goles) de los equipos en función de la hora local del partido.

Resultado esperado: validar si home_win_rate_at_hour / away_win_rate_at_hour
son features con suficiente señal para incluirlas en los modelos.

Usage:
    cd scripts
    set -a; source .env; set +a
    python eda_hour_analysis.py                    # Todas las ligas domésticas
    python eda_hour_analysis.py --league 8         # Solo LaLiga
    python eda_hour_analysis.py --league 8 17 23 --min-matches 5
    python eda_hour_analysis.py --no-plots         # Solo texto, sin gráficos
"""

import argparse
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_connection, MATCHES_TABLE, LEAGUES_TABLE

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

LEAGUE_NAMES = {8: "LaLiga", 17: "Premier League", 23: "Serie A"}
DOMESTIC_LEAGUES = [8, 17, 23]

# Hora local → etiqueta de franja
def hour_to_slot(h: int) -> str:
    if h < 13:
        return "Mañana (<13h)"
    elif h < 17:
        return "Tarde (13-16h)"
    elif h < 20:
        return "Tarde-noche (17-19h)"
    else:
        return "Noche (≥20h)"

SLOT_ORDER = ["Mañana (<13h)", "Tarde (13-16h)", "Tarde-noche (17-19h)", "Noche (≥20h)"]
DAY_NAMES  = {1: "Dom", 2: "Lun", 3: "Mar", 4: "Mié", 5: "Jue", 6: "Vie", 7: "Sáb"}


# ─────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────

def load_matches_with_hour(league_ids: list[int]) -> pd.DataFrame:
    """
    Devuelve un DataFrame a nivel de partido con:
    MatchId, LeagueId, homeTeam, awayTeam, Year,
    home_goals, away_goals, result (1/X/2),
    hour_of_day, day_of_week, time_slot.

    JOIN entre Leagues (stats) y Matches (hora local).
    Solo incluye partidos con MatchDateLocal no nulo.
    """
    placeholders = ",".join(["%s"] * len(league_ids))
    sql = f"""
        SELECT
            l.MatchId,
            l.LeagueId,
            l.homeTeam,
            l.awayTeam,
            l.Year,
            HOUR(m.MatchDateLocal)      AS hour_of_day,
            DAYOFWEEK(m.MatchDateLocal) AS day_of_week,
            SUM(CASE WHEN l.name = 'Goals'
                      AND l.period IN ('1ST', '2ND')
                     THEN CAST(l.homeValue AS DECIMAL) ELSE 0 END) AS home_goals,
            SUM(CASE WHEN l.name = 'Goals'
                      AND l.period IN ('1ST', '2ND')
                     THEN CAST(l.awayValue AS DECIMAL) ELSE 0 END) AS away_goals
        FROM {LEAGUES_TABLE} l
        JOIN {MATCHES_TABLE} m ON l.MatchId = m.MatchId
        WHERE l.name = 'Goals'
          AND m.MatchDateLocal IS NOT NULL
          AND l.LeagueId IN ({placeholders})
        GROUP BY l.MatchId, l.LeagueId, l.homeTeam, l.awayTeam, l.Year,
                 hour_of_day, day_of_week
        ORDER BY l.LeagueId, l.Year, l.MatchId
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, league_ids)
    rows = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️  No se encontraron partidos con MatchDateLocal. ¿Se recolectaron los metadatos?")
        sys.exit(1)

    df["home_goals"] = df["home_goals"].astype(float).astype(int)
    df["away_goals"] = df["away_goals"].astype(float).astype(int)
    df["result"] = df.apply(
        lambda r: "1" if r.home_goals > r.away_goals
        else ("2" if r.home_goals < r.away_goals else "X"),
        axis=1
    )
    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    df["time_slot"]   = df["hour_of_day"].map(hour_to_slot)
    df["day_name"]    = df["day_of_week"].map(DAY_NAMES)
    df["league_name"] = df["LeagueId"].map(LEAGUE_NAMES)
    return df


# ─────────────────────────────────────────────
# Análisis global por hora
# ─────────────────────────────────────────────

def global_hour_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Win rates globales (todas las ligas) por hora."""
    grp = df.groupby("hour_of_day").agg(
        matches   = ("MatchId", "count"),
        home_wins = ("result", lambda s: (s == "1").sum()),
        draws     = ("result", lambda s: (s == "X").sum()),
        away_wins = ("result", lambda s: (s == "2").sum()),
        avg_goals = ("home_goals", lambda s: (s + df.loc[s.index, "away_goals"]).mean()),
    ).reset_index()
    grp["home_win_rate"] = grp["home_wins"] / grp["matches"]
    grp["draw_rate"]     = grp["draws"]     / grp["matches"]
    grp["away_win_rate"] = grp["away_wins"] / grp["matches"]
    return grp.sort_values("hour_of_day")


def global_slot_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Win rates globales por franja horaria."""
    grp = df.groupby("time_slot").agg(
        matches   = ("MatchId", "count"),
        home_wins = ("result", lambda s: (s == "1").sum()),
        draws     = ("result", lambda s: (s == "X").sum()),
        away_wins = ("result", lambda s: (s == "2").sum()),
        avg_goals = ("home_goals", lambda s: (s + df.loc[s.index, "away_goals"]).mean()),
    ).reset_index()
    grp["home_win_rate"] = grp["home_wins"] / grp["matches"]
    grp["draw_rate"]     = grp["draws"]     / grp["matches"]
    grp["away_win_rate"] = grp["away_wins"] / grp["matches"]
    grp["time_slot"]     = pd.Categorical(grp["time_slot"], categories=SLOT_ORDER, ordered=True)
    return grp.sort_values("time_slot")


# ─────────────────────────────────────────────
# Análisis por equipo
# ─────────────────────────────────────────────

def team_hour_stats(df: pd.DataFrame, min_matches: int = 3) -> pd.DataFrame:
    """
    Para cada combinación (equipo, rol, hora) calcula:
    win_rate, draw_rate, loss_rate, avg_goals, n_matches.

    Devuelve solo filas con >= min_matches partidos.
    """
    records = []

    # Partidos como local
    home = df[["MatchId", "homeTeam", "LeagueId", "league_name", "hour_of_day",
               "time_slot", "result", "home_goals", "away_goals"]].copy()
    home = home.rename(columns={"homeTeam": "team", "home_goals": "goals_for",
                                 "away_goals": "goals_against"})
    home["role"]   = "home"
    home["win"]    = (home["result"] == "1").astype(int)
    home["draw"]   = (home["result"] == "X").astype(int)
    home["loss"]   = (home["result"] == "2").astype(int)

    # Partidos como visitante
    away = df[["MatchId", "awayTeam", "LeagueId", "league_name", "hour_of_day",
               "time_slot", "result", "home_goals", "away_goals"]].copy()
    away = away.rename(columns={"awayTeam": "team", "away_goals": "goals_for",
                                 "home_goals": "goals_against"})
    away["role"]   = "away"
    away["win"]    = (away["result"] == "2").astype(int)
    away["draw"]   = (away["result"] == "X").astype(int)
    away["loss"]   = (away["result"] == "1").astype(int)

    combined = pd.concat([home, away], ignore_index=True)

    grp = combined.groupby(["team", "role", "hour_of_day", "time_slot", "league_name"]).agg(
        n_matches    = ("MatchId", "count"),
        wins         = ("win", "sum"),
        draws        = ("draw", "sum"),
        losses       = ("loss", "sum"),
        avg_goals_for    = ("goals_for", "mean"),
        avg_goals_against= ("goals_against", "mean"),
    ).reset_index()

    grp["win_rate"]  = grp["wins"]  / grp["n_matches"]
    grp["draw_rate"] = grp["draws"] / grp["n_matches"]
    grp["loss_rate"] = grp["losses"]/ grp["n_matches"]
    grp["avg_goals"] = grp["avg_goals_for"] + grp["avg_goals_against"]

    return grp[grp["n_matches"] >= min_matches].copy()


def top_pattern_teams(team_stats: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Equipos con mayor variación de win_rate entre horas (home y away).
    Alta variación = patrón por horario más pronunciado → mejor candidato como feature.
    """
    records = []
    for (team, role, league), grp in team_stats.groupby(["team", "role", "league_name"]):
        if len(grp) < 2:
            continue
        wr_std  = grp["win_rate"].std()
        wr_max  = grp["win_rate"].max()
        wr_min  = grp["win_rate"].min()
        wr_diff = wr_max - wr_min
        records.append({
            "team": team, "role": role, "league": league,
            "hours_available": len(grp),
            "win_rate_std": round(wr_std, 3),
            "win_rate_max": round(wr_max, 3),
            "win_rate_min": round(wr_min, 3),
            "win_rate_range": round(wr_diff, 3),
        })
    result = pd.DataFrame(records).sort_values("win_rate_range", ascending=False)
    return result.head(top_n)


# ─────────────────────────────────────────────
# Test estadístico: ¿es significativo el patrón?
# ─────────────────────────────────────────────

def chi2_hour_vs_result(df: pd.DataFrame) -> None:
    """Chi-cuadrado: ¿la distribución de resultados es independiente de la hora?"""
    print("\n📊 Test chi-cuadrado: hora del partido vs resultado")
    print("   H0: el resultado es independiente de la hora")
    print("   p < 0.05 → la hora SÍ influye estadísticamente\n")

    for league_name, grp in df.groupby("league_name"):
        contingency = pd.crosstab(grp["hour_of_day"], grp["result"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        significance = "✅ SIGNIFICATIVO" if p < 0.05 else "❌ no significativo"
        print(f"   {league_name:20s}  χ²={chi2:.2f}  p={p:.4f}  ({significance})")


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_global_hour(hour_stats: pd.DataFrame, output_dir: Path) -> None:
    """Gráfico de barras apiladas: win/draw/loss rate por hora (global)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = hour_stats["hour_of_day"]
    x = np.arange(len(hours))
    width = 0.6

    ax.bar(x, hour_stats["home_win_rate"], width, label="Local gana",     color="#2196F3", alpha=0.85)
    ax.bar(x, hour_stats["draw_rate"],     width, bottom=hour_stats["home_win_rate"],
           label="Empate", color="#9E9E9E", alpha=0.85)
    ax.bar(x, hour_stats["away_win_rate"], width,
           bottom=hour_stats["home_win_rate"] + hour_stats["draw_rate"],
           label="Visitante gana", color="#F44336", alpha=0.85)

    # Número de partidos encima
    for i, (_, row) in enumerate(hour_stats.iterrows()):
        ax.text(i, 1.01, f"n={int(row['matches'])}", ha="center", va="bottom", fontsize=7, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Hora local del partido")
    ax.set_ylabel("Proporción de resultados")
    ax.set_title("Distribución de resultados por hora (todas las ligas)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = output_dir / "01_global_hour_results.png"
    fig.savefig(path, dpi=150)
    print(f"   💾 {path}")
    plt.close(fig)


def plot_home_win_rate_by_league_and_hour(df: pd.DataFrame, output_dir: Path) -> None:
    """Líneas: home win rate por hora, una línea por liga."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, role, role_label in zip(axes, ["home", "away"], ["Local", "Visitante"]):
        for league_name, grp in df.groupby("league_name"):
            col = "result"
            if role == "home":
                win_val = "1"
            else:
                win_val = "2"
            stats_grp = grp.groupby("hour_of_day").apply(
                lambda g: pd.Series({
                    "win_rate": (g["result"] == win_val).mean(),
                    "n": len(g)
                })
            ).reset_index()
            # Solo horas con ≥ 5 partidos
            stats_grp = stats_grp[stats_grp["n"] >= 5]
            ax.plot(stats_grp["hour_of_day"], stats_grp["win_rate"],
                    marker="o", label=league_name, linewidth=2)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel("Hora local")
        ax.set_ylabel("Win rate")
        ax.set_title(f"Win rate {role_label} por hora y liga")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "02_win_rate_by_league_hour.png"
    fig.savefig(path, dpi=150)
    print(f"   💾 {path}")
    plt.close(fig)


def plot_heatmap_team_hour(team_stats: pd.DataFrame, role: str,
                            league_name: str, output_dir: Path,
                            min_teams: int = 5) -> None:
    """Heatmap: equipos × hora → win rate."""
    sub = team_stats[(team_stats["role"] == role) &
                     (team_stats["league_name"] == league_name)].copy()
    if sub.empty or sub["team"].nunique() < min_teams:
        return

    pivot = sub.pivot_table(index="team", columns="hour_of_day",
                            values="win_rate", aggfunc="mean")
    # Ordenar equipos por win rate media descendente
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2),
                                    max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0%", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={"label": "Win rate"})
    ax.set_title(f"Win rate como {role} por hora — {league_name}")
    ax.set_xlabel("Hora local")
    ax.set_ylabel("")
    plt.tight_layout()
    fname = f"03_heatmap_{role}_{league_name.lower().replace(' ', '_')}.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150)
    print(f"   💾 {path}")
    plt.close(fig)


def plot_slot_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Distribución de partidos por franja horaria y liga."""
    counts = df.groupby(["league_name", "time_slot"]).size().reset_index(name="n")
    counts["time_slot"] = pd.Categorical(counts["time_slot"],
                                          categories=SLOT_ORDER, ordered=True)
    counts = counts.sort_values(["league_name", "time_slot"])

    fig, ax = plt.subplots(figsize=(10, 5))
    leagues = counts["league_name"].unique()
    x = np.arange(len(SLOT_ORDER))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, (league, color) in enumerate(zip(leagues, colors)):
        sub = counts[counts["league_name"] == league].set_index("time_slot")["n"]
        sub = sub.reindex(SLOT_ORDER, fill_value=0)
        ax.bar(x + i * width, sub.values, width, label=league, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(SLOT_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Número de partidos")
    ax.set_title("Distribución de partidos por franja horaria")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "04_slot_distribution.png"
    fig.savefig(path, dpi=150)
    print(f"   💾 {path}")
    plt.close(fig)


def plot_goals_by_hour(df: pd.DataFrame, output_dir: Path) -> None:
    """Goles totales medios por hora."""
    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    grp = df.groupby("hour_of_day").agg(
        avg_goals=("total_goals", "mean"),
        n=("MatchId", "count")
    ).reset_index()
    grp = grp[grp["n"] >= 5]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(grp["hour_of_day"], grp["avg_goals"], color="#FF9800", alpha=0.85, width=0.6)
    ax.axhline(grp["avg_goals"].mean(), color="grey", linestyle="--",
               label=f"Media global: {grp['avg_goals'].mean():.2f}")
    ax.set_xlabel("Hora local del partido")
    ax.set_ylabel("Goles medios por partido")
    ax.set_title("Media de goles totales por hora")
    ax.set_xticks(grp["hour_of_day"])
    ax.set_xticklabels([f"{h:02d}:00" for h in grp["hour_of_day"]])
    ax.legend()
    plt.tight_layout()
    path = output_dir / "05_avg_goals_by_hour.png"
    fig.savefig(path, dpi=150)
    print(f"   💾 {path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="EDA: rendimiento de equipos por hora del partido")
    parser.add_argument("--league", type=int, nargs="+", default=DOMESTIC_LEAGUES,
                        help="IDs de liga a analizar (default: 8 17 23)")
    parser.add_argument("--min-matches", type=int, default=3,
                        help="Mínimo de partidos por (equipo, hora) para incluir en análisis (default: 3)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Omitir generación de gráficos")
    parser.add_argument("--output-dir", type=str, default="eda_output",
                        help="Directorio de salida para gráficos (default: eda_output)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    league_labels = [LEAGUE_NAMES.get(lid, str(lid)) for lid in args.league]
    print("\n" + "=" * 60)
    print("🕐 EDA: RENDIMIENTO POR HORA DEL PARTIDO")
    print("=" * 60)
    print(f"Ligas     : {', '.join(league_labels)}")
    print(f"Min matches: {args.min_matches}")
    print()

    # ── Carga ─────────────────────────────────────────────────
    print("📥 Cargando datos...")
    df = load_matches_with_hour(args.league)
    total = len(df)
    with_hour = df["hour_of_day"].notna().sum()
    print(f"   Total partidos  : {total}")
    print(f"   Con hora local  : {with_hour}")
    print(f"   Sin hora local  : {total - with_hour}  (excluidos del análisis)")
    print(f"\n   Distribución por liga:")
    for league_name, grp in df.groupby("league_name"):
        print(f"   · {league_name:20s} {len(grp):>5} partidos")

    # ── Distribución de horas ──────────────────────────────────
    print("\n📊 Horas de comienzo más frecuentes:")
    hour_counts = df["hour_of_day"].value_counts().sort_index()
    for h, n in hour_counts.items():
        bar = "█" * int(n / total * 100)
        print(f"   {h:02d}:00  {n:>5} partidos  {bar}")

    # ── Análisis global ────────────────────────────────────────
    print("\n📊 Win rates globales por hora:")
    hour_stats = global_hour_analysis(df)
    print(hour_stats[["hour_of_day", "matches", "home_win_rate",
                       "draw_rate", "away_win_rate", "avg_goals"]].to_string(index=False))

    print("\n📊 Win rates globales por franja horaria:")
    slot_stats = global_slot_analysis(df)
    print(slot_stats[["time_slot", "matches", "home_win_rate",
                       "draw_rate", "away_win_rate", "avg_goals"]].to_string(index=False))

    # ── Test estadístico ──────────────────────────────────────
    chi2_hour_vs_result(df)

    # ── Análisis por equipo ───────────────────────────────────
    print("\n📊 Calculando estadísticas por equipo y hora...")
    team_stats = team_hour_stats(df, min_matches=args.min_matches)
    print(f"   Combinaciones (equipo, rol, hora) con ≥{args.min_matches} partidos: {len(team_stats)}")

    print("\n🏆 Top 10 equipos con mayor variación de win rate por hora (local):")
    top_home = top_pattern_teams(team_stats[team_stats["role"] == "home"], top_n=10)
    print(top_home.to_string(index=False))

    print("\n🏆 Top 10 equipos con mayor variación de win rate por hora (visitante):")
    top_away = top_pattern_teams(team_stats[team_stats["role"] == "away"], top_n=10)
    print(top_away.to_string(index=False))

    # ── Conclusión sobre feature ──────────────────────────────
    print("\n" + "─" * 60)
    print("💡 EVALUACIÓN DE LA FEATURE home/away_win_rate_at_hour")
    print("─" * 60)
    global_var = hour_stats["home_win_rate"].std()
    print(f"   Desviación estándar del home win rate entre horas: {global_var:.3f}")
    if global_var > 0.03:
        print("   ✅ Variación notable → la feature tiene señal potencial")
    else:
        print("   ⚠️  Variación baja → señal débil, evaluar con cautela")

    if not args.no_plots:
        print("\n🖼️  Generando gráficos...")
        plot_global_hour(hour_stats, output_dir)
        plot_home_win_rate_by_league_and_hour(df, output_dir)
        plot_slot_distribution(df, output_dir)
        plot_goals_by_hour(df, output_dir)
        for league_name in df["league_name"].unique():
            plot_heatmap_team_hour(team_stats, "home", league_name, output_dir)
            plot_heatmap_team_hour(team_stats, "away", league_name, output_dir)
        print(f"\n   Gráficos guardados en: {output_dir.resolve()}")

    print("\n✅ Análisis completado.\n")


if __name__ == "__main__":
    main()

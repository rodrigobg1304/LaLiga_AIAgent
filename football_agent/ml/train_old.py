import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from football_agent.db import run_query, TABLE, get_standings

OVER_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]
ELO_DIFF_THRESHOLD = 20

FEATURES = [
    "Goals", "Expected goals", "Total shots", "Shots on target",
    "Big chances", "Ball possession", "Accurate passes", "Passes",
    "Total saves", "Goalkeeper saves", "Clearances", "Interceptions",
    "Tackles won", "Fouls", "Duels"
]

LEAGUE_NAMES = {"8": "laliga", "17": "premier", "23": "seriea"}


def load_match_stats() -> pd.DataFrame:
    placeholders = ",".join(["%s"] * len(FEATURES))
    sql = f"""
        SELECT matchId, homeTeam, awayTeam, Year, Round, leagueId,
               name,
               SUM(CAST(homeValue AS DECIMAL)) AS homeValue,
               SUM(CAST(awayValue AS DECIMAL)) AS awayValue
        FROM {TABLE}
        WHERE name IN ({placeholders})
        GROUP BY matchId, homeTeam, awayTeam, Year, Round, leagueId, name
        ORDER BY Year, CAST(Round AS SIGNED)
    """
    rows = run_query(sql, tuple(FEATURES))
    return pd.DataFrame(rows)


def build_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["matchId", "homeTeam", "awayTeam", "Year", "Round", "leagueId"],
        columns="name",
        values=["homeValue", "awayValue"],
        aggfunc="first"
    ).reset_index()

    wide.columns = [
        f"{col[1]}_{col[0]}" if col[1] else col[0]
        for col in wide.columns
    ]

    wide["home_goals"]  = wide.get("Goals_homeValue", 0)
    wide["away_goals"]  = wide.get("Goals_awayValue", 0)
    wide["total_goals"] = wide["home_goals"] + wide["away_goals"]
    wide["resultado"]   = np.where(
        wide["home_goals"] > wide["away_goals"], "1",
        np.where(wide["home_goals"] < wide["away_goals"], "2", "X")
    )
    for t in OVER_THRESHOLDS:
        col = f"over_{str(t).replace('.', '_')}"
        wide[col] = (wide["total_goals"] > t).astype(int)

    return wide


def get_proxy_teams(team, standings_df, league_id, year):
    """
    Obtiene equipos vecinos en la tabla para usar como proxy.
    - Equipos a ±2 posiciones de distancia
    - Si es líder (pos 1): 3 equipos por debajo
    - Si es colista (última pos): 3 equipos por arriba

    Returns: lista de equipos vecinos
    """
    standings_df = pd.DataFrame(standings_df)

    team_row = standings_df[
        (standings_df["team"] == team) &
        (standings_df["leagueId"] == league_id) &
        (standings_df["year"] == year)
        ]

    if team_row.empty:
        return []

    position = team_row.iloc[0]["position"]
    max_position = standings_df[
        (standings_df["leagueId"] == league_id) &
        (standings_df["year"] == year)
        ]["position"].max()

    # Determinar rango de vecinos
    if position == 1:  # Líder
        lower = position
        upper = position + 3
    elif position == max_position:  # Colista
        lower = position - 3
        upper = position
    else:  # Normal
        lower = position - 2
        upper = position + 2

    neighbors = standings_df[
        (standings_df["leagueId"] == league_id) &
        (standings_df["year"] == year) &
        (standings_df["position"] >= lower) &
        (standings_df["position"] <= upper) &
        (standings_df["team"] != team)
        ]["team"].tolist()

    return neighbors


def compute_team_rolling_avg(wide: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    wide = wide.sort_values(["Year", "Round"]).reset_index(drop=True)
    records = []

    for _, row in wide.iterrows():
        home = row["homeTeam"]
        away = row["awayTeam"]
        year = row["Year"]
        rnd  = row["Round"]
        league_id = row["leagueId"]

        def before(df):
            return df[
                (df["Year"] < year) | ((df["Year"] == year) & (df["Round"] < rnd))
            ]

        home_prev = before(wide[wide["homeTeam"] == home]).tail(n)
        away_prev = before(wide[wide["awayTeam"] == away]).tail(n)
        home_all  = before(wide[(wide["homeTeam"] == home)])
        away_all  = before(wide[(wide["awayTeam"] == away)])
        h2h       = wide[
            ((wide["homeTeam"] == home) & (wide["awayTeam"] == away))
        ].tail(5)

        if len(h2h) <= 2 and year == "25/26":
            print(f"DEBUG: Entrando en h2h con equipos {home} y {away}. Temporada {year}")
            year_temp = "25/26"
            standings_temp = get_standings(leagueId=league_id, year=year_temp)
            home_neighbours = get_proxy_teams(team=home, standings_df=standings_temp, league_id=league_id,
                                              year=year_temp)
            away_neighbours = get_proxy_teams(team=away, standings_df=standings_temp, league_id=league_id,
                                              year=year_temp)

            h2h = before(wide[(wide["homeTeam"].isin(home_neighbours)) &
                              (wide["awayTeam"].isin(away_neighbours))
                         ])

        record = {
            "league_id": league_id,
            "matchId":   row["matchId"],
            "homeTeam":  home,
            "awayTeam":  away,
            "year":      year,
            "round":     rnd,
            "resultado": row["resultado"],
            "over_0_5":  row["over_0_5"],
            "over_1_5":  row["over_1_5"],
            "over_2_5":  row["over_2_5"],
            "over_3_5":  row["over_3_5"],
        }

        # ── Goles medios ──
        # Media histórica completa
        home_goals_scored_all = home_all["home_goals"].mean()
        home_goals_conceded_all = home_all["away_goals"].mean()
        away_goals_scored_all = away_all["away_goals"].mean()
        away_goals_conceded_all = away_all["home_goals"].mean()

        # Media reciente (últimos n)
        home_goals_scored_recent = home_prev["home_goals"].mean()
        home_goals_conceded_recent = home_prev["away_goals"].mean()
        away_goals_scored_recent = away_prev["away_goals"].mean()
        away_goals_conceded_recent = away_prev["home_goals"].mean()

        # Combinación ponderada: dar más peso a la tendencia reciente
        # 65% reciente, 35% histórico
        home_goals_scored   = home_goals_scored_all * 0.35 + home_goals_scored_recent * 0.65
        home_goals_conceded = home_goals_conceded_all * 0.35 + home_goals_conceded_recent * 0.65
        away_goals_scored = away_goals_scored_all * 0.35 + away_goals_scored_recent * 0.65
        away_goals_conceded = away_goals_conceded_all * 0.35 + away_goals_conceded_recent * 0.65

        record["home_avg_goals_scored"]   = home_goals_scored
        record["home_avg_goals_conceded"] = home_goals_conceded
        record["away_avg_goals_scored"]   = away_goals_scored
        record["away_avg_goals_conceded"] = away_goals_conceded
        record["xG_match"] = (
            home_goals_scored + away_goals_conceded +
            away_goals_scored + home_goals_conceded
        ) / 2

        # ── Over rates históricas ──
        for t in OVER_THRESHOLDS:
            col = f"over_{str(t).replace('.', '_')}"
            # Home team - ponderado
            home_over_recent = (home_prev[col] == 1).mean() if col in home_prev.columns else 0
            home_over_hist = (home_all[col] == 1).mean() if col in home_all.columns else 0
            home_over = 0.65 * home_over_recent + 0.35 * home_over_hist

            # Away team - ponderado
            away_over_recent = (away_prev[col] == 1).mean() if col in away_prev.columns else 0
            away_over_hist = (away_all[col] == 1).mean() if col in away_all.columns else 0
            away_over = 0.65 * away_over_recent + 0.35 * away_over_hist

            # H2H - sin cambios por ahora
            h2h_over = (h2h[col] == 1).mean() if (col in h2h.columns and len(h2h) > 0) else 0

            record[f"home_{col}_rate"] = home_over
            record[f"away_{col}_rate"] = away_over
            record[f"h2h_{col}_rate"] = h2h_over
            record[f"combined_{col}_rate"] = (home_over + away_over) / 2

        # ── Distribución de goles (diferencia over_1.5 vs over_2.5) ──
        # Features CONTINUAS para que el modelo distinga thresholds cercanos.
        # Sin estas, home_over_1_5_rate y home_over_2_5_rate pueden ser idénticas
        # cuando el equipo marcó 2+ goles en todos sus últimos partidos.

        # Desviación estándar - ponderada
        home_goals_scored_std_recent = home_prev["home_goals"].astype(float).std() if len(home_prev) > 1 else 0
        home_goals_scored_std_hist = home_all["home_goals"].astype(float).std() if len(home_all) > 1 else 0
        record["home_goals_scored_std"] = 0.65 * home_goals_scored_std_recent + 0.35 * home_goals_scored_std_hist

        home_goals_conceded_std_recent = home_prev["away_goals"].astype(float).std() if len(home_prev) > 1 else 0
        home_goals_conceded_std_hist = home_all["away_goals"].astype(float).std() if len(home_all) > 1 else 0
        record["home_goals_conceded_std"] = 0.65 * home_goals_conceded_std_recent + 0.35 * home_goals_conceded_std_hist

        away_goals_scored_std_recent = away_prev["away_goals"].astype(float).std() if len(away_prev) > 1 else 0
        away_goals_scored_std_hist = away_all["away_goals"].astype(float).std() if len(away_all) > 1 else 0
        record["away_goals_scored_std"] = 0.65 * away_goals_scored_std_recent + 0.35 * away_goals_scored_std_hist

        away_goals_conceded_std_recent = away_prev["home_goals"].astype(float).std() if len(away_prev) > 1 else 0
        away_goals_conceded_std_hist = away_all["home_goals"].astype(float).std() if len(away_all) > 1 else 0
        record["away_goals_conceded_std"] = 0.65 * away_goals_conceded_std_recent + 0.35 * away_goals_conceded_std_hist

        for goals in [0, 1, 2, 3]:
            # Home scored
            home_scored_recent = (home_prev["home_goals"] == goals).mean()
            home_scored_hist = (home_all["home_goals"] == goals).mean()
            record[f"home_scored_{goals}_rate"] = 0.65 * home_scored_recent + 0.35 * home_scored_hist

            # Away scored
            away_scored_recent = (away_prev["away_goals"] == goals).mean()
            away_scored_hist = (away_all["away_goals"] == goals).mean()
            record[f"away_scored_{goals}_rate"] = 0.65 * away_scored_recent + 0.35 * away_scored_hist

            # Home conceded
            home_conceded_recent = (home_prev["away_goals"] == goals).mean()
            home_conceded_hist = (home_all["away_goals"] == goals).mean()
            record[f"home_conceded_{goals}_rate"] = 0.65 * home_conceded_recent + 0.35 * home_conceded_hist

            # Away conceded
            away_conceded_recent = (away_prev["home_goals"] == goals).mean()
            away_conceded_hist = (away_all["home_goals"] == goals).mean()
            record[f"away_conceded_{goals}_rate"] = 0.65 * away_conceded_recent + 0.35 * away_conceded_hist

            # expected_total_goals ya usa las variables ponderadas
        record["expected_total_goals"] = (home_goals_scored + away_goals_conceded +
                                          away_goals_scored + home_goals_conceded) / 2

        if len(home_all) > 0:
            record["home_max_total_goals"] = home_all["home_goals"].max()
            record["home_min_total_goals"] = home_all["home_goals"].min()
        else:
            record["home_max_total_goals"] = 0
            record["home_min_total_goals"] = 0

        if len(away_all) > 0:
            record["away_max_total_goals"] = away_all["home_goals"].max()
            record["away_min_total_goals"] = away_all["home_goals"].min()
        else:
            record["away_max_total_goals"] = 0
            record["away_min_total_goals"] = 0

        # TODO: HACER LUEGO CON H2H
        if len(h2h) > 0:
            h2h_totals = (h2h["home_goals"] + h2h["away_goals"]).astype(float)
            record["h2h_goals_std"]            = h2h_totals.std() if len(h2h) > 1 else 0
            record["h2h_goals_max"]            = h2h_totals.max()
            record["h2h_over_2_5_rate_strict"] = (h2h_totals > 2).mean()
            record["h2h_over_3_5_rate_strict"] = (h2h_totals > 3).mean()
        else:
            record["h2h_goals_std"]            = 0
            record["h2h_goals_max"]            = 0
            record["h2h_over_2_5_rate_strict"] = 0
            record["h2h_over_3_5_rate_strict"] = 0

        # ── xG y tiros ──
        xg_h = "Expected goals_homeValue"
        xg_a = "Expected goals_awayValue"
        record["home_avg_xG"]       = home_all[xg_h].mean() if xg_h in home_all.columns else 0
        record["away_avg_xG"]       = away_all[xg_a].mean() if xg_a in away_all.columns else 0
        record["expected_xG_match"] = record["home_avg_xG"] + record["away_avg_xG"]

        sh = "Total shots_homeValue"
        sa = "Total shots_awayValue"
        record["home_avg_shots"] = home_all[sh].mean() if sh in home_all.columns else 0
        record["away_avg_shots"] = away_all[sa].mean() if sa in away_all.columns else 0

        bch = "Big chances_homeValue"
        bca = "Big chances_awayValue"
        record["home_avg_big_chances"] = home_all[bch].mean() if bch in home_all.columns else 0
        record["away_avg_big_chances"] = away_all[bca].mean() if bca in away_all.columns else 0

        # ── Stats rolling genéricas ──
        for feat in FEATURES:
            hcol = f"{feat}_homeValue"
            acol = f"{feat}_awayValue"
            record[f"home_avg_{feat}"] = home_all[hcol].mean() if hcol in home_all.columns else 0
            record[f"away_avg_{feat}"] = away_all[acol].mean() if acol in away_all.columns else 0

        # ── Win rates como local y visitante ──
        # Ponderación invertida: 65% histórico, 35% forma reciente
        # (las rachas de victorias son más volátiles que el rendimiento de goles)

        # Home team
        home_win_recent = (home_prev["resultado"] == "1").mean()
        home_win_hist = (home_all["resultado"] == "1").mean()
        record["home_win_rate_home"] = 0.35 * home_win_recent + 0.65 * home_win_hist

        home_draw_recent = (home_prev["resultado"] == "X").mean()
        home_draw_hist = (home_all["resultado"] == "X").mean()
        record["home_draw_rate_home"] = 0.35 * home_draw_recent + 0.65 * home_draw_hist

        home_loss_recent = (home_prev["resultado"] == "2").mean()
        home_loss_hist = (home_all["resultado"] == "2").mean()
        record["home_loss_rate_home"] = 0.35 * home_loss_recent + 0.65 * home_loss_hist

        # Away team
        away_win_recent = (away_prev["resultado"] == "2").mean()
        away_win_hist = (away_all["resultado"] == "2").mean()
        record["away_win_rate_away"] = 0.35 * away_win_recent + 0.65 * away_win_hist

        away_draw_recent = (away_prev["resultado"] == "X").mean()
        away_draw_hist = (away_all["resultado"] == "X").mean()
        record["away_draw_rate_away"] = 0.35 * away_draw_recent + 0.65 * away_draw_hist

        away_loss_recent = (away_prev["resultado"] == "1").mean()
        away_loss_hist = (away_all["resultado"] == "1").mean()
        record["away_loss_rate_away"] = 0.35 * away_loss_recent + 0.65 * away_loss_hist

        # ── H2H ──
        if len(h2h) > 0:
            record["h2h_home_wins"] = (h2h["resultado"] == "1").mean()
            record["h2h_draws"]     = (h2h["resultado"] == "X").mean()
            record["h2h_away_wins"] = (h2h["resultado"] == "2").mean()
            record["h2h_avg_goals"] = (h2h["home_goals"] + h2h["away_goals"]).mean()
        else:
            record["h2h_home_wins"] = 0.33
            record["h2h_draws"]     = 0.33
            record["h2h_away_wins"] = 0.33
            record["h2h_avg_goals"] = 0

        # ── Forma reciente (últimos 5) ──
        record["home_form_pts"] = sum(
            3 if r["resultado"] == "1" else (1 if r["resultado"] == "X" else 0)
            for _, r in home_prev.iterrows()
        )
        record["away_form_pts"] = sum(
            3 if r["resultado"] == "2" else (1 if r["resultado"] == "X" else 0)
            for _, r in away_prev.iterrows()
        )

        records.append(record)

    return pd.DataFrame(records)


def compute_elo(wide: pd.DataFrame, k: int = 32, scale: int = 600, home_advantage: int = 100) -> dict:
    elo = {}
    elo_history = {}

    for _, row in wide.sort_values(["Year", "Round"]).iterrows():
        home = (row["homeTeam"], row["leagueId"])
        away = (row["awayTeam"], row["leagueId"])
        elo.setdefault(home, 1500)
        elo.setdefault(away, 1500)

        # Aplicar ventaja local
        elo_home_adjusted = elo[home] + home_advantage

        exp_home = 1 / (1 + 10 ** ((elo[away] - elo_home_adjusted) / scale))

        if row["resultado"] == "1":
            sh, sa = 1, 0
        elif row["resultado"] == "2":
            sh, sa = 0, 1
        else:
            sh, sa = 0.5, 0.5

        elo_history[row["matchId"]] = {
            "elo_home": elo[home],
            "elo_away": elo[away],
            "elo_diff": elo[home] - elo[away],
            "exp_home_win_prob": exp_home  # Añadir probabilidad
        }

        elo[home] += k * (sh - exp_home)
        elo[away] += k * (sa - (1 - exp_home))

    return elo_history


class HybridCalibratedModel:
    def __init__(self, base_model, regressors, classes):
        self.base_model = base_model
        self.regressors = regressors
        self.classes_   = classes

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)
        cal = np.column_stack([
            self.regressors[i].predict(raw[:, i])
            for i in range(len(self.regressors))
        ])
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return cal / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def temporal_split(data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    data_sorted  = data.sort_values(["year", "round"]).reset_index(drop=True)
    n            = len(data_sorted)
    train_end    = int(n * train_ratio)
    val_end      = int(n * (train_ratio + val_ratio))

    train = data_sorted.iloc[:train_end]
    val   = data_sorted.iloc[train_end:val_end]
    test  = data_sorted.iloc[val_end:]

    print(f"  Split temporal → train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test


def safe_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    counts = pd.Series(y_train).value_counts()
    min_count = counts.min()

    if min_count < 6:
        print(f"  ⚠️  Clase minoritaria con {min_count} muestras — SMOTE omitido")
        return X_train, y_train

    sm = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    print(f"  SMOTE: {len(X_train)} → {len(X_bal)} muestras de entrenamiento")
    return X_bal, y_bal


def enforce_monotonicity(over_probs: dict) -> dict:
    result   = {}
    prev_val = 1.0
    for t in sorted(over_probs.keys()):
        val        = min(over_probs[t], prev_val)
        result[t]  = val
        prev_val   = val
    return result


def calibrate_over_model(model, X_val: pd.DataFrame, y_val: pd.Series):
    proba_val = model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_val, y_val)
    return iso


def print_over_distribution(wide: pd.DataFrame):
    print("\n📊 Distribución Over/Under en el dataset:")
    total = len(wide)
    for t in OVER_THRESHOLDS:
        col   = f"over_{str(t).replace('.', '_')}"
        count = wide[col].sum()
        pct   = count / total * 100
        print(f"  Over {t}: {count}/{total} partidos ({pct:.1f}%)")
        if pct < 25:
            print(f"  ⚠️  Clase minoritaria (<25%) — usar class_weight='balanced'")


def train():
    print("📥 Cargando datos...")
    df   = load_match_stats()
    wide = build_wide_format(df)

    print_over_distribution(wide)

    print("\n⚙️  Calculando rolling features...")
    data = compute_team_rolling_avg(wide)
    print(f"✅ Dataset: {len(data)} partidos con histórico suficiente")

    elo_history = compute_elo(wide)
    elo_df = pd.DataFrame.from_dict(elo_history, orient="index").reset_index()
    elo_df.columns = ["matchId", "elo_home", "elo_away", "elo_diff", "exp_home_win_prob"]
    data = data.merge(elo_df, on="matchId", how="left")

    # Features nuevas de distribución de goles
    extra_over_features = [
        "home_goals_scored_std", "home_goals_conceded_std",
        "away_goals_scored_std", "away_goals_conceded_std",
        "expected_total_goals",
        "home_max_total_goals", "home_min_total_goals",
        "away_max_total_goals", "away_min_total_goals",
        "h2h_goals_std", "h2h_goals_max",
        "h2h_over_2_5_rate_strict", "h2h_over_3_5_rate_strict",
        *[f"home_scored_{g}_rate"   for g in [0, 1, 2, 3]],
        *[f"away_scored_{g}_rate"   for g in [0, 1, 2, 3]],
        *[f"home_conceded_{g}_rate" for g in [0, 1, 2, 3]],
        *[f"away_conceded_{g}_rate" for g in [0, 1, 2, 3]],
    ]

    feature_cols = [c for c in data.columns if
                    c.startswith("home_avg_") or
                    c.startswith("away_avg_") or
                    c.startswith("home_") or
                    c.startswith("away_") or
                    c in ["elo_home", "elo_away", "elo_diff",
                          "home_form_pts", "away_form_pts",
                          "h2h_home_wins", "h2h_draws",
                          "h2h_away_wins", "h2h_avg_goals",
                          "combined_over_0_5_rate", "combined_over_1_5_rate",
                          "combined_over_2_5_rate", "combined_over_3_5_rate",
                          "xG_match", "expected_xG_match"]]

    # Añadir extras que no caigan ya en el filtro de home_/away_
    for f in extra_over_features:
        if f in data.columns and f not in feature_cols:
            feature_cols.append(f)

    for league_id, league_name in LEAGUE_NAMES.items():
        league_data = data[data["league_id"] == int(league_id)].copy()
        if len(league_data) < 50:
            print(f"\n⚠️  {league_name}: datos insuficientes ({len(league_data)}), saltando.")
            continue

        print(f"\n{'=' * 55}")
        print(f"  Entrenando {league_name.upper()} — {len(league_data)} partidos")
        print(f"{'=' * 55}")

        train_data, val_data, test_data = temporal_split(league_data)

        X_train = train_data[feature_cols].fillna(0).astype(float)
        X_val   = val_data[feature_cols].fillna(0).astype(float)
        X_test  = test_data[feature_cols].fillna(0).astype(float)

        # ── Modelo resultado ──
        le       = LabelEncoder()
        y_all    = le.fit_transform(league_data["resultado"])
        y_train  = le.transform(train_data["resultado"])
        y_val_r  = le.transform(val_data["resultado"])
        y_test_r = le.transform(test_data["resultado"])

        X_train_bal, y_train_bal = safe_smote(X_train, y_train)

        model_result = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss"
        )
        model_result.fit(X_train_bal, y_train_bal)

        y_pred_base = model_result.predict(X_test)
        print(f"\n🎯 Resultado (base) — Accuracy: {accuracy_score(y_test_r, y_pred_base):.2%}")
        print(classification_report(y_test_r, y_pred_base, target_names=le.classes_))

        proba_val  = model_result.predict_proba(X_val)
        n_classes  = len(le.classes_)
        iso_regs   = []
        for i in range(n_classes):
            target_i = (y_val_r == i).astype(int)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(proba_val[:, i], target_i)
            iso_regs.append(iso)

        calibrated_model = HybridCalibratedModel(model_result, iso_regs, le.classes_)

        y_pred_cal    = calibrated_model.predict(X_test)
        y_test_labels = le.inverse_transform(y_test_r)
        print(f"\n✅ Resultado (calibrado) — Accuracy: {accuracy_score(y_test_labels, y_pred_cal):.2%}")
        print(classification_report(y_test_labels, y_pred_cal, target_names=le.classes_))

        os.makedirs("ml/models", exist_ok=True)
        with open(f"ml/models/model_result_{league_name}.pkl", "wb") as f:
            pickle.dump({
                "model":              model_result,
                "calibrated_model":   calibrated_model,
                "encoder":            le,
                "features":           feature_cols,
                "elo_diff_threshold": ELO_DIFF_THRESHOLD,
            }, f)

        # ── Modelos Over/Under ──
        over_models = {}
        print(f"\n⚽ Entrenando modelos Over/Under para {league_name}...")

        for t in OVER_THRESHOLDS:
            col     = f"over_{str(t).replace('.', '_')}"
            y_tr    = train_data[col].astype(int)
            y_val_o = val_data[col].astype(int)
            y_te    = test_data[col].astype(int)

            pct_pos      = y_tr.mean() * 100
            use_balanced = pct_pos < 35 or pct_pos > 65
            apply_smote  = 35 <= pct_pos <= 65

            if apply_smote:
                X_tr_bal, y_tr_bal = safe_smote(X_train, y_tr)
            else:
                X_tr_bal, y_tr_bal = X_train, y_tr
                print(f"  Over {t}: {pct_pos:.1f}% — SMOTE omitido (baseline naive más fuerte)")

            print(f"  Over {t}: {pct_pos:.1f}% positivos en train "
                  f"{'→ scale_pos_weight activo' if use_balanced else ''}")

            model_over = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=3, gamma=0.1, eval_metric="logloss",
                scale_pos_weight=(1 - y_tr.mean()) / y_tr.mean() if use_balanced else 1
            )
            model_over.fit(X_tr_bal, y_tr_bal)

            iso_over = calibrate_over_model(model_over, X_val, y_val_o)

            y_pred_over = model_over.predict(X_test)
            acc = accuracy_score(y_te, y_pred_over)
            print(f"  Over {t} — Accuracy: {acc:.2%}")

            over_models[str(t)] = {
                "model":      model_over,
                "calibrator": iso_over,
                "features":   feature_cols,
            }

        with open(f"ml/models/models_over_under_{league_name}.pkl", "wb") as f:
            pickle.dump(over_models, f)

        print(f"\n✅ Modelos {league_name} guardados.")


if __name__ == "__main__":
    train()
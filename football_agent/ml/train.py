import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from football_agent.db import run_query, TABLE

# ─────────────────────────────────────────────
# 0. DEFINIR VARIABLES
# ─────────────────────────────────────────────
OVER_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]

# ─────────────────────────────────────────────
# 1. EXTRAER DATOS
# ─────────────────────────────────────────────

FEATURES = [
    "Goals", "Expected goals", "Total shots", "Shots on target",
    "Big chances", "Ball possession", "Accurate passes", "Passes",
    "Total saves", "Goalkeeper saves", "Clearances", "Interceptions",
    "Tackles won", "Fouls", "Duels"
]

def load_match_stats() -> pd.DataFrame:
    """Carga todos los partidos con sus stats en formato wide."""
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
    """Convierte de long a wide: una fila por partido con todas las stats."""
    wide = df.pivot_table(
        index=["matchId", "homeTeam", "awayTeam", "Year", "Round", "leagueId"],
        columns="name",
        values=["homeValue", "awayValue"],
        aggfunc="first"
    ).reset_index()

    # Aplanar columnas
    wide.columns = [
        f"{col[1]}_{col[0]}" if col[1] else col[0]
        for col in wide.columns
    ]

    # Calcular resultado
    wide["home_goals"] = wide.get("Goals_homeValue", 0)
    wide["away_goals"] = wide.get("Goals_awayValue", 0)
    wide["total_goals"] = wide["home_goals"] + wide["away_goals"]
    wide["resultado"]  = np.where(
        wide["home_goals"] > wide["away_goals"], "1",
        np.where(wide["home_goals"] < wide["away_goals"], "2", "X")
    )
    # Over/Under para cada threshold
    for t in OVER_THRESHOLDS:
        col = f"over_{str(t).replace('.', '_')}"
        wide[col] = (wide["total_goals"] > t).astype(int)

    return wide


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_team_rolling_avg(wide: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Para cada partido calcula la media de las últimas N jornadas
    de cada equipo como features del modelo.
    """
    wide = wide.sort_values(["Year", "Round"])
    records = []

    for _, row in wide.iterrows():
        home = row["homeTeam"]
        away = row["awayTeam"]
        year = row["Year"]
        rnd  = row["Round"]

        # ── Partidos anteriores como local ──
        home_prev = wide[
            (wide["homeTeam"] == home) &
            ((wide["Year"] < year) | ((wide["Year"] == year) & (wide["Round"] < rnd)))
        ].tail(n)

        # ── Partidos anteriores como visitante ──
        away_prev = wide[
            (wide["awayTeam"] == away) &
            ((wide["Year"] < year) | ((wide["Year"] == year) & (wide["Round"] < rnd)))
        ].tail(n)

        # ── Partidos anteriores equipo local (en casa o fuera) ──
        home_all = wide[
            ((wide["homeTeam"] == home) | (wide["awayTeam"] == home)) &
            ((wide["Year"] < year) | ((wide["Year"] == year) & (wide["Round"] < rnd)))
        ].tail(n)

        # ── Partidos anteriores equipo visitante (en casa o fuera) ──
        away_all = wide[
            ((wide["homeTeam"] == away) | (wide["awayTeam"] == away)) &
            ((wide["Year"] < year) | ((wide["Year"] == year) & (wide["Round"] < rnd)))
        ].tail(n)

        # ── Head to head ──
        h2h = wide[
            ((wide["homeTeam"] == home) & (wide["awayTeam"] == away)) &
            ((wide["Year"] < year) | ((wide["Year"] == year) & (wide["Round"] < rnd)))
        ].tail(5)

        if len(home_prev) < 3 or len(away_prev) < 3:
            continue

        record = {
            "league_id": row['leagueId'],
            "matchId":   row["matchId"],
            "homeTeam":  home,
            "awayTeam":  away,
            "year":      year,
            "round":     rnd,
            "resultado": row["resultado"],
            "over_0_5": row["over_0_5"],
            "over_1_5": row["over_1_5"],
            "over_2_5": row["over_2_5"],
            "over_3_5": row["over_3_5"],
        }

        # ── Features Over 2.5 ──
        home_goals_scored    = home_prev["home_goals"].mean() if len(home_prev) > 0 else 0
        home_goals_conceded  = home_prev["away_goals"].mean() if len(home_prev) > 0 else 0
        away_goals_scored    = away_prev["away_goals"].mean() if len(away_prev) > 0 else 0
        away_goals_conceded  = away_prev["home_goals"].mean() if len(away_prev) > 0 else 0

        record["home_avg_goals_scored"]   = home_goals_scored
        record["home_avg_goals_conceded"] = home_goals_conceded
        record["away_avg_goals_scored"]   = away_goals_scored
        record["away_avg_goals_conceded"] = away_goals_conceded

        record["xG_match"] = (home_goals_scored + away_goals_conceded +
                              away_goals_scored + home_goals_conceded) / 2

        # Over rate histórica para cada threshold
        for t in OVER_THRESHOLDS:
            col = f"over_{str(t).replace('.', '_')}"
            if col in home_prev.columns:
                record[f"home_{col}_rate"] = (home_prev[col] == 1).mean()
            if col in away_prev.columns:
                record[f"away_{col}_rate"] = (away_prev[col] == 1).mean()
            if col in h2h.columns and len(h2h) > 0:
                record[f"h2h_{col}_rate"] = (h2h[col] == 1).mean()
            record[f"combined_{col}_rate"] = (record.get(f"home_{col}_rate", 0) + record.get(f"away_{col}_rate", 0)) / 2

        # xG medio de ambos equipos
        xg_home_col = "Expected goals_homeValue"
        xg_away_col = "Expected goals_awayValue"
        if xg_home_col in home_prev.columns:
            record["home_avg_xG"] = home_prev[xg_home_col].mean()
        if xg_away_col in away_prev.columns:
            record["away_avg_xG"] = away_prev[xg_away_col].mean()

        if xg_home_col in home_prev.columns and xg_away_col in away_prev.columns:
            record["expected_xG_match"] = (
                home_prev[xg_home_col].mean() + away_prev[xg_away_col].mean()
            )

        # Tiros totales medios
        shots_home_col = "Total shots_homeValue"
        shots_away_col = "Total shots_awayValue"
        if shots_home_col in home_prev.columns:
            record["home_avg_shots"] = home_prev[shots_home_col].mean()
        if shots_away_col in away_prev.columns:
            record["away_avg_shots"] = away_prev[shots_away_col].mean()

        # Big chances medias
        bc_home_col = "Big chances_homeValue"
        bc_away_col = "Big chances_awayValue"
        if bc_home_col in home_prev.columns:
            record["home_avg_big_chances"] = home_prev[bc_home_col].mean()
        if bc_away_col in away_prev.columns:
            record["away_avg_big_chances"] = away_prev[bc_away_col].mean()

        record["h2h_avg_goals"] = (h2h["home_goals"] + h2h["away_goals"]).mean() if len(h2h) > 0 else 0

        # ── Features stats rolling (local y visitante) ──
        for feat in FEATURES:
            hcol = f"{feat}_homeValue"
            acol = f"{feat}_awayValue"
            if hcol in home_prev.columns:
                record[f"home_avg_{feat}"] = home_prev[hcol].mean()
            if acol in away_prev.columns:
                record[f"away_avg_{feat}"] = away_prev[acol].mean()

        # ── Win rate como local (últimos N) ──
        record["home_win_rate_home"]  = (home_prev["resultado"] == "1").mean() if len(home_prev) > 0 else 0
        record["home_draw_rate_home"] = (home_prev["resultado"] == "X").mean() if len(home_prev) > 0 else 0
        record["home_loss_rate_home"] = (home_prev["resultado"] == "2").mean() if len(home_prev) > 0 else 0

        # ── Win rate como visitante (últimos N) ──
        record["away_win_rate_away"]  = (away_prev["resultado"] == "2").mean() if len(away_prev) > 0 else 0
        record["away_draw_rate_away"] = (away_prev["resultado"] == "X").mean() if len(away_prev) > 0 else 0
        record["away_loss_rate_away"] = (away_prev["resultado"] == "1").mean() if len(away_prev) > 0 else 0

        # ── Win rate global (local + visitante combinado) ──
        if len(home_all) > 0:
            home_wins_all = sum(
                1 for _, r in home_all.iterrows()
                if (r["homeTeam"] == home and r["resultado"] == "1") or
                   (r["awayTeam"] == home and r["resultado"] == "2")
            )
            record["home_win_rate_global"] = home_wins_all / len(home_all)
            record["home_avg_goals_scored_global"] = sum(
                r["home_goals"] if r["homeTeam"] == home else r["away_goals"]
                for _, r in home_all.iterrows()
            ) / len(home_all)
            record["home_avg_goals_conceded_global"] = sum(
                r["away_goals"] if r["homeTeam"] == home else r["home_goals"]
                for _, r in home_all.iterrows()
            ) / len(home_all)

        if len(away_all) > 0:
            away_wins_all = sum(
                1 for _, r in away_all.iterrows()
                if (r["homeTeam"] == away and r["resultado"] == "1") or
                   (r["awayTeam"] == away and r["resultado"] == "2")
            )
            record["away_win_rate_global"] = away_wins_all / len(away_all)
            record["away_avg_goals_scored_global"] = sum(
                r["home_goals"] if r["homeTeam"] == away else r["away_goals"]
                for _, r in away_all.iterrows()
            ) / len(away_all)
            record["away_avg_goals_conceded_global"] = sum(
                r["away_goals"] if r["homeTeam"] == away else r["home_goals"]
                for _, r in away_all.iterrows()
            ) / len(away_all)

        # ── Head to head ──
        if len(h2h) > 0:
            record["h2h_home_wins"] = (h2h["resultado"] == "1").mean()
            record["h2h_draws"]     = (h2h["resultado"] == "X").mean()
            record["h2h_away_wins"] = (h2h["resultado"] == "2").mean()
            record["h2h_avg_goals"] = (h2h["home_goals"] + h2h["away_goals"]).mean()
        else:
            record["h2h_home_wins"] = 0
            record["h2h_draws"]     = 0
            record["h2h_away_wins"] = 0
            record["h2h_avg_goals"] = 0

        # ── Forma reciente (últimos 5, independiente de N) ──
        home_form = wide[
            (wide["homeTeam"] == home) &
            ((wide["Year"] < year) | ((wide["Year"] == year) & (wide["Round"] < rnd)))
        ].tail(5)
        away_form = wide[
            (wide["awayTeam"] == away) &
            ((wide["Year"] < year) | ((wide["Year"] == year) & (wide["Round"] < rnd)))
        ].tail(5)

        record["home_form_pts"] = sum(
            3 if r["resultado"] == "1" else (1 if r["resultado"] == "X" else 0)
            for _, r in home_form.iterrows()
        )
        record["away_form_pts"] = sum(
            3 if r["resultado"] == "2" else (1 if r["resultado"] == "X" else 0)
            for _, r in away_form.iterrows()
        )

        records.append(record)

    return pd.DataFrame(records)


def compute_elo(wide: pd.DataFrame, k: int = 20) -> dict:
    """Calcula el Elo de cada equipo partido a partido."""
    elo = {}
    elo_history = {}

    for _, row in wide.sort_values(["Year", "Round"]).iterrows():
        home = (row["homeTeam"], row["leagueId"])
        away = (row["awayTeam"], row["leagueId"])

        elo.setdefault(home, 1500)
        elo.setdefault(away, 1500)

        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 - exp_home

        if row["resultado"] == "1":
            score_home, score_away = 1, 0
        elif row["resultado"] == "2":
            score_home, score_away = 0, 1
        else:
            score_home, score_away = 0.5, 0.5

        # Guardar Elo ANTES del partido como feature
        elo_history[row["matchId"]] = {
            "elo_home": elo[home],
            "elo_away": elo[away],
            "elo_diff": elo[home] - elo[away]
        }

        elo[home] += k * (score_home - exp_home)
        elo[away] += k * (score_away - exp_away)

    return elo_history

# ─────────────────────────────────────────────
# 3. CALIBRACIÓN ISOTÓNICA
# ─────────────────────────────────────────────

class HybridCalibratedModel:
    """
    Wrapper de calibración isotónica manual compatible con todas las versiones de sklearn.
    Alternativa a CalibratedClassifierCV(cv="prefit") que no está disponible en versiones antiguas.
    Entrena un IsotonicRegression por clase sobre las probabilidades raw del modelo base,
    sin reentrenar el XGBoost subyacente.
    """
    def __init__(self, base_model, regressors, classes):
        self.base_model = base_model
        self.regressors = regressors
        self.classes_   = classes

    def predict_proba(self, X):
        import numpy as np
        raw = self.base_model.predict_proba(X)
        cal = np.column_stack([
            self.regressors[i].predict(raw[:, i])
            for i in range(len(self.regressors))
        ])
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return cal / row_sums

    def predict(self, X):
        import numpy as np
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ─────────────────────────────────────────────
# 4. ENTRENAR MODELOS
# ─────────────────────────────────────────────

# Umbral de elo_diff para considerar un partido como favorito claro
ELO_DIFF_THRESHOLD = 20

def train():
    print("📥 Cargando datos...")
    df   = load_match_stats()
    wide = build_wide_format(df)

    # ── CORRECCIÓN: compute_team_rolling_avg se llama UNA sola vez ──
    data = compute_team_rolling_avg(wide)
    print(f"✅ Dataset: {len(data)} partidos con histórico suficiente")

    # Añadir Elo
    elo_history = compute_elo(wide)
    elo_df = pd.DataFrame.from_dict(elo_history, orient="index").reset_index()
    elo_df.columns = ["matchId", "elo_home", "elo_away", "elo_diff"]
    data = data.merge(elo_df, on="matchId", how="left")

    feature_cols = [c for c in data.columns if
                    c.startswith("home_avg_") or
                    c.startswith("away_avg_") or
                    c.startswith("home_") or
                    c.startswith("away_") or
                    c in ["elo_home", "elo_away", "elo_diff",
                          "home_form_pts", "away_form_pts",
                          "h2h_home_wins", "h2h_draws",
                          "h2h_away_wins", "h2h_avg_goals",
                          "combined_0_5_rate", "combined_1_5_rate",
                          "combined_2_5_rate", "combined_3_5_rate",
                          "xG_match", "expected_xG_match"]]

    LEAGUE_NAMES = {"8": "laliga", "17": "premier", "23": "seriea"}

    for league_id, league_name in LEAGUE_NAMES.items():
        league_data = data[data["league_id"] == int(league_id)].copy()
        print(f"\n{'=' * 55}")
        print(f"  Entrenando {league_name.upper()} — {len(league_data)} partidos")
        print(f"{'=' * 55}")

        X = league_data[feature_cols].fillna(0).astype(float)

        # ── Modelo resultado ──
        le = LabelEncoder()
        y_result = le.fit_transform(league_data["resultado"])

        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X, y_result)

        # Split 70% train / 15% val / 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_bal, y_bal, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # ── 1. Entrenar XGBoost base ──
        model_result = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss"
        )
        model_result.fit(X_train, y_train)

        y_pred_base = model_result.predict(X_test)
        print(f"\n🎯 Resultado (base) — Accuracy: {accuracy_score(y_test, y_pred_base):.2%}")
        print(classification_report(y_test, y_pred_base, target_names=le.classes_))

        # ── 2. Calibrar sobre el conjunto de validación ──
        # Usamos un pipeline manual para no reentrenar el XGBoost base:
        # obtenemos las probabilidades raw del modelo ya entrenado sobre X_val,
        # y entrenamos un IsotonicRegression por clase para mapearlas a probs calibradas.
        from sklearn.isotonic import IsotonicRegression

        proba_val = model_result.predict_proba(X_val)
        n_classes = len(le.classes_)

        iso_regressors = []
        for i in range(n_classes):
            target_i = (y_val == i).astype(int)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(proba_val[:, i], target_i)
            iso_regressors.append(iso)

        calibrated_model = HybridCalibratedModel(model_result, iso_regressors, le.classes_)

        y_pred_cal = calibrated_model.predict(X_test)
        y_test_labels = le.inverse_transform(y_test)
        print(f"\n Resultado (calibrado) - Accuracy: {accuracy_score(y_test_labels, y_pred_cal):.2%}")
        print(classification_report(y_test_labels, y_pred_cal, target_names=le.classes_))
        # ── 3. Guardar: modelo base + calibrado + encoder + features + umbral elo ──
        os.makedirs("ml/models", exist_ok=True)
        with open(f"ml/models/model_result_{league_name}.pkl", "wb") as f:
            pickle.dump({
                "model":            model_result,       # XGBoost sin calibrar
                "calibrated_model": calibrated_model,   # XGBoost + capa isotonic
                "encoder":          le,
                "features":         feature_cols,
                "elo_diff_threshold": ELO_DIFF_THRESHOLD  # umbral para confidence
            }, f)

        # ── Modelos Over/Under ──
        over_models = {}
        for t in OVER_THRESHOLDS:
            col = f"over_{str(t).replace('.', '_')}"
            y_over = league_data[col].astype(int)

            sm2 = SMOTE(random_state=42)
            X_bal2, y_bal2 = sm2.fit_resample(X, y_over)
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X_bal2, y_bal2, test_size=0.2, random_state=42
            )

            model_over = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=3, gamma=0.1, eval_metric="logloss"
            )
            model_over.fit(X_train2, y_train2)
            y_pred2 = model_over.predict(X_test2)

            acc = accuracy_score(y_test2, y_pred2)
            print(f"⚽ Over {t} — Accuracy: {acc:.2%}")

            over_models[str(t)] = {"model": model_over, "features": feature_cols}

        with open(f"ml/models/models_over_under_{league_name}.pkl", "wb") as f:
            pickle.dump(over_models, f)

        print(f"\n✅ Modelos {league_name} guardados")


if __name__ == "__main__":
    train()
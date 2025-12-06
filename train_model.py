# ========================= train_model.py — vEnsemble_ADPoisson ==========================
# Objectif :
#   - prédire le total de buts
#   - baseline Poisson attaque/défense (non entraînée)
#   - modèle global ML (CatBoost + HGB)
#   - ensemble final = combinaison optimale (ML + Poisson + xg_exp_total)
#   - backtest chrono + test final, sauvegarde artefacts
# ================================================================================

# -------------------------- CONFIG --------------------------

import os
import json
from decimal import Decimal
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "model_files")  # <- même nom que ton dossier GitHub
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------- UTILS ---------------------------

def np_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_block(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": np_rmse(y_true, y_pred),
        "n": int(len(y_true)),
    }


def best_ensemble_weights(
    y: np.ndarray,
    pred_ml: np.ndarray,
    pred_poisson: np.ndarray,
    xg_exp: np.ndarray,
    grid: List[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Cherche (w1, w2, w3) qui minimisent la MAE de :
        pred = w1 * pred_ml + w2 * pred_poisson + w3 * xg_exp
    via une petite grid-search sur les poids.
    Les poids sont normalisés pour que w1 + w2 + w3 = 1.
    """
    if grid is None:
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    best_mae = 1e9
    best_weights = (1.0, 0.0, 0.0)

    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                if w1 + w2 + w3 == 0:
                    continue
                s = w1 + w2 + w3
                w1n, w2n, w3n = w1 / s, w2 / s, w3 / s
                pred = w1n * pred_ml + w2n * pred_poisson + w3n * xg_exp
                mae = mean_absolute_error(y, pred)
                if mae < best_mae:
                    best_mae = mae
                    best_weights = (w1n, w2n, w3n)

    return best_weights[0], best_weights[1], best_weights[2], float(best_mae)


# ----------------------- DATA LOADING -----------------------

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

query = """
    SELECT
        m.game_id,
        m.date::date AS date_match,
        m.competition,
        m.equipe_domicile,
        m.equipe_exterieur,

        sg1.moyenne_xg_dom AS xg_dom,
        sg1.tirs           AS tirs_dom,

        sg2.moyenne_xg_ext AS xg_ext,
        sg2.tirs           AS tirs_ext,

        s.buts_dom,
        s.buts_ext,
        s.buts_dom + s.buts_ext AS total_buts
    FROM matchs_v2 m
    JOIN stats_globales_v2 sg1
      ON m.equipe_domicile = sg1.equipe
     AND m.competition = sg1.competition
     AND m.saison = sg1.saison
    JOIN stats_globales_v2 sg2
      ON m.equipe_exterieur = sg2.equipe
     AND m.competition = sg2.competition
     AND m.saison = sg2.saison
    JOIN stats_matchs_v2 s
      ON m.game_id = s.game_id
    WHERE s.buts_dom IS NOT NULL
      AND s.buts_ext IS NOT NULL
"""
cur.execute(query)
rows = cur.fetchall()
cols = [col.name for col in cur.description]
conn.close()

df = pd.DataFrame(rows, columns=cols)

# Convertir Decimal → float si besoin
for c in df.columns:
    if df[c].notna().any() and isinstance(df[c].dropna().iloc[0], Decimal):
        df[c] = df[c].astype(float)

df["date_match"] = pd.to_datetime(df["date_match"])

# (Optionnel) filtrer sur dernières saisons
# df = df[df["date_match"] >= pd.to_datetime("2019-08-01")].reset_index(drop=True)


# ----------------------- FEATURE ENGINEERING -----------------------

df_hist = df[[
    "date_match", "competition",
    "equipe_domicile", "equipe_exterieur",
    "buts_dom", "buts_ext"
]].copy()
df_hist["total_buts"] = df_hist["buts_dom"] + df_hist["buts_ext"]


def forme_split(equipe: str, date_ref, role: str, n: int = 5):
    """
    Forme buts marqués / encaissés sur les n derniers matchs avant date_ref.
    role = "home" ou "away"
    """
    q = (df_hist["date_match"] < date_ref)

    if role == "home":
        q &= (df_hist["equipe_domicile"] == equipe)
    else:
        q &= (df_hist["equipe_exterieur"] == equipe)

    m = df_hist.loc[q].sort_values("date_match", ascending=False).head(n)
    if len(m) == 0:
        return 0.0, 0.0

    if role == "home":
        bm = m["buts_dom"].values
        be = m["buts_ext"].values
    else:
        bm = m["buts_ext"].values
        be = m["buts_dom"].values

    return float(np.mean(bm)), float(np.mean(be))


def league_avg_goals(competition: str, date_ref, window: int = 60) -> float:
    """
    Moyenne de buts par match dans la ligue sur les 'window' derniers matchs avant date_ref.
    """
    q = (df_hist["competition"] == competition) & (df_hist["date_match"] < date_ref)
    m = df_hist.loc[q].sort_values("date_match", ascending=False).head(window)
    if len(m) == 0:
        return 2.5
    return float(m["total_buts"].mean())


FEATURES: List[str] = [
    "forme_home_buts_marques",
    "forme_home_buts_encaisses",
    "forme_away_buts_marques",
    "forme_away_buts_encaisses",
    "attaque_home",
    "defense_home",
    "attaque_away",
    "defense_away",
    "xg_dom",
    "xg_ext",
    "tirs_dom",
    "tirs_ext",
    "league_avg_goals_60d",
    "xg_exp_total",
]

records: List[Dict[str, Any]] = []

for _, r in df.sort_values("date_match").iterrows():
    dom = r["equipe_domicile"]
    ext = r["equipe_exterieur"]
    dref = r["date_match"]
    comp = r["competition"]

    bm_home, be_home = forme_split(dom, dref, role="home")
    bm_away, be_away = forme_split(ext, dref, role="away")

    attaque_home = bm_home
    defense_home = be_home
    attaque_away = bm_away
    defense_away = be_away

    xg_exp_total = attaque_home * defense_away + attaque_away * defense_home

    league_avg = league_avg_goals(comp, dref, window=60)

    rec = {
        "game_id": r["game_id"],
        "date_match": dref,
        "competition": comp,
        "y_total": float(r["total_buts"]),

        "forme_home_buts_marques": bm_home,
        "forme_home_buts_encaisses": be_home,
        "forme_away_buts_marques": bm_away,
        "forme_away_buts_encaisses": be_away,

        "attaque_home": attaque_home,
        "defense_home": defense_home,
        "attaque_away": attaque_away,
        "defense_away": defense_away,

        "xg_dom": float(r["xg_dom"] or 0.3),
        "xg_ext": float(r["xg_ext"] or 0.3),

        "tirs_dom": float(r["tirs_dom"] or 0.0),
        "tirs_ext": float(r["tirs_ext"] or 0.0),

        "league_avg_goals_60d": league_avg,
        "xg_exp_total": xg_exp_total,
    }

    records.append(rec)

dfX = pd.DataFrame(records)

# clamp tirs (anti-outliers)
dfX["tirs_dom"] = dfX["tirs_dom"].clip(0, 25)
dfX["tirs_ext"] = dfX["tirs_ext"].clip(0, 25)

X_all = dfX[FEATURES].astype(float).values
y_all = dfX["y_total"].astype(float).values
dates = dfX["date_match"].values
xg_exp_vec = dfX["xg_exp_total"].astype(float).values


# ----------------------------- BASELINE POISSON -----------------------------

def baseline_poisson_row(row: pd.Series) -> float:
    """
    Poisson attaque/défense simple :
        λ_home ≈ (bm_home + be_away) / 2
        λ_away ≈ (bm_away + be_home) / 2
    """
    bm_home = row["forme_home_buts_marques"]
    be_home = row["forme_home_buts_encaisses"]
    bm_away = row["forme_away_buts_marques"]
    be_away = row["forme_away_buts_encaisses"]

    lam_home = 0.5 * (bm_home + be_away)
    lam_away = 0.5 * (bm_away + be_home)
    lam_total = max(0.0, lam_home + lam_away)
    return lam_total

baseline_all = dfX.apply(baseline_poisson_row, axis=1).values


# --------------------------- SPLITS CHRONO ---------------------------

order = np.argsort(dates)
X_all = X_all[order]
y_all = y_all[order]
xg_exp_vec = xg_exp_vec[order]
baseline_all = baseline_all[order]
dfX = dfX.iloc[order].reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=5)
folds = list(tscv.split(X_all))

trainval_idx, test_idx = folds[-1]
X_trainval, y_trainval = X_all[trainval_idx], y_all[trainval_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
baseline_trainval = baseline_all[trainval_idx]
baseline_test = baseline_all[test_idx]
xg_exp_trainval = xg_exp_vec[trainval_idx]
xg_exp_test = xg_exp_vec[test_idx]


# --------------------------- BACKTEST (GLOBAL ML) ---------------------------

def backtest_ml_global():
    rows = []

    for fold_i, (tr, val) in enumerate(folds[:-1]):  # on exclut le dernier (test final)
        X_tr, X_val = X_all[tr], X_all[val]
        y_tr, y_val = y_all[tr], y_all[val]
        base_val = baseline_all[val]

        # CatBoost
        model_cat = CatBoostRegressor(
            depth=6,
            learning_rate=0.06,
            iterations=700,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        model_cat.fit(X_tr, y_tr)
        pred_cat_val = model_cat.predict(X_val)

        # HGB
        model_hgb = HistGradientBoostingRegressor(
            max_depth=6,
            max_leaf_nodes=31,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )
        model_hgb.fit(X_tr, y_tr)
        pred_hgb_val = model_hgb.predict(X_val)

        pred_ml_val = 0.5 * (pred_cat_val + pred_hgb_val)

        rows.append({
            "fold": fold_i,
            "baseline": metrics_block(y_val, base_val),
            "ml_global": metrics_block(y_val, pred_ml_val),
        })

    def avg(model_key: str, metric_key: str) -> float:
        vals = [r[model_key][metric_key] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "baseline": {
            "mae": avg("baseline", "mae"),
            "rmse": avg("baseline", "rmse"),
        },
        "ml_global": {
            "mae": avg("ml_global", "mae"),
            "rmse": avg("ml_global", "rmse"),
        },
        "n_folds": len(rows),
    }

    return rows, summary


back_rows, back_summary = backtest_ml_global()

print("\n=== Backtest ML Global (moyenne des folds, hors test final) ===")
print(json.dumps(back_summary, indent=2, ensure_ascii=False))


# --------------------------- TRAIN FINAL (ML GLOBAL) ---------------------------

# CatBoost final
model_cat_final = CatBoostRegressor(
    depth=6,
    learning_rate=0.06,
    iterations=900,
    loss_function="RMSE",
    random_seed=42,
    verbose=False,
)
model_cat_final.fit(X_trainval, y_trainval)

# HGB final
model_hgb_final = HistGradientBoostingRegressor(
    max_depth=6,
    max_leaf_nodes=31,
    learning_rate=0.05,
    min_samples_leaf=20,
    random_state=42,
)
model_hgb_final.fit(X_trainval, y_trainval)

pred_cat_trainval = model_cat_final.predict(X_trainval)
pred_hgb_trainval = model_hgb_final.predict(X_trainval)
pred_ml_trainval = 0.5 * (pred_cat_trainval + pred_hgb_trainval)


# --------------------------- ENSEMBLE WEIGHTS ---------------------------

w1, w2, w3, mae_ensemble_trainval = best_ensemble_weights(
    y_trainval,
    pred_ml_trainval,
    baseline_trainval,
    xg_exp_trainval,
    grid=[0.0, 0.25, 0.5, 0.75, 1.0],
)

print("\n=== Poids ensemble optimaux sur trainval ===")
print(json.dumps({
    "w_ml": w1,
    "w_poisson": w2,
    "w_xg_exp": w3,
    "mae_ensemble_trainval": mae_ensemble_trainval,
}, indent=2, ensure_ascii=False))


# --------------------------- TEST FINAL ---------------------------

pred_cat_test = model_cat_final.predict(X_test)
pred_hgb_test = model_hgb_final.predict(X_test)
pred_ml_test = 0.5 * (pred_cat_test + pred_hgb_test)

pred_ensemble_test = w1 * pred_ml_test + w2 * baseline_test + w3 * xg_exp_test

test_metrics = {
    "baseline": metrics_block(y_test, baseline_test),
    "ml_global": metrics_block(y_test, pred_ml_test),
    "ensemble": metrics_block(y_test, pred_ensemble_test),
}

print("\n=== Test final ===")
print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


# --------------------------- SAVE ARTEFACTS ---------------------------

joblib.dump(model_cat_final, os.path.join(OUT_DIR, "model_cat_total_goals.pkl"))
joblib.dump(model_hgb_final, os.path.join(OUT_DIR, "model_hgb_total_goals.pkl"))

with open(os.path.join(OUT_DIR, "FEATURES_TOTAL_BUTS.json"), "w", encoding="utf-8") as f:
    json.dump(FEATURES, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "ensemble_weights_and_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "weights": {
                "w_ml": w1,
                "w_poisson": w2,
                "w_xg_exp": w3,
            },
            "backtest_summary": back_summary,
            "mae_ensemble_trainval": mae_ensemble_trainval,
            "test_final": test_metrics,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n=== DONE: modèles & métriques sauvegardés dans", OUT_DIR, "===")
print("Contenu du dossier :", os.listdir(OUT_DIR))
